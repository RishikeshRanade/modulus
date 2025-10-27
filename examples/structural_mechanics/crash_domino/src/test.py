# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This code defines a distributed pipeline for testing the DoMINO model on
CFD datasets. It includes the instantiating the DoMINO model and datapipe,
automatically loading the most recent checkpoint, reading the VTP/VTU/STL
testing files, calculation of parameters required for DoMINO model and
evaluating the model in parallel using DistributedDataParallel across multiple
GPUs. This is a common recipe that enables training of combined models for surface
and volume as well either of them separately. The model predictions are loaded in
the the VTP/VTU files and saved in the specified directory. The eval tab in
config.yaml can be used to specify the input and output directories.
"""

import os, re
import time

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# This will set up the cupy-ecosystem and pytorch to share memory pools
from physicsnemo.utils.memory import unified_gpu_memory

import numpy as np
import cupy as cp

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable

import pandas as pd
import pyvista as pv

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

import vtk
from vtk.util import numpy_support

from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.cae.domino_crash_datapipe import DoMINODataPipe
from physicsnemo.models.domino_crash.model import DoMINO
from physicsnemo.models.domino_crash.geometry_rep import scale_sdf
from physicsnemo.utils.domino.utils import *
from physicsnemo.utils.domino.vtk_file_utils import *
from physicsnemo.utils.sdf import signed_distance_field
from physicsnemo.utils.neighbors import knn
from utils import ScalingFactors, load_scaling_factors
from utils import get_time_series_data

def loss_fn(output, target):
    masked_loss = torch.mean(((output - target) ** 2.0), (0, 1, 2))
    loss = torch.mean(masked_loss)
    return loss


def test_step(data_dict, model, device, cfg, vol_factors, surf_factors):
    avg_tloss_vol = 0.0
    avg_tloss_surf = 0.0
    running_tloss_vol = 0.0
    running_tloss_surf = 0.0

    if cfg.model.model_type == "volume" or cfg.model.model_type == "combined":
        output_features_vol = True
    else:
        output_features_vol = None

    if cfg.model.model_type == "surface" or cfg.model.model_type == "combined":
        output_features_surf = True
    else:
        output_features_surf = None

    with torch.no_grad():
        point_batch_size = 256000
        # data_dict = dict_to_device(data_dict, device)

        # Non-dimensionalization factors
        length_scale = data_dict["length_scale"]

        global_params_values = data_dict["global_params_values"]
        global_params_reference = data_dict["global_params_reference"]
        stress = global_params_reference[:, 0, :]

        # STL nodes
        geo_centers = data_dict["geometry_coordinates"]

        # Bounding box grid
        s_grid = data_dict["surf_grid"]
        sdf_surf_grid = data_dict["sdf_surf_grid"]
        # Scaling factors
        surf_max = data_dict["surface_min_max"][:, 1]
        surf_min = data_dict["surface_min_max"][:, 0]

        if output_features_surf is not None:
            # Represent geometry on bounding box
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = model.geo_rep_surface(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

        if output_features_surf is not None:
            # Next calculate surface predictions
            # Sampled points on surface
            surface_mesh_centers = data_dict["surface_mesh_centers"]
            surface_normals = data_dict["surface_normals"]
            surface_areas = data_dict["surface_areas"]

            # Neighbors of sampled points on surface
            surface_mesh_neighbors = data_dict["surface_mesh_neighbors"]
            surface_neighbors_normals = data_dict["surface_neighbors_normals"]
            surface_neighbors_areas = data_dict["surface_neighbors_areas"]
            surface_areas = torch.unsqueeze(surface_areas, -1)
            surface_neighbors_areas = torch.unsqueeze(surface_neighbors_areas, -1)
            pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]
            num_points = surface_mesh_centers.shape[1]
            subdomain_points = int(np.floor(num_points / point_batch_size))

            target_surf = data_dict["surface_fields"]
            prediction_surf = torch.zeros_like(target_surf)

            start_time = time.time()

            for p in range(subdomain_points + 1):
                start_idx = p * point_batch_size
                end_idx = (p + 1) * point_batch_size
                with torch.no_grad():
                    target_batch = target_surf[:, start_idx:end_idx]
                    surface_mesh_centers_batch = surface_mesh_centers[
                        :, start_idx:end_idx
                    ]
                    surface_mesh_neighbors_batch = surface_mesh_neighbors[
                        :, start_idx:end_idx
                    ]
                    surface_normals_batch = surface_normals[:, start_idx:end_idx]
                    surface_neighbors_normals_batch = surface_neighbors_normals[
                        :, start_idx:end_idx
                    ]
                    surface_areas_batch = surface_areas[:, start_idx:end_idx]
                    surface_neighbors_areas_batch = surface_neighbors_areas[
                        :, start_idx:end_idx
                    ]
                    pos_surface_center_of_mass_batch = pos_surface_center_of_mass[
                        :, start_idx:end_idx
                    ]

                    if cfg.model.transient:
                        geo_encoding_local_all = []
                        for i in range(surface_mesh_centers.shape[1]):
                            geo_encoding_local_i = model.surface_local_geo_encodings(
                                0.5 * encoding_g_surf, surface_mesh_centers_batch[:, i, :, :3], s_grid
                            )
                            geo_encoding_local_all.append(torch.unsqueeze(geo_encoding_local_i, 1))
                        geo_encoding_local = torch.cat(geo_encoding_local_all, dim=1)
                    else:
                        geo_encoding_local = model.surface_local_geo_encodings(
                            0.5 * encoding_g_surf,
                            surface_mesh_centers_batch,
                            s_grid,
                        )
                    pos_encoding = model.fc_p_surf(pos_surface_center_of_mass_batch)

                    if cfg.model.transient_scheme == "implicit":
                        for i in range(cfg.model.integration_steps):
                            if i == 0:
                                surface_mesh_centers_batch_i = surface_mesh_centers_batch[:, i]
                                surface_mesh_neighbors_batch_i = surface_mesh_neighbors_batch[:, i]
                                prediction_surf[:, i, start_idx:end_idx] = surface_mesh_centers_batch_i[:, :, :3]
                            else:
                                surface_mesh_centers_batch_i[:, :, :3] += tpredictions_batch
                                for j in range(surface_mesh_neighbors_batch_i.shape[2]):
                                    surface_mesh_neighbors_batch_i[:, :, j, :3] += tpredictions_batch

                                prediction_surf[:, i, start_idx:end_idx] = tpredictions_batch

                            tpredictions_batch = model.solution_calculator_surf(
                                surface_mesh_centers_batch_i,
                                geo_encoding_local[:, i],
                                pos_encoding[:, i],
                                surface_mesh_neighbors_batch_i,
                                surface_normals_batch[:, i],
                                surface_neighbors_normals_batch[:, i],
                                surface_areas_batch[:, i],
                                surface_neighbors_areas_batch[:, i],
                                global_params_values,
                                global_params_reference,
                            )
                    else:
                        for i in range(surface_mesh_centers.shape[1]):
                            tpredictions_batch = model.solution_calculator_surf(
                                surface_mesh_centers_batch[:, i],
                                geo_encoding_local[:, i],
                                pos_encoding[:, i],
                                surface_mesh_neighbors_batch[:, i],
                                surface_normals_batch[:, i],
                                surface_neighbors_normals_batch[:, i],
                                surface_areas_batch[:, i],
                                surface_neighbors_areas_batch[:, i],
                                global_params_values,
                                global_params_reference,
                            )
                            prediction_surf[:, i, start_idx:end_idx] = tpredictions_batch

            if cfg.model.normalization == "min_max_scaling":
                prediction_surf = unnormalize(
                    prediction_surf, surf_factors[0], surf_factors[1]
                )
            elif cfg.model.normalization == "mean_std_scaling":
                prediction_surf = unstandardize(
                    prediction_surf, surf_factors[0], surf_factors[1]
                )

        else:
            prediction_surf = None

    return None, prediction_surf


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    input_path = cfg.eval.test_path

    model_type = cfg.model.model_type

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    if model_type == "volume" or model_type == "combined":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
        num_vol_vars = 0
        for j in volume_variable_names:
            if cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1
    else:
        num_vol_vars = None

    if model_type == "surface" or model_type == "combined":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
    else:
        num_surf_vars = None

    global_features = 0
    global_params_names = list(cfg.variables.global_parameters.keys())
    for param in global_params_names:
        if cfg.variables.global_parameters[param].type == "vector":
            global_features += len(cfg.variables.global_parameters[param].reference)
        else:
            global_features += 1

    ######################################################
    # Get scaling factors - precompute them if this fails!
    ######################################################
    pickle_path = os.path.join(cfg.data.scaling_factors)

    vol_factors, surf_factors = load_scaling_factors(cfg)
    print("Vol factors:", vol_factors)
    print("Surf factors:", surf_factors)

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        global_features=global_features,
        model_parameters=cfg.model,
    ).to(dist.device)

    model = torch.compile(model, disable=True)

    checkpoint = torch.load(
        to_absolute_path(os.path.join(cfg.resume_dir, cfg.eval.checkpoint_name)),
        map_location=dist.device,
    )

    model.load_state_dict(checkpoint)

    print("Model loaded")

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        model = model.module

    dirnames = get_filenames(input_path)
    dev_id = torch.cuda.current_device()
    num_files = int(len(dirnames) / dist.world_size)
    dirnames_per_gpu = dirnames[int(num_files * dev_id) : int(num_files * (dev_id + 1))]

    pred_save_path = cfg.eval.save_path

    if dist.rank == 0:
        create_directory(pred_save_path)

    l2_surface_all = []
    l2_volume_all = []
    aero_forces_all = []
    for count, dirname in enumerate(dirnames_per_gpu):
        filepath = os.path.join(input_path, dirname)
        tag = int(re.findall(r"(\w+?)(\d+)", dirname)[0][1])

        # Read STL
        reader = pv.get_reader(filepath)
        mesh_stl = reader.read()
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[
            :, 1:
        ]  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        length_scale = np.array(
            np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0)),
            dtype=np.float32,
        )
        length_scale = torch.from_numpy(length_scale).to(torch.float32).to(dist.device)
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"], dtype=np.float32)
        stl_centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)

        # Convert to torch tensors and load on device
        stl_vertices = torch.from_numpy(stl_vertices).to(torch.float32).to(dist.device)
        stl_sizes = torch.from_numpy(stl_sizes).to(torch.float32).to(dist.device)
        stl_centers = torch.from_numpy(stl_centers).to(torch.float32).to(dist.device)
        mesh_indices_flattened = (
            torch.from_numpy(mesh_indices_flattened).to(torch.int32).to(dist.device)
        )

        # Center of mass calculation
        center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes)

        s_max = (
            torch.from_numpy(np.asarray(cfg.data.bounding_box_surface.max))
            .to(torch.float32)
            .to(dist.device)
        )
        s_min = (
            torch.from_numpy(np.asarray(cfg.data.bounding_box_surface.min))
            .to(torch.float32)
            .to(dist.device)
        )

        nx, ny, nz = cfg.model.interp_res

        surf_grid = create_grid(
            s_max, s_min, torch.from_numpy(np.asarray([nx, ny, nz])).to(dist.device)
        )

        normed_stl_vertices_cp = normalize(stl_vertices, s_max, s_min)
        surf_grid_normed = normalize(surf_grid, s_max, s_min)

        # SDF calculation on the grid using WARP
        time_start = time.time()
        sdf_surf_grid, _ = signed_distance_field(
            normed_stl_vertices_cp,
            mesh_indices_flattened,
            surf_grid_normed,
            use_sign_winding_number=True,
        )

        surf_grid_max_min = torch.stack([s_min, s_max])

        # Get global parameters and global parameters scaling from config.yaml
        global_params_names = list(cfg.variables.global_parameters.keys())
        global_params_reference = {
            name: cfg.variables.global_parameters[name]["reference"]
            for name in global_params_names
        }
        global_params_types = {
            name: cfg.variables.global_parameters[name]["type"]
            for name in global_params_names
        }
        stress = global_params_reference["stress"]

        # Arrange global parameters reference in a list, ensuring it is flat
        global_params_reference_list = []
        for name, type in global_params_types.items():
            if type == "vector":
                global_params_reference_list.extend(global_params_reference[name])
            elif type == "scalar":
                global_params_reference_list.append(global_params_reference[name])
            else:
                raise ValueError(
                    f"Global parameter {name} not supported for  this dataset"
                )
        global_params_reference = np.array(
            global_params_reference_list, dtype=np.float32
        )
        global_params_reference = torch.from_numpy(global_params_reference).to(
            dist.device
        )

        # Define the list of global parameter values for each simulation.
        # Note: The user must ensure that the values provided here correspond to the
        # `global_parameters` specified in `config.yaml` and that these parameters
        # exist within each simulation file.
        global_params_values_list = []
        for key in global_params_types.keys():
            if key == "stress":
                global_params_values_list.append(stress)
            else:
                raise ValueError(f"Global parameter {key} not supported for  this dataset")
        global_params_values_list = np.array(
            global_params_values_list, dtype=np.float32
        )
        global_params_values = torch.from_numpy(global_params_values_list).to(
            dist.device
        )

        # Read VTP
        if model_type == "surface" or model_type == "combined":
            cell_data = mesh_stl.point_data_to_cell_data()

            if cfg.model.mesh_type == "node":
                timesteps, surface_fields, magnitude_data = get_time_series_data(mesh_stl, data_prefix="displacement")
                surface_coordinates = mesh_stl.points
            else:
                surface_coordinates = cell_data.cell_centers().points
                timesteps, surface_fields, magnitude_data = get_time_series_data(cell_data, data_prefix="displacement")
            
            num_timesteps = len(timesteps)
            num_points = surface_coordinates.shape[0]

            t_max = np.amax(timesteps)  
            t_min = np.amin(timesteps)
            timesteps = torch.from_numpy(timesteps).to(torch.float32).to(dist.device)
            timesteps = normalize(timesteps, t_max, t_min)
            timesteps = repeat_array(timesteps, num_points, axis=1, new_axis=True)
            timesteps = torch.unsqueeze(timesteps, axis=-1)

            surface_normals = np.array(cell_data.cell_normals, dtype=np.float32)
            surface_sizes = cell_data.compute_cell_sizes(
                length=False, area=True, volume=False
            )
            surface_sizes = np.array(surface_sizes.cell_data["Area"], dtype=np.float32)

            # Normalize cell normals
            surface_normals = (
                surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
            )

            surface_coordinates_all = []
            surface_normals_all = []
            surface_sizes_all = []
            for i in range(1, surface_fields.shape[0]):
                surface_coordinates_all.append(surface_coordinates + surface_fields[i])
                surface_normals_all.append(surface_normals)
                surface_sizes_all.append(surface_sizes)
            surface_coordinates_all = np.asarray(surface_coordinates_all)
            surface_normals_all = np.asarray(surface_normals_all)
            surface_sizes_all = np.asarray(surface_sizes_all)

            surface_coordinates = np.concatenate([np.expand_dims(surface_coordinates, 0), surface_coordinates_all], axis=0)
            surface_normals = np.concatenate([np.expand_dims(surface_normals, 0), surface_normals_all], axis=0)
            surface_sizes = np.concatenate([np.expand_dims(surface_sizes, 0), surface_sizes_all], axis=0)

            surface_coordinates = (
                torch.from_numpy(surface_coordinates).to(torch.float32).to(dist.device)
            )
            surface_normals = (
                torch.from_numpy(surface_normals).to(torch.float32).to(dist.device)
            )
            surface_sizes = (
                torch.from_numpy(surface_sizes).to(torch.float32).to(dist.device)
            )
            surface_fields = (
                torch.from_numpy(surface_fields).to(torch.float32).to(dist.device)
            )

            if cfg.model.num_neighbors_surface > 1:
                time_start = time.time()
                ii, dd = knn(
                    points=surface_coordinates[0],
                    queries=surface_coordinates[0],
                    k=cfg.model.num_neighbors_surface,
                )

                surface_neighbors = surface_coordinates[:, ii]
                surface_neighbors = surface_neighbors[:, :, 1:]

                timesteps_neighbors = repeat_array(timesteps, cfg.model.num_neighbors_surface-1, axis=2, new_axis=True)

                if cfg.model.mesh_type == "element":
                    surface_neighbors_normals = surface_normals[:, ii]
                    surface_neighbors_normals = surface_neighbors_normals[:, :, 1:]
                    surface_neighbors_sizes = surface_sizes[:, ii]
                    surface_neighbors_sizes = surface_neighbors_sizes[:, :, 1:]
                else:
                    surface_neighbors_normals = surface_normals
                    surface_neighbors_sizes = surface_sizes

            else:
                surface_neighbors = surface_coordinates
                surface_neighbors_normals = surface_normals
                surface_neighbors_sizes = surface_sizes

            if cfg.data.normalize_coordinates:
                surface_coordinates = normalize(surface_coordinates, s_max, s_min)
                surf_grid = normalize(surf_grid, s_max, s_min)
                center_of_mass_normalized = normalize(center_of_mass, s_max, s_min)
                surface_neighbors = normalize(surface_neighbors, s_max, s_min)
            else:
                center_of_mass_normalized = center_of_mass
            pos_surface_center_of_mass = surface_coordinates - center_of_mass_normalized

            surface_coordinates = torch.cat([surface_coordinates, timesteps], axis=-1)
            if cfg.model.num_neighbors_surface > 1:
                surface_neighbors = torch.cat([surface_neighbors, timesteps_neighbors], axis=-1)
            else:
                surface_neighbors = surface_neighbors

        else:
            surface_coordinates = None
            surface_fields = None
            surface_sizes = None
            surface_normals = None
            surface_neighbors = None
            surface_neighbors_normals = None
            surface_neighbors_sizes = None
            pos_surface_center_of_mass = None

        geom_centers = stl_vertices
        # print(f"Geom centers max: {np.amax(geom_centers, axis=0)}, min: {np.amin(geom_centers, axis=0)}")

        
        if model_type == "surface":
            data_dict = {
                "pos_surface_center_of_mass": pos_surface_center_of_mass,
                "geometry_coordinates": geom_centers,
                "surf_grid": surf_grid,
                "sdf_surf_grid": sdf_surf_grid,
                "surface_mesh_centers": surface_coordinates,
                "surface_mesh_neighbors": surface_neighbors,
                "surface_normals": surface_normals,
                "surface_neighbors_normals": surface_neighbors_normals,
                "surface_areas": surface_sizes,
                "surface_neighbors_areas": surface_neighbors_sizes,
                "surface_fields": surface_fields,
                "surface_min_max": surf_grid_max_min,
                "length_scale": length_scale,
                "global_params_values": torch.unsqueeze(global_params_values, -1),
                "global_params_reference": torch.unsqueeze(global_params_reference, -1),
            }
        else:
            raise ValueError(f"Model type: {model_type} not supported yet")

        data_dict = {key: torch.unsqueeze(value, 0) for key, value in data_dict.items()}

        prediction_vol, prediction_surf = test_step(
            data_dict, model, dist.device, cfg, vol_factors, surf_factors
        )

        prediction_surf = prediction_surf[0].reshape(num_timesteps, num_points, prediction_surf.shape[-1])
        surface_fields = surface_fields.reshape(num_timesteps, num_points, surface_fields.shape[-1])

        surface_coordinates_initial = surface_coordinates[0, :, :3]

        for i in range(num_timesteps):
            surface_fields[i, :, :] = surface_fields[i, :, :] + surface_coordinates_initial

        if cfg.model.transient_scheme == "implicit":
            for i in range(num_timesteps):
                if i == 0:
                    d_prediction_surf = surface_coordinates_initial
                else:
                    d_prediction_surf = prediction_surf[i-1, :, :]
                prediction_surf[i, :, :] = prediction_surf[i, :, :] + d_prediction_surf
        elif cfg.model.transient_scheme == "explicit":
            for i in range(num_timesteps):
                prediction_surf[i, :, :] += surface_coordinates_initial
        else:
            raise ValueError(f"Invalid transient scheme: {cfg.model.transient_scheme}")

        vtp_pred_save_path = os.path.join(
            pred_save_path, dirname[:-4], "predicted"
        )
        create_directory(vtp_pred_save_path)
        vtp_true_save_path = os.path.join(
            pred_save_path, dirname[:-4], "true"
        )
        create_directory(vtp_true_save_path)
        
        prediction_surf = prediction_surf.cpu().numpy()
        surface_fields = surface_fields.cpu().numpy()
        surface_coordinates_initial = surface_coordinates_initial.cpu().numpy()
        timesteps = unnormalize(timesteps, t_max, t_min)
        timesteps = timesteps.cpu().numpy()
        if prediction_surf is not None:

            mesh_stl.clear_cell_data()
            mesh_stl.clear_point_data()

            mesh_stl_deformed_new = mesh_stl.copy()
            initial_field_pred = mesh_stl_deformed_new.points
            initial_field_true = mesh_stl_deformed_new.points

            for i in range(1, cfg.model.integration_steps + 1):
                vtp_pred_save_path_new = os.path.join(
                        vtp_pred_save_path, f"boundary_predicted_{i}.vtp"
                    )
                vtp_true_save_path_new = os.path.join(
                    vtp_true_save_path, f"boundary_true_{i}.vtp"
                )
                vector_field_name = f"displacement"

                initial_field_pred_new = prediction_surf[i, :, :]
                initial_field_true_new = surface_fields[i, :, :]

                mesh_stl_deformed_new.points = initial_field_pred_new
                mesh_stl_deformed_new[vector_field_name] = prediction_surf[i, :, :] - surface_coordinates_initial
                mesh_stl_deformed_new.save(vtp_pred_save_path_new)
                mesh_stl_deformed_new.points = initial_field_true_new
                mesh_stl_deformed_new[vector_field_name] = surface_fields[i, :, :] - surface_coordinates_initial
                mesh_stl_deformed_new.save(vtp_true_save_path_new)

        pvd_content = """<?xml version="1.0"?>
        <VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
        <Collection>
        """
        for timestep in range(1, cfg.model.integration_steps + 1):
            pvd_content += f'    <DataSet timestep="{timestep}" file="{os.path.basename(f"boundary_predicted_{timestep}.vtp")}"/>\n'
        pvd_content += """  </Collection>
        </VTKFile>
        """
       
        pvd_filename = os.path.join(os.path.join(vtp_pred_save_path, "predicted.pvd"))
        with open(pvd_filename, "w") as f:
            f.write(pvd_content)

        pvd_content = """<?xml version="1.0"?>
        <VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
        <Collection>
        """
        for timestep in range(1, cfg.model.integration_steps + 1):
            pvd_content += f'    <DataSet timestep="{timestep}" file="{os.path.basename(f"boundary_true_{timestep}.vtp")}"/>\n'
        pvd_content += """  </Collection>
        </VTKFile>
        """
        pvd_filename = os.path.join(os.path.join(vtp_true_save_path, "truth.pvd"))
        with open(pvd_filename, "w") as f:
            f.write(pvd_content)

        # # Predict deformation
        # vtp_pred_save_path = os.path.join(
        #     pred_save_path, dirname[:-4], "predicted_deformed"
        # )
        # create_directory(vtp_pred_save_path)
        # vtp_true_save_path = os.path.join(
        #     pred_save_path, dirname[:-4], "true_deformed"
        # )
        # create_directory(vtp_true_save_path)
        # mesh_stl_deformed = mesh_stl.copy()
        # initial_field_pred = mesh_stl_deformed.points
        # initial_field_true = mesh_stl_deformed.points

        # for i in range(1, cfg.model.integration_steps + 1):
        #     vtp_pred_save_path_new = os.path.join(
        #             vtp_pred_save_path, f"boundary_predicted_{i}.vtp"
        #         )
        #     vtp_true_save_path_new = os.path.join(
        #         vtp_true_save_path, f"boundary_true_{i}.vtp"
        #     )
        #     vector_field_name = f"displacement"

        #     initial_field_pred += (prediction_surf[i, :, :] - prediction_surf[i-1, :, :])
        #     initial_field_true += (surface_fields[i, :, :] - surface_fields[i-1, :, :])

        #     mesh_stl_deformed.points = initial_field_pred
        #     mesh_stl_deformed[vector_field_name] = prediction_surf[i, :, :]
        #     mesh_stl_deformed.save(vtp_pred_save_path_new)
        #     mesh_stl_deformed.points = initial_field_true
        #     mesh_stl_deformed[vector_field_name] = surface_fields[i, :, :]
        #     mesh_stl_deformed.save(vtp_true_save_path_new)

        # pvd_content = """<?xml version="1.0"?>
        # <VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
        # <Collection>
        # """
        # for timestep in range(1, 21):
        #     pvd_content += f'    <DataSet timestep="{timestep}" file="{os.path.basename(f"boundary_predicted_{timestep}.vtp")}"/>\n'
        # pvd_content += """  </Collection>
        # </VTKFile>
        # """
       
        # pvd_filename = os.path.join(os.path.join(vtp_pred_save_path, "predicted.pvd"))
        # with open(pvd_filename, "w") as f:
        #     f.write(pvd_content)

        # pvd_content = """<?xml version="1.0"?>
        # <VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
        # <Collection>
        # """
        # for timestep in range(1, 21):
        #     pvd_content += f'    <DataSet timestep="{timestep}" file="{os.path.basename(f"boundary_true_{timestep}.vtp")}"/>\n'
        # pvd_content += """  </Collection>
        # </VTKFile>
        # """
        # pvd_filename = os.path.join(os.path.join(vtp_true_save_path, "truth.pvd"))
        # with open(pvd_filename, "w") as f:
        #     f.write(pvd_content)

        if prediction_surf is not None:

            for ii in range(surface_fields.shape[0]):
                print("Timestep:", ii)
                l2_gt = np.mean(np.square(surface_fields[ii] - surface_coordinates_initial), (0))
                l2_error = np.mean(np.square(prediction_surf[ii] - surface_fields[ii] - surface_coordinates_initial), (0))
                l2_surface_all.append(np.sqrt(l2_error / l2_gt))

                error_max = (np.max(np.abs(prediction_surf[ii]), axis=(0)) - np.amax(abs(surface_fields[ii] - surface_coordinates_initial), axis=(0)))/np.amax(np.abs(surface_fields[ii] - surface_coordinates_initial), axis=(0))
                pred_displacement_mag = np.sqrt(np.sum(np.square(prediction_surf[ii] - surface_coordinates_initial), axis=(1)))
                true_displacement_mag = np.sqrt(np.sum(np.square(surface_fields[ii] - surface_coordinates_initial), axis=(1)))
                # print(true_displacement_mag.shape, pred_displacement_mag.shape)
                l2_gt_displacement_mag = np.mean(np.square(true_displacement_mag), (0))
                l2_error_displacement_mag = np.mean(np.square(pred_displacement_mag - true_displacement_mag), (0))
                error_max_displacement = (np.max(np.abs(pred_displacement_mag), axis=(0)) - np.amax(abs(true_displacement_mag), axis=(0)))/np.amax(np.abs(true_displacement_mag), axis=(0))

                print(
                    "Surface L-2 norm:",
                    dirname,
                    np.sqrt(l2_error) / np.sqrt(l2_gt),
                )
                print(
                    "Surface mse:",
                    dirname,
                    l2_error,
                )
                print(
                    "Surface error max:",
                    dirname,
                    error_max,
                )
                print(
                    "Displacement L-2 norm:",
                    dirname,
                    np.sqrt(l2_error_displacement_mag) / np.sqrt(l2_gt_displacement_mag),
                )
                print(
                    "Displacement mse:",
                    dirname,
                    l2_error_displacement_mag,
                )
                print(
                    "Displacement error max:",
                    dirname,
                    error_max_displacement,
                )

    l2_surface_all = np.asarray(l2_surface_all)  # num_files, 4
    l2_surface_mean = np.mean(l2_surface_all, 0)
    print(
        f"Mean over all samples, surface={l2_surface_mean}"
    )


if __name__ == "__main__":
    main()
