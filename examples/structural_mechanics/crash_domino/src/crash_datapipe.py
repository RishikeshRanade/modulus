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
This is the datapipe to read OpenFoam files (vtp/vtu/stl) and save them as point clouds 
in npy format. 

"""

import time, random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable, Dict

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from physicsnemo.utils.domino.utils import *
from torch.utils.data import Dataset
from utils import extract_index_from_filename, extract_time_series_info, get_time_series_data

class CrashDataset(Dataset):
    """
    Datapipe for converting openfoam dataset to npy

    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        surface_variables: Optional[list] = [
            "pMean",
            "wallShearStress",
        ],
        volume_variables: Optional[list] = ["UMean", "pMean"],
        global_params_types: Optional[dict] = {
            "inlet_velocity": "vector",
            "air_density": "scalar",
        },
        global_params_reference: Optional[dict] = {
            "inlet_velocity": [30.0],
            "air_density": 1.226,
        },
        device: int = 0,
        model_type=None,
    ):
        if isinstance(input_dir, str):
            input_dir = Path(input_dir)
        input_dir = input_dir.expanduser()

        self.data_path = input_dir

        assert self.data_path.exists(), f"Path {self.data_path} does not exist"

        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"

        self.filenames = get_filenames(self.data_path)
        random.shuffle(self.filenames)
        self.indices = np.array(len(self.filenames))

        self.surface_variables = surface_variables
        self.volume_variables = volume_variables

        self.global_params_types = global_params_types
        self.global_params_reference = global_params_reference

        self.stream_velocity = 0.0

        self.stress = self.global_params_reference["stress"]

        self.device = device
        self.model_type = model_type

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cfd_filename = self.filenames[idx]
        file_index = extract_index_from_filename(cfd_filename)

        displacement_dir = self.data_path / f"run{file_index}_displacement.vtp"
        print("displacement_dir: ", displacement_dir)

        mesh_displacement = pv.read(displacement_dir)

        stl_vertices = mesh_displacement.points
       
        mesh_indices_flattened = np.array(mesh_displacement.faces).reshape((-1, 4))[:, 1:].flatten()  # Assuming triangular elements
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))

        stl_sizes = mesh_displacement.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"])
        stl_centers = np.array(mesh_displacement.cell_centers().points)

        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))

        # print(mesh)
        cell_data = mesh_displacement.point_data_to_cell_data()
        surface_coordinates_centers = cell_data.cell_centers().points
        surface_normals = np.array(cell_data.cell_normals)
        surface_sizes = cell_data.compute_cell_sizes(
            length=False, area=True, volume=False
        )
        surface_sizes = np.array(surface_sizes.cell_data["Area"])
        timesteps, displacement_data, magnitude_data = get_time_series_data(mesh_displacement, data_prefix="displacement")
        surface_fields = displacement_data
        surface_coordinates = mesh_displacement.points
        # print(surface_fields.shape)
        # print(surface_fields[1].max(), surface_fields[1].min())
        # print(surface_fields[-1].max(), surface_fields[-1].min())
        # exit()

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
 
        # print("surface_coordinates.shape: ", surface_coordinates.shape)
        # print("surface_normals.shape: ", surface_normals.shape)
        # print("surface_sizes.shape: ", surface_sizes.shape)
        
        # Arrange global parameters reference in a list based on the type of the parameter
        global_params_reference_list = []
        for name, type in self.global_params_types.items():
            if type == "vector":
                global_params_reference_list.extend(self.global_params_reference[name])
            elif type == "scalar":
                global_params_reference_list.append(self.global_params_reference[name])
            else:
                raise ValueError(
                    f"Global parameter {name} not supported for  this dataset"
                )
        global_params_reference = np.array(
            global_params_reference_list, dtype=np.float32
        )

        # Prepare the list of global parameter values for each simulation file
        # Note: The user must ensure that the values provided here correspond to the
        # `global_parameters` specified in `config.yaml` and that these parameters
        # exist within each simulation file.
        global_params_values_list = []
        for key in self.global_params_types.keys():
            if key == "stress":
                global_params_values_list.extend(
                    self.global_params_reference["stress"]
                )
            else:
                raise ValueError(
                    f"Global parameter {key} not supported for  this dataset"
                )
        global_params_values = np.array(global_params_values_list, dtype=np.float32)

        # Add the parameters to the dictionary
        return {
            "stl_coordinates": np.float32(stl_vertices),
            "stl_centers": np.float32(stl_centers),
            "stl_faces": np.float32(mesh_indices_flattened),
            "stl_areas": np.float32(stl_sizes),
            "surface_mesh_centers": np.float32(surface_coordinates),
            "surface_normals": np.float32(surface_normals),
            "surface_areas": np.float32(surface_sizes),
            "surface_fields": np.float32(surface_fields),
            "surface_features": np.float32(surface_coordinates), # This can be thickness and material properties on nodes
            "geometry_features": np.float32(stl_vertices), # This can be thickness and material properties on nodes
            "timesteps": np.float32(timesteps),
            "filename": cfd_filename,
            "global_params_values": global_params_values,
            "global_params_reference": global_params_reference,
        }


if __name__ == "__main__":
    fm_data = DSDataset(
        data_path="/code/aerofoundationdata/",
        phase="train",
        volume_variables=["UMean", "pMean", "nutMean"],
        surface_variables=["pMean", "wallShearStress", "nutMean"],
        global_params_types={"inlet_velocity": "vector", "air_density": "scalar"},
        global_params_reference={"inlet_velocity": [30.0], "air_density": 1.226},
        sampling=False,
        sample_in_bbox=False,
    )
    d_dict = fm_data[1]
