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
This code contains the DoMINO model architecture.
The DoMINO class contains an architecture to model surface
quantities
"""

import torch
import torch.nn as nn

from physicsnemo.models.layers import FourierMLP, get_activation
from physicsnemo.models.unet import UNet

from .encodings import (
    MultiGeometryEncoding,
)
from .geometry_rep import GeometryRep, scale_sdf
from .mlps import AggregationModel
from .solutions import SolutionCalculatorSurface

# @dataclass
# class MetaData(ModelMetaData):
#     name: str = "DoMINO"
#     # Optimization
#     jit: bool = False
#     cuda_graphs: bool = True
#     amp: bool = True
#     # Inference
#     onnx_cpu: bool = True
#     onnx_gpu: bool = True
#     onnx_runtime: bool = True
#     # Physics informed
#     var_dim: int = 1
#     func_torch: bool = False
#     auto_grad: bool = False


class DoMINO(nn.Module):
    """
    DoMINO model architecture for predicting surface quantities.

    The DoMINO (Deep Operational Modal Identification and Nonlinear Optimization) model
    is designed to model surface physical quantities in aerodynamic
    simulations. It can operate in two modes:
    1. Surface-only: Predicting only surface quantities required for crash

    The model uses a combination of:
    - Geometry representation modules
    - Neural network basis functions
    - Parameter encoding
    - Local and global geometry processing
    - Aggregation models for final prediction

    Parameters
    ----------
    input_features : int
        Number of point input features
    output_features_surf : int, optional
        Number of output features on surface
    model_parameters
        Model parameters controlled by config.yaml

    Example
    -------
    >>> from physicsnemo.models.domino.model import DoMINO
    >>> import torch, os
    >>> from hydra import compose, initialize
    >>> from omegaconf import OmegaConf
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> cfg = OmegaConf.register_new_resolver("eval", eval)
    >>> with initialize(version_base="1.3", config_path="examples/cfd/external_aerodynamics/domino/src/conf"):
    ...    cfg = compose(config_name="config")
    >>> cfg.model.model_type = "combined"
    >>> model = DoMINO(
    ...         input_features=3,
    ...         output_features_vol=5,
    ...         output_features_surf=4,
    ...         model_parameters=cfg.model
    ...     ).to(device)

    Warp ...
    >>> bsize = 1
    >>> nx, ny, nz = cfg.model.interp_res
    >>> num_neigh = 7
    >>> global_features = 2
    >>> pos_normals_closest_vol = torch.randn(bsize, 100, 3).to(device)
    >>> pos_normals_com_vol = torch.randn(bsize, 100, 3).to(device)
    >>> pos_normals_com_surface = torch.randn(bsize, 100, 3).to(device)
    >>> geom_centers = torch.randn(bsize, 100, 3).to(device)
    >>> grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    >>> surf_grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    >>> sdf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    >>> sdf_surf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    >>> sdf_nodes = torch.randn(bsize, 100, 1).to(device)
    >>> surface_coordinates = torch.randn(bsize, 100, 3).to(device)
    >>> surface_neighbors = torch.randn(bsize, 100, num_neigh, 3).to(device)
    >>> surface_normals = torch.randn(bsize, 100, 3).to(device)
    >>> surface_neighbors_normals = torch.randn(bsize, 100, num_neigh, 3).to(device)
    >>> surface_sizes = torch.rand(bsize, 100).to(device) + 1e-6 # Note this needs to be > 0.0
    >>> surface_neighbors_areas = torch.rand(bsize, 100, num_neigh).to(device) + 1e-6
    >>> surf_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    >>> global_params_values = torch.randn(bsize, global_features, 1).to(device)
    >>> global_params_reference = torch.randn(bsize, global_features, 1).to(device)
    >>> input_dict = {
    ...            "pos_surface_center_of_mass": pos_normals_com_surface,
    ...            "geometry_coordinates": geom_centers,
    ...            "grid": grid,
    ...            "surf_grid": surf_grid,
    ...            "sdf_grid": sdf_grid,
    ...            "sdf_surf_grid": sdf_surf_grid,
    ...            "sdf_nodes": sdf_nodes,
    ...            "surface_mesh_centers": surface_coordinates,
    ...            "surface_mesh_neighbors": surface_neighbors,
    ...            "surface_normals": surface_normals,
    ...            "surface_neighbors_normals": surface_neighbors_normals,
    ...            "surface_areas": surface_sizes,
    ...            "surface_neighbors_areas": surface_neighbors_areas,
    ...            "surface_min_max": surf_grid_max_min,
    ...            "global_params_reference": global_params_values,
    ...            "global_params_values": global_params_reference,
    ...        }
    >>> output = model(input_dict)
    >>> print(f"{output[0].shape}, {output[1].shape}")
    torch.Size([1, 100, 5]), torch.Size([1, 100, 4])
    """

    def __init__(
        self,
        input_features: int,
        output_features_vol: int | None = None,
        output_features_surf: int | None = None,
        global_features: int = 2,
        nodal_surface_features: int = 0,
        nodal_geometry_features: int = 0,
        model_parameters=None,
    ):
        """
        Initialize the DoMINO model.

        Args:
            input_features: Number of input feature dimensions for point data
            output_features_surf: Number of output features for surface quantities (None for volume-only mode)
            transient: Whether the model is transient
            tranient_scheme: The scheme to use for the transient model
            model_parameters: Configuration parameters for the model
            nodal_surface_features: Number of nodal surface features
            nodal_geometry_features: Number of nodal geometry features

        Raises:
            ValueError: If output_features_surf are None
        """
        super().__init__()

        self.output_features_surf = output_features_surf
        self.num_sample_points_surface = model_parameters.num_neighbors_surface
        self.integration_steps = model_parameters.integration_steps
        self.integration_scheme = model_parameters.transient_scheme
        self.transient = model_parameters.transient
        self.activation_processor = (
            model_parameters.geometry_rep.geo_processor.activation
        )
        self.nodal_surface_features = nodal_surface_features
        self.nodal_geometry_features = nodal_geometry_features
        h = 8
        in_channels = (
            2
            + len(model_parameters.geometry_rep.geo_conv.surface_radii)
        )
        out_channels_surf = 1 + len(
            model_parameters.geometry_rep.geo_conv.surface_radii
        )
        self.combined_unet_surf = UNet(
            in_channels=in_channels,
            out_channels=out_channels_surf,
                model_depth=3,
                feature_map_channels=[
                    h,
                    2 * h,
                    4 * h,
                ],
                num_conv_blocks=1,
                kernel_size=3,
                stride=1,
                conv_activation=self.activation_processor,
                padding=1,
                padding_mode="zeros",
                pooling_type="MaxPool3d",
                pool_size=2,
                normalization="layernorm",
                use_attn_gate=True,
                attn_decoder_feature_maps=[4 * h, 2 * h],
                attn_feature_map_channels=[2 * h, h],
                attn_intermediate_channels=4 * h,
                gradient_checkpointing=True,
            )
        self.global_features = global_features

        if self.output_features_surf is None:
            raise ValueError(
                "`output_features_surf` must be specified"
            )
        if hasattr(model_parameters, "solution_calculation_mode"):
            if model_parameters.solution_calculation_mode not in [
                "one-loop",
                "two-loop",
            ]:
                raise ValueError(
                    f"Invalid solution_calculation_mode: {model_parameters.solution_calculation_mode}, select 'one-loop' or 'two-loop'."
                )
            self.solution_calculation_mode = model_parameters.solution_calculation_mode
        else:
            self.solution_calculation_mode = "two-loop"
        self.num_variables_surf = output_features_surf
        self.grid_resolution = model_parameters.interp_res
        self.use_surface_normals = model_parameters.use_surface_normals
        self.use_surface_area = model_parameters.use_surface_area
        self.encode_parameters = model_parameters.encode_parameters
        self.geo_encoding_type = model_parameters.geometry_encoding_type

        if self.use_surface_normals:
            if not self.use_surface_area:
                input_features_surface = input_features + 3
            else:
                input_features_surface = input_features + 4
        else:
            input_features_surface = input_features

        if self.encode_parameters:
            # Defining the parameter model
            base_layer_p = model_parameters.parameter_model.base_layer
            self.parameter_model = FourierMLP(
                input_features=self.global_features,
                fourier_features=model_parameters.parameter_model.fourier_features,
                num_modes=model_parameters.parameter_model.num_modes,
                base_layer=model_parameters.parameter_model.base_layer,
                activation=get_activation(model_parameters.parameter_model.activation),
            )
        else:
            base_layer_p = 0

        self.geo_rep_surface = GeometryRep(
            input_features=input_features,
            radii=model_parameters.geometry_rep.geo_conv.surface_radii,
            neighbors_in_radius=model_parameters.geometry_rep.geo_conv.surface_neighbors_in_radius,
            hops=model_parameters.geometry_rep.geo_conv.surface_hops,
            sdf_scaling_factor=model_parameters.geometry_rep.geo_processor.surface_sdf_scaling_factor,
            model_parameters=model_parameters,
            nodal_geometry_features=nodal_geometry_features,
        )

        if self.transient:
            input_features_surface = input_features_surface + 1 # Adding one for the time step
            input_features = input_features + 1 # Adding one for the time step

        # Basis functions for surface
        base_layer_nn = model_parameters.nn_basis_functions.base_layer
        if self.output_features_surf is not None:
            self.nn_basis_surf = nn.ModuleList()
            for _ in range(
                self.num_variables_surf
            ):  # Have the same basis function for each variable
                self.nn_basis_surf.append(
                    FourierMLP(
                        input_features=input_features_surface + self.nodal_surface_features,
                        base_layer=model_parameters.nn_basis_functions.base_layer,
                        fourier_features=model_parameters.nn_basis_functions.fourier_features,
                        num_modes=model_parameters.nn_basis_functions.num_modes,
                        activation=get_activation(
                            model_parameters.nn_basis_functions.activation
                        ),
                        # model_parameters=model_parameters.nn_basis_functions,
                    )
                )

        # Positional encoding
        position_encoder_base_neurons = model_parameters.position_encoder.base_neurons
        self.activation = get_activation(model_parameters.activation)
        self.use_sdf_in_basis_func = model_parameters.use_sdf_in_basis_func
        
        if self.output_features_surf is not None:
            inp_pos_surf = 3

            self.fc_p_surf = FourierMLP(
                input_features=inp_pos_surf,
                fourier_features=model_parameters.position_encoder.fourier_features,
                num_modes=model_parameters.position_encoder.num_modes,
                base_layer=model_parameters.position_encoder.base_neurons,
                activation=get_activation(model_parameters.position_encoder.activation),
            )

        # Create a set of local geometry encodings for the surface data:
        self.surface_local_geo_encodings = MultiGeometryEncoding(
            radii=model_parameters.geometry_local.surface_radii,
            neighbors_in_radius=model_parameters.geometry_local.surface_neighbors_in_radius,
            geo_encoding_type=self.geo_encoding_type,
            n_upstream_radii=len(model_parameters.geometry_rep.geo_conv.surface_radii),
            base_layer=512,
            activation=get_activation(model_parameters.local_point_conv.activation),
            grid_resolution=self.grid_resolution,
        )

        # Aggregation model
        if self.output_features_surf is not None:
            # Surface
            base_layer_geo_surf = 0
            for j in model_parameters.geometry_local.surface_neighbors_in_radius:
                base_layer_geo_surf += j

            self.agg_model_surf = nn.ModuleList()
            for _ in range(self.num_variables_surf):
                self.agg_model_surf.append(
                    AggregationModel(
                        input_features=position_encoder_base_neurons
                        + base_layer_nn
                        + base_layer_geo_surf
                        + base_layer_p,
                        output_features=1,
                        base_layer=model_parameters.aggregation_model.base_layer,
                        activation=get_activation(
                            model_parameters.aggregation_model.activation
                        ),
                    )
                )

            self.solution_calculator_surf = SolutionCalculatorSurface(
                num_variables=self.num_variables_surf,
                num_sample_points=self.num_sample_points_surface,
                use_surface_normals=self.use_surface_normals,
                use_surface_area=self.use_surface_area,
                encode_parameters=self.encode_parameters,
                parameter_model=self.parameter_model
                if self.encode_parameters
                else None,
                aggregation_model=self.agg_model_surf,
                nn_basis=self.nn_basis_surf,
            )

    def forward(self, data_dict):
        # Loading STL inputs, bounding box grids, precomputed SDF and scaling factors

        # STL nodes
        geo_centers = data_dict["geometry_coordinates"]

        # Bounding box grid
        s_grid = data_dict["surf_grid"]
        sdf_surf_grid = data_dict["sdf_surf_grid"]

        # Parameters
        global_params_values = data_dict["global_params_values"]
        global_params_reference = data_dict["global_params_reference"]

        if "surface_features" in data_dict.keys():
            surface_features = data_dict["surface_features"]
            if surface_features.shape[-1] != self.nodal_surface_features:
                raise ValueError(f"Surface features must have {self.nodal_surface_features} features")
        else:
            surface_features = None
        
        if "geometry_features" in data_dict.keys():
            geometry_features = data_dict["geometry_features"]
            if geometry_features.shape[-1] != self.nodal_geometry_features:
                raise ValueError(f"Geometry features must have {self.nodal_geometry_features} features")
        else:
            geometry_features = None

        if self.output_features_surf is not None:
            # Represent geometry on bounding box
            # Scaling factors
            if "surface_min_max" in data_dict.keys():
                surf_max = data_dict["surface_min_max"][:, 1]
                surf_min = data_dict["surface_min_max"][:, 0]
                geo_centers_surf = (
                    2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
                )
            else:
                geo_centers_surf = geo_centers

            encoding_g_surf = self.geo_rep_surface(
                geo_centers_surf, s_grid, sdf_surf_grid, geometry_features=geometry_features
            )

            # Positional encoding based on center of mass of geometry to surface node
            pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]
            encoding_node_surf = pos_surface_center_of_mass

            # Calculate positional encoding on surface centers
            encoding_node_surf = self.fc_p_surf(encoding_node_surf)

        if self.output_features_surf is not None:
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

            if self.transient:
                # Calculate local geometry encoding for surface
                encoding_g_surf_all = []
                for i in range(surface_mesh_centers.shape[1]):
                    encoding_g_surf_i = self.surface_local_geo_encodings(
                        0.5 * encoding_g_surf, surface_mesh_centers[:, i, :, :3], s_grid
                    )
                    encoding_g_surf_all.append(torch.unsqueeze(encoding_g_surf_i, 1))
                encoding_g_surf = torch.cat(encoding_g_surf_all, dim=1)
            else:
                # Calculate local geometry encoding for surface
                encoding_g_surf = self.surface_local_geo_encodings(
                    0.5 * encoding_g_surf, surface_mesh_centers, s_grid
                )

            # Approximate solution on surface cell center
            if self.integration_scheme == "implicit":
                output_surf_all = []
                for i in range(self.integration_steps):
                    if i == 0:
                        surface_mesh_centers_i = surface_mesh_centers[:, i]
                        surface_mesh_neighbors_i = surface_mesh_neighbors[:, i]
                    else:
                        surface_mesh_centers_i[:, :, :3] += output_surf
                        for j in range(surface_mesh_neighbors_i.shape[2]):
                            surface_mesh_neighbors_i[:, :, j, :3] += output_surf

                    if surface_features is not None:
                        surface_features_i = surface_features[:, i]
                    else:
                        surface_features_i = None

                    output_surf = self.solution_calculator_surf(
                        surface_mesh_centers_i,
                        encoding_g_surf[:, i],
                        encoding_node_surf[:, i],
                        surface_mesh_neighbors_i,
                        surface_normals[:, i],
                        surface_neighbors_normals[:, i],
                        surface_areas[:, i],
                        surface_neighbors_areas[:, i],
                        global_params_values,
                        global_params_reference,
                        surface_features_i,
                    )
                    output_surf_all.append(torch.unsqueeze(output_surf, 1))
                output_surf = torch.cat(output_surf_all, dim=1)
            else:
                output_surf_all = []
                for i in range(surface_mesh_centers.shape[1]):
                    if surface_features is not None:
                        surface_features_i = surface_features[:, i]
                    else:
                        surface_features_i = None
                    output_surf = self.solution_calculator_surf(
                            surface_mesh_centers[:, i],
                            encoding_g_surf[:, i],
                            encoding_node_surf[:, i],
                            surface_mesh_neighbors[:, i],
                            surface_normals[:, i],
                            surface_neighbors_normals[:, i],
                            surface_areas[:, i],
                            surface_neighbors_areas[:, i],
                            global_params_values,
                            global_params_reference,
                            surface_features_i,
                        )
                    output_surf_all.append(torch.unsqueeze(output_surf, 1))
                output_surf = torch.cat(output_surf_all, dim=1)
        else:
            output_surf = None

        return output_surf
