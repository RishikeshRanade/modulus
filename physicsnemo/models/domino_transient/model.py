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
The DoMINO class contains an architecture to model both surface and
volume quantities together as well as separately (controlled using
the config.yaml file)
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
from .solutions import SolutionCalculatorSurface, SolutionCalculatorVolume

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
    DoMINO model architecture for predicting both surface and volume quantities.

    The DoMINO (Deep Operational Modal Identification and Nonlinear Optimization) model
    is designed to model both surface and volume physical quantities in aerodynamic
    simulations. It can operate in three modes:
    1. Surface-only: Predicting only surface quantities
    2. Volume-only: Predicting only volume quantities
    3. Combined: Predicting both surface and volume quantities

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
    output_features_vol : int, optional
        Number of output features in volume
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
    >>> volume_coordinates = torch.randn(bsize, 100, 3).to(device)
    >>> vol_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    >>> surf_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    >>> global_params_values = torch.randn(bsize, global_features, 1).to(device)
    >>> global_params_reference = torch.randn(bsize, global_features, 1).to(device)
    >>> input_dict = {
    ...            "pos_volume_closest": pos_normals_closest_vol,
    ...            "pos_volume_center_of_mass": pos_normals_com_vol,
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
    ...            "volume_mesh_centers": volume_coordinates,
    ...            "volume_min_max": vol_grid_max_min,
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
        nodal_volume_features: int = 0,
        nodal_geometry_features: int = 0,
        model_parameters=None,
    ):
        """
        Initialize the DoMINO model.

        Args:
            input_features: Number of input feature dimensions for point data
            output_features_vol: Number of output features for volume quantities (None for surface-only mode)
            output_features_surf: Number of output features for surface quantities (None for volume-only mode)
            transient: Whether the model is transient
            tranient_scheme: The scheme to use for the transient model
            model_parameters: Configuration parameters for the model

        Raises:
            ValueError: If both output_features_vol and output_features_surf are None
        """
        super().__init__()

        self.output_features_vol = output_features_vol
        self.output_features_surf = output_features_surf
        self.num_sample_points_surface = model_parameters.num_neighbors_surface
        self.num_sample_points_volume = model_parameters.num_neighbors_volume
        self.integration_steps = model_parameters.integration_steps
        self.integration_scheme = model_parameters.transient_scheme
        self.transient = model_parameters.transient
        self.activation_processor = (
            model_parameters.geometry_rep.geo_processor.activation
        )
        self.nodal_surface_features = nodal_surface_features
        self.nodal_volume_features = nodal_volume_features
        self.nodal_geometry_features = nodal_geometry_features
        
        self.global_features = global_features

        if self.output_features_vol is None and self.output_features_surf is None:
            raise ValueError(
                "At least one of `output_features_vol` or `output_features_surf` must be specified"
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
        self.num_variables_vol = output_features_vol
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

        self.geo_rep_volume = GeometryRep(
            input_features=input_features,
            radii=model_parameters.geometry_rep.geo_conv.volume_radii,
            neighbors_in_radius=model_parameters.geometry_rep.geo_conv.volume_neighbors_in_radius,
            hops=model_parameters.geometry_rep.geo_conv.volume_hops,
            sdf_scaling_factor=model_parameters.geometry_rep.geo_processor.volume_sdf_scaling_factor,
            model_parameters=model_parameters,
            nodal_geometry_features=nodal_geometry_features,
        )

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

        # Basis functions for surface and volume
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

        if self.output_features_vol is not None:
            self.nn_basis_vol = nn.ModuleList()
            for _ in range(
                self.num_variables_vol
            ):  # Have the same basis function for each variable
                self.nn_basis_vol.append(
                    FourierMLP(
                        input_features=input_features + self.nodal_volume_features,
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
        self.sdf_scaling_factor = (
            model_parameters.geometry_rep.geo_processor.volume_sdf_scaling_factor
        )
        if self.output_features_vol is not None:
            inp_pos_vol = (
                7 + len(self.sdf_scaling_factor)
                if model_parameters.use_sdf_in_basis_func
                else 3
            )

            self.fc_p_vol = FourierMLP(
                input_features=inp_pos_vol,
                fourier_features=model_parameters.position_encoder.fourier_features,
                num_modes=model_parameters.position_encoder.num_modes,
                base_layer=model_parameters.position_encoder.base_neurons,
                activation=get_activation(model_parameters.position_encoder.activation),
            )

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

        # Create a set of local geometry encodings for the surface data:
        self.volume_local_geo_encodings = MultiGeometryEncoding(
            radii=model_parameters.geometry_local.volume_radii,
            neighbors_in_radius=model_parameters.geometry_local.volume_neighbors_in_radius,
            geo_encoding_type=self.geo_encoding_type,
            n_upstream_radii=len(model_parameters.geometry_rep.geo_conv.volume_radii),
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

        if self.output_features_vol is not None:
            # Volume
            base_layer_geo_vol = 0
            for j in model_parameters.geometry_local.volume_neighbors_in_radius:
                base_layer_geo_vol += j

            self.agg_model_vol = nn.ModuleList()
            for _ in range(self.num_variables_vol):
                self.agg_model_vol.append(
                    AggregationModel(
                        input_features=position_encoder_base_neurons
                        + base_layer_nn
                        + base_layer_geo_vol
                        + base_layer_p,
                        output_features=1,
                        base_layer=model_parameters.aggregation_model.base_layer,
                        activation=get_activation(
                            model_parameters.aggregation_model.activation
                        ),
                    )
                )
            if hasattr(model_parameters, "return_volume_neighbors"):
                return_volume_neighbors = model_parameters.return_volume_neighbors
            else:
                return_volume_neighbors = False

            self.solution_calculator_vol = SolutionCalculatorVolume(
                num_variables=self.num_variables_vol,
                num_sample_points=self.num_sample_points_volume,
                noise_intensity=50,
                return_volume_neighbors=return_volume_neighbors,
                encode_parameters=self.encode_parameters,
                parameter_model=self.parameter_model
                if self.encode_parameters
                else None,
                aggregation_model=self.agg_model_vol,
                nn_basis=self.nn_basis_vol,
            )

    def _validate_and_extract_surface_properties(self, data_dict):
        """
        Validate and extract surface properties from data dictionary.
        
        Args:
            data_dict: Input data dictionary
            
        Returns:
            Tuple of (surface_areas, surface_normals, surface_neighbors_areas, surface_neighbors_normals)
        """
        surface_areas = None
        surface_normals = None
        surface_neighbors_areas = None
        surface_neighbors_normals = None
        
        if "surface_areas" in data_dict:
            surface_areas = data_dict["surface_areas"]
            surface_neighbors_areas = data_dict["surface_neighbors_areas"]
            surface_areas = torch.unsqueeze(surface_areas, -1)
            surface_neighbors_areas = torch.unsqueeze(surface_neighbors_areas, -1)
        if "surface_normals" in data_dict:
            surface_normals = data_dict["surface_normals"]
            surface_neighbors_normals = data_dict["surface_neighbors_normals"]

        return surface_areas, surface_normals, surface_neighbors_areas, surface_neighbors_normals
    
    def _validate_and_extract_nodal_features(self, data_dict):
        """
        Validate and extract nodal features from data dictionary.
        """
        surface_features = None
        volume_features = None
        geometry_features = None
        
        if "surface_features" in data_dict:
            surface_features = data_dict["surface_features"]
            if surface_features.shape[-1] != self.nodal_surface_features:
                raise ValueError(
                    f"Surface features must have {self.nodal_surface_features} features"
                )
                
        if "volume_features" in data_dict:
            volume_features = data_dict["volume_features"]
            if volume_features.shape[-1] != self.nodal_volume_features:
                raise ValueError(
                    f"Volume features must have {self.nodal_volume_features} features"
                )
                
        if "geometry_features" in data_dict:
            geometry_features = data_dict["geometry_features"]
            if geometry_features.shape[-1] != self.nodal_geometry_features:
                raise ValueError(
                    f"Geometry features must have {self.nodal_geometry_features} features"
                )
                
        return surface_features, volume_features, geometry_features
    
    def _compute_volume_positional_encoding(self, data_dict):
        """
        Compute positional encodings for volume domain.
        
        Args:
            data_dict: Input data dictionary containing:
                - sdf_nodes: SDF values at volume nodes
                - pos_volume_closest: Positions of closest surface points
                - pos_volume_center_of_mass: Positions relative to geometry center of mass
            
        Returns:
            Positional encoding tensor for volume nodes
        """
        # Compute SDF-based features
        sdf_nodes = data_dict["sdf_nodes"]
        scaled_sdf_nodes = [
            scale_sdf(sdf_nodes, scaling) for scaling in self.sdf_scaling_factor
        ]
        scaled_sdf_nodes = torch.cat(scaled_sdf_nodes, dim=-1)

        # Compute positional encodings
        pos_volume_closest = data_dict["pos_volume_closest"]
        pos_volume_center_of_mass = data_dict["pos_volume_center_of_mass"]
        
        if self.use_sdf_in_basis_func:
            encoding_node_vol = torch.cat(
                (sdf_nodes, scaled_sdf_nodes, pos_volume_closest, pos_volume_center_of_mass),
                dim=-1,
            )
        else:
            encoding_node_vol = pos_volume_center_of_mass

        # Apply positional encoder network
        encoding_node_vol = self.fc_p_vol(encoding_node_vol)

        return encoding_node_vol
    
    def _compute_volume_encodings(self, data_dict, geo_centers, geometry_features):
        """
        Compute geometry encodings for volume domain.
        
        Args:
            data_dict: Input data dictionary
            geo_centers: Geometry center coordinates
            geometry_features: Optional geometry features
            
        Returns:
            Tuple of (encoding_g_vol, p_grid)
        """
        # Computational domain grid
        p_grid = data_dict["grid"]
        sdf_grid = data_dict["sdf_grid"]
        
        # Normalize geometry centers based on volume domain
        if "volume_min_max" in data_dict:
            vol_max = data_dict["volume_min_max"][:, 1]
            vol_min = data_dict["volume_min_max"][:, 0]
            geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
        else:
            geo_centers_vol = geo_centers

        # Compute geometry representation encoding
        encoding_g_vol = self.geo_rep_volume(
            geo_centers_vol, p_grid, sdf_grid, geometry_features=geometry_features
        )

        return encoding_g_vol, p_grid
    
    def _compute_surface_positional_encoding(self, data_dict):
        """
        Compute positional encodings for surface domain.
        
        Args:
            data_dict: Input data dictionary containing:
                - pos_surface_center_of_mass: Positions relative to geometry center of mass
            
        Returns:
            Positional encoding tensor for surface nodes
        """
        # Compute positional encoding
        pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]
        encoding_node_surf = self.fc_p_surf(pos_surface_center_of_mass)

        return encoding_node_surf
    
    def _compute_surface_encodings(self, data_dict, geo_centers, geometry_features):
        """
        Compute geometry encodings for surface domain.
        
        Args:
            data_dict: Input data dictionary
            geo_centers: Geometry center coordinates
            geometry_features: Optional geometry features
            
        Returns:
            Tuple of (encoding_g_surf, s_grid, sdf_surf_grid)
        """
        # Surface grid
        s_grid = data_dict["surf_grid"]
        sdf_surf_grid = data_dict["sdf_surf_grid"]
        
        # Normalize geometry centers based on surface domain
        if "surface_min_max" in data_dict:
            surf_max = data_dict["surface_min_max"][:, 1]
            surf_min = data_dict["surface_min_max"][:, 0]
            geo_centers_surf = 2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
        else:
            geo_centers_surf = geo_centers

        # Compute geometry representation encoding
        encoding_g_surf = self.geo_rep_surface(
            geo_centers_surf, s_grid, sdf_surf_grid, geometry_features=geometry_features
        )

        return encoding_g_surf, s_grid, sdf_surf_grid
    
    def _compute_volume_local_encodings(self, encoding_g_vol, volume_mesh_centers, p_grid):
        """
        Compute local geometry encodings for volume mesh.
        
        Args:
            encoding_g_vol: Global volume geometry encoding
            volume_mesh_centers: Volume mesh center coordinates
            p_grid: Volume grid
            
        Returns:
            Local volume geometry encodings
        """
        if self.transient:
            encoding_g_vol_all = []
            for i in range(volume_mesh_centers.shape[1]):
                encoding_g_vol_i = self.volume_local_geo_encodings(
                    0.5 * encoding_g_vol, volume_mesh_centers[:, i, :, :3], p_grid
                )
                encoding_g_vol_all.append(torch.unsqueeze(encoding_g_vol_i, 1))
            return torch.cat(encoding_g_vol_all, dim=1)
        else:
            return self.volume_local_geo_encodings(
                0.5 * encoding_g_vol, volume_mesh_centers, p_grid
            )
    
    def _compute_surface_local_encodings(self, encoding_g_surf, surface_mesh_centers, s_grid):
        """
        Compute local geometry encodings for surface mesh.
        
        Args:
            encoding_g_surf: Global surface geometry encoding
            surface_mesh_centers: Surface mesh center coordinates
            s_grid: Surface grid
            
        Returns:
            Local surface geometry encodings
        """
        if self.transient:
            encoding_g_surf_all = []
            for i in range(surface_mesh_centers.shape[1]):
                encoding_g_surf_i = self.surface_local_geo_encodings(
                    0.5 * encoding_g_surf, surface_mesh_centers[:, i, :, :3], s_grid
                )
                encoding_g_surf_all.append(torch.unsqueeze(encoding_g_surf_i, 1))
            return torch.cat(encoding_g_surf_all, dim=1)
        else:
            return self.surface_local_geo_encodings(
                0.5 * encoding_g_surf, surface_mesh_centers, s_grid
            )
    
    def _compute_volume_output_implicit(
        self,
        volume_mesh_centers,
        encoding_g_vol,
        encoding_node_vol,
        global_params_values,
        global_params_reference,
        volume_features,
    ):
        """
        Compute volume output using implicit integration scheme.
        
        Args:
            volume_mesh_centers: Volume mesh center coordinates
            encoding_g_vol: Volume geometry encodings
            encoding_node_vol: Volume node encodings
            global_params_values: Global parameter values
            global_params_reference: Global parameter references
            volume_features: Optional volume features
            
        Returns:
            Volume output tensor
        """
        output_vol_all = []
        volume_mesh_centers_i = None
        
        for i in range(self.integration_steps):
            if i == 0:
                volume_mesh_centers_i = volume_mesh_centers[:, i]
            else:
                volume_mesh_centers_i[:, :, :3] += output_vol

            volume_features_i = volume_features[:, i] if volume_features is not None else None

            output_vol = self.solution_calculator_vol(
                volume_mesh_centers_i,
                encoding_g_vol[:, i],
                encoding_node_vol[:, i],
                global_params_values,
                global_params_reference,
                volume_features_i,
            )
            output_vol_all.append(torch.unsqueeze(output_vol, 1))
            
        return torch.cat(output_vol_all, dim=1)
    
    def _compute_volume_output_explicit(
        self,
        volume_mesh_centers,
        encoding_g_vol,
        encoding_node_vol,
        global_params_values,
        global_params_reference,
        volume_features,
    ):
        """
        Compute volume output using explicit integration scheme.
        
        Args:
            volume_mesh_centers: Volume mesh center coordinates
            encoding_g_vol: Volume geometry encodings
            encoding_node_vol: Volume node encodings
            global_params_values: Global parameter values
            global_params_reference: Global parameter references
            volume_features: Optional volume features
            
        Returns:
            Volume output tensor
        """
        output_vol_all = []
        
        for i in range(volume_mesh_centers.shape[1]):
            volume_features_i = volume_features[:, i] if volume_features is not None else None
            
            output_vol = self.solution_calculator_vol(
                volume_mesh_centers[:, i],
                encoding_g_vol[:, i],
                encoding_node_vol[:, i],
                global_params_values,
                global_params_reference,
                volume_features_i,
            )
            output_vol_all.append(torch.unsqueeze(output_vol, 1))
            
        return torch.cat(output_vol_all, dim=1)
    
    def _compute_surface_output_implicit(
        self,
        surface_mesh_centers,
        surface_mesh_neighbors,
        surface_normals,
        surface_neighbors_normals,
        surface_areas,
        surface_neighbors_areas,
        encoding_g_surf,
        encoding_node_surf,
        global_params_values,
        global_params_reference,
        surface_features,
    ):
        """
        Compute surface output using implicit integration scheme.
        
        Args:
            surface_mesh_centers: Surface mesh center coordinates
            surface_mesh_neighbors: Surface mesh neighbor coordinates
            surface_normals: Surface normal vectors
            surface_neighbors_normals: Surface neighbor normal vectors
            surface_areas: Surface element areas
            surface_neighbors_areas: Surface neighbor areas
            encoding_g_surf: Surface geometry encodings
            encoding_node_surf: Surface node encodings
            global_params_values: Global parameter values
            global_params_reference: Global parameter references
            surface_features: Optional surface features
            
        Returns:
            Surface output tensor
        """
        output_surf_all = []
        surface_mesh_centers_i = None
        surface_mesh_neighbors_i = None
        
        for i in range(self.integration_steps):
            if i == 0:
                surface_mesh_centers_i = surface_mesh_centers[:, i]
                surface_mesh_neighbors_i = surface_mesh_neighbors[:, i]
            else:
                surface_mesh_centers_i[:, :, :3] += output_surf
                for j in range(surface_mesh_neighbors_i.shape[2]):
                    surface_mesh_neighbors_i[:, :, j, :3] += output_surf

            surface_features_i = surface_features[:, i] if surface_features is not None else None

            output_surf = self.solution_calculator_surf(
                surface_mesh_centers_i,
                encoding_g_surf[:, i],
                encoding_node_surf[:, i],
                surface_mesh_neighbors_i,
                surface_normals[:, i] if surface_normals is not None else None,
                surface_neighbors_normals[:, i] if surface_neighbors_normals is not None else None,
                surface_areas[:, i] if surface_areas is not None else None,
                surface_neighbors_areas[:, i] if surface_neighbors_areas is not None else None,
                global_params_values,
                global_params_reference,
                surface_features_i,
            )
            output_surf_all.append(torch.unsqueeze(output_surf, 1))
            
        return torch.cat(output_surf_all, dim=1)
    
    def _compute_surface_output_explicit(
        self,
        surface_mesh_centers,
        surface_mesh_neighbors,
        surface_normals,
        surface_neighbors_normals,
        surface_areas,
        surface_neighbors_areas,
        encoding_g_surf,
        encoding_node_surf,
        global_params_values,
        global_params_reference,
        surface_features,
    ):
        """
        Compute surface output using explicit integration scheme.
        
        Args:
            surface_mesh_centers: Surface mesh center coordinates
            surface_mesh_neighbors: Surface mesh neighbor coordinates
            surface_normals: Surface normal vectors
            surface_neighbors_normals: Surface neighbor normal vectors
            surface_areas: Surface element areas
            surface_neighbors_areas: Surface neighbor areas
            encoding_g_surf: Surface geometry encodings
            encoding_node_surf: Surface node encodings
            global_params_values: Global parameter values
            global_params_reference: Global parameter references
            surface_features: Optional surface features
            
        Returns:
            Surface output tensor
        """
        output_surf_all = []
        
        for i in range(surface_mesh_centers.shape[1]):
            surface_features_i = surface_features[:, i] if surface_features is not None else None
            
            output_surf = self.solution_calculator_surf(
                surface_mesh_centers[:, i],
                encoding_g_surf[:, i],
                encoding_node_surf[:, i],
                surface_mesh_neighbors[:, i],
                surface_normals[:, i] if surface_normals is not None else None,
                surface_neighbors_normals[:, i] if surface_neighbors_normals is not None else None,
                surface_areas[:, i] if surface_areas is not None else None,
                surface_neighbors_areas[:, i] if surface_neighbors_areas is not None else None,
                global_params_values,
                global_params_reference,
                surface_features_i,
            )
            output_surf_all.append(torch.unsqueeze(output_surf, 1))
            
        return torch.cat(output_surf_all, dim=1)

    def forward(self, data_dict):
        """
        Forward pass of the DoMINO model.
        
        Args:
            data_dict: Dictionary containing all input data including:
                - geometry_coordinates: Geometry center coordinates
                - surf_grid: Surface grid
                - sdf_surf_grid: Surface SDF grid
                - sdf_grid: Volume SDF grid
                - grid: Volume grid
                - volume_mesh_centers: Volume mesh center coordinates
                - surface_mesh_centers: Surface mesh center coordinates
                - surface_normals: Surface normal vectors
                - surface_areas: Surface element areas
                - surface_mesh_neighbors: Surface mesh neighbor coordinates
                - surface_neighbors_normals: Surface neighbor normal vectors
                - surface_neighbors_areas: Surface neighbor areas
                - volume_mesh_centers: Volume mesh center coordinates
                - surface_mesh_centers: Surface mesh center coordinates
                - (optional) surface_normals: Surface normal vectors
                - (optional) surface_areas: Surface element areas
                - surface_mesh_neighbors: Surface mesh neighbor coordinates
                - (optional) surface_neighbors_normals: Surface neighbor normal vectors
                - (optional) surface_neighbors_areas: Surface neighbor areas
                - global_params_values: Global parameter values
                - global_params_reference: Global parameter references
                - (optional) surface_features, volume_features, geometry_features
                - (optional) volume-specific data if output_features_vol is not None
                - (optional) surface-specific data if output_features_surf is not None
                
        Returns:
            Tuple of (output_vol, output_surf) where each can be None if not computed
        """
        # Extract base inputs
        geo_centers = data_dict["geometry_coordinates"]
        global_params_values = data_dict["global_params_values"]
        global_params_reference = data_dict["global_params_reference"]
        
        # Validate and extract features
        surface_features, volume_features, geometry_features = (
            self._validate_and_extract_nodal_features(data_dict)
        )

        surface_areas, surface_normals, surface_neighbors_areas, surface_neighbors_normals = (
            self._validate_and_extract_surface_properties(data_dict)
        )

        # Compute volume outputs if required
        output_vol = None
        if self.output_features_vol is not None:
            # Compute volume geometry encodings
            encoding_g_vol, p_grid = self._compute_volume_encodings(
                data_dict, geo_centers, geometry_features
            )
            
            # Compute volume positional encodings
            encoding_node_vol = self._compute_volume_positional_encoding(data_dict)
            
            # Get volume mesh data
            volume_mesh_centers = data_dict["volume_mesh_centers"]
            
            # Compute local geometry encodings
            encoding_g_vol = self._compute_volume_local_encodings(
                encoding_g_vol, volume_mesh_centers, p_grid
            )
            
            # Compute volume solution based on integration scheme
            if self.integration_scheme == "implicit":
                output_vol = self._compute_volume_output_implicit(
                    volume_mesh_centers,
                    encoding_g_vol,
                    encoding_node_vol,
                    global_params_values,
                    global_params_reference,
                    volume_features,
                )
            else:
                output_vol = self._compute_volume_output_explicit(
                    volume_mesh_centers,
                    encoding_g_vol,
                    encoding_node_vol,
                    global_params_values,
                    global_params_reference,
                    volume_features,
                )

        # Compute surface outputs if required
        output_surf = None
        if self.output_features_surf is not None:
            # Compute surface geometry encodings
            encoding_g_surf, s_grid, _ = self._compute_surface_encodings(
                data_dict, geo_centers, geometry_features
            )
            
            # Compute surface positional encodings
            encoding_node_surf = self._compute_surface_positional_encoding(data_dict)
            
            # Get surface mesh data
            surface_mesh_centers = data_dict["surface_mesh_centers"]
            surface_mesh_neighbors = data_dict["surface_mesh_neighbors"]
            
            # Compute local geometry encodings
            encoding_g_surf = self._compute_surface_local_encodings(
                encoding_g_surf, surface_mesh_centers, s_grid
            )
            
            # Compute surface solution based on integration scheme
            if self.integration_scheme == "implicit":
                output_surf = self._compute_surface_output_implicit(
                    surface_mesh_centers,
                    surface_mesh_neighbors,
                    surface_normals,
                    surface_neighbors_normals,
                    surface_areas,
                    surface_neighbors_areas,
                    encoding_g_surf,
                    encoding_node_surf,
                    global_params_values,
                    global_params_reference,
                    surface_features,
                )
            else:
                output_surf = self._compute_surface_output_explicit(
                    surface_mesh_centers,
                    surface_mesh_neighbors,
                    surface_normals,
                    surface_neighbors_normals,
                    surface_areas,
                    surface_neighbors_areas,
                    encoding_g_surf,
                    encoding_node_surf,
                    global_params_values,
                    global_params_reference,
                    surface_features,
                )

        return output_vol, output_surf
