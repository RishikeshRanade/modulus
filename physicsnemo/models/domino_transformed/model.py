# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

r"""
DoMINO Model Architecture.

The DoMINO class contains an architecture to model both surface and
volume quantities together as well as separately (controlled using
the config.yaml file).
"""

from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.models.unet import UNet
from physicsnemo.nn import FourierMLP, get_activation

from .config import DEFAULT_MODEL_PARAMS, Config
from .fusion import HierarchicalFusion
from .geometry_rep import GeometryRep
from .mlps import AggregationModel
from .solutions import (
    SolutionCalculatorSurface,
    SolutionCalculatorVolume,
    apply_parameter_encoding,
)


def _fusion_geometry_dims(fusion_cfg: Any) -> tuple[Any, Any]:
    """Return (geometry_dim_volume, geometry_dim_surface) from fusion config."""
    if fusion_cfg is None:
        return None, None
    return (
        getattr(fusion_cfg, "geometry_dim_volume", None),
        getattr(fusion_cfg, "geometry_dim_surface", None),
    )


def _fusion_aggregation_input_dim(fusion_cfg: Any) -> int:
    """Aggregation model input dimension when using fusion only (fusion_dim)."""
    if fusion_cfg is None:
        raise ValueError("fusion config is required for DoMINO (fusion-only mode)")
    return fusion_cfg.fusion_dim


def _build_fusion_module(
    fusion_cfg: Any,
    base_layer_nn: int,
    geometry_dim: int,
    global_dim: int,
) -> HierarchicalFusion | None:
    """Build HierarchicalFusion for one branch if fusion_cfg enables it."""
    if fusion_cfg is None or not getattr(fusion_cfg, "use_hierarchical_fusion", False):
        return None
    return HierarchicalFusion(
        basis_dim=base_layer_nn,
        geometry_dim=geometry_dim,
        global_dim=global_dim,
        fusion_dim=fusion_cfg.fusion_dim,
        stage1_hidden=getattr(fusion_cfg, "stage1_hidden", None),
        film_hidden=getattr(fusion_cfg, "film_hidden", None),
    )


class DoMINO(Module):
    r"""
    DoMINO model architecture for predicting both surface and volume quantities.

    The DoMINO (Deep Operational Modal Identification and Nonlinear Optimization) model
    is designed to model both surface and volume physical quantities in aerodynamic
    simulations. It can operate in three modes:

    1. Surface-only: Predicting only surface quantities
    2. Volume-only: Predicting only volume quantities
    3. Combined: Predicting both surface and volume quantities

    The model uses a combination of:

    - Geometry representation modules via :class:`~physicsnemo.models.domino_transformed.geometry_rep.GeometryRep`
    - Neural network basis functions via :class:`~physicsnemo.nn.FourierMLP`
    - Parameter encoding
    - Local and global geometry processing
    - Aggregation models for final prediction

    Parameters
    ----------
    input_features : int
        Number of point input features (typically 3 for x, y, z coordinates).
    output_features_vol : int, optional, default=None
        Number of output features in volume. Set to ``None`` for surface-only mode.
    output_features_surf : int, optional, default=None
        Number of output features on surface. Set to ``None`` for volume-only mode.
    global_features : int, optional, default=2
        Number of global parameter features for conditioning.
    model_parameters : Any, optional, default=None
        Model parameters controlled by config.yaml. Contains nested configuration
        for geometry representation, neural network basis functions, aggregation
        model, position encoder, and geometry local settings.

    Forward
    -------
    data_dict : dict[str, torch.Tensor]
        Dictionary containing input tensors with the following keys:

        - ``"geometry_coordinates"``: Geometry centers of shape :math:`(B, N_{geo}, 3)`
        - ``"sdf_nodes"``: SDF at volume mesh nodes (when using SDF in basis), shape :math:`(B, N_{vol}, 4)` (SDF + closest point)
        - ``"pos_volume_closest"``: Closest surface point to volume nodes of shape :math:`(B, N_{vol}, 3)`
        - ``"pos_volume_center_of_mass"``: Center of mass to volume nodes of shape :math:`(B, N_{vol}, 3)`
        - ``"pos_surface_center_of_mass"``: Center of mass to surface nodes of shape :math:`(B, N_{surf}, 3)`
        - ``"surface_mesh_centers"``: Surface mesh center coordinates of shape :math:`(B, N_{surf}, 3)`
        - ``"surface_mesh_neighbors"``: Surface mesh neighbor coordinates of shape :math:`(B, N_{surf}, K, 3)`
        - ``"surface_normals"``: Surface normals of shape :math:`(B, N_{surf}, 3)`
        - ``"surface_neighbors_normals"``: Surface neighbor normals of shape :math:`(B, N_{surf}, K, 3)`
        - ``"surface_areas"``: Surface cell areas of shape :math:`(B, N_{surf})`
        - ``"surface_neighbors_areas"``: Surface neighbor areas of shape :math:`(B, N_{surf}, K)`
        - ``"volume_mesh_centers"``: Volume mesh center coordinates of shape :math:`(B, N_{vol}, 3)`
        - ``"volume_min_max"``: Volume bounding box min/max of shape :math:`(B, 2, 3)`
        - ``"surface_min_max"``: Surface bounding box min/max of shape :math:`(B, 2, 3)`
        - ``"global_params_values"``: Global parameters (same as in solutions), shape :math:`(B, N_{params}, 1)`
        - ``"global_params_reference"``: Global parameter references for normalization, shape :math:`(B, N_{params}, 1)`

    Outputs
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        A tuple containing:

        - Volume output tensor of shape :math:`(B, N_{vol}, D_{vol})` or ``None`` if volume-only mode is disabled
        - Surface output tensor of shape :math:`(B, N_{surf}, D_{surf})` or ``None`` if surface-only mode is disabled

    Example
    -------
    >>> from physicsnemo.models.domino_transformed.model import DoMINO
    >>> from physicsnemo.models.domino_transformed.config import DEFAULT_MODEL_PARAMS
    >>> import torch
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> cfg = DEFAULT_MODEL_PARAMS  # already has model_type "combined"
    >>> model = DoMINO(
    ...         input_features=3,
    ...         output_features_vol=5,
    ...         output_features_surf=4,
    ...         model_parameters=cfg
    ...     ).to(device)
    >>> bsize = 1
    >>> nx, ny, nz = cfg.interp_res
    >>> num_neigh = cfg.num_neighbors_surface
    >>> global_features = 2
    >>> pos_normals_closest_vol = torch.randn(bsize, 100, 3).to(device)
    >>> pos_normals_com_vol = torch.randn(bsize, 100, 3).to(device)
    >>> pos_normals_com_surface = torch.randn(bsize, 100, 3).to(device)
    >>> geom_centers = torch.randn(bsize, 100, 3).to(device)
    >>> sdf_nodes = torch.randn(bsize, 100, 4).to(device)
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
    ...            "global_params_values": global_params_values,
    ...            "global_params_reference": global_params_reference,
    ...        }
    >>> output = model(input_dict)
    >>> print(f"{output[0].shape}, {output[1].shape}")
    torch.Size([1, 100, 5]), torch.Size([1, 100, 4])

    Note
    ----
    At least one of ``output_features_vol`` or ``output_features_surf`` must be specified.
    """

    def __init__(
        self,
        input_features: int,
        output_features_vol: int | None = None,
        output_features_surf: int | None = None,
        global_features: int = 2,
        model_parameters: Any = None,
    ):
        super().__init__(meta=ModelMetaData(name="DoMINO"))

        # Convert model_parameters to Config, using defaults if None
        if model_parameters is None:
            model_parameters = DEFAULT_MODEL_PARAMS
        elif not isinstance(model_parameters, Config):
            model_parameters = Config.from_hydra(model_parameters)
        # Update stored __init__ args so checkpoint JSON serialization uses the Config
        self._args["__args__"]["model_parameters"] = model_parameters

        self.output_features_vol = output_features_vol
        self.output_features_surf = output_features_surf
        self.num_sample_points_surface = model_parameters.num_neighbors_surface
        self.num_sample_points_volume = model_parameters.num_neighbors_volume
        self.activation_processor = (
            model_parameters.geometry_rep.geo_processor.activation
        )

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
        self.use_surface_normals = model_parameters.use_surface_normals
        self.use_surface_area = model_parameters.use_surface_area
        self.encode_parameters = model_parameters.encode_parameters

        if self.use_surface_normals:
            if not self.use_surface_area:
                input_features_surface = input_features + 3
            else:
                input_features_surface = input_features + 4
        else:
            input_features_surface = input_features

        if self.encode_parameters:
            # Only build parameter_model when encode_parameters is True (config.model.encode_parameters).
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
            self.parameter_model = None  # Used only when encode_parameters is True

        fusion_cfg = getattr(model_parameters, "fusion", None)
        geo_vol_out, geo_surf_out = _fusion_geometry_dims(fusion_cfg)
        self.geo_rep_volume = GeometryRep(
            input_features=input_features,
            radii=model_parameters.geometry_rep.geo_conv.volume_radii,
            neighbors_in_radius=model_parameters.geometry_rep.geo_conv.volume_neighbors_in_radius,
            model_parameters=model_parameters,
            output_dim=geo_vol_out,
        )

        self.geo_rep_surface = GeometryRep(
            input_features=input_features,
            radii=model_parameters.geometry_rep.geo_conv.surface_radii,
            neighbors_in_radius=model_parameters.geometry_rep.geo_conv.surface_neighbors_in_radius,
            model_parameters=model_parameters,
            output_dim=geo_surf_out,
        )

        # Basis functions for surface and volume
        base_layer_nn = model_parameters.nn_basis_functions.base_layer
        # Fusion-only: require hierarchical fusion
        if fusion_cfg is None or not getattr(fusion_cfg, "use_hierarchical_fusion", False):
            raise ValueError(
                "DoMINO requires fusion.use_hierarchical_fusion=True (fusion-only mode)"
            )
        self.use_hierarchical_fusion = True
        # Shared basis per branch (one FourierMLP for all variables; aggregation is per-variable)
        if self.output_features_surf is not None:
            self.nn_basis_surf = FourierMLP(
                input_features=input_features_surface,
                base_layer=model_parameters.nn_basis_functions.base_layer,
                fourier_features=model_parameters.nn_basis_functions.fourier_features,
                num_modes=model_parameters.nn_basis_functions.num_modes,
                activation=get_activation(
                    model_parameters.nn_basis_functions.activation
                ),
            )

        if self.output_features_vol is not None:
            self.nn_basis_vol = FourierMLP(
                input_features=input_features,
                base_layer=model_parameters.nn_basis_functions.base_layer,
                fourier_features=model_parameters.nn_basis_functions.fourier_features,
                num_modes=model_parameters.nn_basis_functions.num_modes,
                activation=get_activation(
                    model_parameters.nn_basis_functions.activation
                ),
            )

        # Positional encoding
        self.activation = get_activation(model_parameters.activation)
        self.use_sdf = model_parameters.use_sdf

        # Fusion + aggregation for surface (fusion-only)
        if self.output_features_surf is not None:
            global_dim_surf = (
                base_layer_p
                if self.encode_parameters
                else (fusion_cfg.global_embed_dim if fusion_cfg else 0)
            )
            self.fusion_module_surf = _build_fusion_module(
                fusion_cfg,
                base_layer_nn,
                getattr(fusion_cfg, "geometry_dim_surface", 0) or 0,
                global_dim_surf,
            )
            agg_input_surf = _fusion_aggregation_input_dim(fusion_cfg)
            agg_base = model_parameters.aggregation_model.base_layer
            agg_layers = getattr(
                model_parameters.aggregation_model, "num_hidden_layers", 2
            )
            self.agg_model_surf = nn.ModuleList(
                AggregationModel(
                    input_features=agg_input_surf,
                    output_features=1,
                    base_layer=agg_base,
                    activation=get_activation(
                        model_parameters.aggregation_model.activation
                    ),
                    num_hidden_layers=agg_layers,
                )
                for _ in range(self.num_variables_surf)
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
                fusion_module=self.fusion_module_surf,
            )

        if not self.encode_parameters:
            self.global_embed_proj = nn.Linear(
                self.global_features, fusion_cfg.global_embed_dim
            )
        else:
            self.global_embed_proj = None

        # Fusion + aggregation for volume (fusion-only)
        if self.output_features_vol is not None:
            global_dim_vol = (
                base_layer_p
                if self.encode_parameters
                else (fusion_cfg.global_embed_dim if fusion_cfg else 0)
            )
            self.fusion_module_vol = _build_fusion_module(
                fusion_cfg,
                base_layer_nn,
                getattr(fusion_cfg, "geometry_dim_volume", 0) or 0,
                global_dim_vol,
            )
            agg_input_vol = _fusion_aggregation_input_dim(fusion_cfg)
            agg_base = model_parameters.aggregation_model.base_layer
            agg_layers = getattr(
                model_parameters.aggregation_model, "num_hidden_layers", 2
            )
            self.agg_model_vol = nn.ModuleList(
                AggregationModel(
                    input_features=agg_input_vol,
                    output_features=1,
                    base_layer=agg_base,
                    activation=get_activation(
                        model_parameters.aggregation_model.activation
                    ),
                    num_hidden_layers=agg_layers,
                )
                for _ in range(self.num_variables_vol)
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
                fusion_module=self.fusion_module_vol,
            )

    def _get_global_embedding(
        self,
        mesh_centers: Float[torch.Tensor, "batch num_points 3"],
        global_params_values: Float[torch.Tensor, "batch num_params 1"],
        global_params_reference: Float[torch.Tensor, "batch num_params 1"],
    ) -> Float[torch.Tensor, "batch num_points global_dim"]:
        """Compute global embedding for fusion: (B, N, global_dim)."""
        if self.encode_parameters:
            param_enc = apply_parameter_encoding(
                mesh_centers,
                global_params_values,
                global_params_reference,
            )
            return self.parameter_model(param_enc)
        B, N = mesh_centers.shape[0], mesh_centers.shape[1]
        gp = global_params_values.squeeze(-1).unsqueeze(1).expand(B, N, -1)
        return self.global_embed_proj(gp)

    def _forward_volume(
        self,
        data_dict: dict[str, Float[torch.Tensor, "..."]],
        geo_centers: Float[torch.Tensor, "batch num_geo 3"],
        global_params_values: Float[torch.Tensor, "batch num_params 1"],
        global_params_reference: Float[torch.Tensor, "batch num_params 1"],
    ):
        """Volume branch: geometry encoding + fusion + aggregation."""
        volume_mesh_centers = data_dict["volume_mesh_centers"]
        encoding_g_vol = self.geo_rep_volume(geo_centers, volume_mesh_centers)
        if self.use_sdf:
            sdf_nodes = data_dict["sdf_nodes"]
            sdf_nodes[:, :, -3:] = volume_mesh_centers - sdf_nodes[:, :, -3:]
        global_embedding = self._get_global_embedding(
            volume_mesh_centers,
            global_params_values,
            global_params_reference,
        )
        return self.solution_calculator_vol(
            volume_mesh_centers,
            encoding_g_vol,
            encoding_node=None,
            global_embedding=global_embedding,
            global_params_values=global_params_values,
            global_params_reference=global_params_reference,
        )

    def _forward_surface(
        self,
        data_dict: dict[str, Float[torch.Tensor, "..."]],
        encoding_g_surf: Float[torch.Tensor, "..."],
        global_params_values: Float[torch.Tensor, "batch num_params 1"],
        global_params_reference: Float[torch.Tensor, "batch num_params 1"],
    ):
        """Surface branch: geometry encoding + fusion + aggregation."""
        surface_mesh_centers = data_dict["surface_mesh_centers"]
        surface_normals = data_dict["surface_normals"]
        surface_areas = data_dict["surface_areas"].unsqueeze(-1)
        surface_mesh_neighbors = data_dict["surface_mesh_neighbors"]
        surface_neighbors_normals = data_dict["surface_neighbors_normals"]
        surface_neighbors_areas = data_dict["surface_neighbors_areas"].unsqueeze(-1)
        global_embedding = self._get_global_embedding(
            surface_mesh_centers,
            global_params_values,
            global_params_reference,
        )
        return self.solution_calculator_surf(
            surface_mesh_centers,
            encoding_g_surf,
            encoding_node=None,
            global_embedding=global_embedding,
            surface_mesh_neighbors=surface_mesh_neighbors,
            surface_normals=surface_normals,
            surface_neighbors_normals=surface_neighbors_normals,
            surface_areas=surface_areas,
            surface_neighbors_areas=surface_neighbors_areas,
            global_params_values=global_params_values,
            global_params_reference=global_params_reference,
        )

    def forward(
        self,
        data_dict: dict[str, Float[torch.Tensor, "..."]],
    ) -> tuple[
        Float[torch.Tensor, "batch num_vol out_vol"] | None,
        Float[torch.Tensor, "batch num_surf out_surf"] | None,
    ]:
        r"""
        Perform forward pass of the DoMINO model.

        Parameters
        ----------
        data_dict : dict[str, torch.Tensor]
            Dictionary containing input tensors. See class docstring for required keys.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            Tuple of (volume_output, surface_output). Either may be ``None`` if
            the corresponding output mode is disabled.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            required_keys = [
                "geometry_coordinates",
                "global_params_values",
                "global_params_reference",
            ]
            if self.output_features_vol is not None:
                required_keys.extend(
                    [
                        "volume_mesh_centers",
                    ]
                )
            if self.output_features_surf is not None:
                required_keys.extend(
                    [
                        "surface_mesh_centers",
                        "surface_mesh_neighbors",
                        "surface_normals",
                        "surface_neighbors_normals",
                        "surface_areas",
                        "surface_neighbors_areas",
                    ]
                )

            missing_keys = [k for k in required_keys if k not in data_dict]
            if missing_keys:
                raise ValueError(f"Missing required keys in data_dict: {missing_keys}")

        geo_centers = data_dict["geometry_coordinates"]
        global_params_values = data_dict["global_params_values"]
        global_params_reference = data_dict["global_params_reference"]

        if self.output_features_vol is not None:
            output_vol = self._forward_volume(
                data_dict,
                geo_centers,
                global_params_values,
                global_params_reference,
            )
        else:
            output_vol = None

        if self.output_features_surf is not None:
            surface_mesh_centers = data_dict["surface_mesh_centers"]
            encoding_g_surf = self.geo_rep_surface(
                geo_centers, surface_mesh_centers
            )
            output_surf = self._forward_surface(
                data_dict,
                encoding_g_surf,
                global_params_values,
                global_params_reference,
            )
        else:
            output_surf = None

        return output_vol, output_surf
