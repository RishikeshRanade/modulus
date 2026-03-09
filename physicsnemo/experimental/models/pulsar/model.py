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
Pulsar model architecture (PDE Unstructured Latent Simulator with Anchored Attention based Representations).

This module defines a novel architecture for incompressible flow prediction on
unstructured surface and volume point clouds with geometry- and BC-conditioned
latent attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.experimental.models.pulsar.bq_injector import BQFeatureInjector
from physicsnemo.experimental.models.pulsar.context import (
    ContextSliceBuilder,
    GeometryContextEncoder,
    GlobalParamsContextEncoder,
)
from physicsnemo.nn import FourierMLP, Mlp, get_activation

from physicsnemo.experimental.models.pulsar.layers import GlobalConditioner
from physicsnemo.experimental.models.pulsar.physics import PulsarPhysicsStack


class PulsarMetaData(ModelMetaData):
    """Metadata for the Pulsar model."""

    name: str = "Pulsar"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class PulsarModel(Module):
    r"""
    Pulsar model for unstructured surface and volume point clouds.

    This architecture separates geometry and boundary condition (BC) encoding from
    the physics prediction stack. Geometry and BC tokens are pooled into context
    tokens and projected onto physical state slices used by GALE blocks.

    Parameters
    ----------
    n_hidden : int, optional, default=256
        Hidden dimension used across all internal embeddings.
    n_layers : int, optional, default=4
        Number of anchor-slice blocks in the physics stack.
    n_head : int, optional, default=8
        Number of attention heads.
    slice_num : int, optional, default=32
        Number of physical state slices for context projection.
    dropout : float, optional, default=0.0
        Dropout rate used in blocks and token pooling.
    geo_context_tokens : int, optional, default=64
        Number of pooled geometry context tokens.
    bc_context_tokens : int, optional, default=16
        Number of pooled BC context tokens.
    geometry_fourier_features : bool, optional, default=True
        Whether to apply Fourier features to geometry coordinates.
    geometry_num_modes : int, optional, default=8
        Number of Fourier modes for geometry encoding.
    global_dim : int, optional, default=2
        Dimension of global parameter conditioning.
    output_dim_volume : int, optional, default=5
        Number of predicted volume output channels.
    output_dim_surface : int, optional, default=4
        Number of predicted surface output channels.
    enable_volume : bool, optional, default=True
        Whether to predict volume fields.
    enable_surface : bool, optional, default=True
        Whether to predict surface fields.
    enable_force_head : bool, optional, default=False
        Whether to enable aerodynamic force prediction head.
    force_dim : int, optional, default=3
        Number of force components to predict.
    use_bq_features : bool, optional, default=True
        Whether to add ball-query local features to token embeddings.
    geo_bq_radii : list[float], optional, default=[0.05, 0.25]
        Radii for multi-scale geometry ball-query features.
    geo_bq_neighbors : list[int], optional, default=[8, 32]
        Neighbor counts for each geometry ball-query radius.
    surf_bq_radii : list[float], optional, default=[0.05, 0.25]
        Radii for multi-scale surface ball-query features.
    surf_bq_neighbors : list[int], optional, default=[8, 32]
        Neighbor counts for each surface ball-query radius.
    vol_bq_radii : list[float], optional, default=[0.05, 0.25]
        Radii for multi-scale volume ball-query features.
    vol_bq_neighbors : list[int], optional, default=[8, 32]
        Neighbor counts for each volume ball-query radius.
    geo_surf_bq_radii : list[float], optional, default=[0.05, 0.25]
        Radii for geometry-to-surface cross ball-query features.
    geo_surf_bq_neighbors : list[int], optional, default=[8, 32]
        Neighbor counts for each geometry-to-surface cross radius.
    geo_vol_bq_radii : list[float], optional, default=[0.05, 0.25]
        Radii for geometry-to-volume cross ball-query features.
    geo_vol_bq_neighbors : list[int], optional, default=[8, 32]
        Neighbor counts for each geometry-to-volume cross radius.
    surf_geo_bq_radii : list[float], optional, default=[0.05, 0.25]
        Radii for surface-to-geometry cross ball-query features.
    surf_geo_bq_neighbors : list[int], optional, default=[8, 32]
        Neighbor counts for each surface-to-geometry cross radius.
    vol_geo_bq_radii : list[float], optional, default=[0.05, 0.25]
        Radii for volume-to-geometry cross ball-query features.
    vol_geo_bq_neighbors : list[int], optional, default=[8, 32]
        Neighbor counts for each volume-to-geometry cross radius.
    n_hidden_local : int, optional, default=64
        Hidden dimension per ball-query scale.
    use_multigrid_pooler : bool, optional, default=False
        Whether to use multi-grid pooling for geometry tokens.
    multigrid_grid_sizes : list[tuple[int, int, int]], optional
        Voxel grid sizes for multi-grid pooling.
    multigrid_tokens_per_level : int | list[int], optional
        Number of pooled tokens per grid level.
    anchor_grid_sizes : list[tuple[int, int, int]], optional
        Anchor grid sizes per level for anchor-slice attention.
    anchor_radii : list[float], optional
        Anchor attention radii per level.
    anchor_gale_layers : int, optional, default=1
        Number of GALE layers applied to anchor tokens.
    anchor_learnable : bool, optional, default=True
        Whether to use learnable anchor coordinates.
    use_anchor_context : bool, optional, default=False
        Whether to add anchor-derived context slices to the global context.
    anchor_gale_slice_num : int, optional, default=32
        Slice count for anchor GALE blocks.
    anchor_gale_mlp_ratio : int, optional, default=4
        MLP ratio for anchor GALE blocks.
    act : str, optional, default="gelu"
        Activation function name.

    Forward
    -------
    geometry_points : torch.Tensor
        STL-derived geometry points of shape :math:`(B, N_g, 3)`.
    surface_points : torch.Tensor
        Surface point cloud of shape :math:`(B, N_s, 3)`.
    volume_points : torch.Tensor
        Volume point cloud of shape :math:`(B, N_v, 3)`.
    bc_values : torch.Tensor
        Boundary condition parameters of shape :math:`(B, N_{bc}, D_{bc})`.

    Outputs
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of:
        - Volume predictions of shape :math:`(B, N_v, D_{vol})`
        - Surface predictions of shape :math:`(B, N_s, D_{surf})`

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.pulsar import PulsarModel
    >>> model = PulsarModel(n_hidden=128, output_dim_volume=5, output_dim_surface=4)
    >>> geometry_points = torch.randn(2, 2048, 3)
    >>> surface_points = torch.randn(2, 1024, 3)
    >>> volume_points = torch.randn(2, 4096, 3)
    >>> bc_values = torch.randn(2, 4, 2)
    >>> pred_vol, pred_surf = model(
    ...     geometry_points=geometry_points,
    ...     surface_points=surface_points,
    ...     volume_points=volume_points,
    ...     bc_values=bc_values,
    ... )
    >>> pred_vol.shape, pred_surf.shape
    (torch.Size([2, 4096, 5]), torch.Size([2, 1024, 4]))
    """

    def __init__(
        self,
        n_hidden: int = 256,
        n_layers: int = 4,
        n_head: int = 8,
        slice_num: int = 32,
        dropout: float = 0.0,
        geo_context_tokens: int = 64,
        bc_context_tokens: int = 16,
        geometry_fourier_features: bool = True,
        geometry_num_modes: int = 8,
        global_dim: int = 2,
        output_dim_volume: int = 5,
        output_dim_surface: int = 4,
        enable_volume: bool = True,
        enable_surface: bool = True,
        enable_force_head: bool = False,
        force_dim: int = 3,
        use_bq_features: bool = True,
        geo_bq_radii: list[float] | None = None,
        geo_bq_neighbors: list[int] | None = None,
        surf_bq_radii: list[float] | None = None,
        surf_bq_neighbors: list[int] | None = None,
        vol_bq_radii: list[float] | None = None,
        vol_bq_neighbors: list[int] | None = None,
        geo_surf_bq_radii: list[float] | None = None,
        geo_surf_bq_neighbors: list[int] | None = None,
        geo_vol_bq_radii: list[float] | None = None,
        geo_vol_bq_neighbors: list[int] | None = None,
        surf_geo_bq_radii: list[float] | None = None,
        surf_geo_bq_neighbors: list[int] | None = None,
        vol_geo_bq_radii: list[float] | None = None,
        vol_geo_bq_neighbors: list[int] | None = None,
        n_hidden_local: int = 64,
        use_multigrid_pooler: bool = False,
        multigrid_grid_sizes: list[tuple[int, int, int]] | None = None,
        multigrid_tokens_per_level: int | list[int] = 32,
        anchor_grid_sizes: list[tuple[int, int, int]] | None = None,
        anchor_radii: list[float] | None = None,
        anchor_gale_layers: int = 1,
        anchor_learnable: bool = True,
        use_anchor_context: bool = False,
        anchor_gale_slice_num: int = 32,
        anchor_gale_mlp_ratio: int = 4,
        act: str = "gelu",
    ) -> None:
        super().__init__(meta=PulsarMetaData())
        self.hidden_dim = n_hidden
        self.num_layers = n_layers
        self.num_heads = n_head
        self.slice_num = slice_num
        self.geo_context_tokens = geo_context_tokens
        self.bc_context_tokens = bc_context_tokens
        self.global_dim = global_dim
        self.output_dim_volume = output_dim_volume
        self.output_dim_surface = output_dim_surface
        self.enable_volume = enable_volume
        self.enable_surface = enable_surface
        self.enable_force_head = enable_force_head
        self.force_dim = force_dim
        self.use_bq_features = use_bq_features
        self.geo_bq_radii = geo_bq_radii or [0.05, 0.25]
        self.geo_bq_neighbors = geo_bq_neighbors or [8, 32]
        self.surf_bq_radii = surf_bq_radii or [0.05, 0.25]
        self.surf_bq_neighbors = surf_bq_neighbors or [8, 32]
        self.vol_bq_radii = vol_bq_radii or [0.05, 0.25]
        self.vol_bq_neighbors = vol_bq_neighbors or [8, 32]
        self.geo_surf_bq_radii = geo_surf_bq_radii or [0.05, 0.25]
        self.geo_surf_bq_neighbors = geo_surf_bq_neighbors or [8, 32]
        self.geo_vol_bq_radii = geo_vol_bq_radii or [0.05, 0.25]
        self.geo_vol_bq_neighbors = geo_vol_bq_neighbors or [8, 32]
        self.surf_geo_bq_radii = surf_geo_bq_radii or [0.05, 0.25]
        self.surf_geo_bq_neighbors = surf_geo_bq_neighbors or [8, 32]
        self.vol_geo_bq_radii = vol_geo_bq_radii or [0.05, 0.25]
        self.vol_geo_bq_neighbors = vol_geo_bq_neighbors or [8, 32]
        self.bq_hidden_dim = n_hidden_local
        self.use_multigrid_pooler = use_multigrid_pooler
        self.multigrid_grid_sizes = multigrid_grid_sizes or [(16, 16, 16), (8, 8, 8)]
        self.multigrid_tokens_per_level = multigrid_tokens_per_level
        self.anchor_grid_sizes = anchor_grid_sizes or [(16, 16, 16), (8, 8, 8)]
        self.anchor_radii = anchor_radii or [0.2, 0.5]
        self.anchor_gale_layers = anchor_gale_layers
        self.anchor_learnable = anchor_learnable
        self.use_anchor_context = use_anchor_context
        self.anchor_gale_slice_num = anchor_gale_slice_num
        self.anchor_gale_mlp_ratio = anchor_gale_mlp_ratio
        self.activation = act

        act_layer = get_activation(act)

        if len(self.geo_bq_radii) != len(self.geo_bq_neighbors):
            raise ValueError(
                "geo_bq_radii and geo_bq_neighbors must have the same length, got "
                f"{len(self.geo_bq_radii)} and {len(self.geo_bq_neighbors)}"
            )
        if len(self.surf_bq_radii) != len(self.surf_bq_neighbors):
            raise ValueError(
                "surf_bq_radii and surf_bq_neighbors must have the same length, got "
                f"{len(self.surf_bq_radii)} and {len(self.surf_bq_neighbors)}"
            )
        if len(self.vol_bq_radii) != len(self.vol_bq_neighbors):
            raise ValueError(
                "vol_bq_radii and vol_bq_neighbors must have the same length, got "
                f"{len(self.vol_bq_radii)} and {len(self.vol_bq_neighbors)}"
            )
        if len(self.geo_surf_bq_radii) != len(self.geo_surf_bq_neighbors):
            raise ValueError(
                "geo_surf_bq_radii and geo_surf_bq_neighbors must have the same length, got "
                f"{len(self.geo_surf_bq_radii)} and {len(self.geo_surf_bq_neighbors)}"
            )
        if len(self.geo_vol_bq_radii) != len(self.geo_vol_bq_neighbors):
            raise ValueError(
                "geo_vol_bq_radii and geo_vol_bq_neighbors must have the same length, got "
                f"{len(self.geo_vol_bq_radii)} and {len(self.geo_vol_bq_neighbors)}"
            )
        if len(self.surf_geo_bq_radii) != len(self.surf_geo_bq_neighbors):
            raise ValueError(
                "surf_geo_bq_radii and surf_geo_bq_neighbors must have the same length, got "
                f"{len(self.surf_geo_bq_radii)} and {len(self.surf_geo_bq_neighbors)}"
            )
        if len(self.vol_geo_bq_radii) != len(self.vol_geo_bq_neighbors):
            raise ValueError(
                "vol_geo_bq_radii and vol_geo_bq_neighbors must have the same length, got "
                f"{len(self.vol_geo_bq_radii)} and {len(self.vol_geo_bq_neighbors)}"
            )

        # Context encoders
        self.geometry_context = GeometryContextEncoder(
            hidden_dim=n_hidden,
            geo_context_tokens=geo_context_tokens,
            geometry_fourier_features=geometry_fourier_features,
            geometry_num_modes=geometry_num_modes,
            dropout=dropout,
            use_multigrid_pooler=self.use_multigrid_pooler,
            multigrid_grid_sizes=self.multigrid_grid_sizes,
            multigrid_tokens_per_level=self.multigrid_tokens_per_level,
            activation=act,
        )
        self.global_params_context = GlobalParamsContextEncoder(
            hidden_dim=n_hidden,
            bc_context_tokens=bc_context_tokens,
            bc_dim=global_dim,
            dropout=dropout,
            activation=act,
        )
        self.context_slices = ContextSliceBuilder(
            hidden_dim=n_hidden,
            slice_num=slice_num,
            num_heads=n_head,
            dropout=dropout,
            use_anchor_context=self.use_anchor_context,
            num_context_tokens=geo_context_tokens + bc_context_tokens,
        )

        # Optional ball-query feature injection
        self.bq_injector = BQFeatureInjector(
            hidden_dim=n_hidden,
            bq_hidden_dim=n_hidden_local,
            geo_bq_radii=self.geo_bq_radii,
            geo_bq_neighbors=self.geo_bq_neighbors,
            surf_bq_radii=self.surf_bq_radii,
            surf_bq_neighbors=self.surf_bq_neighbors,
            vol_bq_radii=self.vol_bq_radii,
            vol_bq_neighbors=self.vol_bq_neighbors,
            geo_surf_bq_radii=self.geo_surf_bq_radii,
            geo_surf_bq_neighbors=self.geo_surf_bq_neighbors,
            geo_vol_bq_radii=self.geo_vol_bq_radii,
            geo_vol_bq_neighbors=self.geo_vol_bq_neighbors,
            surf_geo_bq_radii=self.surf_geo_bq_radii,
            surf_geo_bq_neighbors=self.surf_geo_bq_neighbors,
            vol_geo_bq_radii=self.vol_geo_bq_radii,
            vol_geo_bq_neighbors=self.vol_geo_bq_neighbors,
        )

        # Encode surface/volume point clouds
        if self.enable_surface:
            self.surface_encoder = FourierMLP(
                input_features=3,
                base_layer=self.hidden_dim,
                fourier_features=True,
                num_modes=geometry_num_modes,
                activation=act_layer,
            )
        else:
            self.surface_encoder = None

        if self.enable_volume:
            self.volume_encoder = FourierMLP(
                input_features=3,
                base_layer=self.hidden_dim,
                fourier_features=True,
                num_modes=geometry_num_modes,
                activation=act_layer,
            )
        else:
            self.volume_encoder = None

        # Apply global conditioning before physics blocks
        self.global_conditioner = GlobalConditioner(self.hidden_dim, dropout=dropout)

        # Physics stack (anchor-slice attention only; only build enabled branch to avoid DDP unused-param errors)
        self.physics_stack = PulsarPhysicsStack(
            hidden_dim=n_hidden,
            num_heads=n_head,
            num_layers=n_layers,
            dropout=dropout,
            anchor_grid_sizes=self.anchor_grid_sizes,
            anchor_radii=self.anchor_radii,
            anchor_gale_layers=self.anchor_gale_layers,
            anchor_gale_slice_num=self.anchor_gale_slice_num,
            anchor_gale_mlp_ratio=self.anchor_gale_mlp_ratio,
            anchor_learnable=self.anchor_learnable,
            activation=act,
            enable_surface=self.enable_surface,
            enable_volume=self.enable_volume,
        )

        # Output heads
        if self.enable_surface:
            self.surface_head = Mlp(
                in_features=self.hidden_dim,
                hidden_features=[self.hidden_dim, self.hidden_dim],
                out_features=output_dim_surface,
                act_layer=act_layer,
                drop=dropout,
            )
        else:
            self.surface_head = None

        if self.enable_volume:
            self.volume_head = Mlp(
                in_features=self.hidden_dim,
                hidden_features=[self.hidden_dim, self.hidden_dim],
                out_features=output_dim_volume,
                act_layer=act_layer,
                drop=dropout,
            )
        else:
            self.volume_head = None
        self.force_head = None
        if self.enable_force_head:
            self.force_head = Mlp(
                in_features=self.hidden_dim,
                hidden_features=[self.hidden_dim, self.hidden_dim],
                out_features=force_dim,
                act_layer=act_layer,
                drop=dropout,
            )

    def forward(
        self,
        geometry_points: Float[torch.Tensor, "batch num_geo 3"],
        surface_points: Float[torch.Tensor, "batch num_surf 3"] | None = None,
        volume_points: Float[torch.Tensor, "batch num_vol 3"] | None = None,
        global_params_values: Float[torch.Tensor, "batch num_bc global_dim"] | None = None,
        bc_values: Float[torch.Tensor, "batch num_bc global_dim"] | None = None,
        return_forces: bool = False,
    ) -> (
        tuple[
            Float[torch.Tensor, "batch num_vol out_vol"] | None,
            Float[torch.Tensor, "batch num_surf out_surf"] | None,
        ]
        | tuple[
            Float[torch.Tensor, "batch num_vol out_vol"] | None,
            Float[torch.Tensor, "batch num_surf out_surf"] | None,
            Float[torch.Tensor, "batch force_dim"],
        ]
    ):
        r"""
        Forward pass of the Pulsar model.

        Parameters
        ----------
        geometry_points : torch.Tensor
            STL-derived geometry points of shape :math:`(B, N_g, 3)`.
        surface_points : torch.Tensor
            Surface point cloud of shape :math:`(B, N_s, 3)`.
        volume_points : torch.Tensor
            Volume point cloud of shape :math:`(B, N_v, 3)`.
        global_params_values : torch.Tensor | None
            Global parameter conditioning of shape :math:`(B, N_{bc}, D_{bc})`.
        bc_values : torch.Tensor | None
            Alias for ``global_params_values`` for backward compatibility.
        return_forces : bool, optional, default=False
            Whether to return the aerodynamic force prediction.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor] or tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Volume and surface predictions with shapes
            :math:`(B, N_v, D_{vol})` and :math:`(B, N_s, D_{surf})`.
            If ``return_forces=True``, also returns force predictions of shape
            :math:`(B, D_{force})`.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if geometry_points.ndim != 3 or geometry_points.shape[-1] != 3:
                raise ValueError(
                    "Expected geometry_points of shape (B, N_g, 3), got "
                    f"{tuple(geometry_points.shape)}"
                )
            if self.enable_surface:
                if surface_points is None:
                    raise ValueError("surface_points is required when enable_surface=True")
                if surface_points.ndim != 3 or surface_points.shape[-1] != 3:
                    raise ValueError(
                        "Expected surface_points of shape (B, N_s, 3), got "
                        f"{tuple(surface_points.shape)}"
                    )
            if self.enable_volume:
                if volume_points is None:
                    raise ValueError("volume_points is required when enable_volume=True")
                if volume_points.ndim != 3 or volume_points.shape[-1] != 3:
                    raise ValueError(
                        "Expected volume_points of shape (B, N_v, 3), got "
                        f"{tuple(volume_points.shape)}"
                    )
            if global_params_values is None:
                global_params_values = bc_values
            if global_params_values is None:
                raise ValueError("global_params_values is required")
            if global_params_values.ndim != 3 or global_params_values.shape[-1] != self.global_dim:
                raise ValueError(
                    "Expected global_params_values of shape (B, N_bc, D_bc), got "
                    f"{tuple(global_params_values.shape)}"
                )
            if return_forces and not self.enable_force_head:
                raise ValueError(
                    "return_forces=True requires enable_force_head=True"
                )

        # Encode geometry/surface/volume point tokens
        geo_tokens = self.geometry_context.encode(geometry_points)
        surf_tokens = self.surface_encoder(surface_points) if self.enable_surface else None
        vol_tokens = self.volume_encoder(volume_points) if self.enable_volume else None

        if self.use_bq_features:
            geo_tokens, surf_tokens, vol_tokens = self.bq_injector(
                geometry_points=geometry_points,
                surface_points=surface_points,
                volume_points=volume_points,
                geo_tokens=geo_tokens,
                surf_tokens=surf_tokens,
                vol_tokens=vol_tokens,
            )

        # Build context tokens/slices
        pooled_geo = self.geometry_context.pool(geometry_points, geo_tokens)
        pooled_bc = self.global_params_context(global_params_values)
        context_tokens = torch.cat([pooled_geo, pooled_bc], dim=1)

        anchor_context_tokens = None
        if self.use_anchor_context:
            anchor_inputs = []
            if self.enable_surface:
                anchor_inputs.append(surf_tokens)
            if self.enable_volume:
                anchor_inputs.append(vol_tokens)
            if len(anchor_inputs) == 0:
                raise ValueError(
                    "use_anchor_context=True requires enable_surface or enable_volume"
                )
            anchor_context_tokens = torch.cat(anchor_inputs, dim=1)

        context_slices = self.context_slices(
            context_tokens=context_tokens,
            anchor_tokens=anchor_context_tokens,
        )

        # Apply global conditioning
        if self.enable_surface:
            surf_tokens = self.global_conditioner(surf_tokens, context_tokens)
        if self.enable_volume:
            vol_tokens = self.global_conditioner(vol_tokens, context_tokens)

        # Run physics blocks with shared context
        surf_tokens, vol_tokens = self.physics_stack(
            surface_points=surface_points,
            volume_points=volume_points,
            surface_tokens=surf_tokens,
            volume_tokens=vol_tokens,
            context_slices=context_slices,
            enable_surface=self.enable_surface,
            enable_volume=self.enable_volume,
        )

        # Project to outputs
        surf_out = self.surface_head(surf_tokens) if self.enable_surface else None
        vol_out = self.volume_head(vol_tokens) if self.enable_volume else None

        if self.enable_force_head:
            # Predict global forces from pooled context
            global_context = context_tokens.mean(dim=1)  # (B, C)
            forces = self.force_head(global_context)
            if return_forces:
                return vol_out, surf_out, forces

        return vol_out, surf_out


__all__ = ["PulsarModel", "PulsarMetaData"]

