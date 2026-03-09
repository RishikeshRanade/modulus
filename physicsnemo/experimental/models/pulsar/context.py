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
Pulsar context encoding modules.

This module defines the geometry and global-parameter encoders that construct
context tokens and their projection to physical slices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.experimental.models.geotransolver.context_projector import (
    ContextProjector,
)
from physicsnemo.nn import FourierMLP, Mlp, get_activation

from physicsnemo.experimental.models.pulsar.layers import (
    LearnedTokenPooler,
    MultiGridTokenPooler,
)


class GeometryContextEncoder(Module):
    r"""
    Geometry context encoder for Pulsar.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for context tokens.
    geo_context_tokens : int
        Number of geometry context tokens.
    geometry_fourier_features : bool
        Whether to apply Fourier features to geometry coordinates.
    geometry_num_modes : int
        Number of Fourier modes for geometry encoding.
    dropout : float, optional, default=0.0
        Dropout rate for context encoders.
    use_multigrid_pooler : bool, optional, default=False
        Whether to use multi-grid pooling for geometry tokens.
    multigrid_grid_sizes : list[tuple[int, int, int]]
        Voxel grid sizes for multi-grid pooling.
    multigrid_tokens_per_level : int | list[int]
        Number of pooled tokens per grid level.
    slice_num : int, optional, default=32
        Number of physical slices.
    num_heads : int, optional, default=8
        Number of attention heads.
    activation : str, optional, default="gelu"
        Activation function name.

    Forward
    -------
    geometry_points : torch.Tensor
        Geometry points of shape :math:`(B, N_g, 3)`.
    global_params_values : torch.Tensor
        Not used; see :class:`GlobalParamsContextEncoder`.

    Outputs
    -------
    torch.Tensor
        Geometry context tokens of shape :math:`(B, S, C)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.pulsar.context import GeometryContextEncoder
    >>> ctx = GeometryContextEncoder(hidden_dim=64, geo_context_tokens=16)
    >>> geometry = torch.randn(2, 512, 3)
    >>> tokens = ctx(geometry)
    >>> tokens.shape
    torch.Size([2, 16, 64])
    """

    def __init__(
        self,
        hidden_dim: int,
        geo_context_tokens: int,
        geometry_fourier_features: bool,
        geometry_num_modes: int,
        dropout: float = 0.0,
        use_multigrid_pooler: bool = False,
        multigrid_grid_sizes: list[tuple[int, int, int]] | None = None,
        multigrid_tokens_per_level: int | list[int] = 32,
        activation: str = "gelu",
    ) -> None:
        super().__init__(meta=None)
        self.hidden_dim = hidden_dim

        act_layer = get_activation(activation)

        self.geometry_encoder = FourierMLP(
            input_features=3,
            base_layer=hidden_dim,
            fourier_features=geometry_fourier_features,
            num_modes=geometry_num_modes,
            activation=act_layer,
        )
        if use_multigrid_pooler:
            self.geo_pooler = MultiGridTokenPooler(
                input_dim=hidden_dim,
                grid_sizes=multigrid_grid_sizes or [(16, 16, 16), (8, 8, 8)],
                tokens_per_level=multigrid_tokens_per_level,
                dropout=dropout,
            )
        else:
            self.geo_pooler = LearnedTokenPooler(
                input_dim=hidden_dim,
                num_context_tokens=geo_context_tokens,
                dropout=dropout,
            )

    def forward(
        self,
        geometry_points: Float[torch.Tensor, "batch num_geo 3"],
    ) -> Float[torch.Tensor, "batch context_tokens hidden_dim"]:
        r"""
        Build geometry context tokens.

        Parameters
        ----------
        geometry_points : torch.Tensor
            Geometry points of shape :math:`(B, N_g, 3)`.
        Returns
        -------
        torch.Tensor
            Geometry context tokens.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if geometry_points.ndim != 3 or geometry_points.shape[-1] != 3:
                raise ValueError(
                    "Expected geometry_points of shape (B, N_g, 3), got "
                    f"{tuple(geometry_points.shape)}"
                )

        geo_tokens = self.encode(geometry_points)
        return self.pool(geometry_points, geo_tokens)

    def encode(
        self,
        geometry_points: Float[torch.Tensor, "batch num_geo 3"],
    ) -> Float[torch.Tensor, "batch num_geo hidden_dim"]:
        r"""
        Encode geometry points into per-point tokens.
        """
        return self.geometry_encoder(geometry_points)

    def pool(
        self,
        geometry_points: Float[torch.Tensor, "batch num_geo 3"],
        geo_tokens: Float[torch.Tensor, "batch num_geo hidden_dim"],
    ) -> Float[torch.Tensor, "batch context_tokens hidden_dim"]:
        r"""
        Pool per-point geometry tokens into context tokens.
        """
        if isinstance(self.geo_pooler, MultiGridTokenPooler):
            return self.geo_pooler(geometry_points, geo_tokens)
        return self.geo_pooler(geo_tokens)


class GlobalParamsContextEncoder(Module):
    r"""
    Global parameter context encoder for Pulsar.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for global parameter tokens.
    bc_context_tokens : int
        Number of global parameter context tokens.
    bc_dim : int
        Dimension of the global parameter conditioning.
    dropout : float, optional, default=0.0
        Dropout rate for the encoder.
    activation : str, optional, default="gelu"
        Activation function name.

    Forward
    -------
    global_params_values : torch.Tensor
        Global parameters of shape :math:`(B, N_{bc}, D_{bc})`.

    Outputs
    -------
    torch.Tensor
        Global parameter context tokens of shape :math:`(B, S, C)`.
    """

    def __init__(
        self,
        hidden_dim: int,
        bc_context_tokens: int,
        bc_dim: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__(meta=None)
        self.hidden_dim = hidden_dim
        self.bc_dim = bc_dim
        act_layer = get_activation(activation)
        self.bc_encoder = Mlp(
            in_features=bc_dim,
            hidden_features=[hidden_dim, hidden_dim],
            out_features=hidden_dim,
            act_layer=act_layer,
            drop=dropout,
        )
        self.bc_pooler = LearnedTokenPooler(
            input_dim=hidden_dim,
            num_context_tokens=bc_context_tokens,
            dropout=dropout,
        )

    def forward(
        self,
        global_params_values: Float[torch.Tensor, "batch num_bc bc_dim"],
    ) -> Float[torch.Tensor, "batch context_tokens hidden_dim"]:
        r"""
        Build global parameter context tokens.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if global_params_values.ndim != 3 or global_params_values.shape[-1] != self.bc_dim:
                raise ValueError(
                    "Expected global_params_values of shape (B, N_bc, D_bc), got "
                    f"{tuple(global_params_values.shape)}"
                )
        bc_tokens = self.bc_encoder(global_params_values)
        return self.bc_pooler(bc_tokens)


class ContextSliceBuilder(Module):
    r"""
    Context slice builder for Pulsar.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for context tokens.
    slice_num : int
        Number of physical slices.
    num_heads : int
        Number of attention heads.
    dropout : float, optional, default=0.0
        Dropout rate.
    use_anchor_context : bool, optional, default=False
        Whether to fuse anchor-derived context into the global slices.
    num_context_tokens : int | None, optional, default=None
        Number of context tokens (e.g. geo + bc). Required when
        ``use_anchor_context=True`` to pool anchor tokens before fusion.

    Forward
    -------
    context_tokens : torch.Tensor
        Context tokens of shape :math:`(B, S, C)`.
    anchor_tokens : torch.Tensor | None
        Optional anchor tokens of shape :math:`(B, N_a, C)`.

    Outputs
    -------
    torch.Tensor
        Context slices of shape :math:`(B, H, S, D)`.
    """

    def __init__(
        self,
        hidden_dim: int,
        slice_num: int,
        num_heads: int,
        dropout: float = 0.0,
        use_anchor_context: bool = False,
        num_context_tokens: int | None = None,
    ) -> None:
        super().__init__(meta=None)
        self.use_anchor_context = use_anchor_context
        if use_anchor_context and num_context_tokens is None:
            raise ValueError(
                "num_context_tokens is required when use_anchor_context=True"
            )
        self.context_projector = ContextProjector(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim,
            dropout=dropout,
            slice_num=slice_num,
            use_te=False,
            plus=False,
        )
        self.anchor_context_projector = ContextProjector(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim,
            dropout=dropout,
            slice_num=slice_num,
            use_te=False,
            plus=False,
        )
        self.anchor_context_fuser = nn.Linear(hidden_dim * 2, hidden_dim)
        if use_anchor_context and num_context_tokens is not None:
            self.anchor_pooler = LearnedTokenPooler(
                input_dim=hidden_dim,
                num_context_tokens=num_context_tokens,
                dropout=dropout,
            )
        else:
            self.anchor_pooler = None

    def forward(
        self,
        context_tokens: Float[torch.Tensor, "batch context_tokens hidden_dim"],
        anchor_tokens: Float[torch.Tensor, "batch num_anchor hidden_dim"] | None = None,
    ) -> Float[torch.Tensor, "batch heads slices dim"]:
        r"""
        Build context slices from context tokens.
        """
        context_slices = self.context_projector(context_tokens)
        if self.use_anchor_context:
            if anchor_tokens is None:
                raise ValueError(
                    "use_anchor_context=True requires anchor_tokens input"
                )
            anchor_slices = self.anchor_context_projector(anchor_tokens)
            # Pool anchor tokens to same length as context_tokens for fusion
            pooled_anchor = self.anchor_pooler(anchor_tokens)
            fused_context = torch.cat([context_tokens, pooled_anchor], dim=-1)
            fused_context = self.anchor_context_fuser(fused_context)
            fused_slices = self.context_projector(fused_context)
            context_slices = 0.5 * (fused_slices + anchor_slices)
        return context_slices


__all__ = [
    "GeometryContextEncoder",
    "GlobalParamsContextEncoder",
    "ContextSliceBuilder",
]

