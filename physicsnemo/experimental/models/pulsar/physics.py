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
Pulsar physics stacks.

This module provides the physics backbone logic for Pulsar using
anchor-slice attention blocks.
"""

from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.experimental.models.pulsar.attention import AnchorSliceBlock


class PulsarPhysicsStack(Module):
    r"""
    Physics stack for Pulsar (anchor-slice attention only).

    Parameters
    ----------
    hidden_dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of physics layers.
    dropout : float
        Dropout rate.
    anchor_grid_sizes : list[tuple[int, int, int]]
        Anchor grid sizes per level.
    anchor_radii : list[float]
        Anchor radii per level.
    anchor_gale_layers : int, optional, default=1
        Number of GALE layers applied to anchor tokens.
    anchor_gale_slice_num : int, optional, default=32
        Slice count for anchor GALE blocks.
    anchor_gale_mlp_ratio : int, optional, default=4
        MLP ratio for anchor GALE blocks.
    anchor_learnable : bool, optional, default=True
        Whether anchors are learnable.
    activation : str, optional, default="gelu"
        Activation function name.
    enable_surface : bool, optional, default=True
        Whether to create surface anchor blocks (avoids unused params when volume-only).
    enable_volume : bool, optional, default=True
        Whether to create volume anchor blocks (avoids unused params when surface-only).

    Forward
    -------
    surface_points : torch.Tensor | None
        Surface points of shape :math:`(B, N_s, 3)` if enabled.
    volume_points : torch.Tensor | None
        Volume points of shape :math:`(B, N_v, 3)` if enabled.
    surface_tokens : torch.Tensor | None
        Surface tokens of shape :math:`(B, N_s, C)` if enabled.
    volume_tokens : torch.Tensor | None
        Volume tokens of shape :math:`(B, N_v, C)` if enabled.
    context_slices : torch.Tensor
        Context slices of shape :math:`(B, H, S, D)`.
    enable_surface : bool
        Whether surface tokens are enabled.
    enable_volume : bool
        Whether volume tokens are enabled.

    Outputs
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        Updated surface and volume tokens.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        anchor_grid_sizes: list[tuple[int, int, int]],
        anchor_radii: list[float],
        anchor_gale_layers: int,
        anchor_gale_slice_num: int,
        anchor_gale_mlp_ratio: int,
        anchor_learnable: bool,
        activation: str,
        enable_surface: bool = True,
        enable_volume: bool = True,
    ) -> None:
        super().__init__(meta=None)
        block_kw = dict(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            grid_sizes=anchor_grid_sizes,
            anchor_radii=anchor_radii,
            dropout=dropout,
            gale_layers=anchor_gale_layers,
            gale_slice_num=anchor_gale_slice_num,
            gale_mlp_ratio=anchor_gale_mlp_ratio,
            learnable_anchors=anchor_learnable,
            activation=activation,
        )
        if enable_surface:
            self.surf_anchor_blocks = nn.ModuleList(
                [AnchorSliceBlock(**block_kw) for _ in range(num_layers)]
            )
        else:
            self.surf_anchor_blocks = nn.ModuleList()
        if enable_volume:
            self.vol_anchor_blocks = nn.ModuleList(
                [AnchorSliceBlock(**block_kw) for _ in range(num_layers)]
            )
        else:
            self.vol_anchor_blocks = nn.ModuleList()

    def forward(
        self,
        surface_points: Float[torch.Tensor, "batch num_surf 3"] | None,
        volume_points: Float[torch.Tensor, "batch num_vol 3"] | None,
        surface_tokens: Float[torch.Tensor, "batch num_surf hidden_dim"] | None,
        volume_tokens: Float[torch.Tensor, "batch num_vol hidden_dim"] | None,
        context_slices: Float[torch.Tensor, "batch heads slices dim"],
        enable_surface: bool,
        enable_volume: bool,
    ) -> tuple[
        Float[torch.Tensor, "batch num_surf hidden_dim"] | None,
        Float[torch.Tensor, "batch num_vol hidden_dim"] | None,
    ]:
        r"""
        Run the physics stack.

        Parameters
        ----------
        surface_points : torch.Tensor | None
            Surface points of shape :math:`(B, N_s, 3)`.
        volume_points : torch.Tensor | None
            Volume points of shape :math:`(B, N_v, 3)`.
        surface_tokens : torch.Tensor | None
            Surface tokens of shape :math:`(B, N_s, C)`.
        volume_tokens : torch.Tensor | None
            Volume tokens of shape :math:`(B, N_v, C)`.
        context_slices : torch.Tensor
            Context slices of shape :math:`(B, H, S, D)`.
        enable_surface : bool
            Whether surface tokens are enabled.
        enable_volume : bool
            Whether volume tokens are enabled.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            Updated surface and volume tokens.
        """
        if enable_surface:
            for block in self.surf_anchor_blocks:
                surface_tokens = block(
                    surface_points, surface_tokens, context_slices
                )
        if enable_volume:
            for block in self.vol_anchor_blocks:
                volume_tokens = block(
                    volume_points, volume_tokens, context_slices
                )

        return surface_tokens, volume_tokens


__all__ = ["PulsarPhysicsStack"]

