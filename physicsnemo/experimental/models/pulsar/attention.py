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
Anchor-slice attention for Pulsar.

This module implements a hierarchical anchor attention mechanism that combines
local anchor tokens with global slice attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.experimental.models.geotransolver.gale import GALE_block
from physicsnemo.nn import Mlp, get_activation


class MultiheadMaskedCrossAttention(Module):
    r"""
    Multi-head cross attention with radius-based masking.

    Supports optional chunking over the query dimension to reduce peak memory
    when N_q * N_k is large (e.g. anchors x surface points).

    Parameters
    ----------
    hidden_dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout : float, optional, default=0.0
        Dropout rate for attention weights.
    query_chunk_size : int or None, optional, default=512
        If set, process query tokens in chunks of this size to limit peak
        memory. Use when N_q * N_k is large (e.g. anchor attention with
        many points). None disables chunking (original behavior).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        query_chunk_size: int | None = 512,
    ) -> None:
        super().__init__(meta=None)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query_chunk_size = query_chunk_size

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_tokens: Float[torch.Tensor, "batch query_tokens hidden_dim"],
        key_tokens: Float[torch.Tensor, "batch key_tokens hidden_dim"],
        query_coords: Float[torch.Tensor, "batch query_tokens 3"],
        key_coords: Float[torch.Tensor, "batch key_tokens 3"],
        radius: float,
    ) -> Float[torch.Tensor, "batch query_tokens hidden_dim"]:
        r"""
        Apply masked cross attention.

        Parameters
        ----------
        query_tokens : torch.Tensor
            Query tokens of shape :math:`(B, N_q, C)`.
        key_tokens : torch.Tensor
            Key/value tokens of shape :math:`(B, N_k, C)`.
        query_coords : torch.Tensor
            Query coordinates of shape :math:`(B, N_q, 3)`.
        key_coords : torch.Tensor
            Key coordinates of shape :math:`(B, N_k, 3)`.
        radius : float
            Mask radius for attention.

        Returns
        -------
        torch.Tensor
            Output tokens of shape :math:`(B, N_q, C)`.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if query_tokens.ndim != 3 or key_tokens.ndim != 3:
                raise ValueError(
                    "Expected query_tokens and key_tokens to be 3D tensors, got "
                    f"{query_tokens.ndim}D and {key_tokens.ndim}D"
                )
            if query_coords.ndim != 3 or key_coords.ndim != 3:
                raise ValueError(
                    "Expected query_coords and key_coords to be 3D tensors, got "
                    f"{query_coords.ndim}D and {key_coords.ndim}D"
                )
            if query_coords.shape[-1] != 3 or key_coords.shape[-1] != 3:
                raise ValueError("Expected last dim of coords to be 3")

        bsz, q_len, _ = query_tokens.shape
        k_len = key_tokens.shape[1]

        chunk_size = self.query_chunk_size
        if chunk_size is None or q_len <= chunk_size:
            # Single pass: original implementation
            q = self.q_proj(query_tokens).view(bsz, q_len, self.num_heads, self.head_dim)
            k = self.k_proj(key_tokens).view(bsz, k_len, self.num_heads, self.head_dim)
            v = self.v_proj(key_tokens).view(bsz, k_len, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3)  # (B, H, Q, D)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            dist2 = torch.cdist(query_coords, key_coords) ** 2  # (B, Q, K)
            radius2 = radius * radius
            mask = dist2 > radius2
            mask = mask[:, None, :, :].expand(-1, self.num_heads, -1, -1)
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout.p, is_causal=False
            )
            attn = attn.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.hidden_dim)
            return self.out_proj(attn)

        # Chunked over query dimension to limit peak memory (B, H, chunk, K)
        k = self.k_proj(key_tokens).view(bsz, k_len, self.num_heads, self.head_dim)
        v = self.v_proj(key_tokens).view(bsz, k_len, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, H, K, D)
        v = v.permute(0, 2, 1, 3)
        radius2 = radius * radius
        out_chunks = []
        for start in range(0, q_len, chunk_size):
            end = min(start + chunk_size, q_len)
            q_chunk = query_tokens[:, start:end, :]
            q_coords_chunk = query_coords[:, start:end, :]
            q = self.q_proj(q_chunk).view(bsz, end - start, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3)  # (B, H, chunk, D)
            dist2 = torch.cdist(q_coords_chunk, key_coords) ** 2  # (B, chunk, K)
            mask = dist2 > radius2
            mask = mask[:, None, :, :].expand(-1, self.num_heads, -1, -1)
            attn_chunk = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout.p, is_causal=False
            )
            attn_chunk = attn_chunk.permute(0, 2, 1, 3).contiguous().view(
                bsz, end - start, self.hidden_dim
            )
            out_chunks.append(attn_chunk)
        attn = torch.cat(out_chunks, dim=1)
        return self.out_proj(attn)


class AnchorSliceBlock(Module):
    r"""
    Hierarchical anchor-slice attention block.

    Parameters
    ----------
    hidden_dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    grid_sizes : list[tuple[int, int, int]]
        Anchor grid sizes per level.
    anchor_radii : list[float]
        Mask radii per level.
    dropout : float, optional, default=0.0
        Dropout rate.
    gale_layers : int, optional, default=1
        Number of GALE layers applied to anchors.
    gale_slice_num : int, optional, default=32
        Number of slices in GALE blocks.
    gale_mlp_ratio : int, optional, default=4
        MLP ratio in GALE blocks.
    learnable_anchors : bool, optional, default=True
        Whether to use learnable anchor coordinates.
    activation : str, optional, default="gelu"
        Activation function name for anchor MLP.

    Forward
    -------
    point_coords : torch.Tensor
        Point coordinates of shape :math:`(B, N, 3)`.
    point_tokens : torch.Tensor
        Point tokens of shape :math:`(B, N, C)`.
    context_slices : torch.Tensor
        Context slices of shape :math:`(B, H, S, D)`.

    Outputs
    -------
    torch.Tensor
        Updated point tokens of shape :math:`(B, N, C)`.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        grid_sizes: list[tuple[int, int, int]],
        anchor_radii: list[float],
        dropout: float = 0.0,
        gale_layers: int = 1,
        gale_slice_num: int = 32,
        gale_mlp_ratio: int = 4,
        learnable_anchors: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__(meta=None)
        if len(grid_sizes) != len(anchor_radii):
            raise ValueError(
                "grid_sizes and anchor_radii must have the same length, got "
                f"{len(grid_sizes)} and {len(anchor_radii)}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.grid_sizes = grid_sizes
        self.anchor_radii = anchor_radii
        self.learnable_anchors = learnable_anchors

        act_layer = get_activation(activation)
        self.anchor_mlp = Mlp(
            in_features=3,
            hidden_features=[hidden_dim, hidden_dim],
            out_features=hidden_dim,
            act_layer=act_layer,
            drop=dropout,
        )
        self.anchor_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.point_to_anchor = MultiheadMaskedCrossAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        self.anchor_to_point = MultiheadMaskedCrossAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        self.gale_blocks = nn.ModuleList(
            [
                GALE_block(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    act=activation,
                    mlp_ratio=gale_mlp_ratio,
                    slice_num=gale_slice_num,
                    last_layer=False,
                    use_te=False,
                    plus=False,
                    context_dim=hidden_dim,
                )
                for _ in range(gale_layers)
            ]
        )

        # Learned anchor coordinates in [-1, 1]^3; mapped to [0, 1] for placement.
        # Anchors are placed in the bbox of the input point cloud (min/max computed
        # on the fly), so they stay in the same coordinate system as the points.
        self.anchor_coords = nn.ParameterList()
        for grid_size in grid_sizes:
            num_anchors = grid_size[0] * grid_size[1] * grid_size[2]
            self.anchor_coords.append(
                nn.Parameter(torch.empty(num_anchors, 3).uniform_(-1.0, 1.0))
            )

    def _anchor_grid(
        self,
        coords: Float[torch.Tensor, "batch points 3"],
        grid_size: tuple[int, int, int],
        level_idx: int,
    ) -> Float[torch.Tensor, "batch anchors 3"]:
        # Place anchors in the bbox of the point cloud (computed on the fly).
        mins = coords.min(dim=1).values
        maxs = coords.max(dim=1).values
        scale = (maxs - mins)[:, None, :].clamp_min(1e-6)
        offset = mins[:, None, :]

        if self.learnable_anchors:
            base = self.anchor_coords[level_idx].to(coords.dtype)  # (A, 3) in [-1, 1]
            base = (base + 1.0) * 0.5  # fraction in [0, 1]; anchors = offset + scale*base in [mins, maxs]
        else:
            nx, ny, nz = grid_size
            grid_x = torch.linspace(
                0.0, 1.0, nx, device=coords.device, dtype=coords.dtype
            )
            grid_y = torch.linspace(
                0.0, 1.0, ny, device=coords.device, dtype=coords.dtype
            )
            grid_z = torch.linspace(
                0.0, 1.0, nz, device=coords.device, dtype=coords.dtype
            )
            base = torch.stack(
                torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), dim=-1
            )
            base = base.reshape(-1, 3)

        anchors = offset + scale * base[None, ...]
        return anchors

    def forward(
        self,
        point_coords: Float[torch.Tensor, "batch points 3"],
        point_tokens: Float[torch.Tensor, "batch points hidden_dim"],
        context_slices: Float[torch.Tensor, "batch heads slices dim"],
    ) -> Float[torch.Tensor, "batch points hidden_dim"]:
        r"""
        Apply hierarchical anchor-slice attention.

        Anchor positions are computed on the fly from the min/max of
        ``point_coords``, so no external reference origin or scale is needed.

        Parameters
        ----------
        point_coords : torch.Tensor
            Point coordinates of shape :math:`(B, N, 3)`.
        point_tokens : torch.Tensor
            Point tokens of shape :math:`(B, N, C)`.
        context_slices : torch.Tensor
            Context slices of shape :math:`(B, H, S, D)`.

        Returns
        -------
        torch.Tensor
            Updated point tokens of shape :math:`(B, N, C)`.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if point_coords.ndim != 3 or point_coords.shape[-1] != 3:
                raise ValueError(
                    "Expected point_coords of shape (B, N, 3), got "
                    f"{tuple(point_coords.shape)}"
                )
            if point_tokens.ndim != 3:
                raise ValueError(
                    "Expected point_tokens of shape (B, N, C), got "
                    f"{tuple(point_tokens.shape)}"
                )

        updated = point_tokens
        for level_idx, (grid_size, radius) in enumerate(
            zip(self.grid_sizes, self.anchor_radii)
        ):
            anchors = self._anchor_grid(point_coords, grid_size, level_idx)
            anchor_tokens = self.anchor_mlp(anchors)
            anchor_tokens = anchor_tokens + self.point_to_anchor(
                anchor_tokens, updated, anchors, point_coords, radius
            )
            anchor_tokens, _ = self.anchor_self_attn(
                anchor_tokens, anchor_tokens, anchor_tokens
            )

            for block in self.gale_blocks:
                anchor_tokens = block((anchor_tokens,), context_slices)[0]

            updated = updated + self.anchor_to_point(
                updated, anchor_tokens, point_coords, anchors, radius
            )

        return updated


__all__ = ["MultiheadMaskedCrossAttention", "AnchorSliceBlock"]

