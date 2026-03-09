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
Pulsar building blocks.

This module defines reusable layers for the Pulsar architecture, including
learned token pooling and global conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module


class LearnedTokenPooler(Module):
    r"""
    Learned token pooling for variable-length point clouds.

    This module uses learnable query tokens to pool a variable-length set of
    input tokens into a fixed number of context tokens.

    Parameters
    ----------
    input_dim : int
        Input feature dimension for tokens.
    num_context_tokens : int
        Number of context tokens to produce.
    dropout : float, optional, default=0.0
        Dropout applied to attention weights.

    Forward
    -------
    x : torch.Tensor
        Input tokens of shape :math:`(B, N, C)`.

    Outputs
    -------
    torch.Tensor
        Context tokens of shape :math:`(B, S, C)` where :math:`S` is
        ``num_context_tokens``.
    """

    def __init__(
        self,
        input_dim: int,
        num_context_tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(meta=None)
        self.input_dim = input_dim
        self.num_context_tokens = num_context_tokens
        self.scale = input_dim**-0.5

        self.query_tokens = nn.Parameter(
            torch.randn(num_context_tokens, input_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Float[torch.Tensor, "batch tokens channels"]
    ) -> Float[torch.Tensor, "batch context_tokens channels"]:
        r"""
        Pool input tokens into a fixed set of context tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape :math:`(B, N, C)`.

        Returns
        -------
        torch.Tensor
            Context tokens of shape :math:`(B, S, C)`.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected x to have shape (B, N, C), got {tuple(x.shape)}"
                )
            if x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected x with last dim {self.input_dim}, got {x.shape[-1]}"
                )

        # Build attention logits using learned queries
        queries = self.query_tokens[None, :, :].expand(
            x.shape[0], -1, -1
        )  # (B, S, C)
        attn_logits = torch.matmul(queries, x.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn_logits, dim=-1)  # (B, S, N)
        attn = self.dropout(attn)

        # Pool input tokens into context tokens
        context = torch.matmul(attn, x)  # (B, S, C)
        return context


class GlobalConditioner(Module):
    r"""
    Global conditioning module for point embeddings.

    This module aggregates context tokens into a single global vector and
    applies a learned affine transformation to point embeddings.

    Parameters
    ----------
    hidden_dim : int
        Dimension of point and context embeddings.
    dropout : float, optional, default=0.0
        Dropout applied after conditioning.

    Forward
    -------
    point_tokens : torch.Tensor
        Point embeddings of shape :math:`(B, N, C)`.
    context_tokens : torch.Tensor
        Context tokens of shape :math:`(B, S, C)`.

    Outputs
    -------
    torch.Tensor
        Conditioned point embeddings of shape :math:`(B, N, C)`.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__(meta=None)
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        point_tokens: Float[torch.Tensor, "batch tokens hidden_dim"],
        context_tokens: Float[torch.Tensor, "batch context_tokens hidden_dim"],
    ) -> Float[torch.Tensor, "batch tokens hidden_dim"]:
        r"""
        Apply global conditioning to point tokens.

        Parameters
        ----------
        point_tokens : torch.Tensor
            Point embeddings of shape :math:`(B, N, C)`.
        context_tokens : torch.Tensor
            Context tokens of shape :math:`(B, S, C)`.

        Returns
        -------
        torch.Tensor
            Conditioned point embeddings of shape :math:`(B, N, C)`.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if point_tokens.ndim != 3:
                raise ValueError(
                    f"Expected point_tokens to have shape (B, N, C), got {tuple(point_tokens.shape)}"
                )
            if context_tokens.ndim != 3:
                raise ValueError(
                    f"Expected context_tokens to have shape (B, S, C), got {tuple(context_tokens.shape)}"
                )
            if point_tokens.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"Expected point_tokens last dim {self.hidden_dim}, got {point_tokens.shape[-1]}"
                )
            if context_tokens.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"Expected context_tokens last dim {self.hidden_dim}, got {context_tokens.shape[-1]}"
                )

        # Aggregate context tokens and apply affine conditioning
        global_context = context_tokens.mean(dim=1)  # (B, C)
        shift = self.proj(global_context)  # (B, C)
        conditioned = point_tokens + shift[:, None, :]
        return self.dropout(conditioned)


class MultiGridTokenPooler(Module):
    r"""
    Multi-grid learned token pooling for point clouds.

    This module builds pooled context tokens at multiple spatial scales by
    aggregating point tokens into voxel bins and applying learned token pooling
    per grid level.

    Parameters
    ----------
    input_dim : int
        Input feature dimension for tokens.
    grid_sizes : list[tuple[int, int, int]]
        Voxel grid sizes for each level as ``(N_x, N_y, N_z)``.
    tokens_per_level : int | list[int]
        Number of pooled tokens per level. If an int, the same number is used
        for each level.
    dropout : float, optional, default=0.0
        Dropout applied to attention weights.

    Forward
    -------
    coords : torch.Tensor
        Point coordinates of shape :math:`(B, N, 3)`.
    tokens : torch.Tensor
        Point tokens of shape :math:`(B, N, C)`.

    Outputs
    -------
    torch.Tensor
        Context tokens of shape :math:`(B, S, C)` where :math:`S` is the sum
        of tokens across all levels.
    """

    def __init__(
        self,
        input_dim: int,
        grid_sizes: list[tuple[int, int, int]],
        tokens_per_level: int | list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__(meta=None)
        if isinstance(tokens_per_level, int):
            tokens_per_level = [tokens_per_level] * len(grid_sizes)
        if len(tokens_per_level) != len(grid_sizes):
            raise ValueError(
                "tokens_per_level must match grid_sizes length, got "
                f"{len(tokens_per_level)} and {len(grid_sizes)}"
            )

        self.input_dim = input_dim
        self.grid_sizes = grid_sizes
        self.tokens_per_level = tokens_per_level
        self.poolers = nn.ModuleList(
            [
                LearnedTokenPooler(
                    input_dim=input_dim,
                    num_context_tokens=num_tokens,
                    dropout=dropout,
                )
                for num_tokens in tokens_per_level
            ]
        )

    def forward(
        self,
        coords: Float[torch.Tensor, "batch points 3"],
        tokens: Float[torch.Tensor, "batch points channels"],
    ) -> Float[torch.Tensor, "batch context_tokens channels"]:
        r"""
        Pool tokens across multiple grid levels.

        Parameters
        ----------
        coords : torch.Tensor
            Point coordinates of shape :math:`(B, N, 3)`.
        tokens : torch.Tensor
            Point tokens of shape :math:`(B, N, C)`.

        Returns
        -------
        torch.Tensor
            Context tokens of shape :math:`(B, S, C)` where :math:`S` is the
            sum of tokens across all levels.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if coords.ndim != 3 or coords.shape[-1] != 3:
                raise ValueError(
                    "Expected coords of shape (B, N, 3), got "
                    f"{tuple(coords.shape)}"
                )
            if tokens.ndim != 3:
                raise ValueError(
                    "Expected tokens of shape (B, N, C), got "
                    f"{tuple(tokens.shape)}"
                )
            if coords.shape[0] != tokens.shape[0] or coords.shape[1] != tokens.shape[1]:
                raise ValueError(
                    "coords and tokens must share batch and point dimensions, got "
                    f"{tuple(coords.shape)} and {tuple(tokens.shape)}"
                )

        pooled_levels = []
        batch_size = coords.shape[0]
        for (nx, ny, nz), pooler in zip(self.grid_sizes, self.poolers):
            level_tokens = []
            for b in range(batch_size):
                points = coords[b]
                feats = tokens[b]
                mins = points.min(dim=0).values
                maxs = points.max(dim=0).values
                denom = (maxs - mins).clamp_min(1e-6)
                normalized = (points - mins) / denom

                ix = torch.clamp((normalized[:, 0] * nx).floor(), 0, nx - 1).long()
                iy = torch.clamp((normalized[:, 1] * ny).floor(), 0, ny - 1).long()
                iz = torch.clamp((normalized[:, 2] * nz).floor(), 0, nz - 1).long()
                linear = ix * (ny * nz) + iy * nz + iz
                num_bins = nx * ny * nz

                sums = torch.zeros(
                    num_bins, feats.shape[-1], device=feats.device, dtype=feats.dtype
                )
                counts = torch.zeros(
                    num_bins, 1, device=feats.device, dtype=feats.dtype
                )
                sums.index_add_(0, linear, feats)
                counts.index_add_(
                    0, linear, torch.ones_like(linear, dtype=feats.dtype)[:, None]
                )
                mask = counts.squeeze(-1) > 0
                bin_feats = sums[mask] / counts[mask]

                pooled = pooler(bin_feats[None, ...])  # (1, S, C)
                level_tokens.append(pooled)

            pooled_levels.append(torch.cat(level_tokens, dim=0))

        return torch.cat(pooled_levels, dim=1)


__all__ = ["LearnedTokenPooler", "GlobalConditioner", "MultiGridTokenPooler"]

