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
Ball-query feature extraction for Pulsar.

This module provides helper layers to build self/cross BQ feature stacks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.experimental.models.geotransolver.context_projector import (
    GeometricFeatureProcessor,
)


class BQFeatureStack(Module):
    r"""
    Multi-scale BQ feature stack with projection.

    Parameters
    ----------
    radii : list[float]
        Radii for multi-scale ball-query features.
    neighbors : list[int]
        Neighbor counts for each radius (must match ``radii`` length).
    feature_dim : int
        Feature dimension for key points.
    hidden_dim : int
        Hidden dimension per BQ scale.
    out_dim : int
        Projection output dimension.

    Forward
    -------
    query_points : torch.Tensor
        Query points of shape :math:`(B, N, 3)`.
    key_points : torch.Tensor
        Key points of shape :math:`(B, M, 3)`.

    Outputs
    -------
    torch.Tensor
        Projected BQ features of shape :math:`(B, N, D_{out})`.
    """

    def __init__(
        self,
        radii: list[float],
        neighbors: list[int],
        feature_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__(meta=None)
        if len(radii) != len(neighbors):
            raise ValueError(
                "radii and neighbors must have the same length, got "
                f"{len(radii)} and {len(neighbors)}"
            )

        self.radii = radii
        self.neighbors = neighbors
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.processors = nn.ModuleList(
            [
                GeometricFeatureProcessor(
                    radius=radius,
                    neighbors_in_radius=neighbors_in_radius,
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                )
                for radius, neighbors_in_radius in zip(radii, neighbors)
            ]
        )
        self.proj = nn.Linear(hidden_dim * len(radii), out_dim)

    def forward(
        self,
        query_points: Float[torch.Tensor, "batch query_points 3"],
        key_points: Float[torch.Tensor, "batch key_points 3"],
    ) -> Float[torch.Tensor, "batch query_points out_dim"]:
        r"""
        Compute projected BQ features for query points.

        Parameters
        ----------
        query_points : torch.Tensor
            Query points of shape :math:`(B, N, 3)`.
        key_points : torch.Tensor
            Key points of shape :math:`(B, M, 3)`.

        Returns
        -------
        torch.Tensor
            Projected features of shape :math:`(B, N, D_{out})`.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if query_points.ndim != 3 or query_points.shape[-1] != 3:
                raise ValueError(
                    "Expected query_points of shape (B, N, 3), got "
                    f"{tuple(query_points.shape)}"
                )
            if key_points.ndim != 3 or key_points.shape[-1] != 3:
                raise ValueError(
                    "Expected key_points of shape (B, M, 3), got "
                    f"{tuple(key_points.shape)}"
                )

        bq_features = torch.cat(
            [processor(query_points, key_points) for processor in self.processors],
            dim=-1,
        )
        return self.proj(bq_features)


__all__ = ["BQFeatureStack"]

