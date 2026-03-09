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
Ball-query feature injection modules for Pulsar.
"""

from __future__ import annotations

import torch
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.experimental.models.pulsar.bq_features import BQFeatureStack


class BQFeatureInjector(Module):
    r"""
    Inject multi-scale ball-query features into geometry/surface/volume tokens.

    Parameters
    ----------
    hidden_dim : int
        Token embedding dimension.
    bq_hidden_dim : int
        Hidden dimension for BQ feature stacks.
    geo_bq_radii : list[float]
        Radii for geometry self BQ.
    geo_bq_neighbors : list[int]
        Neighbor counts for geometry self BQ.
    surf_bq_radii : list[float]
        Radii for surface self BQ.
    surf_bq_neighbors : list[int]
        Neighbor counts for surface self BQ.
    vol_bq_radii : list[float]
        Radii for volume self BQ.
    vol_bq_neighbors : list[int]
        Neighbor counts for volume self BQ.
    geo_surf_bq_radii : list[float]
        Radii for geometry->surface cross BQ.
    geo_surf_bq_neighbors : list[int]
        Neighbor counts for geometry->surface cross BQ.
    geo_vol_bq_radii : list[float]
        Radii for geometry->volume cross BQ.
    geo_vol_bq_neighbors : list[int]
        Neighbor counts for geometry->volume cross BQ.
    surf_geo_bq_radii : list[float]
        Radii for surface->geometry cross BQ.
    surf_geo_bq_neighbors : list[int]
        Neighbor counts for surface->geometry cross BQ.
    vol_geo_bq_radii : list[float]
        Radii for volume->geometry cross BQ.
    vol_geo_bq_neighbors : list[int]
        Neighbor counts for volume->geometry cross BQ.
    """

    def __init__(
        self,
        hidden_dim: int,
        bq_hidden_dim: int,
        geo_bq_radii: list[float],
        geo_bq_neighbors: list[int],
        surf_bq_radii: list[float],
        surf_bq_neighbors: list[int],
        vol_bq_radii: list[float],
        vol_bq_neighbors: list[int],
        geo_surf_bq_radii: list[float],
        geo_surf_bq_neighbors: list[int],
        geo_vol_bq_radii: list[float],
        geo_vol_bq_neighbors: list[int],
        surf_geo_bq_radii: list[float],
        surf_geo_bq_neighbors: list[int],
        vol_geo_bq_radii: list[float],
        vol_geo_bq_neighbors: list[int],
    ) -> None:
        super().__init__(meta=None)

        # Self BQ stacks
        self.geo_bq_self_stack = BQFeatureStack(
            radii=geo_bq_radii,
            neighbors=geo_bq_neighbors,
            feature_dim=3,
            hidden_dim=bq_hidden_dim,
            out_dim=hidden_dim,
        )
        self.surf_bq_self_stack = BQFeatureStack(
            radii=surf_bq_radii,
            neighbors=surf_bq_neighbors,
            feature_dim=3,
            hidden_dim=bq_hidden_dim,
            out_dim=hidden_dim,
        )
        self.vol_bq_self_stack = BQFeatureStack(
            radii=vol_bq_radii,
            neighbors=vol_bq_neighbors,
            feature_dim=3,
            hidden_dim=bq_hidden_dim,
            out_dim=hidden_dim,
        )

        # Cross BQ stacks (geometry <-> surface/volume)
        self.geo_bq_surf_cross_stack = BQFeatureStack(
            radii=geo_surf_bq_radii,
            neighbors=geo_surf_bq_neighbors,
            feature_dim=3,
            hidden_dim=bq_hidden_dim,
            out_dim=hidden_dim,
        )
        self.geo_bq_vol_cross_stack = BQFeatureStack(
            radii=geo_vol_bq_radii,
            neighbors=geo_vol_bq_neighbors,
            feature_dim=3,
            hidden_dim=bq_hidden_dim,
            out_dim=hidden_dim,
        )
        self.surf_bq_cross_stack = BQFeatureStack(
            radii=surf_geo_bq_radii,
            neighbors=surf_geo_bq_neighbors,
            feature_dim=3,
            hidden_dim=bq_hidden_dim,
            out_dim=hidden_dim,
        )
        self.vol_bq_cross_stack = BQFeatureStack(
            radii=vol_geo_bq_radii,
            neighbors=vol_geo_bq_neighbors,
            feature_dim=3,
            hidden_dim=bq_hidden_dim,
            out_dim=hidden_dim,
        )

    def forward(
        self,
        geometry_points: Float[torch.Tensor, "batch num_geo 3"],
        surface_points: Float[torch.Tensor, "batch num_surf 3"] | None,
        volume_points: Float[torch.Tensor, "batch num_vol 3"] | None,
        geo_tokens: Float[torch.Tensor, "batch num_geo hidden_dim"],
        surf_tokens: Float[torch.Tensor, "batch num_surf hidden_dim"] | None,
        vol_tokens: Float[torch.Tensor, "batch num_vol hidden_dim"] | None,
    ) -> tuple[
        Float[torch.Tensor, "batch num_geo hidden_dim"],
        Float[torch.Tensor, "batch num_surf hidden_dim"] | None,
        Float[torch.Tensor, "batch num_vol hidden_dim"] | None,
    ]:
        r"""
        Inject BQ features into the provided tokens.
        """
        geo_tokens = geo_tokens + self.geo_bq_self_stack(
            geometry_points, geometry_points
        )
        if surface_points is not None and surf_tokens is not None:
            geo_cross_features = self.geo_bq_surf_cross_stack(
                geometry_points, surface_points
            )
            if geo_cross_features.shape[1] == geo_tokens.shape[1]:
                geo_tokens = geo_tokens + geo_cross_features
            surf_tokens = surf_tokens + self.surf_bq_self_stack(
                surface_points, surface_points
            )
            surf_cross_features = self.surf_bq_cross_stack(
                surface_points, geometry_points
            )
            if surf_cross_features.shape[1] == surf_tokens.shape[1]:
                surf_tokens = surf_tokens + surf_cross_features
        if volume_points is not None and vol_tokens is not None:
            geo_vol_cross_features = self.geo_bq_vol_cross_stack(
                geometry_points, volume_points
            )
            if geo_vol_cross_features.shape[1] == geo_tokens.shape[1]:
                geo_tokens = geo_tokens + geo_vol_cross_features
            vol_tokens = vol_tokens + self.vol_bq_self_stack(
                volume_points, volume_points
            )
            vol_cross_features = self.vol_bq_cross_stack(
                volume_points, geometry_points
            )
            if vol_cross_features.shape[1] == vol_tokens.shape[1]:
                vol_tokens = vol_tokens + vol_cross_features

        return geo_tokens, surf_tokens, vol_tokens


__all__ = ["BQFeatureInjector"]

