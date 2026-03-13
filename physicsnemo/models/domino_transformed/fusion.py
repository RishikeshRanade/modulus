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
DoMINO Hierarchical Fusion Module.

Combines basis, geometry encoding, and global embedding via:
  Stage 1: project basis + geometry to common dim, add, LayerNorm.
  Stage 2: FiLM conditioning with global: out = gamma * stage1 + beta.
"""

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module


class HierarchicalFusion(Module):
    r"""
    Hierarchical fusion of three streams: basis, geometry encoding, and global embedding.

    Stage 1: Project basis and geometry to a common dimension and combine with LayerNorm.
    Stage 2: FiLM (Feature-wise Linear Modulation) with global embedding:
    :math:`\gamma, \beta = \mathrm{MLP}(\mathrm{global})`, output = :math:`\gamma \odot \mathrm{stage1} + \beta`.

    Parameters
    ----------
    basis_dim : int
        Dimension of the basis stream (f_b).
    geometry_dim : int
        Dimension of the geometry encoding stream (f_g).
    global_dim : int
        Dimension of the global embedding stream (f_p).
    fusion_dim : int
        Output dimension (d) of the fused representation.
    stage1_hidden : int | None, optional
        If set, use an MLP for geometry projection with this hidden dim; else a single Linear.
    film_hidden : int | None, optional
        Hidden dimension for the FiLM MLP. Defaults to fusion_dim if None.

    Forward
    -------
    basis : torch.Tensor
        Shape (B, N, f_b) or (B, N, K, f_b) when K is the number of spatial samples per point.
    geometry : torch.Tensor
        Shape (B, N, f_g). Will be expanded to match basis if basis has an extra dimension.
    global_embed : torch.Tensor
        Shape (B, N, f_p). Will be expanded to match basis if basis has an extra dimension.

    Outputs
    -------
    torch.Tensor
        Fused representation of shape (B, N, d) or (B, N, K, d) to match basis.
    """

    def __init__(
        self,
        basis_dim: int,
        geometry_dim: int,
        global_dim: int,
        fusion_dim: int,
        stage1_hidden: int | None = None,
        film_hidden: int | None = None,
    ):
        super().__init__(meta=None)
        self.fusion_dim = fusion_dim
        film_hidden = film_hidden or fusion_dim

        # Stage 1: project basis and geometry to fusion_dim
        self.proj_basis = nn.Linear(basis_dim, fusion_dim)
        if stage1_hidden is not None:
            self.proj_geometry = nn.Sequential(
                nn.Linear(geometry_dim, stage1_hidden),
                nn.GELU(),
                nn.Linear(stage1_hidden, fusion_dim),
            )
        else:
            self.proj_geometry = nn.Linear(geometry_dim, fusion_dim)
        self.stage1_ln = nn.LayerNorm(fusion_dim)

        # Stage 2: FiLM from global
        self.film_mlp = nn.Sequential(
            nn.Linear(global_dim, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, 2 * fusion_dim),
        )

    def forward(
        self,
        basis: Float[torch.Tensor, "batch num_points ... basis_dim"],
        geometry: Float[torch.Tensor, "batch num_points geometry_dim"],
        global_embed: Float[torch.Tensor, "batch num_points global_dim"],
    ) -> Float[torch.Tensor, "batch num_points ... fusion_dim"]:
        # Handle optional extra dimension (e.g. B, N, K, f_b for center + neighbors)
        need_expand = basis.ndim == 4
        if need_expand:
            b, n, k, _ = basis.shape
            basis_flat = basis.reshape(b * n * k, -1)
            # Expand geometry and global to (B, N, K, dim) then flatten
            geom_exp = geometry.unsqueeze(2).expand(-1, -1, k, -1)
            global_exp = global_embed.unsqueeze(2).expand(-1, -1, k, -1)
            geometry_flat = geom_exp.reshape(b * n * k, -1)
            global_flat = global_exp.reshape(b * n * k, -1)
        else:
            basis_flat = basis
            geometry_flat = geometry
            global_flat = global_embed

        # Stage 1
        proj_b = self.proj_basis(basis_flat)
        proj_g = self.proj_geometry(geometry_flat)
        combined = self.stage1_ln(proj_b + proj_g)

        # Stage 2: FiLM
        gamma_beta = self.film_mlp(global_flat)
        gamma = gamma_beta[..., : self.fusion_dim]
        beta = gamma_beta[..., self.fusion_dim :]
        # Optional: scale >= 1 for stability
        gamma = 1.0 + torch.nn.functional.softplus(gamma)
        out = gamma * combined + beta

        if need_expand:
            out = out.reshape(b, n, k, self.fusion_dim)
        return out
