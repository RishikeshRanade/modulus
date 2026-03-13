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
DoMINO Geometry Representation Modules.

This module contains geometry representation layers for the DoMINO model architecture,
including SDF scaling, geometry convolution, and geometry processing modules.

Token-direct path: cross ball query from tokens to geometry (distance, direction,
relative position), then Transolver (slice attention) on tokens to produce
geometry encoding (B, N_tokens, C).
"""

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.models.transolver.transolver import TransolverBlock
from physicsnemo.models.unet import UNet
from physicsnemo.nn import BQWarp, Mlp, fourier_encode, get_activation
from physicsnemo.nn.functional import radius_search


class GeoConvOut(Module):
    r"""
    Geometry layer to project STL geometry data onto regular grids.

    This module processes ball-query outputs through an MLP to produce
    a grid-based geometry representation.

    Parameters
    ----------
    input_features : int
        Number of input feature dimensions (typically 3 for x, y, z coordinates).
    neighbors_in_radius : int
        Number of neighbors to consider within the ball query radius.
    model_parameters : Any
        Configuration parameters containing:
        - ``base_neurons``: Number of neurons in hidden layers
        - ``fourier_features``: Whether to use Fourier feature encoding
        - ``num_modes``: Number of Fourier modes if using Fourier features
        - ``base_neurons_in``: Output feature dimension
        - ``activation``: Activation function name
    grid_resolution : list[int], optional, default=[256, 96, 64]
        Resolution of the output grid as :math:`[N_x, N_y, N_z]`.

    Forward
    -------
    x : torch.Tensor
        Input tensor containing coordinates of neighboring points
        of shape :math:`(B, N_x \cdot N_y \cdot N_z, K, 3)` where :math:`K`
        is ``neighbors_in_radius``.
    grid : torch.Tensor
        Input tensor represented as a grid of shape :math:`(B, N_x, N_y, N_z, 3)`.

    Outputs
    -------
    torch.Tensor
        Processed geometry features of shape :math:`(B, D_{out}, N_x, N_y, N_z)`
        where :math:`D_{out}` is ``base_neurons_in``.

    See Also
    --------
    :class:`~physicsnemo.nn.Mlp` : MLP module used for feature processing.
    :class:`GeometryRep` : Uses this module for multi-scale geometry encoding.
    """

    def __init__(
        self,
        input_features: int,
        neighbors_in_radius: int,
        model_parameters: Any,
        grid_resolution: list[int] | None = None,
    ):
        super().__init__(meta=None)
        if grid_resolution is None:
            grid_resolution = [256, 96, 64]
        base_neurons = model_parameters.base_neurons
        self.fourier_features = model_parameters.fourier_features
        self.num_modes = model_parameters.num_modes

        if self.fourier_features:
            input_features_calculated = (
                input_features * (1 + 2 * self.num_modes) * neighbors_in_radius
            )
        else:
            input_features_calculated = input_features * neighbors_in_radius

        self.mlp = Mlp(
            in_features=input_features_calculated,
            hidden_features=[base_neurons, base_neurons // 2],
            out_features=model_parameters.base_neurons_in,
            act_layer=get_activation(model_parameters.activation),
            drop=0.0,
        )

        self.grid_resolution = grid_resolution

        self.activation = get_activation(model_parameters.activation)

        self.neighbors_in_radius = neighbors_in_radius

        if self.fourier_features:
            self.register_buffer(
                "freqs", torch.exp(torch.linspace(0, math.pi, self.num_modes))
            )

    def forward(
        self,
        x: Float[torch.Tensor, "batch grid_points neighbors 3"],
        grid: Float[torch.Tensor, "batch nx ny nz 3"],
        radius: float = 0.025,
        neighbors_in_radius: int = 10,
    ) -> Float[torch.Tensor, "batch out_features nx ny nz"]:
        r"""
        Process and project geometric features onto a 3D grid.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing coordinates of the neighboring points
            of shape :math:`(B, N_x \cdot N_y \cdot N_z, K, 3)`.
        grid : torch.Tensor
            Input tensor represented as a grid of shape :math:`(B, N_x, N_y, N_z, 3)`.
        radius : float, optional, default=0.025
            Ball query radius (unused, kept for API compatibility).
        neighbors_in_radius : int, optional, default=10
            Number of neighbors (unused, kept for API compatibility).

        Returns
        -------
        torch.Tensor
            Processed geometry features of shape :math:`(B, D_{out}, N_x, N_y, N_z)`.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            if x.ndim != 4 or x.shape[-1] != 3:
                raise ValueError(
                    f"Expected x of shape (B, N, K, 3), got shape {tuple(x.shape)}"
                )
            if grid.ndim != 5 or grid.shape[-1] != 3:
                raise ValueError(
                    f"Expected grid of shape (B, Nx, Ny, Nz, 3), "
                    f"got shape {tuple(grid.shape)}"
                )

        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )
        grid = grid.reshape(1, nx * ny * nz, 3, 1)

        # Rearrange input to flatten spatial and neighbor dimensions
        x = rearrange(
            x, "b x y z -> b x (y z)", x=nx * ny * nz, y=self.neighbors_in_radius, z=3
        )

        # Apply Fourier feature encoding if enabled
        if self.fourier_features:
            facets = torch.cat((x, fourier_encode(x, self.freqs)), axis=-1)
        else:
            facets = x

        # Process through MLP with tanh activation
        x = F.tanh(self.mlp(facets))

        # Reshape to 3D grid format
        x = rearrange(x, "b (x y z) c -> b c x y z", x=nx, y=ny, z=nz)

        return x


class CrossBallQueryFeatures(Module):
    r"""
    Cross ball-query from token positions to geometry points with rich neighborhood features.

    For each token (query), finds geometry points within radius and computes per-neighbor:
    distance, unit direction (token -> neighbor), and relative position. Flattens and
    projects via MLP to a fixed output dimension. Supports batch by processing batch
    index 0 only (radius_search is batch-1); callers can loop or vmap.

    Parameters
    ----------
    radius : float
        Ball query radius.
    neighbors_in_radius : int
        Max neighbors per query.
    out_dim : int
        Output feature dimension per token.
    hidden_dim : int, optional
        MLP hidden dimension. Defaults to out_dim * 2.
    eps : float, optional
        Epsilon for normalizing direction. Default 1e-6.
    """

    def __init__(
        self,
        radius: float,
        neighbors_in_radius: int,
        out_dim: int,
        hidden_dim: int | None = None,
        eps: float = 1e-3,
    ):
        super().__init__(meta=None)
        self.radius = radius
        self.neighbors_in_radius = neighbors_in_radius
        self.eps = eps
        # Per neighbor: 1 (dist) + 3 (direction) + 3 (relative position) = 7
        feat_per_neighbor = 7
        in_dim = neighbors_in_radius * feat_per_neighbor
        hidden_dim = hidden_dim or (out_dim * 2)
        self.mlp = Mlp(
            in_features=in_dim,
            hidden_features=[hidden_dim, hidden_dim // 2],
            out_features=out_dim,
            act_layer=nn.GELU(),
            drop=0.0,
        )

    def forward(
        self,
        geometry: Float[torch.Tensor, "batch num_geo 3"],
        tokens: Float[torch.Tensor, "batch num_tokens 3"],
    ) -> Float[torch.Tensor, "batch num_tokens out_dim"]:
        # radius_search is batch-1; loop over batch
        B, N_geo, _ = geometry.shape
        N_tok = tokens.shape[1]
        out_list = []
        for b in range(B):
            inds, neighbor_pts, dists = radius_search(
                geometry[b],
                tokens[b],
                self.radius,
                self.neighbors_in_radius,
                return_dists=True,
                return_points=True,
            )
            # inds (N_tok, K), neighbor_pts (N_tok, K, 3), dists (N_tok, K)
            q = tokens[b].unsqueeze(1)
            direction = (neighbor_pts - q)
            rel_pos = neighbor_pts - q
            feats = torch.cat(
                (dists.unsqueeze(-1), direction, rel_pos), dim=-1
            )
            feats = feats.reshape(N_tok, -1)
            out_b = self.mlp(feats)
            out_list.append(out_b)

        return torch.stack(out_list, dim=0)


class GeometryRep(Module):
    r"""
    Geometry representation module that processes STL geometry data.

    This module constructs a multiscale representation of geometry by:

    1. Computing multi-scale geometry encoding for local and global context
       using ball queries at different radii
    2. Processing signed distance field (SDF) data for surface information
    3. Optionally combining encodings using cross-attention

    The combined encoding enables the model to reason about both local and global
    geometric properties.

    Parameters
    ----------
    input_features : int
        Number of input feature dimensions (typically 3 for x, y, z coordinates).
    radii : Sequence[float]
        List of radii for multi-scale ball queries.
    neighbors_in_radius : Sequence[int]
        List of neighbor counts for each radius scale.
    model_parameters : Any, optional, default=None
        Configuration parameters for geometry representation, containing
        nested config for ``geometry_rep.geo_conv`` and ``geometry_rep.geo_processor``.

    Forward
    -------
    x : torch.Tensor
        Geometry point cloud of shape :math:`(B, N_{geo}, 3)`.
    tokens : torch.Tensor
        Query token positions of shape :math:`(B, N_{tokens}, 3)`.

    Outputs
    -------
    torch.Tensor
        Token-directed geometry encoding of shape :math:`(B, N_{tokens}, C)`.

    Example
    -------
    >>> import torch
    >>> # GeometryRep requires model_parameters configuration
    >>> # See DoMINO model for typical usage

    See Also
    --------
    :class:`CrossBallQueryFeatures` : Cross ball query from tokens to geometry.
    :class:`~physicsnemo.models.transolver.transolver.TransolverBlock` : Slice attention on tokens.
    """

    def __init__(
        self,
        input_features: int,
        radii: Sequence[float],
        neighbors_in_radius: Sequence[int],
        model_parameters: Any = None,
        output_dim: int | None = None,
    ):
        super().__init__(meta=None)
        geometry_rep = model_parameters.geometry_rep
        self.activation_conv = get_activation(geometry_rep.geo_conv.activation)
        self.activation_processor = geometry_rep.geo_processor.activation
        self.use_token_direct = getattr(geometry_rep, "use_token_direct", False)
        self.radii = list(radii)
        self.neighbors_in_radius = list(neighbors_in_radius)

        token_hidden = getattr(geometry_rep, "token_hidden_dim", 256)
        token_out = output_dim or getattr(
            geometry_rep, "token_out_dim", token_hidden
        )
        transolver_depth = getattr(geometry_rep, "transolver_depth", 2)
        slice_num = getattr(geometry_rep, "transolver_slice_num", 32)
        num_heads = getattr(geometry_rep, "transolver_num_heads", 4)
        dropout = getattr(geometry_rep, "transolver_dropout", 0.0)
        use_te = getattr(geometry_rep, "transolver_use_te", False)
        self.token_out_dim = token_out
        scale_out = max(token_hidden // len(radii), 32)
        self.cross_bq = nn.ModuleList([
            CrossBallQueryFeatures(
                radius=radii[j],
                neighbors_in_radius=neighbors_in_radius[j],
                out_dim=scale_out,
                hidden_dim=token_hidden,
            )
            for j in range(len(radii))
        ])
        scale_dim = scale_out * len(radii)
        self.token_merge = nn.Linear(scale_dim, token_hidden)
        self.transolver_blocks = nn.ModuleList()
        for _ in range(transolver_depth - 1):
            self.transolver_blocks.append(
                TransolverBlock(
                    num_heads=num_heads,
                    hidden_dim=token_hidden,
                    dropout=dropout,
                    act="gelu",
                    mlp_ratio=4,
                    last_layer=False,
                    slice_num=slice_num,
                    spatial_shape=None,
                    use_te=use_te,
                    plus=False,
                )
            )
        self.transolver_blocks.append(
            TransolverBlock(
                num_heads=num_heads,
                hidden_dim=token_hidden,
                dropout=dropout,
                act="gelu",
                mlp_ratio=4,
                last_layer=True,
                out_dim=token_out,
                slice_num=slice_num,
                spatial_shape=None,
                use_te=use_te,
                plus=False,
            )
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch num_geo 3"],
        tokens: Float[torch.Tensor, "batch num_tokens 3"],
    ) -> Float[torch.Tensor, "batch channels encoding_features"]:
        r"""
        Process geometry data to create a comprehensive representation.

        This method combines short-range, long-range, and SDF-based geometry
        encodings to create a rich representation of the geometry.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing geometric point data of shape :math:`(B, N_{geo}, 3)`.
        tokens : torch.Tensor
            Input tensor containing token data of shape :math:`(B, N_{tokens}, 3)`.

        Returns
        -------
        torch.Tensor
            Geometry encoding on tokens, shape :math:`(B, N_{tokens}, C)`.
        """

        # Multi-scale cross ball query: tokens query geometry
        scale_feats = []
        for layer in self.cross_bq:
            scale_feats.append(layer(x, tokens))
        h = torch.cat(scale_feats, dim=-1)
        h = self.token_merge(h)
        for block in self.transolver_blocks:
            h = block(h)
        return h
