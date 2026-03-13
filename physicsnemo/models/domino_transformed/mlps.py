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
DoMINO-Transformed MLP Modules.

This module contains the aggregation MLP used to combine basis and
geometry encodings for final predictions.
"""

import torch.nn as nn

from physicsnemo.nn import Mlp


class AggregationModel(Mlp):
    r"""
    Neural network module to aggregate local geometry encoding with basis functions.

    This module combines basis function representations with geometry encodings
    to predict the final output quantities. It serves as the final prediction layer
    that integrates all available information sources.

    The architecture is an MLP with configurable depth (default 2 hidden layers)
    and width. Kept shallow/narrow by default since it only maps fused features
    to a scalar per variable.

    Parameters
    ----------
    input_features : int
        Number of input features (e.g. fusion_dim).
    output_features : int
        Number of output features (typically 1 for scalar field prediction).
    base_layer : int
        Number of neurons in each hidden layer.
    activation : nn.Module
        The activation function to use between layers.
    num_hidden_layers : int, optional
        Number of hidden layers. Default 2. Use 2--3 for readout; more is rarely needed.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, N, D_{in})` where :math:`B` is batch size,
        :math:`N` is the number of points, and :math:`D_{in}` is ``input_features``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, N, D_{out})` where :math:`D_{out}` is
        ``output_features``.

    Example
    -------
    >>> import torch
    >>> from physicsnemo.models.domino_transformed.mlps import AggregationModel
    >>> from physicsnemo.nn import get_activation
    >>> model = AggregationModel(
    ...     input_features=128,
    ...     output_features=1,
    ...     base_layer=512,
    ...     activation=get_activation("gelu"),
    ... )
    >>> x = torch.randn(2, 100, 128)
    >>> output = model(x)
    >>> output.shape
    torch.Size([2, 100, 1])

    See Also
    --------
    :class:`~physicsnemo.nn.Mlp` : Base MLP class.
    :class:`~physicsnemo.models.domino_transformed.solutions.SolutionCalculatorVolume` : Uses this for volume predictions.
    :class:`~physicsnemo.models.domino_transformed.solutions.SolutionCalculatorSurface` : Uses this for surface predictions.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        base_layer: int,
        activation: nn.Module,
        num_hidden_layers: int = 2,
    ):
        hidden_features = [base_layer] * num_hidden_layers

        super().__init__(
            in_features=input_features,
            hidden_features=hidden_features,
            out_features=output_features,
            act_layer=activation,
            drop=0.0,
        )
