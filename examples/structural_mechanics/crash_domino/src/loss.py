# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import torch
from typing import Literal, Any

from physicsnemo.utils.domino.utils import unnormalize

from typing import Literal, Any

import torch.cuda.nvtx as nvtx

from physicsnemo.utils.domino.utils import *


def loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    padded_value: float = -10,
) -> torch.Tensor:
    """Calculate mean squared error or root mean squared error with masking for padded values.

    Args:
        output: Predicted values from the model
        target: Ground truth values
        loss_type: Type of loss to calculate ("mse" or "rmse")
        padded_value: Value used for padding in the tensor

    Returns:
        Calculated loss as a scalar tensor
    """
    mask = abs(target - padded_value) > 1e-3

    if loss_type == "rmse":
        dims = (0, 1)
    else:
        dims = None

    num = torch.sum(mask * (output - target) ** 2.0, dims)
    if loss_type == "rmse":
        denom = torch.sum(mask * (target - torch.mean(target, (0, 1))) ** 2.0, dims)
        loss = torch.mean(num / denom)
    elif loss_type == "mse":
        denom = torch.sum(mask)
        loss = torch.mean(num / denom)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss



def compute_loss_dict(
    prediction_vol: torch.Tensor,
    prediction_surf: torch.Tensor,
    batch_inputs: dict,
    loss_fn_type: dict,
    integral_scaling_factor: float,
    surf_loss_scaling: float,
    vol_loss_scaling: float,
    first_deriv: torch.nn.Module | None = None,
    eqn: Any = None,
    bounding_box: torch.Tensor | None = None,
    vol_factors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the loss terms in a single function call.

    Computes:
    - Volume loss if prediction_vol is not None
    - Surface loss if prediction_surf is not None
    - Integral loss if prediction_surf is not None
    - Total loss as a weighted sum of the above

    Returns:
    - Total loss as a scalar tensor
    - Dictionary of loss terms (for logging, etc)
    """
    nvtx.range_push("Loss Calculation")
    total_loss_terms = []
    loss_dict = {}

    if prediction_vol is not None:
        target_vol = batch_inputs["volume_fields"]
            
        loss_vol = loss_fn(
            prediction_vol,
            target_vol,
            loss_fn_type.loss_type,
            padded_value=-10,
        )
        loss_dict["loss_vol"] = loss_vol
        total_loss_terms.append(loss_vol)

    if prediction_surf is not None:
        target_surf = batch_inputs["surface_fields"]

        loss_surf = loss_fn(
            prediction_surf,
            target_surf,
            loss_fn_type.loss_type,
        )

        if loss_fn_type.loss_type == "mse":
            loss_surf = loss_surf * surf_loss_scaling

        total_loss_terms.append(loss_surf)
        loss_dict["loss_surf"] = loss_surf

    total_loss = sum(total_loss_terms)
    loss_dict["total_loss"] = total_loss
    nvtx.range_pop()

    return total_loss, loss_dict
