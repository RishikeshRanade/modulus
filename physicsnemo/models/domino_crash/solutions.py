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

"""
This code contains the DoMINO model architecture.
The DoMINO class contains an architecture to model surface
quantities
"""

from collections import defaultdict

import torch
import torch.nn as nn


def apply_parameter_encoding(
    mesh_centers: torch.Tensor,
    global_params_values: torch.Tensor,
    global_params_reference: torch.Tensor,
) -> torch.Tensor:
    processed_parameters = []
    for k in range(global_params_values.shape[1]):
        param = torch.unsqueeze(global_params_values[:, k, :], 1)
        ref = torch.unsqueeze(global_params_reference[:, k, :], 1)
        param = param.expand(
            param.shape[0],
            mesh_centers.shape[1],
            param.shape[2],
        )
        param = param / (ref + 1e-6)
        processed_parameters.append(param)
    processed_parameters = torch.cat(processed_parameters, axis=-1)

    return processed_parameters

class SolutionCalculatorSurface(nn.Module):
    """
    Module to calculate the output solution of the DoMINO Model for surface data.
    """

    def __init__(
        self,
        num_variables: int,
        num_sample_points: int,
        encode_parameters: bool,
        use_surface_normals: bool,
        use_surface_area: bool,
        parameter_model: nn.Module | None,
        aggregation_model: nn.ModuleList,
        nn_basis: nn.ModuleList,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.num_sample_points = num_sample_points
        self.encode_parameters = encode_parameters
        self.use_surface_normals = use_surface_normals
        self.use_surface_area = use_surface_area
        self.parameter_model = parameter_model
        self.aggregation_model = aggregation_model
        self.nn_basis = nn_basis

        if self.encode_parameters:
            if self.parameter_model is None:
                raise ValueError(
                    "Parameter model is required when encode_parameters is True"
                )

    def forward(
        self,
        surface_mesh_centers: torch.Tensor,
        encoding_g: torch.Tensor,
        encoding_node: torch.Tensor,
        surface_mesh_neighbors: torch.Tensor,
        surface_normals: torch.Tensor,
        surface_neighbors_normals: torch.Tensor,
        surface_areas: torch.Tensor,
        surface_neighbors_areas: torch.Tensor,
        global_params_values: torch.Tensor,
        global_params_reference: torch.Tensor,
        surface_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Function to approximate solution given the neighborhood information"""

        if self.encode_parameters:
            param_encoding = apply_parameter_encoding(
                surface_mesh_centers, global_params_values, global_params_reference
            )
            param_encoding = self.parameter_model(param_encoding)

        centers_inputs = [
            surface_mesh_centers,
        ]
        neighbors_inputs = [
            surface_mesh_neighbors,
        ]

        if self.use_surface_normals:
            centers_inputs.append(surface_normals)
            if self.num_sample_points > 1:
                neighbors_inputs.append(surface_neighbors_normals)

        if self.use_surface_area:
            centers_inputs.append(torch.log(surface_areas) / 10)
            if self.num_sample_points > 1:
                neighbors_inputs.append(torch.log(surface_neighbors_areas) / 10)

        surface_mesh_centers = torch.cat(centers_inputs, dim=-1)
        surface_mesh_neighbors = torch.cat(neighbors_inputs, dim=-1)

        for f in range(self.num_variables):
            for p in range(self.num_sample_points):
                if p == 0:
                    surf_m_c = surface_mesh_centers
                else:
                    surf_m_c = surface_mesh_neighbors[:, :, p - 1] + 1e-6
                    noise = surface_mesh_centers - surf_m_c
                    dist = torch.norm(noise, dim=-1, keepdim=True)

                if surface_features is not None:
                    surf_m_c = torch.cat((surf_m_c, surface_features), dim=-1)
                basis_f = self.nn_basis[f](surf_m_c)
                output = torch.cat((basis_f, encoding_node, encoding_g), dim=-1)
                
                if self.encode_parameters:
                    output = torch.cat((output, param_encoding), dim=-1)
                if p == 0:
                    output_center = self.aggregation_model[f](output)
                else:
                    if p == 1:
                        output_neighbor = self.aggregation_model[f](output) * (
                            1.0 / dist
                        )
                        dist_sum = 1.0 / dist
                    else:
                        output_neighbor += self.aggregation_model[f](output) * (
                            1.0 / dist
                        )
                        dist_sum += 1.0 / dist
            if self.num_sample_points > 1:
                output_res = 0.5 * output_center + 0.5 * output_neighbor / dist_sum
            else:
                output_res = output_center
            if f == 0:
                output_all = output_res
            else:
                output_all = torch.cat((output_all, output_res), dim=-1)

        return output_all
