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

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from transformer_engine import pytorch as te
except ImportError:
    te = None

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module
from physicsnemo.utils.neighbors import radius_search

def get_activation(activation: Literal["relu", "gelu"]) -> Callable:
    """
    Return a PyTorch activation function corresponding to the given name.
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Activation function {activation} not found")


def fourier_encode(coords, num_freqs):
    """Function to caluculate fourier features"""
    # Create a range of frequencies
    freqs = torch.exp(torch.linspace(0, math.pi, num_freqs, device=coords.device))
    # Generate sine and cosine features
    features = [torch.sin(coords * f) for f in freqs] + [
        torch.cos(coords * f) for f in freqs
    ]
    ret = torch.cat(features, dim=-1)
    return ret


def fourier_encode_vectorized(coords, freqs):
    """Vectorized Fourier feature encoding"""
    D = coords.shape[-1]
    F = freqs.shape[0]

    # freqs = torch.exp(torch.linspace(0, math.pi, num_freqs, device=coords.device))  # [F]
    freqs = freqs[None, None, :, None]  # reshape to [*, F, 1] for broadcasting

    coords = coords.unsqueeze(-2)  # [*, 1, D]
    scaled = (coords * freqs).reshape(*coords.shape[:-2], D * F)  # [*, D, F]
    features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # [*, D, 2F]

    return features.reshape(*coords.shape[:-2], D * 2 * F)  # [*, D * 2F]

class BallQuery(nn.Module):
    """
    Warp-based ball-query layer for finding neighboring points within a specified radius.

    This layer uses an accelerated ball query implementation to efficiently find points
    within a specified radius of query points.
    """

    def __init__(
        self,
        radius: float = 0.25,
        neighbors_in_radius: int = 10,
    ):
        """
        Initialize the BQWarp layer.

        Args:
            radius: Radius for ball query operation
            neighbors_in_radius: Maximum number of neighbors to return within radius
        """
        super().__init__()
        self.radius = radius
        self.neighbors_in_radius = neighbors_in_radius

    def forward(
        self, pc1: torch.Tensor, pc2: torch.Tensor, reverse_mapping: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs ball query operation to find neighboring points and their features.

        This method uses the Warp-accelerated ball query implementation to find points
        within a specified radius. It can operate in two modes:
        - Forward mapping: Find points from x that are near p_grid points (reverse_mapping=False)
        - Reverse mapping: Find points from p_grid that are near x points (reverse_mapping=True)

        Args:
            pc1: Tensor of shape (batch_size, num_points1, 3+features) containing point coordinates
               and their features
            pc2: Tensor of shape (batch_size, num_points2, 3+features) containing point coordinates
            reverse_mapping: Boolean flag to control the direction of the mapping:
                            - True: Find pc2 points near pc1 points
                            - False: Find pc1 points near pc2 points

        Returns:
            tuple containing:
                - mapping: Tensor containing indices of neighboring points
                - outputs: Tensor containing coordinates of the neighboring points
        """

        if reverse_mapping:
            mapping, outputs = radius_search(
                pc1[0],
                pc2[0],
                self.radius,
                self.neighbors_in_radius,
                return_points=True,
            )
            mapping = mapping.unsqueeze(0)
            outputs = outputs.unsqueeze(0)
        else:
            mapping, outputs = radius_search(
                pc2[0],
                pc1[0],
                self.radius,
                self.neighbors_in_radius,
                return_points=True,
            )
            mapping = mapping.unsqueeze(0)
            outputs = outputs.unsqueeze(0)

        return mapping, outputs

class UnstructuredConvBlock(nn.Module):
    """
    An Unstructured Convolution block, followed by an optional normalization and activation.

    Parameters:
    ----------
        input_features (int): Number of channels in the input.
        output_features (int): Number of channels produced by the convolution.
        activation (Optional[str]): Type of activation to use. Default is 'relu'.
        normalization (Optional[str]): Type of normalization to use. Default is 'groupnorm'.
        normalization_args (dict): Arguments for the normalization layer.
        feature_map_channels (List[int]): Number of channels for each conv block in the encoder and decoder.
        fourier_features (bool): Whether to use fourier features.
        num_modes (int): Number of modes for fourier features.
        bq_radii (List[int]): Radii for ball query.
        bq_kernel_size (List[int]): Kernel size for ball query.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        activation: Optional[str] = "relu",
        normalization: Optional[str] = "groupnorm",
        normalization_args: Optional[dict] = None,
        feature_map_channels: List[int] = [32, 16, 8],
        fourier_features: bool = True,
        num_modes: int = 4,
        bq_radii: List[int] = [0.05, 0.25, 1.0],
        bq_kernel_size: List[int] = [10, 10, 10],
    ):
        super().__init__()
        self.feature_map_channels = feature_map_channels
        input_map_channels = feature_map_channels[:-1]
        self.fourier_features = fourier_features
        self.bq_radii = bq_radii
        self.bq_kernel_size = bq_kernel_size

        if self.fourier_features:
            input_features_calculated = input_features * (1 + 2 * num_modes)
        else:
            input_features_calculated = input_features

        input_map_channels = input_map_channels.insert(0, input_features_calculated)

        self.fc_maps = nn.ModuleList()
        for i in range(len(feature_map_channels)):
            self.fc_maps.append(
                nn.Linear(input_map_channels[i], feature_map_channels[i])
            )
        self.bq_warp = nn.ModuleList()
        for j in range(len(self.bq_radii)):
            self.bq_warp.append(
                BallQuery(
                    radius=self.bq_radii[j],
                    neighbors_in_radius=self.bq_kernel_size[j],
                )
            )

        self.activation = get_activation(activation)

        if self.fourier_features:
            self.register_buffer(
                "freqs", torch.exp(torch.linspace(0, math.pi, self.num_modes))
            )

        if normalization:
            if normalization == "groupnorm":
                default_args = {"num_groups": 1, "num_channels": output_features}
                norm_args = {
                    **default_args,
                    **(normalization_args if normalization_args else {}),
                }
                self.norm = nn.GroupNorm(**norm_args)
            elif normalization == "batchnorm":
                self.norm = nn.BatchNorm3d(output_features)
            elif normalization == "layernorm":
                self.norm = ReshapedLayerNorm(output_features)
            else:
                raise ValueError(
                    f"Normalization type '{normalization}' is not supported."
                )
        else:
            self.norm = nn.Identity()

    def forward(self, pc1: torch.Tensor, pc2: torch.Tensor, pc1_features: torch.Tensor = None) -> torch.Tensor:
        geo_encoding = []
        # pc1_features has shape (batch_size, num_points_pc1, num_features * num_bq_radii)
        mask = abs(pc1 - 0) > 1e-6
        for j in range(len(self.bq_radii)):
            mapping, k_short = self.bq_warp[j](pc1, pc2) # Mapping between pc1 and pc2

            # k_short has shape (batch_size, num_points_pc2, num_neighbors_in_radius, 3)
            # pc1_features has shape (batch_size, num_points_pc1, num_features * num_bq_radii)

            num_neighbors_in_radius = k_short.shape[2]
            pc1_features = torch.expand
            
            if self.fourier_features:
                facets = torch.cat((k_short, fourier_encode_vectorized(k_short, self.freqs)), axis=-1)
            else:
                facets = k_short
            for i in range(len(self.fc_maps)):
                x = self.fc_maps[i](facets)
                if i == len(self.fc_maps) - 1:
                    x = F.tanh(x)
                else:
                    x = self.activation(x)
            x = torch.sum(x * mask, 2) # [batch_size, num_points_pc2, num_features]
            geo_encoding.append(x)

        geo_encoding = torch.cat(geo_encoding, dim=-1) # [batch_size, num_points_pc2, num_features * num_bq_radii]
        return geo_encoding

# @dataclass
# class MetaData(ModelMetaData):
#     name: str = "pcn"
#     # Optimization
#     jit: bool = False
#     cuda_graphs: bool = True
#     amp: bool = True
#     # Inference
#     onnx_cpu: bool = True
#     onnx_gpu: bool = True
#     onnx_runtime: bool = True
#     # Physics informed
#     var_dim: int = 1
#     func_torch: bool = False
#     auto_grad: bool = False


# class PCN(Module):
#     """
#     PCN model, featuring an encoder-decoder architecture with skip connections.

#     Parameters:
#     ----------
#         input_features (int): Number of channels in the input.
#         output_features (int): Number of channels produced by the convolution.
#         activation (Optional[str]): Type of activation to use. Default is 'relu'.
#         num_blocks (int): Number of blocks in the encoder and decoder.
#         model_parameters (ModelMetaData): Model parameters controlled by config.yaml

#     Returns:
#     -------
#         torch.Tensor: The processed output tensor.
#     """

#     def __init__(
#         self,
#         input_features: int,
#         output_features: int,
#         activation: Optional[str] = "relu",
#         num_blocks: int = 5,
#         model_parameters=None,
#     ):
#         super().__init__()

#         # Construct the encoder
#         self.encoder = EncoderBlock(
#             in_channels=in_channels,
#             feature_map_channels=feature_map_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             model_depth=model_depth,
#             num_conv_blocks=num_conv_blocks,
#             activation=conv_activation,
#             padding=padding,
#             padding_mode=padding_mode,
#             pooling_type=pooling_type,
#             pool_size=pool_size,
#             normalization=normalization,
#             normalization_args=normalization_args,
#         )

#         # Construct the decoder
#         if num_conv_blocks > 1:
#             decoder_feature_maps = feature_map_channels[::-1][
#                 1:
#             ]  # Reverse and discard the first channel
#         else:
#             decoder_feature_maps = feature_map_channels[::-1]
#         self.decoder = DecoderBlock(
#             out_channels=out_channels,
#             feature_map_channels=decoder_feature_maps,
#             kernel_size=kernel_size,
#             stride=stride,
#             model_depth=model_depth - 1,
#             num_conv_blocks=num_conv_blocks,
#             conv_activation=conv_activation,
#             conv_transpose_activation=conv_transpose_activation,
#             padding=padding,
#             padding_mode=padding_mode,
#             normalization=normalization,
#             normalization_args=normalization_args,
#         )

#         # Initialize attention blocks for each skip connection
#         if self.use_attn_gate:
#             self.attention_blocks = nn.ModuleList(
#                 [
#                     AttentionBlock(
#                         F_g=attn_decoder_feature_maps[i],
#                         F_l=attn_feature_map_channels[i],
#                         F_int=attn_intermediate_channels,
#                     )
#                     for i in range(model_depth - 1)
#                 ]
#             )

#     def checkpointed_forward(self, layer, x):
#         """Wrapper to apply gradient checkpointing if enabled."""
#         if self.gradient_checkpointing:
#             return checkpoint.checkpoint(layer, x, use_reentrant=False)
#         return layer(x)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         skip_features = []
#         # Encoding path
#         for layer in self.encoder.layers:
#             if isinstance(layer, Pool3d):
#                 skip_features.append(x)
#             # Apply checkpointing if enabled
#             x = self.checkpointed_forward(layer, x)

#         # Decoding path
#         skip_features = skip_features[::-1]  # Reverse the skip features
#         concats = 0  # Track number of concats
#         for layer in self.decoder.layers:
#             if isinstance(layer, ConvTranspose):
#                 x = self.checkpointed_forward(layer, x)
#                 if self.use_attn_gate:
#                     # Apply attention to the skip connection
#                     skip_att = self.attention_blocks[concats](x, skip_features[concats])
#                     x = torch.cat([x, skip_att], dim=1)
#                 else:
#                     x = torch.cat([x, skip_features[concats]], dim=1)
#                 concats += 1
#             else:
#                 # Apply checkpointing for other layers
#                 x = self.checkpointed_forward(layer, x)

#         return x