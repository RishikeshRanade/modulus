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

"""
This code provides the datapipe for reading the processed npy files,
generating multi-res grids, calculating signed distance fields,
sampling random points in the volume and on surface,
normalizing fields and returning the output tensors as a dictionary.

This datapipe also non-dimensionalizes the fields, so the order in which the variables should
be fixed: velocity, pressure, turbulent viscosity for volume variables and
pressure, wall-shear-stress for surface variables. The different parameters such as
variable names, domain resolution, sampling size etc. are configurable in config.yaml.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Protocol, Sequence, Union

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
from omegaconf import DictConfig
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.utils.data import Dataset

from physicsnemo.datapipes.cae.cae_dataset import (
    CAEDataset,
    compute_mean_std_min_max,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor
from physicsnemo.models.domino_transformed.utils import (
    calculate_center_of_mass,
    create_grid,
    get_filenames,
    normalize,
    pad,
    shuffle_array,
    standardize,
    unnormalize,
    unstandardize,
)
from physicsnemo.nn.functional import knn, signed_distance_field
from physicsnemo.utils.profiling import profile



@dataclass
class DoMINODataConfig:
    """Configuration for DoMINO dataset processing pipeline.

    Attributes:
        data_path: Path to the dataset to load.
        phase: Which phase of data to load ("train", "val", or "test").
        surface_variables: (Surface specific) Names of surface variables.
        surface_points_sample: (Surface specific) Number of surface points to sample per batch.
        num_surface_neighbors: (Surface specific) Number of surface neighbors to consider for nearest neighbors approach.
        surface_sampling_algorithm: (Surface specific) Algorithm to use for surface sampling ("area_weighted" or "random").
        surface_factors: (Surface specific) Non-dimensionalization factors for surface variables.
            If set, and scaling_type is:
            - min_max_scaling -> rescale surface_fields to the min/max set here
            - mean_std_scaling -> rescale surface_fields to the mean and std set here.
        volume_variables: (Volume specific) Names of volume variables.
        volume_points_sample: (Volume specific) Number of volume points to sample per batch.
        volume_sample_from_disk: (Volume specific) If the volume data is in a shuffled state on disk,
            read contiguous chunks of the data rather than the entire volume data.  This greatly
            accelerates IO in bandwidth limited systems or when the volumetric data is very large.
        volume_factors: (Volume specific) Non-dimensionalization factors for volume variables scaling.
            If set, and scaling_type is:
            - min_max_scaling -> rescale volume_fields to the min/max set here
            - mean_std_scaling -> rescale volume_fields to the mean and std set here.
        normalize_coordinates: Whether to normalize coordinates based on min/max values.
        sampling: Whether to downsample the full resolution mesh to fit in GPU memory.
            Surface and volume sampling points are configured separately as:
            - surface.points_sample
            - volume.points_sample
        geom_points_sample: Number of STL points sampled per batch.
            Independent of volume.points_sample and surface.points_sample.
        scaling_type: Scaling type for volume variables.
            If used, will rescale the volume_fields and surface fields outputs.
            Requires volume.factor and surface.factor to be set.
        compute_scaling_factors: Whether to compute scaling factors.
            Not available if caching.
            Many preprocessing pieces are disabled if computing scaling factors.
        caching: Whether this is for caching or serving.
        deterministic: Whether to use a deterministic seed for sampling and random numbers.
        gpu_preprocessing: Whether to do preprocessing on the GPU (False for CPU).
        gpu_output: Whether to return output on the GPU as cupy arrays.
            If False, returns numpy arrays.
            You might choose gpu_preprocessing=True and gpu_output=False if caching.
        shard_grid: Whether to shard the grid across GPUs for domain parallelism.
            Applies to the surf_grid and similiar tensors.
        shard_points: Whether to shard the points across GPUs for domain parallelism.
            Applies to the volume_fields/surface_fields and similiar tensors.
        use_sdf: Whether to compute signed distance field for volume points (for use in basis).
        scale_invariance: If True, divide position-like coordinates by reference_scale (x, y, z).
        reference_scale: Length-3 scale [sx, sy, sz]. Required when scale_invariance is True.
    """

    data_path: Path | None
    phase: Literal["train", "val", "test"]

    # Surface-specific variables:
    surface_variables: Optional[Sequence] = ("pMean", "wallShearStress")
    surface_points_sample: int = 1024
    num_surface_neighbors: int = 11
    surface_sampling_algorithm: str = Literal["area_weighted", "random"]
    surface_factors: Optional[Sequence] = None

    # Volume specific variables:
    volume_variables: Optional[Sequence] = ("UMean", "pMean")
    volume_points_sample: int = 1024
    volume_sample_from_disk: bool = False
    volume_factors: Optional[Sequence] = None

    normalize_coordinates: bool = False
    sampling: bool = False
    geom_points_sample: int = 300000
    scaling_type: Optional[Literal["min_max_scaling", "mean_std_scaling"]] = None
    compute_scaling_factors: bool = False
    caching: bool = False
    deterministic: bool = False
    gpu_preprocessing: bool = True
    gpu_output: bool = True

    shard_grid: bool = False
    shard_points: bool = False
    use_sdf: bool = True

    scale_invariance: bool = False
    reference_scale: list[float] | None = None

    def __post_init__(self):
        if self.data_path is not None:
            # Ensure data_path is a Path object:
            if isinstance(self.data_path, str):
                self.data_path = Path(self.data_path)
            self.data_path = self.data_path.expanduser()

            if not self.data_path.exists():
                raise ValueError(f"Path {self.data_path} does not exist")

            if not self.data_path.is_dir():
                raise ValueError(f"Path {self.data_path} is not a directory")

        # Object if caching settings are impossible:
        if self.caching:
            if self.sampling:
                raise ValueError("Sampling should be False for caching")
            if self.compute_scaling_factors:
                raise ValueError("Compute scaling factors should be False for caching")

        if self.phase not in [
            "train",
            "val",
            "test",
        ]:
            raise ValueError(
                f"phase should be one of ['train', 'val', 'test'], got {self.phase}"
            )
        if self.scaling_type is not None:
            if self.scaling_type not in [
                "min_max_scaling",
                "mean_std_scaling",
            ]:
                raise ValueError(
                    f"scaling_type should be one of ['min_max_scaling', 'mean_std_scaling'], got {self.scaling_type}"
                )

        if self.scale_invariance:
            if self.reference_scale is None:
                raise ValueError(
                    "reference_scale must be set if scale_invariance is enabled"
                )
            self.reference_scale = list(self.reference_scale)
            if len(self.reference_scale) != 3:
                raise ValueError("reference_scale must be a list of 3 floats")
            self.reference_scale = (
                torch.tensor(self.reference_scale, dtype=torch.float32).reshape(1, 3)
            )


##### TODO
# - The SDF normalization here is based on using a normalized mesh and
#   a normalized coordinate.  The alternate method is to normalize to the min/max of the grid.


class DoMINODataPipe(Dataset):
    """
    Datapipe for DoMINO

    Leverages a dataset for the actual reading of the data, and this
    object is responsible for preprocessing the data.

    """

    def __init__(
        self,
        input_path,
        model_type: Literal["surface", "volume", "combined"],
        pin_memory: bool = False,
        **data_config_overrides,
    ):
        # Perform config packaging and validation
        self.config = DoMINODataConfig(data_path=input_path, **data_config_overrides)

        # Set up the distributed manager:
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()

        dist = DistributedManager()

        # Set devices for the preprocessing and IO target
        self.preproc_device = (
            dist.device if self.config.gpu_preprocessing else torch.device("cpu")
        )
        # The cae_dataset will automatically target this device
        # In an async transfer.
        self.output_device = (
            dist.device if self.config.gpu_output else torch.device("cpu")
        )

        # Model type determines whether we process surface, volume, or both.
        self.model_type = model_type

        # Ensure the volume and surface scaling factors are torch tensors
        # and on the right device:
        if self.config.volume_factors is not None:
            if not isinstance(self.config.volume_factors, torch.Tensor):
                self.config.volume_factors = torch.from_numpy(
                    self.config.volume_factors
                )
            self.config.volume_factors = self.config.volume_factors.to(
                self.preproc_device, dtype=torch.float32
            )
        if self.config.surface_factors is not None:
            if not isinstance(self.config.surface_factors, torch.Tensor):
                self.config.surface_factors = torch.from_numpy(
                    self.config.surface_factors
                )
            self.config.surface_factors = self.config.surface_factors.to(
                self.preproc_device, dtype=torch.float32
            )

        self.dataset = None


    @profile
    def downsample_geometry(
        self,
        stl_vertices,
    ) -> torch.Tensor:
        """
        Downsample the geometry to the desired number of points.

        Args:
            stl_vertices: The vertices of the surface.
        """

        if self.config.sampling:
            geometry_points = self.config.geom_points_sample

            geometry_coordinates_sampled, idx_geometry = shuffle_array(
                stl_vertices, geometry_points
            )
            if geometry_coordinates_sampled.shape[0] < geometry_points:
                raise ValueError(
                    "Surface mesh has fewer points than requested sample size"
                )
            geom_centers = geometry_coordinates_sampled
        else:
            geom_centers = stl_vertices

        return geom_centers

    def process_surface(
        self,
        center_of_mass: torch.Tensor,
        surface_coordinates: torch.Tensor,
        surface_normals: torch.Tensor,
        surface_sizes: torch.Tensor,
        surface_fields: torch.Tensor | None,
        scale_factor: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return_dict = {}

        ########################################################################
        # Remove any sizes <= 0:
        ########################################################################
        idx = surface_sizes > 0
        surface_sizes = surface_sizes[idx]
        surface_normals = surface_normals[idx]
        surface_coordinates = surface_coordinates[idx]
        if surface_fields is not None:
            surface_fields = surface_fields[idx]

        ########################################################################
        # Perform Down sampling of the surface fields.
        # Note that we snapshot the full surface coordinates for
        # use in the kNN in the next step.
        ########################################################################

        full_surface_coordinates = surface_coordinates
        full_surface_normals = surface_normals
        full_surface_sizes = surface_sizes

        if self.config.sampling:
            # Perform the down sampling:
            if self.config.surface_sampling_algorithm == "area_weighted":
                weights = surface_sizes
            else:
                weights = None

            surface_coordinates_sampled, idx_surface = shuffle_array(
                surface_coordinates,
                self.config.surface_points_sample,
                weights=weights,
            )

            if surface_coordinates_sampled.shape[0] < self.config.surface_points_sample:
                raise ValueError(
                    "Surface mesh has fewer points than requested sample size"
                )

            # Select out the sampled points for non-neighbor arrays:
            if surface_fields is not None:
                surface_fields = surface_fields[idx_surface]

            # Subsample the normals and sizes:
            surface_normals = surface_normals[idx_surface]
            surface_sizes = surface_sizes[idx_surface]
            # Update the coordinates to the sampled points:
            surface_coordinates = surface_coordinates_sampled

        ########################################################################
        # Perform a kNN on the surface to find the neighbor information
        ########################################################################
        if self.config.num_surface_neighbors > 1:
            # Perform the kNN:
            neighbor_indices, neighbor_distances = knn(
                points=full_surface_coordinates,
                queries=surface_coordinates,
                k=self.config.num_surface_neighbors,
            )
            # print(f"Full surface coordinates shape: {full_surface_coordinates.shape}")
            # Pull out the neighbor elements.
            # Note that `neighbor_indices` is the index into the original,
            # full sized tensors (full_surface_coordinates, etc).
            surface_neighbors = full_surface_coordinates[neighbor_indices][:, 1:]
            surface_neighbors_normals = full_surface_normals[neighbor_indices][:, 1:]
            surface_neighbors_sizes = full_surface_sizes[neighbor_indices][:, 1:]
        else:
            surface_neighbors = surface_coordinates
            surface_neighbors_normals = surface_normals
            surface_neighbors_sizes = surface_sizes

        # Better to normalize everything after the kNN and sampling
        if self.config.normalize_coordinates:
            surface_coordinates -= center_of_mass
            surface_neighbors -= center_of_mass
        if self.config.scale_invariance and scale_factor is not None:
            surface_coordinates = surface_coordinates / scale_factor
            surface_neighbors = surface_neighbors / scale_factor

        ########################################################################
        # Apply scaling to the targets, if desired:
        ########################################################################
        if self.config.scaling_type is not None and surface_fields is not None:
            surface_fields = self.scale_model_targets(
                surface_fields, self.config.surface_factors
            )

        return_dict.update(
            {
                "surface_mesh_centers": surface_coordinates,
                "surface_mesh_neighbors": surface_neighbors,
                "surface_normals": surface_normals,
                "surface_neighbors_normals": surface_neighbors_normals,
                "surface_areas": surface_sizes,
                "surface_neighbors_areas": surface_neighbors_sizes,
            }
        )
        if surface_fields is not None:
            return_dict["surface_fields"] = surface_fields

        return return_dict

    def process_volume(
        self,
        volume_coordinates: torch.Tensor,
        center_of_mass: torch.Tensor,
        stl_vertices: torch.Tensor,
        stl_indices: torch.Tensor,
        volume_fields: torch.Tensor | None,
        scale_factor: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Preprocess the volume data.

        First, if configured, we reject points not in the volume bounding box.

        Next, if sampling is enabled, we sample the volume points and apply that
        sampling to the ground truth too, if it's present.

        """
        
        ########################################################################
        # Apply sampling to the volume coordinates and fields
        ########################################################################

        # If the volume data has been sampled from disk, directly, then
        # still apply sampling.  We over-pull from disk deliberately.
        if self.config.sampling:
            # Generate a series of idx to sample the volume
            # without replacement
            volume_coordinates_sampled, idx_volume = shuffle_array(
                volume_coordinates, self.config.volume_points_sample
            )
            volume_coordinates_sampled = volume_coordinates[idx_volume]
            # In case too few points are in the sampled data (because the
            # inputs were too few), pad the outputs:
            if volume_coordinates_sampled.shape[0] < self.config.volume_points_sample:
                raise ValueError(
                    "Volume mesh has fewer points than requested sample size"
                )

            # Apply the same sampling to the targets, too:
            if volume_fields is not None:
                volume_fields = volume_fields[idx_volume]

            volume_coordinates = volume_coordinates_sampled

        ########################################################################
        # Apply normalization to the coordinates, if desired:
        ########################################################################
        if self.config.normalize_coordinates:
            volume_coordinates -= center_of_mass
            normed_vertices = stl_vertices - center_of_mass
        else:
            normed_vertices = stl_vertices
        if self.config.scale_invariance and scale_factor is not None:
            volume_coordinates = volume_coordinates / scale_factor
            normed_vertices = normed_vertices / scale_factor

        ########################################################################
        # Apply scaling to the targets, if desired:
        ########################################################################
        if self.config.scaling_type is not None and volume_fields is not None:
            volume_fields = self.scale_model_targets(
                volume_fields, self.config.volume_factors
            )

        ########################################################################
        # Compute Signed Distance Function for volumetric quantities
        # Note - the SDF happens here, after volume data processing finishes,
        # because we need to use the (maybe) normalized volume coordinates and grid
        ########################################################################

        # Get the SDF of all the selected volume coordinates,
        # And keep the closest point to each one.
        if self.config.use_sdf:
            sdf_nodes, sdf_node_closest_point = signed_distance_field(
                normed_vertices,
                stl_indices,
                volume_coordinates,
                use_sign_winding_number=True,
            )
            sdf_nodes = sdf_nodes.reshape((-1, 1))
            sdf_nodes = torch.cat((sdf_nodes, sdf_node_closest_point), dim=-1)

        return_dict = {
            "volume_mesh_centers": volume_coordinates,
            "sdf_nodes": sdf_nodes,
        }

        if volume_fields is not None:
            return_dict["volume_fields"] = volume_fields

        return return_dict

    @torch.no_grad()
    def process_data(self, data_dict):
        # Validate that all required keys are present in data_dict
        required_keys = [
            "global_params_values",
            "global_params_reference",
            "stl_coordinates",
            "stl_faces",
            "stl_centers",
            "stl_areas",
        ]
        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in data_dict: {missing_keys}. "
                f"Required keys are: {required_keys}"
            )

        # Start building the preprocessed return dict:
        return_dict = {
            "global_params_values": data_dict["global_params_values"],
            "global_params_reference": data_dict["global_params_reference"],
        }

        # DoMINO's sharded datapipe can be tricky - output shapes are not always
        # so simple to calculate, since much of the datapipe is dynamic.
        # The datset will read in sharded data, to minimize IO.
        # We collect it all locally, here, and then scatter
        # Appropriately for the outputs

        if self.config.shard_grid or self.config.shard_points:
            # Get the mesh:
            mesh = data_dict["stl_coordinates"]._spec.mesh
            local_data_dict = {}
            for key, value in data_dict.items():
                local_data_dict[key] = value.full_tensor()

            data_dict = local_data_dict

        ########################################################################
        # Process the core STL information
        ########################################################################

        # This is a center of mass computation for the stl surface,
        # using the size of each mesh point as weight.
        center_of_mass = calculate_center_of_mass(
            data_dict["stl_centers"], data_dict["stl_areas"]
        )

        scale_factor = (
            self.config.reference_scale if self.config.scale_invariance else None
        )

        # This will apply downsampling if needed to the geometry coordinates
        geom_centers = self.downsample_geometry(
            stl_vertices=data_dict["stl_coordinates"],
        )
        if self.config.normalize_coordinates:
            geom_centers = geom_centers - center_of_mass
        if self.config.scale_invariance and scale_factor is not None:
            geom_centers = geom_centers / scale_factor
        return_dict["geometry_coordinates"] = geom_centers

        ########################################################################
        # Process the surface data
        ########################################################################
        if self.model_type == "surface" or self.model_type == "combined":
            surface_fields_raw = (
                data_dict["surface_fields"] if "surface_fields" in data_dict else None
            )
            surface_dict = self.process_surface(
                center_of_mass=center_of_mass,
                surface_coordinates=data_dict["surface_mesh_centers"],
                surface_normals=data_dict["surface_normals"],
                surface_sizes=data_dict["surface_areas"],
                surface_fields=surface_fields_raw,
                scale_factor=scale_factor,
            )

            return_dict.update(surface_dict)

        ########################################################################
        # Process the volume data
        ########################################################################
        # For volume data, we store this only if normalizing coordinates:
        if self.model_type == "volume" or self.model_type == "combined":
            volume_fields_raw = (
                data_dict["volume_fields"] if "volume_fields" in data_dict else None
            )
            volume_dict = self.process_volume(
                volume_coordinates=data_dict["volume_mesh_centers"],
                center_of_mass=center_of_mass,
                stl_vertices=data_dict["stl_coordinates"],
                stl_indices=data_dict["stl_faces"],
                volume_fields=volume_fields_raw,
                scale_factor=scale_factor,
            )

            return_dict.update(volume_dict)

        # For domain parallelism, shard everything appropriately:
        if self.config.shard_grid or self.config.shard_points:
            # Mesh was defined above!
            output_dict = {}

            # For scattering, we need to know the _global_ index of rank
            # 0 on this mesh:
            global_index = dist.get_global_rank(mesh.get_group(), 0)

            for key, value in return_dict.items():
                grid_placements = (
                    [
                        Shard(0),
                    ]
                    if self.config.shard_grid
                    else [
                        Replicate(),
                    ]
                )
                point_placements = (
                    [
                        Shard(0),
                    ]
                    if self.config.shard_points
                    else [
                        Replicate(),
                    ]
                )
                if key == "volume_min_max":
                    output_dict[key] = ShardTensor.from_local(
                        value,
                        mesh,
                        [
                            Replicate(),
                        ],
                    )
                elif key == "surface_min_max":
                    output_dict[key] = ShardTensor.from_local(
                        value,
                        mesh,
                        [
                            Replicate(),
                        ],
                    )
                elif not isinstance(value, ShardTensor):
                    if "grid" in key:
                        output_dict[key] = scatter_tensor(
                            value.contiguous(),
                            global_index,
                            mesh,
                            grid_placements,
                            global_shape=value.shape,
                            dtype=value.dtype,
                        )
                    else:
                        output_dict[key] = scatter_tensor(
                            value.contiguous(),
                            global_index,
                            mesh,
                            point_placements,
                            global_shape=value.shape,
                            dtype=value.dtype,
                        )
                else:
                    output_dict[key] = value

            return_dict = output_dict

        return return_dict

    def scale_model_targets(
        self, fields: torch.Tensor, factors: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale the model targets based on the configured scaling factors.
        """
        if self.config.scaling_type == "mean_std_scaling":
            field_mean = factors[0]
            field_std = factors[1]
            return standardize(fields, field_mean, field_std)
        elif self.config.scaling_type == "min_max_scaling":
            field_min = factors[1]
            field_max = factors[0]
            return normalize(fields, field_max, field_min)

    def unscale_model_outputs(
        self,
        volume_fields: torch.Tensor | None = None,
        surface_fields: torch.Tensor | None = None,
    ):
        """
        Unscale the model outputs based on the configured scaling factors.

        The unscaling is included here to make it a consistent interface regardless
        of the scaling factors and type used.

        """

        # This is a step to make sure we can apply to sharded outputs:
        if volume_fields is not None and isinstance(volume_fields, ShardTensor):
            volume_spec = volume_fields._spec
            volume_fields = ShardTensor.to_local(volume_fields)
        else:
            volume_spec = None

        if surface_fields is not None and isinstance(surface_fields, ShardTensor):
            surface_spec = surface_fields._spec
            surface_fields = ShardTensor.to_local(surface_fields)
        else:
            surface_spec = None

        if volume_fields is not None:
            if self.config.scaling_type == "mean_std_scaling":
                vol_mean = self.config.volume_factors[0]
                vol_std = self.config.volume_factors[1]
                volume_fields = unstandardize(volume_fields, vol_mean, vol_std)
            elif self.config.scaling_type == "min_max_scaling":
                vol_min = self.config.volume_factors[1]
                vol_max = self.config.volume_factors[0]
                volume_fields = unnormalize(volume_fields, vol_max, vol_min)
        if surface_fields is not None:
            if self.config.scaling_type == "mean_std_scaling":
                surf_mean = self.config.surface_factors[0]
                surf_std = self.config.surface_factors[1]
                surface_fields = unstandardize(surface_fields, surf_mean, surf_std)
            elif self.config.scaling_type == "min_max_scaling":
                surf_min = self.config.surface_factors[1]
                surf_max = self.config.surface_factors[0]
                surface_fields = unnormalize(surface_fields, surf_max, surf_min)

        if volume_spec is not None:
            volume_fields = ShardTensor.from_local(
                volume_fields,
                device_mesh=volume_spec.mesh,
                placements=volume_spec.placements,
                sharding_shapes=volume_spec.sharding_shapes(),
            )
        if surface_spec is not None:
            surface_fields = ShardTensor.from_local(
                surface_fields,
                device_mesh=surface_spec.mesh,
                placements=surface_spec.placements,
                sharding_shapes=surface_spec.sharding_shapes(),
            )

        return volume_fields, surface_fields

    def set_dataset(self, dataset: Iterable) -> None:
        """
        Pass a dataset to the datapipe to enable iterating over both in one pass.
        """
        self.dataset = dataset

        if self.config.scale_invariance and self.config.reference_scale is not None:
            if hasattr(self.dataset, "output_device"):
                self.config.reference_scale = self.config.reference_scale.to(
                    self.dataset.output_device
                )

        if self.config.volume_sample_from_disk:
            # We deliberately double the data to read compared to the sampling size:
            self.dataset.set_volume_sampling_size(
                100 * self.config.volume_points_sample
            )

    def __len__(self):
        if self.dataset is not None:
            return len(self.dataset)
        else:
            return 0

    def __getitem__(self, idx):
        """
        Function for fetching and processing a single file's data.

        Domino, in general, expects one example per file and the files
        are relatively large due to the mesh size.

        Requires the user to have set a dataset via `set_dataset`.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not present")

        # Get the data from the dataset.
        # Under the hood, this may be fetching preloaded data.
        data_dict = self.dataset[idx]

        return self.__call__(data_dict)

    def __call__(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Process the incoming data dictionary.
        - Processes the data
        - moves it to GPU
        - adds a batch dimension

        Args:
            data_dict: Dictionary containing the data to process as torch.Tensors.

        Returns:
            Dictionary containing the processed data as torch.Tensors.

        """
        data_dict = self.process_data(data_dict)

        # If the data is not on the target device, put it there:
        for key, value in data_dict.items():
            if value.device != self.output_device:
                data_dict[key] = value.to(self.output_device)

        # Add a batch dimension to the data_dict
        data_dict = {k: v.unsqueeze(0) for k, v in data_dict.items()}

        return data_dict

    def __iter__(self):
        if self.dataset is None:
            raise ValueError(
                "Dataset is not present, can not use the datapipe as an iterator."
            )

        for i, batch in enumerate(self.dataset):
            yield self.__call__(batch)


def compute_scaling_factors(
    cfg: DictConfig,
    input_path: str,
    target_keys: list[str],
    max_samples=20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Using the dataset at the path, compute the mean, std, min, and max of the target keys.

    Args:
        cfg: Hydra configuration object containing all parameters
        input_path: Path to the dataset to load.
        target_keys: List of keys to compute the mean, std, min, and max of.
        use_cache: (deprecated) This argument has no effect.
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = CAEDataset(
        data_dir=input_path,
        keys_to_read=target_keys,
        keys_to_read_if_available={},
        output_device=device,
    )

    mean, std, min_val, max_val = compute_mean_std_min_max(
        dataset,
        field_keys=target_keys,
        max_samples=max_samples,
    )

    return mean, std, min_val, max_val


class CachedDoMINODataset(Dataset):
    """
    Dataset for reading cached DoMINO data files, with optional resampling.
    Acts as a drop-in replacement for DoMINODataPipe.
    """

    # @nvtx_annotate(message="CachedDoMINODataset __init__")
    def __init__(
        self,
        data_path: Union[str, Path],
        phase: Literal["train", "val", "test"] = "train",
        sampling: bool = False,
        volume_points_sample: Optional[int] = None,
        surface_points_sample: Optional[int] = None,
        geom_points_sample: Optional[int] = None,
        model_type=None,  # Model_type, surface, volume or combined
        deterministic_seed=False,
        surface_sampling_algorithm="area_weighted",
    ):
        super().__init__()

        self.model_type = model_type
        if deterministic_seed:
            np.random.seed(42)

        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path.expanduser()

        if not self.data_path.exists():
            raise AssertionError(f"Path {self.data_path} does not exist")
        if not self.data_path.is_dir():
            raise AssertionError(f"Path {self.data_path} is not a directory")

        self.deterministic_seed = deterministic_seed
        self.sampling = sampling
        self.volume_points = volume_points_sample
        self.surface_points = surface_points_sample
        self.geom_points = geom_points_sample
        self.surface_sampling_algorithm = surface_sampling_algorithm

        self.filenames = get_filenames(self.data_path, exclude_dirs=True)

        total_files = len(self.filenames)

        self.phase = phase
        self.indices = np.array(range(total_files))

        np.random.shuffle(self.indices)

        if not self.filenames:
            raise AssertionError(f"No cached files found in {self.data_path}")

    def __len__(self):
        return len(self.indices)

    # @nvtx_annotate(message="CachedDoMINODataset __getitem__")
    def __getitem__(self, idx):
        if self.deterministic_seed:
            np.random.seed(idx)
        nvtx.range_push("Load cached file")

        index = self.indices[idx]
        cfd_filename = self.filenames[index]

        filepath = self.data_path / cfd_filename
        result = np.load(filepath, allow_pickle=True).item()
        result = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in result.items()
        }

        nvtx.range_pop()
        if not self.sampling:
            return result

        nvtx.range_push("Sample points")

        # Sample volume points if present
        if "volume_mesh_centers" in result and self.volume_points:
            coords_sampled, idx_volume = shuffle_array(
                result["volume_mesh_centers"], self.volume_points
            )
            if coords_sampled.shape[0] < self.volume_points:
                coords_sampled = pad(
                    coords_sampled, self.volume_points, pad_value=-10.0
                )

            result["volume_mesh_centers"] = coords_sampled
            for key in [
                "volume_fields",
                "pos_volume_closest",
                "pos_volume_center_of_mass",
                "sdf_nodes",
            ]:
                if key in result:
                    result[key] = result[key][idx_volume]

        # Sample surface points if present
        if "surface_mesh_centers" in result and self.surface_points:
            if self.surface_sampling_algorithm == "area_weighted":
                coords_sampled, idx_surface = shuffle_array(
                    points=result["surface_mesh_centers"],
                    n_points=self.surface_points,
                    weights=result["surface_areas"],
                )
            else:
                coords_sampled, idx_surface = shuffle_array(
                    result["surface_mesh_centers"], self.surface_points
                )

            if coords_sampled.shape[0] < self.surface_points:
                coords_sampled = pad(
                    coords_sampled, self.surface_points, pad_value=-10.0
                )

            ii = result["neighbor_indices"]
            result["surface_mesh_neighbors"] = result["surface_mesh_centers"][ii]
            result["surface_neighbors_normals"] = result["surface_normals"][ii]
            result["surface_neighbors_areas"] = result["surface_areas"][ii]

            result["surface_mesh_centers"] = coords_sampled

            for key in [
                "surface_fields",
                "surface_areas",
                "surface_normals",
                "pos_surface_center_of_mass",
                "surface_mesh_neighbors",
                "surface_neighbors_normals",
                "surface_neighbors_areas",
            ]:
                if key in result:
                    result[key] = result[key][idx_surface]

            del result["neighbor_indices"]

        # Sample geometry points if present
        if "geometry_coordinates" in result and self.geom_points:
            coords_sampled, _ = shuffle_array(
                result["geometry_coordinates"], self.geom_points
            )
            if coords_sampled.shape[0] < self.geom_points:
                coords_sampled = pad(coords_sampled, self.geom_points, pad_value=-100.0)
            result["geometry_coordinates"] = coords_sampled

        nvtx.range_pop()
        return result


def create_domino_dataset(
    cfg: DictConfig,
    phase: Literal["train", "val", "test"],
    keys_to_read: list[str],
    keys_to_read_if_available: dict[str, torch.Tensor],
    vol_factors: list[float],
    surf_factors: list[float],
    normalize_coordinates: bool = True,
    sampling: bool = True,
    device_mesh: torch.distributed.DeviceMesh | None = None,
    placements: dict[str, torch.distributed.tensor.Placement] | None = None,
):
    model_type = cfg.model.model_type
    if phase == "train":
        input_path = cfg.data.input_dir
        dataloader_cfg = cfg.train.dataloader
    elif phase == "val":
        input_path = cfg.data.input_dir_val
        dataloader_cfg = cfg.val.dataloader
    elif phase == "test":
        input_path = cfg.eval.test_path
        dataloader_cfg = None
    else:
        raise ValueError(f"Invalid phase {phase}")

    if cfg.data_processor.use_cache:
        return CachedDoMINODataset(
            input_path,
            phase=phase,
            sampling=sampling,
            volume_points_sample=cfg.model.volume_points_sample,
            surface_points_sample=cfg.model.surface_points_sample,
            geom_points_sample=cfg.model.geom_points_sample,
            model_type=cfg.model.model_type,
            surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        )
    else:
        # The dataset path works in two pieces:
        # There is a core "dataset" which is loading data and moving to GPU
        # And there is the preprocess step, here.

        # Optionally, and for backwards compatibility, the preprocess
        # object can accept a dataset which will enable it as an iterator.
        # The iteration function will loop over the dataset, preprocess the
        # output, and return it.

        overrides = {}
        if hasattr(cfg.data, "gpu_preprocessing"):
            overrides["gpu_preprocessing"] = cfg.data.gpu_preprocessing

        if hasattr(cfg.data, "gpu_output"):
            overrides["gpu_output"] = cfg.data.gpu_output

        dm = DistributedManager()

        if cfg.data.gpu_preprocessing:
            device = dm.device
            consumer_stream = torch.cuda.default_stream()
        else:
            device = torch.device("cpu")
            consumer_stream = None

        if dataloader_cfg is not None:
            preload_depth = dataloader_cfg.preload_depth
            pin_memory = dataloader_cfg.pin_memory
        else:
            preload_depth = 1
            pin_memory = False

        dataset = CAEDataset(
            data_dir=input_path,
            keys_to_read=keys_to_read,
            keys_to_read_if_available=keys_to_read_if_available,
            output_device=device,
            preload_depth=preload_depth,
            pin_memory=pin_memory,
            device_mesh=device_mesh,
            placements=placements,
            consumer_stream=consumer_stream,
        )

        # Domain parallelism configuration:
        # (By default, the dataset will shard as aggressively as possible,
        # to improve IO speed and prevent bottlenecks - the datapipe
        # has to reshard to the final shape.)

        # NOTE: we can always capture the mesh and placements from the dataset
        # outputs, so no need to pass them here.
        if cfg.get("domain_parallelism", {}).get("domain_size", 1) > 1:
            shard_grid = cfg.get("domain_parallelism", {}).get("shard_grid", False)
            shard_points = cfg.get("domain_parallelism", {}).get("shard_points", False)
            overrides["shard_grid"] = shard_grid
            overrides["shard_points"] = shard_points

        if getattr(cfg.data, "scale_invariance", False):
            overrides["scale_invariance"] = cfg.data.scale_invariance
        if getattr(cfg.data, "reference_scale", None) is not None:
            overrides["reference_scale"] = list(cfg.data.reference_scale)

        datapipe = DoMINODataPipe(
            input_path,
            phase=phase,
            normalize_coordinates=normalize_coordinates,
            sampling=sampling,
            volume_points_sample=cfg.model.volume_points_sample,
            surface_points_sample=cfg.model.surface_points_sample,
            geom_points_sample=cfg.model.geom_points_sample,
            volume_factors=vol_factors,
            surface_factors=surf_factors,
            scaling_type=cfg.model.normalization,
            model_type=model_type,
            volume_sample_from_disk=cfg.data.volume_sample_from_disk,
            num_surface_neighbors=cfg.model.num_neighbors_surface,
            surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
            use_sdf=cfg.model.use_sdf,
            **overrides,
        )

        datapipe.set_dataset(dataset)

        return datapipe


if __name__ == "__main__":
    fm_data = DoMINODataPipe(
        data_path="/code/processed_data/new_models_1/",
        phase="train",
        sampling=False,
    )
