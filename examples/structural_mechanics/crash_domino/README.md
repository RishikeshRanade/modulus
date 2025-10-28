# DoMINO: Decomposable Multi-scale Iterative Neural Operator for Crash Simulation

DoMINO is a local, multi-scale, point-cloud based model architecture to model large-scale
physics problems such as structural mechanics crash simulations. The DoMINO model architecture takes STL
geometries as input and evaluates structural response quantities such as displacement fields
on the surface of structures over time. The DoMINO architecture is designed to be a fast, accurate
and scalable surrogate model for transient structural dynamics simulations.

DoMINO uses local geometric information to predict solutions on discrete points. First,
a global geometry encoding is learnt from point clouds using a multi-scale, iterative
approach. The geometry representation takes into account both short- and long-range
dependencies that are typically encountered in structural dynamics problems. Additional information
such as signed distance field (SDF), positional encoding, and temporal information are used to enrich the global encoding.
Next, discrete points are randomly sampled, a sub-region is constructed around each point
and the local geometry encoding is extracted in this region from the global encoding.
The local geometry information is learnt using dynamic point convolution kernels.
Finally, a computational stencil is constructed dynamically around each discrete point
by sampling random neighboring points within the same sub-region. The local-geometry
encoding and the computational stencil are aggregated to predict the solutions on the
discrete points.

A preprint describing additional details about the model architecture can be found here
[paper](https://arxiv.org/abs/2501.13350).

## Prerequisites

Install the required dependencies by running below:

```bash
pip install -r requirements.txt
```

## Getting started with the Crash Simulation example

### Configuration basics

DoMINO training and testing is managed through YAML configuration files
powered by Hydra. The base configuration file, `config.yaml` is located in `src/conf`
directory.

To select a specific configuration, use the `--config-name` option when running
the scripts.
You can modify configuration options in two ways:

1. **Direct Editing:** Modify the YAML files directly
2. **Command Line Override:** Use Hydra's `++` syntax to override settings at runtime

For example, to change the training epochs (controlled by `train.epochs`):

```bash
python train.py ++train.epochs=200  # Sets number of epochs to 200
```

This modular configuration system allows for flexible experimentation while
maintaining reproducibility.

#### Project logs

Save and track project logs, experiments, tensorboard files etc. by specifying a
project directory with `project.name`. Tag experiments with `expt`.

### Data

#### Dataset details

In this example, the DoMINO model is trained using crash simulation datasets for
structural mechanics applications. The dataset contains transient structural dynamics
simulations of crash scenarios, including geometries and time-series displacement fields.
Each simulation includes:
- STL geometry files representing the initial structure
- Time-series displacement fields on the surface mesh
- Global parameters such as applied stress
- Temporal information capturing the evolution of structural deformation

The data is processed to include multiple timesteps, capturing the transient behavior
of structures under impact or loading conditions.

#### Data Preprocessing

`PhysicsNeMo` has a related project to help with data processing, called
[PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator).
Using `PhysicsNeMo-Curator`, the data needed to train a DoMINO model can be setup easily.
Please refer to
[these instructions on getting started](https://github.com/NVIDIA/physicsnemo-curator?tab=readme-ov-file#what-is-physicsnemo-curator)
with `PhysicsNeMo-Curator`.

The first step for running the DoMINO pipeline requires processing the raw data
(VTP and STL) into either Zarr or NumPy format for training.
Each of the raw simulation files should be in `vtp` (for time-series surface data) and `stl` (for geometry) formats.
The data processing pipeline extracts displacement fields at multiple timesteps and prepares them for training.

Caching is implemented in the DoMINO datapipe.
Optionally, users can run `cache_data.py` to save outputs
of DoMINO datapipe in the `.npy` files. The DoMINO datapipe is set up to calculate
Signed Distance Field and Nearest Neighbor interpolations on-the-fly during
training. Caching will save these as a preprocessing step and can be used in
cases where the **STL surface meshes or VTP time-series data are very large**.
Data processing is parallelized and takes a couple of hours to write all the
processed files.

The final processed dataset should be divided and saved into 2 directories,
for training and validation.

#### Data Scaling factors

DoMINO has several data-specific configuration tools that rely on some
knowledge of the dataset:

- The output fields (the labels) are normalized during training to a mean
  of zero and a standard deviation of one, averaged over the dataset.
  The scaling is controlled by passing the `surface_factors` values to the datapipe.
- The input locations are scaled by, and optionally cropped to, user defined
  bounding boxes for the surface.  Whether cropping occurs, or not,
  is controlled by the `sample_in_bbox` value of the datapipe.  Normalization
  to the bounding box is enabled with `normalize_coordinates`.  By default,
  both are set to true.  The value of the boxes are configured in the
  `config.yaml` file.

> Note: The datapipe module has a helper function `create_domino_dataset`
> with sensible defaults to help create a Domino Datapipe.

To facilitate setting reasonable values of these, you can use the
`compute_statistics.py` script.  This will load the core dataset as defined
in your `config.yaml` file, loop over several events (200, by default), and
both print and store the surface field statistics as well as the
coordinate statistics.

#### Training

Specify the training and validation data paths, bounding box sizes etc. in the
`data` tab and the training configs such as epochs, batch size etc.
in the `train` tab.

#### Testing

The testing is directly carried out on raw files.
Specify the testing configs in the `test` tab.

### Training the DoMINO model

To train and test the DoMINO model on crash simulation datasets, follow these steps:

1. Specify the configuration settings in `conf/config.yaml`.

2. Run `train.py` to start the training. Modify data, train and model keys in config file.
  If using cached data then use `conf/cached.yaml` instead of `conf/config.yaml`.

3. Run `test.py` to test on `.vtp` / `.vtu`. Predictions are written to the same file.
  Modify eval key in config file to specify checkpoint, input and output directory.
  Important to note that the data used for testing is in the raw simulation format and
  should not be processed to `.npy`.

4. Download the validation results (saved in form of point clouds in `.vtp` / `.vtu` format),
   and visualize in Paraview.

**Training Guidelines:**

- Duration: Training time depends on dataset size and complexity
- Checkpointing: Automatically resumes from latest checkpoint if interrupted
- Multi-GPU Support: Compatible with `torchrun` or MPI for distributed training
- If the training crashes because of OOM, modify the points sampled on surface
  `model.surface_points_sample` and time points `model.time_points_sample`
  to manage memory requirements for your GPU
- The DoMINO model for crash simulation focuses on surface displacement fields
  over time. The model can be configured for transient simulations with
  `model.transient: true` and integration scheme with `model.transient_scheme`
  (either "explicit" or "implicit").
- MSE loss for the surface model gives the best results.
- Bounding box is configurable and will depend on the usecase.

### Training with Domain Parallelism

DoMINO has support for training and inference using domain parallelism in PhysicsNeMo,
via the `ShardTensor` mechanisms and pytorch's FSDP tools.  `ShardTensor`, built on
PyTorch's `DTensor` object, is a domain-parallel-aware tensor that can live on multiple
GPUs and perform operations in a numerically consistent way.  For more information
about the techniques of domain parallelism and `ShardTensor`, refer to PhysicsNeMo
tutorials such as [`ShardTensor`](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.distributed.shardtensor.html).

In DoMINO specifically, domain parallelism has been enabled in two ways, which
can be used concurrently or separately.  First, the input sampled surface points
can be sharded to accommodate higher resolution point sampling across multiple timesteps.
Second, the latent space of the model - typically a regularized grid - can be
sharded to reduce computational complexity of the latent processing.  When training
with sharded models in DoMINO, the primary objective is to enable higher
resolution inputs and larger latent spaces without sacrificing substantial compute time.

When configuring DoMINO for sharded training, adjust the following parameters
from `src/conf/config.yaml`:

```yaml
domain_parallelism:
  domain_size: 2
  shard_grid: True
  shard_points: True
```

The `domain_size` represents the number of GPUs used for each batch - setting
`domain_size: 1` is not advised since that is the standard training regime,
but with extra overhead.  `shard_grid` and `shard_points` will enable domain
parallelism over the latent space and input/output points, respectively.

As one last note regarding domain-parallel training: in the phase of the DoMINO
where the output solutions are calculated, the model can used two different
techniques (numerically identical) to calculate the output.  Due to the
overhead of potential communication at each operation, it's recommended to
use the `one-loop` mode with `model.solution_calculation_mode` when doing
sharded training.  This technique launches vectorized kernels with less
launch overhead at the cost of more memory use.  For non-sharded
training, the `two-loop` setting is more optimal. The difference in `one-loop`
or `two-loop` is purely computational, not algorithmic.

### Performance Optimizations

The training and inference scripts for DoMINO contain several performance
enhancements to accelerate the training and usage of the model. In this
section we'll highlight several of them, as well as how to customize them
if needed.

#### Memory Pool Optimizations

The preprocessor of DoMINO requires a computation of k Nearest Neighbors,
which is accelerated via the `cuml` Neighbors tool.  By default, `cuml` and
`torch` both use memory allocation pools to speed up allocating tensors, but
they do not use the same pool.  This means that during preprocessing, it's
possible for the kNN operation to spend a significant amount of time in
memory allocations - and further, it limits the available memory to `torch`.

To mitigate this, by default in DoMINO we use the Rapids Memory Manager
([`rmm`](https://github.com/rapidsai/rmm)).  If, for some reason, you wish
to disable this you can do so with an environment variable:

```bash
export PHYSICSNEMO_DISABLE_RMM=True
```

Or remove this line from the training script:

```python
from physicsnemo.utils.memory import unified_gpu_memory
```

> Note - why not make it configurable?  We have to set up the shared memory
> pool allocation very early in the program, before the config has even
> been read.  So, we enable by default and the opt-out path is via the
> environment.

#### Transient Data Handling

The dataset size for transient crash simulation data can be substantial due to
multiple timesteps. Each simulation includes time-series displacement data across
all surface points. 

DoMINO's data pipeline handles transient data efficiently by:
- Sampling random time points during training via `model.time_points_sample`
- Sampling surface points at each timestep via `model.surface_points_sample`
- Supporting both explicit and implicit time integration schemes

For large time-series datasets, preprocessing with `PhysicsNeMo-Curator` can help
organize the data efficiently. The data reader supports both Zarr and NumPy formats.

#### Overall Performance

DoMINO is a computationally complex and challenging workload.  Over the course
of several releases, we have chipped away at performance bottlenecks to speed
up the training and inference time. We hope these optimizations enable you to explore more
parameters and surrogate models; if there is a performance issue you see,
please open an issue on GitHub.

### Example Training Results

To provide an example of what a successful training should look like, monitor
the following during training:

- **Training Loss**: MSE loss on displacement predictions should decrease over epochs
- **Validation Loss**: Should track training loss without significant divergence
- **L2 Metrics**: Relative L2 error on displacement fields (X, Y, Z components)
- **Displacement Magnitude**: Error in total displacement magnitude across timesteps

The test script will output detailed metrics including:
- L2 norm for each displacement component
- Mean squared error for displacement fields
- Maximum displacement error
- Time-series displacement accuracy across all timesteps

Results can be visualized in Paraview using the generated VTP files with time-series data.


### DoMINO model pipeline for inference on test samples

After training is completed, `test.py` script can be used to run inference on
test samples. Follow the below steps to run the `test.py`

1. Update the config in the `conf/config.yaml` under the `Testing data Configs`
   tab.

2. The test script is designed to run inference on the raw `.stl` and `.vtp`
   files for each test sample. Use the same scaling parameters that
   were generated during the training. Typically this is `outputs/<project.name>/`,
   where `project.name` is as defined in the `config.yaml`. Update the
   `eval.scaling_param_path` accordingly.

3. Run the `test.py`. The test script can be run in parallel as well. Refer to
   the training guidelines for Multi-GPU. Note, for running `test.py` in parallel,
   the number of GPUs chosen must be <= the number of test samples.
   
4. The output will include time-series VTP files showing predicted displacement fields
   at each timestep, which can be loaded in Paraview for visualization.

## Extending DoMINO to a custom dataset

This repository includes examples of **DoMINO** training on crash simulation datasets.
However, many use cases require training **DoMINO** on a **custom dataset**.
The steps below outline the process.

1. Reorganize your dataset to have a consistent directory structure. The
   raw data directory should contain a separate directory for each simulation.
   Each simulation directory needs to contain mainly 2 files: `stl` and `vtp`,
   corresponding to the geometry and time-series surface field information.
   Additional details such as loading conditions, for example applied stress or impact velocity,
   may be added in a separate metadata file, in case these vary from one case to the next.
2. Modify the following parameters in `conf/config.yaml`
   - `project.name`: Specify a name for your project.
   - `exp_tag`: This is the experiment tag.
   - `data_processor.input_dir`: Input directory where the raw simulation dataset is stored.
   - `data_processor.output_dir`: Output directory to save the processed dataset (`.npy`).
   - `data_processor.num_processors`: Number of parallel processors for data processing.
   - `variables.surface`: Variable names of surface fields and fields type (vector or scalar).
     For crash simulations, typically this is `Displacement: vector`.
   - `variables.global_parameters`: Global parameters like stress, material properties, etc.
   - `data.input_dir`: Processed files used for training.
   - `data.input_dir_val`: Processed files used for validation.
   - `data.bounding_box_surface`: Dimensions of bounding box enclosing the biggest geometry
     in dataset. Surface fields are modeled inside this bounding box.
   - `train.epochs`: Set the number of training epochs.
   - `model.surface_points_sample`: Number of points to sample on the surface mesh per epoch
     per batch. Tune based on GPU memory.
   - `model.time_points_sample`: Number of time steps to sample per epoch per batch.
   - `model.geom_points_sample`: Number of points to sample on STL mesh per epoch per batch.
     Ensure point sampled is less than number of points on STL (for coarser STLs).
   - `model.transient`: Set to `true` for transient crash simulations.
   - `model.transient_scheme`: Choose `explicit` or `implicit` time integration.
   - `model.integration_steps`: Number of time steps in the simulation.
   - `eval.test_path`: Path of directory of raw simulation files for testing and verification.
   - `eval.save_path`: Path of directory where the AI predicted simulation files are saved.
   - `eval.checkpoint_name`: Checkpoint name `outputs/{project.name}/models` to evaluate model.
   - `eval.scaling_param_path`: Scaling parameters populated in `outputs/{project.name}`.
3. Before running `process_data.py` to process the data, modify it to match your
   dataset structure. Key modifications include:
   - Non-dimensionalization schemes based on the order of your variables
   - Path definitions for STL geometry files and VTP time-series data
   - Extraction of displacement fields at each timestep
   - Handling of global parameters (stress, loading conditions, etc.)
   
   For example, you may need to define custom path functions:

    ```python
    class CrashSimPaths:
        # Specify the name of the STL in your dataset
        @staticmethod
        def geometry_path(sim_dir: Path) -> Path:
            return sim_dir / "geometry.stl"

        # Specify the name of the VTP with time-series data
        @staticmethod
        def surface_path(sim_dir: Path) -> Path:
            return sim_dir / "displacement_timeseries.vtp"
    ```

4. Before running `train.py`, modify the loss functions in `loss.py` if needed.
   The default configuration uses MSE loss with optional area weighting.
   For crash simulations with displacement fields, the current loss formulation
   works well, but you may want to customize it based on your specific requirements
   (e.g., emphasizing certain displacement components or adding physics-based constraints).

5. Run `test.py` to validate the trained model on test simulations.

The DoMINO model architecture for crash simulations demonstrates the versatility of
the framework for handling transient structural dynamics problems with complex geometries
and time-varying displacement fields.

## References

1. [DoMINO: A Decomposable Multi-scale Iterative Neural Operator for Modeling Large Scale Engineering Simulations](https://arxiv.org/abs/2501.13350)
