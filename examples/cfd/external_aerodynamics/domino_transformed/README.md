# DoMINO-Transformed: External Aerodynamics Example

This example runs the **DoMINO-Transformed** model variant for external aerodynamics. It mirrors the [Domino](../domino) example in layout and workflow but uses:

- **Token-direct geometry encoding**: Geometry is encoded on the same tokens (surface/volume points) via cross ball queries (distance, direction, relative position) and Transolver (slice attention), instead of a grid-based geometry path.
- **Hierarchical fusion**: Basis and geometry encodings are combined in two stages—Stage 1: project, add, LayerNorm; Stage 2: FiLM from global embedding (γ, β)—instead of plain concatenation before the aggregation model.

Data format, variables, training/validation loop, loss, and evaluation are the same as the Domino example. Only the model and datapipe modules differ (`domino_transformed` model and `domino_transformed_datapipe`).

## Prerequisites

```bash
pip install -r requirements.txt
```

## Configuration

- **Config**: `src/conf/config.yaml` (and optionally `src/conf/cached.yaml` for cached data).
- **Model**: Under `model`, the important additions are:
  - `geometry_rep.use_token_direct: true` and token/Transolver options (`token_hidden_dim`, `transolver_depth`, etc.).
  - `fusion.use_hierarchical_fusion: true` with `fusion_dim`, `geometry_dim_volume`, `geometry_dim_surface`, `global_embed_dim`, etc.

Data paths, variables, bounding boxes, domain parallelism, train/val/eval sections follow the same structure as the Domino example.

## Data

Use the same preprocessing and dataset layout as Domino (e.g. [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator), DrivAerML). The DoMINO-Transformed datapipe (`create_domino_dataset` from `physicsnemo.datapipes.cae.domino_transformed_datapipe`) expects the same keys and directory structure; scaling factors and statistics are computed the same way (`compute_statistics.py`).

## Training

1. Set paths and options in `src/conf/config.yaml` (or `cached.yaml` if using cache).
2. Run from `src/`:
   ```bash
   torchrun --nproc_per_node=<num-gpus> train.py
   ```
3. Checkpoints and TensorBoard go to the configured `output` directory.

## Testing and inference

- **test.py**: Run on raw `.vtp`/`.vtu` (same as Domino). Set `eval.test_path`, `eval.save_path`, `eval.checkpoint_name`, `eval.scaling_param_path` in config.
- **inference_on_stl.py**: Run inference from STL only (volume/surface sampling and predictions). Use the same checkpoint and scaling params as in Domino.

## Scripts (aligned with Domino)

| Script | Purpose |
|--------|---------|
| `train.py` | Train DoMINO-Transformed (DDP/FSDP, mixed precision, validation, checkpointing). |
| `test.py` | Evaluate on test data (raw vtp/vtu). |
| `inference_on_stl.py` | Inference from STL geometry. |
| `compute_statistics.py` | Compute and save scaling factors. |
| `cache_data.py` | Build cached dataset for faster loading. |
| `validate_cache.py` | Validate cached dataset. |
| `benchmark_dataloader.py` | Dataloader benchmarking. |
| `shuffle_volumetric_curator_output.py` | Shuffle volumetric data (e.g. after Curator). |

For more detail on data prep, domain parallelism, and training tips, see the [Domino README](../domino/README.md).
