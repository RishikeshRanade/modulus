# Plan: Hierarchical Fusion for Basis + Geometry + Global

## Overview

Implement **Hierarchical Fusion** to combine three streams—basis `(B, N, f_b)`, geometry encoding `(B, N, f_g)`, and global embedding `(B, N, f_p)`—into a single representation `(B, N, d)` for the aggregation model, instead of plain concatenation.

- **Stage 1**: Combine basis + geometry (e.g. project to common dim `d`, add, LayerNorm).
- **Stage 2**: FiLM with global: `γ, β = MLP(global)`, output = `γ * stage1 + β`.

---

## 1. New module: `HierarchicalFusion`

**Location**: New file [modulus/physicsnemo/models/domino_transformed/fusion.py](modulus/physicsnemo/models/domino_transformed/fusion.py).

**Interface**:

- **Inputs**:
  - `basis`: `(B, N, f_b)` — basis function output (e.g. from `nn_basis(positions)`).
  - `geometry`: `(B, N, f_g)` — geometry encoding from GeometryRep (BQ + Transolver).
  - `global_embed`: `(B, N, f_p)` — global parameter embedding (broadcast per token).
- **Output**: `(B, N, d)` where `d` is the fusion dimension (configurable).

**Stage 1 (basis + geometry)**:

- Project both to dimension `d`:
  - `proj_basis = Linear(basis, d)`  (or small MLP if you want).
  - `proj_geometry = Linear(geometry, d)` or `MLP(geometry, d)`.
- Combine: `combined = LayerNorm(proj_basis + proj_geometry)`.
- Optional: use a small MLP on `geometry` for richer mapping: `proj_geometry = MLP(geometry, hidden_dims=[d], out=d)`.

**Stage 2 (FiLM with global)**:

- `gamma_beta = MLP(global_embed)` → output shape `(B, N, 2*d)`.
- Split: `gamma = gamma_beta[..., :d]`, `beta = gamma_beta[..., d:]`.
- Optional: clamp or activate (e.g. `gamma = 1 + softplus(gamma)` so scale is ≥ 1).
- `out = gamma * combined + beta`.

**Constructor**:

- `basis_dim: int`, `geometry_dim: int`, `global_dim: int`, `fusion_dim: int`.
- Optional: `stage1_hidden: int` for geometry MLP, `film_hidden: int` for FiLM MLP.

---

## 2. Config

**Location**: [modulus/physicsnemo/models/domino_transformed/config.py](modulus/physicsnemo/models/domino_transformed/config.py).

Add under a top-level or nested key (e.g. `fusion` or under `model`):

- `fusion_dim` (int): output dimension `d` of the fusion (and thus aggregation model input).
- Optional: `fusion_stage1_hidden`, `fusion_film_hidden` for MLP widths.

Aggregation model input size becomes `fusion_dim` (no longer `position_encoder_base_neurons + base_layer_nn + base_layer_geo_vol + base_layer_p`). So either:

- Set `fusion_dim` to a chosen value (e.g. 256 or 512), and aggregation model is built with `input_features=fusion_dim`, or
- Derive `fusion_dim` from existing dims (e.g. `max(f_b, f_g, f_p)` or `f_b + f_g // 2`) and document it.

---

## 3. Where fusion is called

Fusion must be called **inside the solution calculator** (volume and surface), because:

- **Basis** is computed per spatial location: for each volume mesh center we have one basis value, but for the neighbor aggregation we also compute `basis_f` at **neighbor positions** (perturbed centers). So we have multiple basis values per mesh point (center + neighbors), while geometry and global are one per mesh point.
- So for each `(center or neighbor)` we compute `basis_f = nn_basis(volume_m_c)` and then `fused = HierarchicalFusion(basis_f, encoding_g, global_embedding)`. Here `encoding_g` and `global_embedding` are `(B, N, ...)` and are broadcast/indexed to match the batch of points (same geometry and global for center and all neighbors of that point).

**Flow in solution calculator (volume)**:

1. Precompute or receive: `encoding_g` `(B, N, f_g)`, `global_embedding` `(B, N, f_p)`.
2. In the loop over `volume_m_c_perturbed` (center + neighbors):
   - `basis_f = self.nn_basis[f](volume_m_c)`  → `(B, N, f_b)` or `(B, N, 1, f_b)` depending on stacking.
3. Call `fused = self.fusion(basis_f, encoding_g, global_embedding)` → `(B, N, d)` (or expanded for neighbors).
4. `output = self.aggregation_model[f](fused)` (and optional inverse-distance weighting as now).

So the solution calculator needs to:

- Hold a `HierarchicalFusion` module (built with `f_b`, `f_g`, `f_p`, `d`).
- Accept `encoding_g` and `global_embedding` in `forward` (instead of or in addition to `encoding_node` if we drop it).
- Use `fusion(basis_f, encoding_g, global_embedding)` instead of `concat(basis_f, encoding_node, encoding_g)`.

---

## 4. Model.py changes

**Volume branch**:

- Compute `encoding_g_vol = self.geo_rep_volume(geo_centers, volume_mesh_centers)` → `(B, N_vol, f_g)` (after GeometryRep is updated to token-direct).
- Compute `global_embedding_vol`: e.g. from `global_params_values` / `global_params_reference` and optional `parameter_model`, broadcast to per-token → `(B, N_vol, f_p)`. If `encode_parameters` is True, use existing `parameter_model(apply_parameter_encoding(...))` and expand to `(B, N_vol, f_p)`; else use a simple embedding of global params and expand.
- Do **not** call `volume_local_geo_encodings` (no grid).
- Call `solution_calculator_vol(volume_mesh_centers, encoding_g_vol, global_embedding_vol, global_params_values, global_params_reference)` (and drop `encoding_node_vol` if we no longer use it, or pass it and fold into basis inside the calculator).

**Surface branch**: Same idea: `encoding_g_surf`, `global_embedding_surf`, then `solution_calculator_surf(..., encoding_g_surf, global_embedding_surf, ...)`.

**Aggregation model construction**:

- Replace `input_features=position_encoder_base_neurons + base_layer_nn + base_layer_geo_vol + base_layer_p` with `input_features=model_parameters.fusion.fusion_dim` (or config-derived `d`). Same for surface.

---

## 5. Solution calculator changes

**Volume** ([solutions.py](modulus/physicsnemo/models/domino_transformed/solutions.py) `SolutionCalculatorVolume`):

- **Constructor**: Add `fusion_module: HierarchicalFusion` (optional for backward compat; if `None`, use legacy concat path). Add `encoding_node` as optional so we can keep or drop it.
- **Forward**: Add arguments `encoding_g`, `global_embedding` (and optionally remove or keep `encoding_node`). Inside the loop over perturbed centers:
  - Compute `basis_f = self.nn_basis[f](volume_m_c)`.
  - If fusion is used: `fused = self.fusion_module(basis_f, encoding_g, global_embedding)`; then `output = self.aggregation_model[f](fused)`.
  - If legacy: keep current `output = torch.cat((basis_f, encoding_node, encoding_g), dim=-1)` and `aggregation_model(output)`.
- Ensure shapes: `encoding_g` and `global_embedding` are `(B, N, f_g)` and `(B, N, f_p)`; when `volume_m_c` is stacked as `(B, N, 1, 3)` for a single neighbor type, `basis_f` may be `(B, N, 1, f_b)` — then either expand `encoding_g`/`global_embedding` to `(B, N, 1, f_g/f_p)` for the fusion call or flatten the batch of points so that fusion sees `(B*N*num_samples, f_b)` and `(B*N*num_samples, f_g)` etc., then reshape back. Document the chosen convention.

**Surface** (`SolutionCalculatorSurface`): Mirror the same changes (fusion module, forward args, loop over center/neighbors if any).

---

## 6. Handling `encoding_node`

Current code uses `encoding_node` (positional encoding) concatenated with basis and geometry. Options:

- **A) Fold into basis**: Before fusion, define `basis = concat(basis_f, encoding_node)` so the “basis” stream has dim `f_b + position_encoder_base_neurons`. Fusion then takes (basis, geometry, global). No change to fusion logic; only the first stream is larger.
- **B) Drop encoding_node**: Rely on basis (FourierMLP of position) and geometry + global only. Simpler; try this first and add encoding_node back into basis if needed.
- **C) Fourth stream**: Not recommended; keeps the design more complex.

Recommendation: **B** for the first version; if needed, **A** later.

---

## 7. Implementation order

1. **Add `fusion.py`** with `HierarchicalFusion` (Stage 1: project basis + geometry to `d`, add, LayerNorm; Stage 2: FiLM from global). Unit test with random tensors.
2. **Config**: Add `fusion_dim` (and optional MLP hidden dims) to default config; ensure aggregation model is built with `input_features=fusion_dim`.
3. **Solution calculator (volume)**: Add `fusion_module` to constructor; in forward, accept `encoding_g` and `global_embedding`; in the aggregation loop call `fusion(basis_f, encoding_g, global_embedding)` and pass result to aggregation model. Handle shapes for center vs neighbors (broadcast or expand as above).
4. **Solution calculator (surface)**: Same pattern.
5. **Model.py**: Compute `global_embedding` from global params (and optional parameter_model), pass `encoding_g_vol`/`encoding_g_surf` and `global_embedding` into solution calculators; remove local_geo_encodings call for the new path; fix aggregation model `input_features` to use `fusion_dim`.

---

## 8. Summary

| Item | Action |
|------|--------|
| **New file** | `domino_transformed/fusion.py` with `HierarchicalFusion(basis_dim, geometry_dim, global_dim, fusion_dim)` |
| **Stage 1** | `combined = LayerNorm(Linear(basis, d) + MLP(geometry, d))` |
| **Stage 2** | `gamma, beta = MLP(global).chunk(2); out = gamma * combined + beta` |
| **Config** | `fusion_dim`, optional `fusion_stage1_hidden`, `fusion_film_hidden` |
| **Solution calculator** | Owns `fusion_module`; forward receives `encoding_g`, `global_embedding`; loop: `fused = fusion(basis_f, encoding_g, global_embedding)` → `agg(fused)` |
| **Model.py** | Build fusion and pass into solution calculators; compute and pass `global_embedding`; aggregation `input_features=fusion_dim` |
| **encoding_node** | Omit in first version; optionally fold into basis stream later |

This yields a single, reproducible place (HierarchicalFusion) where the three streams are combined with a good accuracy/generalization tradeoff, and keeps the aggregation model’s role to “combine fused features and neighborhood interactions → solution.”
