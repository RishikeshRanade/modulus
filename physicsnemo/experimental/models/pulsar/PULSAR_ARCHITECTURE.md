# Pulsar Model Architecture and Information Flow

Pulsar (PDE Unstructured Latent Simulator with Anchored Attention based Representations) is a model for unstructured surface and volume point clouds in CFD. It encodes geometry and boundary-condition (BC) information into context, then runs a physics stack over surface/volume tokens conditioned on that context to produce volume and surface field predictions.

---

## 1. High-Level Information Flow

```
geometry_points (B, N_g, 3)     surface_points (B, N_s, 3)    volume_points (B, N_v, 3)    global_params (B, N_bc, D_bc)
         |                              |                            |                              |
         v                              v                            v                              v
  GeometryContextEncoder          FourierMLP (surface)          FourierMLP (volume)      GlobalParamsContextEncoder
         |                              |                            |                              |
    geo_tokens                    surf_tokens                  vol_tokens                    bc_tokens
         |                              |                            |                              |
         +------------------------------+----------------------------+                              |
         |                    BQFeatureInjector (optional)           |                              |
         v                              v                            v                              v
    geo_tokens (enriched)         surf_tokens (enriched)       vol_tokens (enriched)              |
         |                              |                            |                              |
         v                              |                            |                              v
  GeometryContextEncoder.pool     <-----+----------------------------+-------------->  GlobalParamsContextEncoder
  (pooled_geo)                                                          (pooled_bc)
         |                              |                            |                              |
         +------------------------------+----------------------------+------------------------------+
         v
  context_tokens = cat(pooled_geo, pooled_bc)   [optional: anchor_context_tokens]
         |
         v
  ContextSliceBuilder(context_tokens [, anchor_tokens])  -->  context_slices (B, H, S, D)
         |
         v
  GlobalConditioner(surf_tokens, context_tokens)  and  GlobalConditioner(vol_tokens, context_tokens)
         |
         v
  PulsarPhysicsStack(surface_points, volume_points, surface_tokens, volume_tokens, context_slices)
         |
         v
  surface_head(surf_tokens)  -->  surf_out (B, N_s, D_surf)
  volume_head(vol_tokens)     -->  vol_out (B, N_v, D_vol)
  [optional] force_head(mean(context_tokens)) --> forces (B, D_force)
```

---

## 2. Component Map

### 2.1 Inputs (PulsarModel.forward)

| Input | Shape | Description |
|-------|--------|-------------|
| `geometry_points` | (B, N_g, 3) | STL-derived geometry point cloud (e.g. surface mesh). |
| `surface_points` | (B, N_s, 3) | Query points on the surface where surface fields are predicted (required if ``enable_surface=True``). |
| `volume_points` | (B, N_v, 3) | Query points in the volume where volume fields are predicted (required if ``enable_volume=True``). |
| `global_params_values` / `bc_values` | (B, N_bc, D_bc) | Boundary-condition / global parameter conditioning (e.g. AoA, Mach). ``bc_values`` is an alias for backward compatibility. |

---

### 2.2 Context Encoders (context.py)

#### GeometryContextEncoder

- **Role:** Turn geometry point cloud into a fixed set of geometry context tokens.
- **Steps:**
  1. **Encode:** `FourierMLP(geometry_points)` → per-point tokens (B, N_g, C). Optional Fourier features on 3D coords.
  2. **Pool:** Either `LearnedTokenPooler` (learnable queries, attention over tokens) or `MultiGridTokenPooler` (voxelize by coords, then pool per level).
- **Output:** `geo_tokens` (B, N_g, C) from `encode()`; pooled geometry context (B, geo_context_tokens, C) from `pool()`.

#### GlobalParamsContextEncoder

- **Role:** Turn BC/global parameter vectors into a fixed set of BC context tokens.
- **Steps:**
  1. **Encode:** `Mlp(global_params_values)` → (B, N_bc, C).
  2. **Pool:** `LearnedTokenPooler` → (B, bc_context_tokens, C).
- **Output:** Pooled BC context tokens (B, bc_context_tokens, C).

#### ContextSliceBuilder

- **Role:** Convert combined context tokens (and optionally anchor tokens) into **context slices** used by GALE/AnchorSlice blocks.
- **Steps:**
  1. `ContextProjector(context_tokens)` → (B, H, S, D) (multi-head projection to “physical slices”).
  2. If `use_anchor_context`: fuse with anchor-derived slices and average with anchor slices.
- **Output:** `context_slices` (B, H, S, D). These are the global conditioning tensors for the physics stack.

---

### 2.3 Point Cloud Encoders (model.py)

- **Surface encoder:** `FourierMLP(3 → hidden_dim)` on `surface_points` → `surf_tokens` (B, N_s, C). Optional Fourier features.
- **Volume encoder:** `FourierMLP(3 → hidden_dim)` on `volume_points` → `vol_tokens` (B, N_v, C). Optional Fourier features.
- Geometry per-point tokens come from `GeometryContextEncoder.encode(geometry_points)` → `geo_tokens` (B, N_g, C).

---

### 2.4 BQFeatureInjector (bq_injector.py, bq_features.py)

- **Role:** Add multi-scale, neighborhood-aware (ball-query) features to geometry, surface, and volume tokens.
- **Self BQ:** For each domain (geometry, surface, volume), aggregate features from neighbors within radii (e.g. [0.05, 0.25]) and project to hidden dim; add to existing tokens.
- **Cross BQ:** Geometry↔surface and geometry↔volume: query from one point set, keys from the other within radii; project and add to the query-domain tokens.
- **Implementation:** Uses `BQFeatureStack` (multiple `GeometricFeatureProcessor` per radius, then linear projection). Injector adds these features in-place to `geo_tokens`, `surf_tokens`, `vol_tokens`.
- **Output:** Same shapes (B, N_g, C), (B, N_s, C), (B, N_v, C) with enriched features.

---

### 2.5 Pooling and Context Assembly (model.py)

- **Pooled geometry:** `GeometryContextEncoder.pool(geometry_points, geo_tokens)` → (B, geo_context_tokens, C).
- **Pooled BC:** `GlobalParamsContextEncoder(global_params_values)` → (B, bc_context_tokens, C).
- **Combined context:** `context_tokens = cat(pooled_geo, pooled_bc)` along token dimension.
- **Context slices:** `ContextSliceBuilder(context_tokens [, anchor_tokens])` → (B, H, S, D).

Optional: if `use_anchor_context`, anchor tokens are built from concatenated surf/vol tokens and used in `ContextSliceBuilder` to form additional/fused slices.

---

### 2.6 GlobalConditioner (layers.py)

- **Role:** Condition surface/volume tokens on the global context before the physics stack.
- **Mechanism:** Mean-pool `context_tokens` → (B, C); pass through a small MLP (LayerNorm → Linear → SiLU → Linear) to get a shift; add shift to each point token and apply dropout.
- **Output:** Conditioned `surf_tokens` and `vol_tokens` (same shapes).

---

### 2.7 PulsarPhysicsStack (physics.py)

- **Role:** Update surface and volume tokens using shared `context_slices` (geometry + BC conditioning). The stack uses **anchor-slice attention only**: separate stacks of `AnchorSliceBlock` for surface and for volume (no shared GALE over joint surf/vol).

- **Blocks:** For each enabled branch (surface and/or volume), a stack of `AnchorSliceBlock` layers. Each block takes point coordinates, point tokens, and `context_slices` (B, H, S, D). Surface and volume streams do not cross-attend; they share the same context slices.

- **AnchorSliceBlock (attention.py):** For each grid level:
  1. Build **anchor** positions in the bbox of the point cloud (min/max of point coordinates). Anchors are either a regular voxel grid or learnable positions in [-1, 1]³ mapped into the bbox.
  2. `anchor_mlp(anchors)` → anchor tokens.
  3. **Point→anchor:** `MultiheadMaskedCrossAttention` (queries=anchors, keys/values=point tokens), with radius mask so only nearby points attend.
  4. **Anchor self-attention:** Standard MHA on anchor tokens.
  5. **GALE on anchors:** Apply one or more GALE blocks to anchor tokens with `context_slices`.
  6. **Anchor→point:** Masked cross-attention from points to anchors (radius-masked); residual add to point tokens.

- **Output:** Updated surface tokens (B, N_s, C) and/or volume tokens (B, N_v, C) depending on ``enable_surface`` / ``enable_volume``.

---

### 2.8 Output Heads (model.py)

- **Surface head:** `Mlp(surf_tokens)` → (B, N_s, output_dim_surface) (e.g. pressure, skin friction components).
- **Volume head:** `Mlp(vol_tokens)` → (B, N_v, output_dim_volume) (e.g. velocity, pressure in volume).
- **Force head (optional):** `mean(context_tokens, dim=1)` → (B, C); then `Mlp` → (B, force_dim). Used for global aerodynamic forces.

---

## 3. File → Component Summary

| File | Main components |
|------|------------------|
| `model.py` | `PulsarModel`, `PulsarMetaData`; wires encoders, BQ injector, context, conditioner, physics stack, heads. |
| `context.py` | `GeometryContextEncoder`, `GlobalParamsContextEncoder`, `ContextSliceBuilder`. |
| `layers.py` | `LearnedTokenPooler`, `MultiGridTokenPooler`, `GlobalConditioner`. |
| `bq_injector.py` | `BQFeatureInjector` (orchestrates self/cross BQ and token addition). |
| `bq_features.py` | `BQFeatureStack` (multi-scale ball-query + projection). |
| `physics.py` | `PulsarPhysicsStack` (GALE stack or AnchorSlice stacks). |
| `attention.py` | `MultiheadMaskedCrossAttention`, `AnchorSliceBlock`. |

External dependencies:

- `physicsnemo.experimental.models.geotransolver`: `GALE_block`, `ContextProjector`, `GeometricFeatureProcessor`.
- `physicsnemo.nn`: `FourierMLP`, `Mlp`, `get_activation`.

---

## 4. Data Shapes Summary

| Symbol | Shape | Meaning |
|--------|--------|---------|
| B | batch | Batch size. |
| N_g | num_geo | Number of geometry points. |
| N_s | num_surf | Number of surface query points. |
| N_v | num_vol | Number of volume query points. |
| N_bc | num_bc | Number of BC/global parameter vectors. |
| C | hidden_dim | Token dimension (e.g. 256). |
| H | heads | Number of attention heads. |
| S | slices | Number of context slices. |
| D | dim | Slice dimension (typically = C). |
| geo_context_tokens | - | Number of pooled geometry tokens. |
| bc_context_tokens | - | Number of pooled BC tokens. |

---

## 5. Optional Features (Flags)

- **use_bq_features:** Enable BQFeatureInjector (self + cross ball-query).
- **use_multigrid_pooler:** Use MultiGridTokenPooler for geometry instead of LearnedTokenPooler.
- **use_anchor_context:** Feed anchor-derived tokens into ContextSliceBuilder for fused context slices.
- **enable_volume / enable_surface:** Toggle volume/surface branches (at least one must be True).
- **enable_force_head:** Add force prediction from pooled context; use `return_forces=True` in `forward()` to return forces.

The physics stack is always anchor-slice (separate surface and volume `AnchorSliceBlock` stacks); there is no GALE-over-joint-tokens mode.

---

This document describes the **experimental** Pulsar implementation under `physicsnemo/experimental/models/pulsar/`. `PulsarModel` is used in `examples/cfd/external_aerodynamics/transformer_models/`. The `examples/cfd/external_aerodynamics/pulsar/` example may use a different (legacy) “PULSAR” class with UnstructuredConvNet-based encoder/decoders.
