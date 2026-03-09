# Pulsar: PDE Unstructured Latent Simulator with Anchored Attention based Representations

## Overview

Pulsar is an experimental model designed for PDEs on unstructured surface and volume point clouds. The model introduces several novel architectural components that enable effective learning on complex 3D geometries without requiring structured meshes.

## Key Unique Model Components

### 1. Hierarchical Anchor-Based Attention

Pulsar employs a hierarchical anchor-based attention mechanism that provides controllable locality bias for regions with sharp geometric and flow gradients. This is particularly critical for external aerodynamics where flow features can vary dramatically across the geometry:

- Anchors are strategically placed to capture multi-scale flow phenomena
- Hierarchical structure enables coarse-to-fine attention patterns
- Provides better gradient handling around complex geometric features

### 2. Learned Anchor-Based Attention

The model features a learned anchor-based attention system that adaptively determines the most relevant spatial relationships:

- Anchors are learned during training rather than fixed
- Enables the model to discover optimal attention patterns for specific flow regimes
- Provides flexibility to focus on geometry-specific flow characteristics

### 3. Multiscale Cross and Self Ball-Query

Pulsar incorporates multiscale ball-query operations for both self-attention and cross-attention:

- **Self ball-query**: Captures local neighborhood interactions at multiple scales
- **Cross ball-query**: Enables information exchange between surface and volume representations
- **Multiscale approach**: Handles features from fine geometric details to global flow patterns
- Operates efficiently on unstructured point clouds without grid constraints

### 4. Learned Multi-Grid Token Pooling for Geometry and BCs

The model uses sophisticated pooling mechanisms for both geometry and boundary conditions:

- **Multi-grid pooling**: Hierarchically pools geometry information across multiple resolution levels
- **Learned pooling**: Adaptive pooling weights learned during training
- **Separate BC pooling**: Independent encoding of boundary conditions into context tokens
- Captures coarse-to-fine geometric structure without requiring explicit grid discretization

### 5. Combined Volume + Surface Training with Cross Attention

Pulsar uniquely supports unified training on both surface and volume data:

- **Unified architecture**: Single model handles surface-only, volume-only, or combined regimes
- **Cross attention**: Enables information flow between surface and volume representations
- **Shared context encoder**: Geometry and BC encoding shared across surface and volume modes
- **Flexible training**: Can be configured for different data availability scenarios

## Architecture Benefits

The combination of these unique components provides several key advantages:

- **Unstructured geometry handling**: Direct operation on STL-derived point clouds without meshing requirements
- **Multi-scale flow capture**: Hierarchical attention and ball-query operations capture phenomena from local boundary layers to global circulation patterns
- **Adaptive locality**: Learned anchors and attention patterns adapt to specific geometric and flow characteristics
- **Flexible data regimes**: Unified architecture supports various training scenarios (surface-only, volume-only, or combined)
- **Differentiable geometry pipeline**: End-to-end differentiability enables geometry optimization and inverse design applications


## Notes

- Pulsar is experimental and under active development.
- See the example configs under
  `examples/cfd/external_aerodynamics/pulsar` for training and inference
  recipes.

