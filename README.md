# Biohub - Cell Tracking During Development

This repository contains data pipelines, notebooks, and learning documentation for the Kaggle "Biohub - Cell Tracking During Development" competition. The project explores both heuristic volumetric signal processing pipelines and deep learning solutions for tracking cells across 3D time-lapse microscopy datasets.

## Docker Environment

The runtime environment and dependencies are containerized and available at:

```
docker.io/traspi/profile:latest
```

To run a container with this environment:

```bash
docker pull docker.io/traspi/profile:latest
docker run -it --gpus all -v $(pwd):/workspace docker.io/traspi/profile:latest
```

## Repository Structure

- `nbf_mod.ipynb`: Modular end-to-end pipeline implementing multi-scale Difference-of-Gaussians (DoG), sub-voxel centroid refinement, greedy KD-tree tracking, gap closing, division detection, and graph-based submission generation.
- `cell-tracking-refined.ipynb`: Baseline heuristic tracking workflow featuring volumetric filtering and edge linking.
- `train-3d-unet.ipynb`: Training pipeline for a 3D U-Net using MONAI to perform Gaussian heatmap regression on cell centers.
- `inference-3d-unet.ipynb`: Volumetric sliding window inference pipeline using trained 3D U-Net weights to detect and link cells.
- `README.md`: Project documentation, methodology overview, and key technical learnings.

## Methodology

### 1. Data Ingestion and Decompression
- Input datasets are stored in multi-dimensional Zarr format with Blosc2 compression.
- Timepoint volumes are decompressed on-the-fly to manage memory usage efficiently across time series.
- Voxel dimensions are anisotropic (for example, Z resolution is often 4x coarser than XY resolution).

### 2. Detection Pipeline
- Multi-Scale Difference-of-Gaussians (DoG): Filters volumetric intensity responses across multiple spatial scales with anisotropic sigmas (sigma_z = sigma_xy / aniso_ratio).
- Non-Maximum Suppression (NMS): Identifies local peak responses within a 3D neighborhood footprint.
- Sub-Voxel Centroid Refinement: Refines integer peak coordinates using localized intensity-weighted center-of-mass averaging.

### 3. Tracking and Association
- Greedy KD-Tree Association: Computes Euclidean distance associations between consecutive frames within physical coordinate space (microns).
- Gap Closing: Maintains a pool of temporarily lost tracks and attempts reconnection across a configurable temporal window (e.g. up to 3 skipped frames).
- Division Detection: Identifies cell division events when an unmatched parent track bifurcates into two nearby daughter cells satisfying distance and midpoint constraints.
- Track Pruning: Filters disconnected short-lived tracks below a minimum threshold length to eliminate spurious detections.

### 4. Deep Learning Extension (3D U-Net)
- Predicts Gaussian heatmaps centered on ground truth coordinates using MONAI and PyTorch.
- Employs 3D patch-based sliding window inference to handle large volumetric arrays without exceeding GPU VRAM.

## Key Learnings and Technical Insights

1. Handling Spatial Anisotropy:
   Microscopy imaging often exhibits significant axial (Z-axis) stretching relative to lateral (XY) resolution. Scaling sigmas and spatial search radii by physical dimensions (microns) rather than raw voxel indices is essential for accurate peak detection and distance linking.

2. Sub-Voxel Refinement vs. Grid Peaks:
   Applying local intensity-weighted centroid calculations after local maxima detection smooths track trajectories and reduces frame-to-frame jitter.

3. Temporal Gap Closing:
   Cell intensity can fluctuate across frames due to focal drift or photobleaching. Retaining lost tracks in memory for 1 to 3 frames recovers broken tracks and improves edge-to-node connectivity ratios.

4. Heuristic vs. Deep Learning Trade-offs:
   - Classical DoG pipelines execute fast, require no labeled training iterations, and run efficiently on CPU.
   - Deep learning heatmap regression (3D U-Net) achieves superior separation in dense cell clusters where DoG peaks blend together, but requires careful patch sampling and GPU inference budgeting.

5. Competition Constraint Management:
   Processing hundreds of 3D volumes requires strict streaming and memory cleanup: deleting previous timepoints from memory, batching spatial queries with KD-trees, and writing output records progressively.
