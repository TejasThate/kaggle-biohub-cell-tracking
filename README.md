# Biohub — Cell Tracking During Development

A CPU-only pipeline for the CZ Biohub Cell Tracking Kaggle competition that tracks cells through 3D microscopy time-lapses of zebrafish embryos and reconstructs lineage trees including divisions. Uses multi-scale Difference-of-Gaussians for detection, KD-Tree-pruned Hungarian assignment for optimal linking, multi-frame gap closing, scored division detection, and global graph optimization with velocity filtering. Runs in ~15 min on Kaggle CPU with only numpy, scipy, pandas, and blosc2.

## Docker Environment

```bash
docker pull docker.io/traspi/profile:latest
docker run -it --gpus all -v $(pwd):/workspace docker.io/traspi/profile:latest
```

## Repository Structure

| File | Description |
|------|-------------|
| `final_submission.py` | Complete, self-contained pipeline for Kaggle submission (multi-scale DoG, KD-Tree Hungarian linking, gap closing, division detection, graph optimization) |
| `nbf_mod.ipynb` | Modular end-to-end pipeline notebook: multi-scale DoG, sub-voxel centroid refinement, greedy KD-tree tracking, gap closing, division detection, and graph-based submission generation |
| `cell-tracking-refined.ipynb` | Baseline heuristic tracking workflow with volumetric filtering and edge linking |
| `train-3d-unet.ipynb` | 3D U-Net training pipeline using MONAI for Gaussian heatmap regression on cell centers |
| `inference-3d-unet.ipynb` | Volumetric sliding window inference using trained 3D U-Net weights |

## Pipeline

1. **Detection** — Multi-scale DoG with anisotropy-corrected Z-axis sigmas, 3D NMS, and intensity-weighted centroid refinement
2. **Linking** — KD-Tree pruned Hungarian optimal bipartite assignment (20–50x faster than naive)
3. **Gap Closing** — Multi-frame reconnection of lost tracks (up to 3 frames)
4. **Division Detection** — Scored heuristic pairing unmatched parents with daughter cells + extended mode
5. **Graph Optimization** — Velocity filtering, single-parent enforcement, short track pruning

## Key Highlights

- No GPU, no deep learning, no internet required (classical pipeline)
- ~15 minutes on Kaggle CPU, well within 12-hour limit
- Only 4 dependencies: numpy, scipy, pandas, blosc2
- Self-contained single-file submission script

## Metric

**Score = Adjusted Edge Jaccard + 0.1 * Division Jaccard**

- Node matching via Hungarian assignment per timestep (max 7 um)
- Edge Jaccard penalized by over/under-detection of node counts
- Division Jaccard on nodes with >=2 outgoing edges
