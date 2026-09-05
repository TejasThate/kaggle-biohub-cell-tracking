# CZ Biohub - Cell Tracking During Development

A high-performance, timeout-immune submission pipeline for the [CZ Biohub Cell Tracking Kaggle competition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development). Reconstructs 3D+t cell lineage trees with mitosis detection across 199 private test datasets using isotropic multi-scale Difference-of-Gaussians filtering, sub-voxel centroid refinement, sparse KD-tree pruned Hungarian bipartite matching with velocity momentum, fast gap closing, and strict global graph optimization.

---

## Key Highlights

- Isotropic XY 4x Downsampling (1.625 um Grid): Matches the axial Z voxel resolution (1.625 um) to form an isotropic 3D grid. Accelerates 3D convolutions by 32x-50x, smooths granular subcellular noise, and ensures exactly 1 detection per cell nucleus to avoid the competition node-count penalty.
- Sub-Voxel Centroid Refinement: Maps detected isotropic peaks back to full-resolution space and computes intensity-weighted center-of-mass on raw pixel intensities.
- GPU Acceleration with CPU Fallback: Employs CuPy GPU-accelerated separable convolutions when CUDA is available, seamlessly falling back to C-optimized SciPy on CPU.
- Momentum-Guided Hungarian Matching: Integrates constant-velocity position prediction into KD-tree bipartite matching, significantly suppressing identity swaps in streaming embryonic tissue.
- Direct Gap Closing: Connects lost tracks directly across 1 to 3 frame gaps to reappearing detections without hallucinating synthetic intermediate nodes.
- Geometric Division & Extended Division Detection: Identifies cell division events using biological symmetry invariants (sister separation and parent-midpoint proximity) without brittle raw fluorescence intensity gating.
- Strict Global Graph Optimization: Enforces a 15.0 um/frame physical velocity ceiling, single-parent constraint, and track persistence (minimum 3 frames or division involvement).
- Strict Submission Compliance: Enforces integer typing across all coordinate and ID fields in submission.csv.

---

## Pipeline Architecture

```
3D+t Volume -> Fast Quantile Intensity Normalization
                     |
                     v
       Isotropic 4x Downsampling (1.625 um Grid)
                     |
                     v
       Multi-Scale Difference-of-Gaussians (GPU / CPU)
                     |
                     v
       3D Non-Maximum Suppression (Size 3 Isotropic Footprint)
                     |
                     v
       Full-Resolution Centroid Refinement (Raw Voxel Space)
                     |
                     v
       Momentum-Aware KD-Tree Hungarian Linking (dt = 1)
                     |
                     v
       Direct Gap Closing (1 to 3 frame gaps)
                     |
                     v
       Geometric Division & Extended Division Detection
                     |
                     v
       Global Lineage Graph Optimization (Velocity + Min Track Length)
                     |
                     v
       submission.csv (Strict Integer Schema)
```

---

## Repository Structure

| File | Description |
|---|---|
| `final_submission.py` | Primary standalone submission script. Self-contained, auto-detects Kaggle directories, runs multi-process tracking across datasets, and outputs submission.csv. |
| `cell-tracking-refined.ipynb` | Development notebook for heuristic tracking and volumetric filtering. |
| `nbf_mod.ipynb` | Modular notebook exploring tracking and centroid refinement. |
| `train-3d-unet.ipynb` | 3D U-Net training pipeline with MONAI for Gaussian heatmap regression. |
| `inference-3d-unet.ipynb` | Volumetric sliding-window inference using trained 3D U-Net checkpoints. |
| `README.md` | Project documentation, methodology overview, and execution instructions. |

---

## How to Run

### 1. Kaggle Notebook Submission
1. Copy or upload `final_submission.py` into your Kaggle Notebook.
2. Attach competition data: `biohub-cell-tracking-during-development`.
3. Enable GPU (completes in ~10-15 minutes) or CPU (~15-20 minutes).
4. Run the script or notebook; output is saved to `/kaggle/working/submission.csv`.

### 2. Local Execution
```bash
# Automatic test directory detection:
python final_submission.py

# Or specify custom test directory, output path, and worker count:
python final_submission.py --test-dir /path/to/test --output submission.csv --workers 4
```

---

## Submission Format

The output strictly complies with the competition schema:
```csv
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,dataset_1,node,1,0,32,106,107,-1,-1
1,dataset_1,node,2,0,42,186,154,-1,-1
...
100,dataset_1,edge,-1,-1,-1,-1,-1,1,2
```
All columns are strictly integer-typed.
