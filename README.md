# Kaggle Biohub Cell Tracking During Development

This repository contains notebooks for the "Biohub - Cell Tracking During Development" Kaggle competition. It documents the transition from a heuristic baseline to a supervised deep learning solution.

## Description

The project explores two primary approaches to cell tracking in 3D time-lapse microscopy data:
1. Heuristic Approach (`cell-tracking-refined.ipynb`): Uses multi-scale Difference-of-Gaussians (DoG) for cell detection, combined with an optimized KD-tree Hungarian algorithm for linking.
2. Deep Learning Approach (`train-3d-unet.ipynb` & `inference-3d-unet.ipynb`): A supervised approach using a 3D U-Net (via MONAI) to predict Gaussian heatmaps of cell centers, significantly improving detection accuracy in dense and noisy regions.

## Key Learnings

- Heuristic Limitations: While Difference-of-Gaussians is computationally efficient and intuitive, it struggles heavily with noise and dense cell clusters, leading to missed detections or false positives.
- Gaussian Heatmap Regression: Instead of standard binary segmentation, predicting a Gaussian heatmap around each cell center is far more effective for dense point detection tasks.
- 3D U-Net: Medical imaging libraries like MONAI provide robust 3D network architectures and utilities that handle volumetric data elegantly.
- Resource Constraints: Due to Kaggle's execution time limits, training and inference must be split across multiple notebooks using dataset artifacts to pass model weights.
