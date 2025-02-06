# TSDN: Transport-based Stylization for Dynamic NeRF

This repository contains a pipeline for dynamic scene stylization using TSDN based on TiNeuVox. The dataset is converted from [Robust Dynamic Radiance Fields](https://dynamic-nerf.github.io/) to a [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html) compatible format.

## Key Limitations
Our current implementation faces constraints when processing:
- **Long-duration videos** 
- **Scenes with drastic motions**

These limitations stem from fundamental challenges in Dynamic NeRF architectures. We explicitly encourage researchers to explore solutions in these directions and welcome contributions to overcome these constraints.

## Quick Start
1. **Dynamic Reconstruction**  
   Reconstruct the dynamic NeRF scene:
   ```bash
   python run.py
1. **Stylization Generation**  
   Obtain stylized results using:
   ```bash
   python run_TSDN.py
## Implementation Notes
⚠️ Current Constraints

Core parameters are hardcoded in run.py and run_TSDN.py (to be modularized in future releases)

Dataset conversion tools will be released after code refactoring

## Requirements
* lpips
* mmcv
* imageio
* imageio-ffmpeg
* opencv-python
* pytorch_msssim
* torch
* torch_scatter
