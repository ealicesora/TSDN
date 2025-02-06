# TSDN: Transport-based Stylization for Dynamic NeRF

The dataset is build from Robust Dynamic Radiance Fields and turned to a Dnerf dataset for TiNueVox.

First run run.py to resconstruct dynamic nerf scene and use run_TSDN.py to get stylized results.

Most parameter are hardcoded in the files and will fixed after I have time.

## Requirements
* lpips
* mmcv
* imageio
* imageio-ffmpeg
* opencv-python
* pytorch_msssim
* torch
* torch_scatter
