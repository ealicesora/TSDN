# TSDN: Transport-based Stylization for Dynamic NeRF

The dataset is build from Robust Dynamic Radiance Fields and turned to a Dnerf dataset for TiNueVox.
As a result our work is limited in the length and huge motion of the videos input. 
I believe this is casued by the limiation of Dynamic NeRF. I Really expected someone could follow this work and fix this. 


First run run.py to resconstruct dynamic nerf scene and use run_TSDN.py to get stylized results.

Most parameter are hardcoded in the files and will be fixed later
Some tools may help transferring the dataset, and I will upload them once i clean up them.


## Requirements
* lpips
* mmcv
* imageio
* imageio-ffmpeg
* opencv-python
* pytorch_msssim
* torch
* torch_scatter
