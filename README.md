# OpticalFlowAnemones
This repository contains all the scripts and pipelines used to quantify and visualize particle flow around anemone tentacles under control and CPF/CYP-treated conditions, as presented in the manuscript "Pesticide-induced physiological breakdown and bleaching in reef organisms".

## Overview

The analysis includes:
- Video preprocessing and conversion to TIFF image stacks
- Dense optical flow computation and smoothing
- U-Net-based segmentation of anemone tentacles
- Calculation of local order parameters (flow coherence)
- Streamline visualization over original frames
- Statistical analysis and plotting of results

## Repository Structure

```
OpticalFlowAnemones/
├── convert_videos_to_tiff_frames.ipynb # Convert videos to TIFF stacks
├── compute_dense_optical_flow.py # Compute and smooth dense optical flow
├── train_unet.py     # Training U-Net for masks
├── predict_masks.py # Applying U-Net for masking
├── calculate_order.py # Compute flow coherence (local order parameter)
├── plot_streamlines.py # Overlay streamlines and annotations
├── generate_superplot.py # Statistical comparison and plotting
└── README.md
```



## Software Dependencies and Environment

Tested on:
Operating System: Ubuntu 22.04
Python: 3.10

Required packages:
- OpenCV ≥ 4.5
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- matplotlib ≥ 3.5
- seaborn ≥ 0.11
- tifffile ≥ 2021.4
- scikit-image ≥ 0.19
- PyTorch ≥ 1.11 (with CUDA support for training on GPU)
- albumentations ≥ 1.1

Non-standard hardware (optional but recommended):
- GPU with CUDA support (for faster U-Net training and prediction)

## Installation Guide

Clone the repository:
```bash
git clone https://github.com/your-username/OpticalFlowAnemones.git
cd OpticalFlowAnemones
```
Create a virtual environment and install dependencies:

```bash
conda create anemone_analysis python=3.10 -y
conda activate anemone_analysis
pip install -r requirements.txt
```

## Usage


- Convert MP4 videos to TIFF:
```
python convert_videos_to_tiff_frames.py
```

- Compute Optical Flow:
```
python compute_dense_optical_flow.py
python interpolate_smooth_fields.py
```

- Train U-Net for segmentation:
```
python train_unet.py
```

- Generate and apply masks:
```
python predict_masks.py
```

- Calculate Order Parameter:
```
python calculate_order.py
```

- Visualize Streamlines:
```
python plot_streamlines.py 
```

- Analyze and plot statistics:
```
python generate_superplot.py
```

