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
├── video_to_tiff.ipynb # Convert videos to TIFF stacks
├── optical_flow.py # Compute and smooth dense optical flow
├── training.py     # Training U-Net for masks
├── segmentation.py # Applying U-Net for masking
├── order_parameter.py # Compute flow coherence (local order parameter)
├── visualization.py # Overlay streamlines and annotations
├── statistics.py # Statistical comparison and plotting
└── README.md
```

## Requirements

- Python 3.8+
- Libraries:
  - OpenCV
  - NumPy
  - SciPy
  - matplotlib
  - seaborn
  - tifffile
  - scikit-image
  - PyTorch (for U-Net training and inference)
  - albumentations

Create a virtual environment and install dependencies:

```bash
python -m venv env
source env/bin/activate  # or .\\env\\Scripts\\activate on Windows
pip install -r requirements.txt
```

## Usage

```
Convert MP4 videos to TIFF:

Run convert_videos_to_tiff_frames.py

Compute Optical Flow:

Use compute_dense_optical_flow.py

Train U-Net for segmentation:

See train_unet.py 

Generate and apply masks:

Use predict_masks.py

Calculate Order Parameter:

Run calculate_order.py

Visualize Streamlines:

Run plot_streamlines.py to generate TIFF overlays

Analyze and plot statistics:

Use generate_superplot.py
```
