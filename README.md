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

