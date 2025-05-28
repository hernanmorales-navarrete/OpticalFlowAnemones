import os
import numpy as np
import tifffile
from glob import glob
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from skimage.filters import threshold_otsu

# Input paramters
nameId = "Control_optical_flow"
flow_folder   = "/medicina/hmorales/projects/AnemoneTracking/data/processed/"+nameId 
output_folder = "/medicina/hmorales/projects/AnemoneTracking/data/processed/"+nameId+"_smooth/"
smooth_wnd = 3


# Functions

def interpolate_nans(field):
    h, w = field.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    known_points = ~np.isnan(field)
    interp = griddata(
        (X[known_points], Y[known_points]),
        field[known_points],
        (X, Y),
        method='linear'
    )
    # Optional: fill remaining NaNs using nearest if any
    if np.isnan(interp).any():
        interp = griddata(
            (X[known_points], Y[known_points]),
            field[known_points],
            (X, Y),
            method='nearest'
        )
    return interp

def process_frame(args):
    i, flow_frame, smooth_wnd = args
    u = flow_frame[:, :, 0]
    v = flow_frame[:, :, 1]

    # Compute speed and Mask low-magnitude vectors
    magnitude = np.sqrt(u**2 + v**2)
    valid_magnitude = magnitude[~np.isnan(magnitude)]
#    threshold = threshold_otsu(valid_magnitude)
    threshold = 0.05 * np.nanmax(magnitude)
    mask = magnitude < threshold

    u[mask] = np.nan
    v[mask] = np.nan

    # Interpolate
    u = interpolate_nans(u)
    v = interpolate_nans(v)

    # Smooth
    u = gaussian_filter(u, sigma=smooth_wnd)
    v = gaussian_filter(v, sigma=smooth_wnd)

    return i, np.stack([u, v], axis=-1)  # shape: [H, W, 2]

def process_folder(flow_folder, smooth_wnd, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    flow_files = glob(os.path.join(flow_folder, "*.tif"))
    for flow_path in flow_files:
        basename = os.path.basename(flow_path).replace("_dense_flow.tif", "")
        print(f"Processing: {basename}")

        # Load data
        flow_stack = tifffile.imread(flow_path)  # [N-1, H, W, 3]

        # Setup
        num_frames = flow_stack.shape[0]
        smooth_flow_stack = np.zeros_like(flow_stack)
        args_list = [(i, flow_stack[i], smooth_wnd) for i in range(num_frames)]

        # Parallel processing
        with ProcessPoolExecutor() as executor:
            for i, result in tqdm(executor.map(process_frame, args_list), total=num_frames, desc="Processing frames"):
                smooth_flow_stack[i] = result

        # Save masked speed maps
        smooth_flow_stack = np.stack(smooth_flow_stack)
        out_path = os.path.join(output_folder, basename + ".tif")
        tifffile.imwrite(out_path, smooth_flow_stack.astype(np.float32))


# Apply

process_folder(
    flow_folder   = flow_folder,
    smooth_wnd    = smooth_wnd,
    output_folder = output_folder
)

