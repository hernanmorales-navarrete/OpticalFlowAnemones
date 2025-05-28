import os
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter, sobel
from glob import glob
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from concurrent.futures import ProcessPoolExecutor
from functools import partial


nameId = "Control"
# Parameters
flow_folder             = "/medicina/hmorales/projects/AnemoneTracking/data/processed/"+nameId+"_optical_flow_smooth/"
mask_folder             = "/medicina/hmorales/projects/AnemoneTracking/data/processed/"+nameId+"_masks/"
mask_exp_folder         = "/medicina/hmorales/projects/AnemoneTracking/data/processed/"+nameId+"_masks_exp/"
output_order_folder     = "/medicina/hmorales/projects/AnemoneTracking/data/processed/"+nameId+"_orderparam_flow_smooth_exp/"
window_size = 100  # for local order parameter
useMask = True
useExpandedMask = True

os.makedirs(output_order_folder, exist_ok=True)

# Helper: compute local order parameter
def compute_order_parameter(vx, vy, window_size=50):

    magnitude = np.sqrt(vx**2 + vy**2)
    valid_mask = magnitude > 1e-3
    theta = np.zeros_like(vx)
    theta[valid_mask] = np.arctan2(vy[valid_mask], vx[valid_mask])
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    mean_cos = gaussian_filter(np.cos(theta), sigma=window_size)
    mean_sin = gaussian_filter(np.sin(theta), sigma=window_size)
    
    S = np.sqrt(mean_cos**2 + mean_sin**2)
    return S

def compute_order_param_weighted_by_norm_speed(u, v, sigma=50):
    speed = np.sqrt(u**2 + v**2)
    max_speed = np.nanmax(speed)

    # Normalize speed for weighting
    norm_speed = speed / (max_speed + 1e-6)

    theta = np.arctan2(v, u)
    weighted_cos = norm_speed * np.cos(theta)
    weighted_sin = norm_speed * np.sin(theta)

    # Smooth the weighted components
    mean_cos = gaussian_filter(weighted_cos, sigma=sigma)
    mean_sin = gaussian_filter(weighted_sin, sigma=sigma)

    # Final order parameter (not normalized per pixel)
    S_weighted = np.sqrt(mean_cos**2 + mean_sin**2)
    return S_weighted


def process_frame(t, flow_t, mask_t, window_size, useMask):
    vx = flow_t[:, :, 0]
    vy = flow_t[:, :, 1]

    order_param = compute_order_parameter(vx, vy, window_size)
    #order_param = compute_order_param_weighted_by_norm_speed(vx, vy, window_size)
    if useMask:
        order_param = order_param * mask_t

    return order_param  

def compute_stacks(flow, mask, window_size, useMask, base_name=""):
    X, H, W, _ = flow.shape
    order_stack = np.zeros((X, H, W), dtype=np.float32)

    with ProcessPoolExecutor() as executor:
        func = partial(process_frame, window_size=window_size, useMask=useMask)
        results = list(tqdm(
            executor.map(func, range(X), flow, mask),
            total=X,
            desc=f"Processing {base_name}"
        ))

    for t, order_param in enumerate(results):
        order_stack[t] = order_param

    return order_stack  

# Process all optical flow TIFFs
for flow_path in glob(os.path.join(flow_folder, "*.tif")):
    base_name = os.path.basename(flow_path).replace(".tif", "")
    mask_path = os.path.join(mask_folder, f"{base_name}.tif")

    
    # Load flow image: shape (Z, H, W, 2) 
    flow = tifffile.imread(flow_path)  # Expecting shape: (Z, H, W, 2)
    if flow.shape[3] != 2:
        raise ValueError(f"Unexpected shape in {flow_path}: {flow.shape}")
        
    # Load and invert mask: mask==1 means inside (to be zeroed)
    mask = tifffile.imread(mask_path).astype(bool)
    inv_mask = (~mask).astype(np.float32)

    if useExpandedMask:
        mask_exp_path = os.path.join(mask_exp_folder, f"{base_name}.tif")
        mask_exp = tifffile.imread(mask_exp_path).astype(bool)
        inv_mask = mask_exp & (~mask)
        inv_mask = inv_mask.astype(np.float32)
  
    # Compute values
    order_stack = compute_stacks(flow, inv_mask, window_size, useMask, base_name=base_name)

    # Save results
    order_out_path = os.path.join(output_order_folder, f"{base_name}_order.tif")
    
    tifffile.imwrite(order_out_path, order_stack.astype(np.float32))

    print(f"Processed: {base_name}")

