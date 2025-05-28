import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import cv2
from glob import glob
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle

# Input parameters

name = "CPF_CYP"

pixel_sizes = {
    "Ctr_1_F": 0.0028735632,
    "Ctr_3_A": 0.0048076923,
    "Ctr_3_C": 0.0048076923,
    "Ctr_3_D": 0.0048076923,
    "Ctr_3_E": 0.0048076923,
    "Ctr_3_F": 0.0048076923,
    "Ctr_3_G": 0.0048076923,
    "CFP_CYP_2_J": 0.0027777778,
    "CFP_CYP_2_K": 0.0027777778,
    "CFP_CYP_2_L": 0.0027777778,
    "CFP_CYP_3_I": 0.0033333333,
    "CFP_CYP_3_J": 0.0045454545,
    "CFP_CYP_3_R": 0.0046728972,
    "CFP_CYP_3_S": 0.0046728972
}

image_folder ="/medicina/hmorales/projects/AnemoneTracking/data/processed/"+name+"/"
mask_folder  ="/medicina/hmorales/projects/AnemoneTracking/data/processed/"+name+"_masks/"
mask_exp_folder  ="/medicina/hmorales/projects/AnemoneTracking/data/processed/"+name+"_masks/"
flow_folder  ="/medicina/hmorales/projects/AnemoneTracking/data/processed/"+name+"_optical_flow_smooth/"
order_folder ="/medicina/hmorales/projects/AnemoneTracking/data/processed/"+name+"_orderparam_flow_smooth/"
output_folder="/medicina/hmorales/projects/AnemoneTracking/data/processed/output_analysis_optical_flow_smooth/"
time_rate = 1.0/30.0
max_overall_speed = 0.5


def load_mask(mask_path, Inverse):
    mask = tifffile.imread(mask_path)
    if Inverse:
        return (mask == 0).astype(np.uint8)  # Invert: 1 = valid region
    else:
        return (mask > 0).astype(np.uint8) 



def plot_streamlines(u, v, background_img, speed, max_speed, max_speed_log, mean_order, time, pixel_size): 
    h, w = u.shape
    y, x = np.mgrid[0:h, 0:w]

    lw = 5 * np.log1p(speed) / max_speed_log
    norm = Normalize(vmin=0, vmax=max_overall_speed)

    fig, ax = plt.subplots(figsize=(10, 10))
    canvas = FigureCanvas(fig)

    ax.imshow(background_img, cmap='gray')
    ax.streamplot(x, y, u, v, color=speed, density=2, arrowsize=1e-10, linewidth=lw, cmap='plasma', norm=norm)

    # Horizontal colorbar at the bottom
    cbar = fig.colorbar(ax.collections[0], ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label(r'Flow Speed [$\mathrm{mm/s}$]', fontsize=14)
    cbar.ax.tick_params(labelsize=12)


    # Add text for order parameter and time (still upper right corner)
    ax.text(
        0.98, 0.02, f"S = {mean_order:.2f}, t = {time:.2f} s",
        color='white', fontsize=16, ha='right', va='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3')
    )

    # === Add scale bar of 1 mm in upper right ===
    scale_mm = 1
    scale_pixels = scale_mm / pixel_size  # convert to pixels
    bar_length = int(scale_pixels)
    bar_height = 5  # in pixels

    # Position: from top-right with 30 px margin
    x_pos = w - bar_length - 30
    y_pos = 30  # 30 px from top

    rect = Rectangle((x_pos, y_pos), bar_length, bar_height,
                     linewidth=0, edgecolor=None, facecolor='white')
    ax.add_patch(rect)

    # Add label centered above the bar
    ax.text(x_pos + bar_length / 2, y_pos + bar_height + 25, '1 mm',
            color='white', fontsize=14, ha='center', va='bottom',
            bbox=dict(facecolor='black', alpha=0.0, pad=2))

    ax.axis('off')
    fig.tight_layout(pad=0)
    canvas.draw()

    # Convert to image
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def process_folder(flow_folder, image_folder, mask_folder, mask_exp_folder, order_folder, output_folder, time_rate, pixel_sizes, max_overall_speed):
    os.makedirs(output_folder, exist_ok=True)

    flow_files = glob(os.path.join(flow_folder, "*.tif"))
    for flow_path in flow_files:
        basename = os.path.basename(flow_path).replace(".tif", "")
        print(f"Processing: {basename}")

        # Lookup pixel size
        pixel_size = pixel_sizes.get(basename)

        # Output result
        if pixel_size is not None:
            print(f"The pixel size for '{basename}' is {pixel_size}")
        else:
            print(f"'{basename}' not found in the dictionary.")
            
        # Paths
        image_path = os.path.join(image_folder, f"{basename}.tif")
        mask_path = os.path.join(mask_folder, f"{basename}.tif")
        mask_exp_path = os.path.join(mask_exp_folder, f"{basename}.tif")
        order_path = os.path.join(order_folder, f"{basename}.tif")
        out_path = os.path.join(output_folder, f"{basename}.tif")

        # Load data
        flow_stack     = tifffile.imread(flow_path)  # [N-1, H, W, 2]
        original_stack = tifffile.imread(image_path)  # [N, H, W, 3]
        order_param    = tifffile.imread(order_path)  # [N-1, H, W]        
        mask_inv = load_mask(mask_path, True)  # [N, H, W]
        mask = load_mask(mask_path, False)  # [N, H, W]
        mask_exp = load_mask(mask_exp_path, False)  # [N, H, W]

        if mask_folder != mask_exp_folder:
            new_mask = mask_exp & (~mask)
        else:
            print("No expanded ask")
            new_mask = mask_inv

        # Mask the image, flos and order parameters
        new_mask = new_mask[:-1,:,:]
        order_param  = order_param * new_mask.astype(np.float32)
        mask_inv_2 = np.repeat(new_mask[..., np.newaxis], 2, axis=-1)
        flow_stack   = flow_stack * pixel_size / time_rate
        flow_stack  = flow_stack * mask_inv_2.astype(np.float32)

        #mask_3 = np.repeat(mask[..., np.newaxis], 3, axis=-1)
        #original_stack = original_stack * mask_3 
        #original_stack[original_stack == 0] = 255

        #print("flow_stack:", flow_stack.shape, ", ", flow_path)
        #print("original_stack:", original_stack.shape, ", ", image_path)
        #print("mask_inv:", mask_inv.shape, ", ", mask_path)

        allspeed = np.sqrt(flow_stack[:, :, :, 0]**2 + flow_stack[:, :, :, 1]**2)
        max_speed = np.max(allspeed)
        allspeed = np.nan_to_num(allspeed, nan=0.0, posinf=0.0, neginf=0.0)
        log1pallspeed = np.log1p(allspeed)
        max_speed_log = np.max(log1pallspeed)
        print(f'File {basename} , max speed: {max_speed} [mm/s], log1pmax speed: {max_speed_log}')

        max_speed = max_overall_speed
        max_speed_log = np.log1p(max_speed)
        print(f'File {basename} , max speed: {max_speed} [mm/s], log1pmax speed: {max_speed_log}')

        result = []
#        for i in range(flow_stack.shape[0]):
        for i in range(0, flow_stack.shape[0], 5):
            #print(i, " from ", flow_stack.shape[0])
            u = flow_stack[i, :, :, 0]
            v = flow_stack[i, :, :, 1]
            magnitude = allspeed[i, :, :]
            order = order_param[i,:,:]

            # get mean order
            order = order[order > 0]  # Exclude zeros
            if len(order) > 0:
                mean_order = np.mean(order)
            else:
                mean_order = 0  # or np.nan if you prefer to skip

            # Get matching original image frame
            if i < original_stack.shape[0]:
                image_i = original_stack[i]
            else:
                image_i = original_stack[-1]  # fallback


            # Generate streamline overlay
            img = plot_streamlines(u, v, image_i, magnitude, max_speed, max_speed_log, mean_order, i*time_rate, pixel_size)
            result.append(img)

        # Save masked speed maps
        speed_stack = np.stack(result, axis=0)
        tifffile.imwrite(out_path, speed_stack.astype(np.uint8))
        print(f"Saved: {out_path}")


# Calculate


process_folder(
        flow_folder  = flow_folder,
        image_folder = image_folder,
        mask_folder  = mask_folder,
        mask_exp_folder = mask_exp_folder,
        order_folder = order_folder,
        output_folder= output_folder,
        time_rate = time_rate,        
        pixel_sizes = pixel_sizes,
        max_overall_speed = max_overall_speed
)

