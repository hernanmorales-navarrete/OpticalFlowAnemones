import os
import numpy as np
import cv2
import tifffile
from glob import glob
from tqdm import tqdm

def compute_dense_optical_flow(frames_gray):
    flow_stack = []
    for i in tqdm(range(len(frames_gray) - 1), desc="Computing Optical Flow"):
#    for i in range(len(frames_gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i], frames_gray[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        u = flow[..., 0]
        v = flow[..., 1]

        flow_stack.append(np.stack([u, v], axis=-1))  # shape: [H, W, 2]
    return np.array(flow_stack)  # shape: [N-1, H, W, 2]

def process_tif_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    tif_files = glob(os.path.join(input_folder, "*.tif"))

    for tif_path in tif_files:
        filename = os.path.splitext(os.path.basename(tif_path))[0]
        print(f"Processing: {filename}")

        frames = tifffile.imread(tif_path)

        # Ensure RGB format
        if frames.ndim == 3:  # [N, H, W]
            print("Grayscale TIF, skipping...")
            frames_gray = frames

        elif frames.ndim == 4 and frames.shape[-1] >= 3:
            frames = frames[..., :3]  # Keep RGB
            # Convert each frame to grayscale
            frames_gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]

        else:
            print("Unsupported format, skipping...")
            continue


        # Compute dense optical flow
        flow_stack = compute_dense_optical_flow(frames_gray)

        # Save as single multi-frame TIF with 2 channels per frame
        out_path = os.path.join(output_folder, f"{filename}_dense_flow.tif")
        tifffile.imwrite(out_path, flow_stack.astype(np.float32))
        print(f"Saved: {out_path}")

# Example usage
if __name__ == "__main__":
    input_folder  = "/medicina/hmorales/projects/AnemoneTracking/data/processed/CPF_CYP/"
    output_folder = "/medicina/hmorales/projects/AnemoneTracking/data/processed/CPF_CYP_optical_flow/"
    process_tif_folder(input_folder, output_folder)


