import torch
import torch.nn as nn
import numpy as np
from skimage import io
import cv2
from pathlib import Path
import os
from tqdm import tqdm
from patchify import patchify, unpatchify
from skimage.color import rgb2gray
from scipy.ndimage import median_filter, binary_fill_holes
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_dilation

class UNet(nn.Module):
    def __init__(self, in_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = conv_block(128, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)
        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        dec2 = self.upconv2(bottleneck)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))
        return torch.sigmoid(self.conv_final(dec1))


class UNetPredictor:
    def __init__(self, model_path, in_channels, input_dir, output_dir, image_scale=0.5, device=None, pixel_sizes=None, expand_mask=0):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_scale = image_scale
        self.in_channels = in_channels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pixel_sizes = pixel_sizes
        self.expand_mask = expand_mask

        # Load model
        self.model = UNet(in_channels=in_channels).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def resize_by_plane(self, image, inverse=False):
        if image.ndim not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D image stack, got {image.ndim}D")
        z, y, x = image.shape[:3]

        if inverse:
            new_y = round(y / self.image_scale)
            new_x = round(x / self.image_scale)
        else:
            new_y = round(y * self.image_scale)
            new_x = round(x * self.image_scale)

        if image.ndim == 3:
            resized = np.zeros((z, new_y, new_x), dtype=image.dtype)
            for i in range(z):
                resized[i] = cv2.resize(image[i], (new_x, new_y), interpolation=cv2.INTER_CUBIC)
        else:
            resized = np.zeros((z, new_y, new_x, 3), dtype=image.dtype)
            for i in range(z):
                resized[i] = cv2.resize(image[i], (new_x, new_y), interpolation=cv2.INTER_CUBIC)
        return resized

    def preprocess_slice(self, slice_img):
        norm_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        if norm_img.ndim == 2:
            norm_img = torch.from_numpy(norm_img).float().unsqueeze(0)
        else:
            norm_img = torch.from_numpy(norm_img).float().permute(2, 0, 1)
        return norm_img.unsqueeze(0).to(self.device)

    def flip_tensor(self, tensor, mode):
        """Flip tensor with mode: 'none', 'h', 'v', 'hv'."""
        if mode == 'h':
            return torch.flip(tensor, dims=[-1])  # horizontal
        elif mode == 'v':
            return torch.flip(tensor, dims=[-2])  # vertical
        elif mode == 'hv':
            return torch.flip(tensor, dims=[-2, -1])
        return tensor  # 'none'

    def unflip_tensor(self, tensor, mode):
        """Invert the flip operation."""
        return self.flip_tensor(tensor, mode)

    def predict_masks(self, patch_size=(128, 128), batch_size=64):
        tif_files = sorted(self.input_dir.glob("*.tif"))
        if not tif_files:
            print("No .tif files found in", self.input_dir)
            return

        for tif_file in tqdm(tif_files, desc="Predicting"):
            image = io.imread(tif_file)  # shape: (z, y, x, c) or (z, y, x)
            image = self.resize_by_plane(image)

            preds = []
            for i in range(image.shape[0]):
                slice_img = image[i,:,:,:]
                slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)

                original_shape = slice_img.shape[:2]
                if self.in_channels == 1:
                    slice_img = rgb2gray(slice_img)
                # Pad to make divisible by patch size
                pad_y = (patch_size[0] - original_shape[0] % patch_size[0]) % patch_size[0]
                pad_x = (patch_size[1] - original_shape[1] % patch_size[1]) % patch_size[1]
                pad_width = ((0, pad_y), (0, pad_x), (0, 0)) if slice_img.ndim == 3 else ((0, pad_y), (0, pad_x))
                slice_padded = np.pad(slice_img, pad_width, mode='reflect')
                padded_shape = slice_padded.shape[:2]

                # Patchify
                patches = patchify(slice_padded, patch_size + ((3,) if self.in_channels == 3 else ()), step=patch_size[0])
                patch_list = []
                positions = []

                #print("patches shape: ", patches.shape)

                for row in range(patches.shape[0]):
                    for col in range(patches.shape[1]):
                        patch = patches[row, col]
                        if patch .shape[0] == 1:
                        	patch = patch.squeeze(0)
                        #print("patch shape: ", patch.shape)
                        patch_tensor = self.preprocess_slice(patch).squeeze(0)  # [C, H, W]
                        patch_list.append(patch_tensor)
                        positions.append((row, col))

                # Batched prediction
                pred_patches = []
                for b in range(0, len(patch_list), batch_size):
                    batch = torch.stack(patch_list[b:b + batch_size]).to(self.device)  # [B, C, H, W]
                    with torch.no_grad():
                        batch_pred = self.model(batch).squeeze(1).cpu().numpy()  # [B, H, W]
                        batch_pred = (batch_pred > 0.75).astype(np.uint8) * 255
                        pred_patches.extend(batch_pred)

                # Fill predicted patch array
                patch_grid = np.zeros(patches.shape[:2] + patch_size, dtype=np.uint8)
                for (r, c), pred_patch in zip(positions, pred_patches):
                    patch_grid[r, c] = pred_patch

                # Reconstruct and crop
                full_mask = unpatchify(patch_grid, padded_shape)
                full_mask = full_mask[:original_shape[0], :original_shape[1]]
                full_mask = median_filter(full_mask, size=(5, 5))
                full_mask = (full_mask > 0).astype(bool)
                full_mask = remove_small_objects(full_mask, min_size=100)
                full_mask = binary_fill_holes(full_mask)
                full_mask = full_mask.astype(np.uint8) * 255
                preds.append(full_mask)

            # get maks back to oiginal size and binnarize
            preds = np.stack(preds).astype(np.uint8)		
            preds = median_filter(preds, size=(7, 1, 1))
            preds = self.resize_by_plane(preds, True)

            # Dilate mask
            if expand_mask > 0:
                dilated_preds = np.empty_like(preds)
                pixel_size = pixel_sizes.get(tif_file.stem)
                expand_mask_pixels = int(expand_mask/pixel_size)

                if pixel_size is not None:
                    print(f"The pixel size for '{tif_file.stem}' is {pixel_size} expanding {expand_mask_pixels} pixels")
                else:
                    print(f"'{tif_file.stem}' not found in the dictionary.")

                for t in range(preds.shape[0]):
                    frame = preds[t] > 0  # ensure binary mask
                    dilated_frame = binary_dilation(frame, iterations=expand_mask_pixels)
                    dilated_frame = dilated_frame.astype(preds.dtype)
                    dilated_preds[t] = dilated_frame.astype(np.uint8) * 255
                
                preds = dilated_preds

            save_path = self.output_dir / (tif_file.stem + ".tif")
            io.imsave(save_path, preds.astype(np.uint8))

if __name__ == "__main__":

    name = "CPF_CYP"
    expand_mask = 0.3 # in mm
	
    model_path = "/medicina/hmorales/projects/AnemoneTracking/data/training/model_output/RGB_new_best_model.pth"
    input_dir  = "/medicina/hmorales/projects/AnemoneTracking/data/processed/"+name+"/"
    output_dir = "/medicina/hmorales/projects/AnemoneTracking/data/processed/"+name+"_masks_exp/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


    predictor = UNetPredictor(
        model_path=model_path,
        in_channels=3,
        input_dir=input_dir,
        output_dir=output_dir,
        image_scale=0.5,
        device=device,
        pixel_sizes = pixel_sizes,
        expand_mask = expand_mask
    )

    predictor.predict_masks()
