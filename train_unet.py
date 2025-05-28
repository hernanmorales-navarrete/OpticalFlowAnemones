# -*- coding: utf-8 -*-
# Image Segmentation using *UNet*

## 1. Importing Modules & Packages

# Input paramters
srcpath = r"/medicina/hmorales/projects/AnemoneTracking/data/training/"
npatches = 250000
batch_size=64
image_scale = 0.5
in_channels = 3
seedId = 42
GPU_Id = 0

# Training parameters
num_epochs = 100
learning_rate = 0.0001
patience = 10  # Early stopping patience


modelname = ''
if in_channels == 1:
    modelname = 'grayscale'

if in_channels == 3:
    modelname = 'RGB_new'


# Libraries for handling URL and file operations
import urllib                     # Library for handling URL operations
import os                         # Library for interacting with the operating system
import zipfile                    # Library for handling zip file operations
from pathlib import Path

# Libraries for numerical and scientific computation
import numpy as np                # NumPy: manipulation of numerical arrays
import scipy.ndimage as ndi       # The image processing package scipy.ndimage
from scipy.ndimage import gaussian_filter  # Gaussian filter function from scipy.ndimage
from math import log10            # Math library for logarithmic calculations
import random                     # Library for random number generation

# Libraries for data visualization and plotting
import matplotlib.pyplot as plt   # The plotting module matplotlib.pyplot as plt
import seaborn as sns             # Seaborn: data visualization library
import pandas as pd               # Pandas: data manipulation and analysis library
import time                       # Time library for measuring execution time

# Libraries for image processing and handling
from PIL import Image             # Pillow: image processing library
from skimage import io            # Scikit-image: image I/O
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Libraries for machine learning and neural networks
import torch                      # PyTorch: deep learning library
import torch.nn as nn             # PyTorch neural network module
import torch.optim as optim       # PyTorch optimization module
from torch.utils.data import DataLoader, Dataset, random_split  # DataLoader and Dataset modules from PyTorch
from torchvision import transforms  # Transforms module from torchvision for image transformations
from torchsummary import summary    # Summary module from torchsummary for model summary

from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2  # Optional if converting to tensors inside Albumentations



# Commented out IPython magic to ensure Python compatibility.
# Set matplotlib backend
# %matplotlib inline
#%matplotlib inline              # Displays as static figure in code cell output
#%matplotlib notebook            # Displays as interactive figure in code cell output
#%matplotlib qt                  # Displays as interactive figure in a separate window

# Check GPU
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")


device = torch.device(f"cuda:{GPU_Id}" if torch.cuda.is_available() else "cpu")

print("Number of GPUs:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Using:", torch.cuda.get_device_name(device))


# Check memory on the GPU print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Set seeds for reproducibility
def set_seed(seed=42):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# DataLoader setup
def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)


# Set seeds for reproducibility
set_seed(seedId)

def resize_by_plane(image, image_scale):
    """
    Resizes a stack of 2D images (grayscale or RGB) plane by plane.

    Parameters:
        image (np.array): 3D array (z, y, x) for grayscale or 4D array (z, y, x, 3) for RGB
        image_scale (float): scale factor for resizing

    Returns:
        np.array: resized image stack with same number of planes (z)
    """
    if image.ndim not in [3, 4]:
        raise ValueError(f'Expected 3D (grayscale) or 4D (RGB) input, got {image.ndim}D.')

    z = image.shape[0]
    y = image.shape[1]
    x = image.shape[2]
    new_y = round(y * image_scale)
    new_x = round(x * image_scale)

    if image.ndim == 3:
        # Grayscale stack
        resized = np.zeros((z, new_y, new_x), dtype=image.dtype)
        for i in range(z):
            resized[i] = cv2.resize(image[i], (new_x, new_y), interpolation=cv2.INTER_CUBIC)
    else:
        # RGB stack
        resized = np.zeros((z, new_y, new_x, 3), dtype=image.dtype)
        for i in range(z):
            resized[i] = cv2.resize(image[i], (new_x, new_y), interpolation=cv2.INTER_CUBIC)

    return resized

def generate_random_patches(img,
                            mask,
                            patch_shape,
                            num_patches,
                            max_attempts_per_patch=1000):
    """
    Generator that yields random patches from a grayscale or RGB image.

    Parameters:
    - img: 2D or 3D NumPy array (grayscale or RGB image)
    - patch_shape: tuple (patch_height, patch_width)
    - num_patches: total number of patches to generate
    - max_attempts_per_patch: max tries per patch

    Yields:
    - patch: NumPy array of shape (ph, pw) or (ph, pw, 3)
    """

    if img.ndim == 2:
        h, w = img.shape
        channels = False
    elif img.ndim == 3:
        h, w, c = img.shape
        channels = True
    else:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB).")

    ph, pw = patch_shape

    if ph > h or pw > w:
        raise ValueError("Patch size is larger than the image dimensions.")

    h_m, w_m = mask.shape
    if h_m != h or w_m != w:
        raise ValueError("Mask dimensions do not match the image dimensions.")

    generated = 0
    total_attempts = 0
    max_total_attempts = num_patches * max_attempts_per_patch

    while generated < num_patches and total_attempts < max_total_attempts:
        y = np.random.randint(0, h - ph + 1)
        x = np.random.randint(0, w - pw + 1)

        if channels:
            patch = img[y:y + ph, x:x + pw, :]
        else:
            patch = img[y:y + ph, x:x + pw]

        patch_mask = mask[y:y + ph, x:x + pw]

        if np.amax(patch) > 0:
            yield patch, patch_mask

            generated += 1
            total_attempts += 1

    if generated < num_patches:
        print(f'Warning: only generated {generated} of {num_patches} patches.')


def generate_patches_from_tif(images_path, masks_path, in_channels,
                                       patch_shape=(128, 128), num_patches=100):
    """
    For each image and its corresponding mask, generate aligned random patches.
    """
    images_path = Path(images_path)
    masks_path = Path(masks_path)

    nimages = 0

    for image_file in images_path.glob("*.tif"):
        mask_file = masks_path / image_file.name

        if not mask_file.exists():
            print(f"Mask not found for {image_file.name}, skipping.")
            continue
        nimages += 1

    print(f"Found {nimages} images and masks.")


    Npatches = num_patches // nimages + 1

    count = 0
    images_patches = []
    mask_patches   = []

    for image_file in images_path.glob("*.tif"):
        mask_file = masks_path / image_file.name
        if not mask_file.exists():
            print(f"Mask not found for {image_file.name}, skipping.")
            continue

        # Load image and mask
        print("loading :", image_file.name)
        image = io.imread(image_file)
        masks = io.imread(mask_file)

        image = resize_by_plane(image, image_scale)
        masks = resize_by_plane(masks, image_scale)

 #       patches_per_image = max(1, Npatches // image.shape[0] + 1)
        patches_per_image = max(1,  5*(image.shape[1]*image.shape[2]) // (patch_shape[0]*patch_shape[0]))

 #       print(image.shape, masks.shape, patches_per_image)
        for k in range(image.shape[0]):

            img  = image[k,:,:,:]
            if in_channels == 1:
                img = rgb2gray(img)

            mask = masks[k,:,:]
            mask = (mask > 0).astype(np.uint8)

            # Generate patches
            patch_gen = generate_random_patches(img, mask, patch_shape, patches_per_image)
            base_name = image_file.stem

            # Save patches
            for i, (img_patch, mask_patch) in enumerate(patch_gen):

                images_patches.append(img_patch.astype(np.float32))
                mask_patches.append(mask_patch.astype(np.float32))
                count += 1

        print(f"Saved {count + 1} patches ")

    return images_patches, mask_patches

"""
## 2. Importing  data"""

# create a folder for models.
model_dir = os.path.join(srcpath, 'model_output')
os.makedirs(model_dir, exist_ok=True)

images_path = os.path.join(srcpath, 'images')
masks_path = os.path.join(srcpath, 'masks')

train_images, train_masks = generate_patches_from_tif(
    						images_path = images_path,
    						masks_path  = masks_path,
    						in_channels = in_channels,
    						patch_shape=(128, 128),
    						num_patches=npatches
						)

"""## 3. Ploting image example from the training set"""

# Generate training, validation and test images

# Convert lists to NumPy arrays if needed
train_images = np.array(train_images)
train_masks = np.array(train_masks)

# 1. Split into temporary train+val and test sets
train_images_temp, test_images, train_masks_temp, test_masks = train_test_split(
    train_images, train_masks, test_size=0.2, random_state=seedId)

# 2. Split temporary train+val into train and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(
    train_images_temp, train_masks_temp, test_size=0.2, random_state=seedId)  # 20% of 80% = 16%

# Print info data sets
print('------------------------------------------------------------------')

print("Shape of train images: {}".format(train_images.shape))
print("Shape of train masks: {}".format(train_masks.shape))
print("Shape of validation images: {}".format(val_images.shape))
print("Shape of validation masks: {}".format(val_masks.shape))
print("Shape of test images: {}".format(test_images.shape))
print("Shape of test masks: {}".format(test_masks.shape))



## 4. Image Segmentation using UNet

### Load images


augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
])

# Custom dataset for loading images
class SegmentationDataset(Dataset):
    def __init__(self, images, masks, augment=None):
        """
        Args:
            images (list or np.ndarray): List or array of (H, W) or (H, W, 3) images
            masks (list or np.ndarray): List or array of (H, W) masks
            augment (albumentations.Compose, optional): Albumentations transform
        """
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Ensure float32
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # Apply Albumentations augmentations (expects HWC image, HW mask)
        if self.augment:
            augmented = self.augment(image=image, mask=mask, random_state=42)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to tensor: image → [C, H, W], mask → [1, H, W]
        if image.ndim == 2:  # grayscale
            image = torch.from_numpy(image).unsqueeze(0)
        else:  # RGB
            image = torch.from_numpy(image).permute(2, 0, 1)

        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
    
# Load datasets
train_dataset = SegmentationDataset(train_images, train_masks, augment=augment)
val_dataset   = SegmentationDataset(val_images, val_masks)
test_dataset  = SegmentationDataset(test_images, test_masks)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)


# Print the number of elements and batches in each data loader
print('------------------------------------------------------------------')
print(f'Number of elements in train_loader: {len(train_loader.dataset)}')
print(f'Number of batches in train_loader: {len(train_loader)}')

print(f'Number of elements in val_loader: {len(val_loader.dataset)}')
print(f'Number of batches in val_loader: {len(val_loader)}')

print(f'Number of elements in test_loader: {len(test_loader.dataset)}')
print(f'Number of batches in test_loader: {len(test_loader)}')
print('------------------------------------------------------------------')

# Function to plot image pairs
def plot_image_pairs(data_loader, title, Nimages, outdir):
    data_iter = iter(data_loader)
    fig, axes = plt.subplots(nrows=Nimages, ncols=2, figsize=(8, 4 * Nimages))
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(wspace=0.1, hspace=0)

    images, masks = next(data_iter)  # One batch of (image, mask)

    if Nimages > images.shape[0]:
        Nimages = images.shape[0]

    for i in range(Nimages):
        # Image: [C, H, W]
        image_np = images[i].cpu().numpy()
        mask_np = masks[i].cpu().numpy()

        # Convert to [H, W, C] for RGB or [H, W] for grayscale
        if image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))  # RGB
        else:
            image_np = image_np[0]  # Grayscale

        mask_np = np.squeeze(mask_np)

        axes[i, 0].imshow(image_np, cmap=None if image_np.ndim == 3 else 'gray')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].axis('off')

        if i == 0:
            axes[i, 0].set_title('Source Image', fontsize=12)
            axes[i, 1].set_title('Target Mask', fontsize=12)

    # Save the figure
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{title}_patches.png"), dpi=300, bbox_inches='tight')
    plt.show()

# Plotting train and test image pairs
plot_image_pairs(train_loader, modelname+"_Train_Data_Loader", 4, model_dir)
plot_image_pairs(val_loader, modelname+"_Validation_Data_Loader", 4, model_dir)
plot_image_pairs(test_loader, modelname+"_Test_Data_Loader", 4, model_dir)




"""## Define UNet model"""
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

        # Use in_channels in the first encoder block
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

# Initialize model
model = UNet(in_channels)
model = model.to(device)
#summary(model, (in_channels, 128, 128))

"""## Train Model"""
criterion = nn.BCELoss()    # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Store losses
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    start_time = time.time()  # Start time for epoch
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks.float())

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation loop
    model.to(device)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs,  masks.float())
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    epoch_time = time.time() - start_time  # Calculate epoch duration
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f} seconds')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# Plotting the training and validation loss
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig( os.path.join(model_dir, modelname+"_loss_plot.png"), dpi=300, bbox_inches='tight')
plt.show()


# Save the model
torch.save(model.state_dict(), os.path.join(model_dir, modelname+'_best_model.pth'))


# Number of examples to show
NimagesToShow = 4
fig, axes = plt.subplots(nrows=NimagesToShow, ncols=4, figsize=(12, 3 * NimagesToShow))
fig.suptitle("Model evaluation", fontsize=16)
fig.subplots_adjust(wspace=0.1, hspace=0)

model.eval()
with torch.no_grad():
    data_iter = iter(test_loader)
    noisy_images, masks = next(data_iter)
    noisy_images = noisy_images.to(device)
    outputs = model(noisy_images)

    # Move everything to CPU for plotting
    noisy_images = noisy_images.cpu()
    outputs = outputs.cpu()
    masks = masks.cpu()  

    if NimagesToShow > noisy_images.shape[0]:
        NimagesToShow = noisy_images.shape[0]

    for i in range(NimagesToShow):
        # ----- Handle image (grayscale or RGB) -----
        image_np = noisy_images[i].numpy()
        if image_np.shape[0] == 3:  # RGB
            image_np = np.transpose(image_np, (1, 2, 0))
        else:  # Grayscale
            image_np = image_np[0]

        # ----- Handle prediction -----
        pred_np = outputs[i].numpy()
        if pred_np.shape[0] == 3:
            pred_np = np.transpose(pred_np, (1, 2, 0))
        else:
            pred_np = pred_np[0]

        # ----- Handle ground truth (mask) -----
        target_np = masks[i].numpy()
        if target_np.shape[0] == 3:
            target_np = np.transpose(target_np, (1, 2, 0))
        else:
            target_np = target_np[0]

        # ----- Plot -----
        axes[i, 0].imshow(image_np, cmap=None if image_np.ndim == 3 else 'gray')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(pred_np, cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(target_np, cmap='gray')
        axes[i, 2].axis('off')

        if i == 0:
            axes[i, 0].set_title('Source Image', fontsize=12)
            axes[i, 1].set_title('Predicted Image', fontsize=12)
            axes[i, 2].set_title('Target Image', fontsize=12)


plt.savefig(os.path.join(model_dir,modelname+"_precition.png"), dpi=300, bbox_inches='tight')
plt.show()