import imageio
import numpy as np
import lpips
import torch
from skimage.metrics import structural_similarity as ssim
import os

# Load the batch of images
image1_dir = "path/to/batch1"  # Replace with your batch directory
image2_dir = "path/to/batch2"  # Replace with your batch directory

image1_paths = sorted([os.path.join(image1_dir, f) for f in os.listdir(image1_dir) if f.endswith('.png')])
image2_paths = sorted([os.path.join(image2_dir, f) for f in os.listdir(image2_dir) if f.endswith('.png')])

# Ensure both batches have the same number of images
if len(image1_paths) != len(image2_paths):
    raise ValueError("Both batches must have the same number of images.")

# Initialize LPIPS loss model
lpips_loss = lpips.LPIPS(net='alex')  # You can choose 'alex', 'vgg', or 'squeeze'

# Convert images to tensors and normalize to [-1, 1]
def convert_to_tensor(img):
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0)  # Convert to [1, C, H, W]
    img = 2 * img - 1  # Normalize to [-1, 1]
    return img

# PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

import imageio
import numpy as np
import lpips
import torch
from skimage.metrics import structural_similarity as ssim
import os

# Load the batch of images
image1_dir = "path/to/batch1"  # Replace with your batch directory
image2_dir = "path/to/batch2"  # Replace with your batch directory

image1_paths = sorted([os.path.join(image1_dir, f) for f in os.listdir(image1_dir) if f.endswith('.png')])
image2_paths = sorted([os.path.join(image2_dir, f) for f in os.listdir(image2_dir) if f.endswith('.png')])

# Ensure both batches have the same number of images
if len(image1_paths) != len(image2_paths):
    raise ValueError("Both batches must have the same number of images.")

# Initialize LPIPS loss model
lpips_loss = lpips.LPIPS(net='alex')  # You can choose 'alex', 'vgg', or 'squeeze'

# Convert images to tensors and normalize to [-1, 1]
def convert_to_tensor(img):
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0)  # Convert to [1, C, H, W]
    img = 2 * img - 1  # Normalize to [-1, 1]
    return img

# PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

# Initialize accumulators for average values
psnr_total = 0
ssim_total = 0
lpips_total = 0

# Loop through each pair of images
num_images = len(image1_paths)
for img1_path, img2_path in zip(image1_paths, image2_paths):
    image1 = imageio.imread(img1_path)
    image2 = imageio.imread(img2_path)

    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError(f"Input images {img1_path} and {img2_path} must have the same dimensions.")

    # PSNR (Peak Signal-to-Noise Ratio)
    psnr_value = calculate_psnr(image1, image2)
    psnr_total += psnr_value
    print(f"PSNR for {img1_path} and {img2_path}: {psnr_value:.2f} dB")

    # SSIM (Structural Similarity Index)
    ssim_value = ssim(image1, image2, multichannel=True)
    ssim_total += ssim_value
    print(f"SSIM for {img1_path} and {img2_path}: {ssim_value:.4f}")

    # LPIPS (Learned Perceptual Image Patch Similarity)
    tensor1 = convert_to_tensor(image1)
    tensor2 = convert_to_tensor(image2)

    lpips_value = lpips_loss(tensor1, tensor2).item()
    lpips_total += lpips_value
    print(f"LPIPS for {img1_path} and {img2_path}: {lpips_value:.4f}")

# Calculate and print average values
psnr_avg = psnr_total / num_images
ssim_avg = ssim_total / num_images
lpips_avg = lpips_total / num_images

print(f"Average PSNR: {psnr_avg:.2f} dB")
print(f"Average SSIM: {ssim_avg:.4f}")
print(f"Average LPIPS: {lpips_avg:.4f}")

