import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

def calculate_psnr(img1, img2):
    # Assumes images are [0, 1] range
    img1 = img1.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    img2 = img2.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    return psnr_func(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    img1 = img1.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    img2 = img2.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    # Multichannel for RGB
    return ssim_func(img1, img2, data_range=1.0, channel_axis=2)