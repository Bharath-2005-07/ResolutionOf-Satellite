import numpy as np
import torch

def normalize_sentinel2(image):
    """
    Normalizes 16-bit Sentinel-2 reflectance data (0-10000) to [0, 1].
    """
    img = image.astype(np.float32)
    # 10000 is the standard quantification value for Sentinel-2
    img = np.clip(img, 0, 10000) / 10000.0
    return img

def to_tensor(image):
    """
    Converts a numpy image (H, W, C) to a PyTorch tensor (C, H, W).
    """
    if len(image.shape) == 3:
        return torch.from_numpy(image).permute(2, 0, 1)
    else:
        return torch.from_numpy(image).unsqueeze(0)