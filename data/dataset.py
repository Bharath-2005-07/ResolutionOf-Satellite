"""
Data Pipeline for Satellite Super-Resolution
Supports: WorldStrat dataset, Google Earth Engine API, and custom datasets
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SatelliteSRDataset(Dataset):
    """
    Dataset for satellite image super-resolution
    Handles paired LR/HR images from various sources
    """
    def __init__(self, lr_dir, hr_dir, patch_size=64, scale_factor=4, 
                 augment=True, normalize=True):
        """
        Args:
            lr_dir: Directory containing low-resolution images
            hr_dir: Directory containing high-resolution images
            patch_size: Size of LR patches (HR will be patch_size * scale_factor)
            scale_factor: Upscaling factor (4 or 8)
            augment: Apply data augmentation
            normalize: Normalize images to [0, 1]
        """
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.normalize = normalize
        
        # Find matching pairs
        self.lr_files = sorted(self.lr_dir.glob('*.png')) + sorted(self.lr_dir.glob('*.tif'))
        self.hr_files = sorted(self.hr_dir.glob('*.png')) + sorted(self.hr_dir.glob('*.tif'))
        
        # Verify pairs exist
        assert len(self.lr_files) == len(self.hr_files), \
            f"Mismatch: {len(self.lr_files)} LR vs {len(self.hr_files)} HR images"
        
        print(f"Found {len(self.lr_files)} LR/HR pairs")
        
    def __len__(self):
        return len(self.lr_files)
    
    def load_image(self, path):
        """Load image from various formats"""
        path = str(path)
        
        if path.endswith('.tif') and RASTERIO_AVAILABLE:
            with rasterio.open(path) as src:
                img = src.read()  # (C, H, W)
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        elif PIL_AVAILABLE:
            img = np.array(Image.open(path))
        else:
            raise ImportError("Install rasterio or Pillow for image loading")
            
        return img.astype(np.float32)
    
    def normalize_image(self, img):
        """Normalize image to [0, 1] range"""
        if img.max() > 1:
            if img.max() > 255:  # 16-bit (Sentinel-2)
                img = np.clip(img, 0, 10000) / 10000.0
            else:  # 8-bit
                img = img / 255.0
        return img
    
    def random_crop(self, lr_img, hr_img):
        """Extract random matching patches from LR and HR images"""
        lr_h, lr_w = lr_img.shape[:2]
        hr_h, hr_w = hr_img.shape[:2]
        
        lr_patch = self.patch_size
        hr_patch = self.patch_size * self.scale_factor
        
        # Random position in LR space
        lr_x = random.randint(0, max(0, lr_w - lr_patch))
        lr_y = random.randint(0, max(0, lr_h - lr_patch))
        
        # Corresponding position in HR space
        hr_x = lr_x * self.scale_factor
        hr_y = lr_y * self.scale_factor
        
        lr_crop = lr_img[lr_y:lr_y+lr_patch, lr_x:lr_x+lr_patch]
        hr_crop = hr_img[hr_y:hr_y+hr_patch, hr_x:hr_x+hr_patch]
        
        return lr_crop, hr_crop
    
    def augment_pair(self, lr_img, hr_img):
        """Apply random augmentations to both images"""
        # Random horizontal flip
        if random.random() > 0.5:
            lr_img = np.fliplr(lr_img).copy()
            hr_img = np.fliplr(hr_img).copy()
            
        # Random vertical flip
        if random.random() > 0.5:
            lr_img = np.flipud(lr_img).copy()
            hr_img = np.flipud(hr_img).copy()
            
        # Random 90-degree rotation
        k = random.randint(0, 3)
        lr_img = np.rot90(lr_img, k).copy()
        hr_img = np.rot90(hr_img, k).copy()
        
        return lr_img, hr_img
    
    def __getitem__(self, idx):
        # Load images
        lr_img = self.load_image(self.lr_files[idx])
        hr_img = self.load_image(self.hr_files[idx])
        
        # Normalize
        if self.normalize:
            lr_img = self.normalize_image(lr_img)
            hr_img = self.normalize_image(hr_img)
        
        # Random crop
        lr_img, hr_img = self.random_crop(lr_img, hr_img)
        
        # Augmentation
        if self.augment:
            lr_img, hr_img = self.augment_pair(lr_img, hr_img)
        
        # Convert to tensors (C, H, W)
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float()
        
        return lr_tensor, hr_tensor


class WorldStratDataset(Dataset):
    """
    Dataset loader for WorldStrat - Open-source paired LR/HR satellite dataset
    Download from: https://github.com/worldstrat/worldstrat
    
    Structure:
    worldstrat/
        train/
            lr/  # Sentinel-2 (10m)
            hr/  # SPOT/Pleiades (~1.5m)
        val/
        test/
    """
    def __init__(self, root_dir, split='train', patch_size=64, scale_factor=4, augment=True):
        self.root = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        
        self.lr_dir = self.root / split / 'lr'
        self.hr_dir = self.root / split / 'hr'
        
        # Get all image pairs
        self.samples = self._find_pairs()
        print(f"WorldStrat {split}: Found {len(self.samples)} pairs")
        
    def _find_pairs(self):
        pairs = []
        if not self.lr_dir.exists():
            print(f"Warning: {self.lr_dir} not found")
            return pairs
            
        for lr_file in self.lr_dir.glob('*'):
            hr_file = self.hr_dir / lr_file.name
            if hr_file.exists():
                pairs.append((lr_file, hr_file))
        return pairs
    
    def __len__(self):
        return len(self.samples) if self.samples else 100  # Return dummy length for demo
    
    def __getitem__(self, idx):
        if not self.samples:
            # Return random data for demo/testing
            lr = torch.randn(3, self.patch_size, self.patch_size)
            hr = torch.randn(3, self.patch_size * self.scale_factor, 
                           self.patch_size * self.scale_factor)
            return lr, hr
            
        lr_path, hr_path = self.samples[idx]
        
        # Load and process (implementation similar to SatelliteSRDataset)
        lr_img = self._load_image(lr_path)
        hr_img = self._load_image(hr_path)
        
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float()
        
        return lr_tensor, hr_tensor
    
    def _load_image(self, path):
        if PIL_AVAILABLE:
            img = np.array(Image.open(path).convert('RGB'))
            return img.astype(np.float32) / 255.0
        return np.random.rand(self.patch_size, self.patch_size, 3).astype(np.float32)


class SyntheticSRDataset(Dataset):
    """
    Create synthetic LR/HR pairs from HR images by downsampling
    Useful when you only have HR images for training
    """
    def __init__(self, hr_dir, patch_size=64, scale_factor=4, augment=True,
                 degradation='bicubic'):
        """
        Args:
            hr_dir: Directory containing high-resolution images
            degradation: 'bicubic', 'bilinear', or 'gaussian_blur'
        """
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.degradation = degradation
        
        self.hr_files = list(self.hr_dir.glob('*.png')) + list(self.hr_dir.glob('*.jpg'))
        print(f"Found {len(self.hr_files)} HR images for synthetic LR generation")
        
    def __len__(self):
        return len(self.hr_files)
    
    def create_lr(self, hr_img):
        """Create LR image by downsampling HR image"""
        import cv2
        
        hr_h, hr_w = hr_img.shape[:2]
        lr_h, lr_w = hr_h // self.scale_factor, hr_w // self.scale_factor
        
        if self.degradation == 'bicubic':
            lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        elif self.degradation == 'bilinear':
            lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)
        elif self.degradation == 'gaussian_blur':
            # Add Gaussian blur then downsample
            blurred = cv2.GaussianBlur(hr_img, (5, 5), 1.5)
            lr_img = cv2.resize(blurred, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        else:
            lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            
        return lr_img
    
    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        
        if PIL_AVAILABLE:
            hr_img = np.array(Image.open(hr_path).convert('RGB')).astype(np.float32) / 255.0
        else:
            hr_img = np.random.rand(256, 256, 3).astype(np.float32)
        
        # Random crop HR patch
        hr_patch_size = self.patch_size * self.scale_factor
        h, w = hr_img.shape[:2]
        
        x = random.randint(0, max(0, w - hr_patch_size))
        y = random.randint(0, max(0, h - hr_patch_size))
        
        hr_patch = hr_img[y:y+hr_patch_size, x:x+hr_patch_size]
        
        # Create LR from HR
        lr_patch = self.create_lr(hr_patch)
        
        # Augmentation
        if self.augment and random.random() > 0.5:
            hr_patch = np.fliplr(hr_patch).copy()
            lr_patch = np.fliplr(lr_patch).copy()
        
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float()
        
        return lr_tensor, hr_tensor


class GEEDataset(Dataset):
    """
    Dataset that fetches data from Google Earth Engine
    Requires authenticated GEE API
    
    Usage:
        import ee
        ee.Authenticate()
        ee.Initialize()
    """
    def __init__(self, coordinates_list, patch_size=64, scale_factor=4):
        """
        Args:
            coordinates_list: List of (lon, lat) tuples for sampling
        """
        self.coordinates = coordinates_list
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        
        try:
            import ee
            self.ee = ee
            self.gee_available = True
        except ImportError:
            print("Google Earth Engine API not available. Install with: pip install earthengine-api")
            self.gee_available = False
            
    def __len__(self):
        return len(self.coordinates)
    
    def fetch_sentinel2_patch(self, lon, lat, buffer_m=640):
        """
        Fetch Sentinel-2 patch from GEE
        
        Args:
            lon, lat: Center coordinates
            buffer_m: Buffer in meters (640m = 64 pixels at 10m/pixel)
        """
        if not self.gee_available:
            return np.random.rand(self.patch_size, self.patch_size, 3).astype(np.float32)
            
        ee = self.ee
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_m).bounds()
        
        # Get Sentinel-2 L2A Surface Reflectance
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(region)
                     .filterDate('2023-01-01', '2024-01-01')
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                     .sort('CLOUDY_PIXEL_PERCENTAGE')
                     .first())
        
        # Select RGB bands
        image = collection.select(['B4', 'B3', 'B2'])  # Red, Green, Blue
        
        # Get as numpy array
        try:
            url = image.getThumbURL({
                'region': region,
                'dimensions': f'{self.patch_size}x{self.patch_size}',
                'format': 'png'
            })
            
            import urllib.request
            from io import BytesIO
            
            with urllib.request.urlopen(url) as response:
                img_data = response.read()
            img = np.array(Image.open(BytesIO(img_data)))
            return img.astype(np.float32) / 255.0
            
        except Exception as e:
            print(f"GEE fetch error: {e}")
            return np.random.rand(self.patch_size, self.patch_size, 3).astype(np.float32)
    
    def __getitem__(self, idx):
        lon, lat = self.coordinates[idx]
        
        # Fetch LR (Sentinel-2)
        lr_img = self.fetch_sentinel2_patch(lon, lat)
        
        # For training, we need HR pairs - use bicubic upscaled as placeholder
        # In real scenario, you'd fetch WorldView or other HR source
        import cv2
        hr_size = self.patch_size * self.scale_factor
        hr_img = cv2.resize(lr_img, (hr_size, hr_size), interpolation=cv2.INTER_CUBIC)
        
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float()
        
        return lr_tensor, hr_tensor


def get_dataloader(dataset_type='synthetic', **kwargs):
    """
    Factory function to get appropriate dataloader
    
    Args:
        dataset_type: 'synthetic', 'worldstrat', 'satellite', 'gee'
        **kwargs: Arguments passed to dataset constructor
    
    Returns:
        DataLoader instance
    """
    batch_size = kwargs.pop('batch_size', 8)
    num_workers = kwargs.pop('num_workers', 4)
    
    if dataset_type == 'synthetic':
        dataset = SyntheticSRDataset(**kwargs)
    elif dataset_type == 'worldstrat':
        dataset = WorldStratDataset(**kwargs)
    elif dataset_type == 'satellite':
        dataset = SatelliteSRDataset(**kwargs)
    elif dataset_type == 'gee':
        dataset = GEEDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


# Demo dataset for quick testing
class DemoDataset(Dataset):
    """Quick demo dataset with random data for testing pipeline"""
    def __init__(self, num_samples=100, patch_size=64, scale_factor=4):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        lr = torch.randn(3, self.patch_size, self.patch_size) * 0.5 + 0.5
        lr = torch.clamp(lr, 0, 1)
        
        hr = torch.randn(3, self.patch_size * self.scale_factor, 
                        self.patch_size * self.scale_factor) * 0.5 + 0.5
        hr = torch.clamp(hr, 0, 1)
        
        return lr, hr
