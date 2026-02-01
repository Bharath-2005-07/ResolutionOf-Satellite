# 🛰️ Satellite Super-Resolution - Data Preparation Guide

## Overview

This project requires **real satellite imagery** for training. Here are all the ways to get data:

---

## 📥 Data Sources

### **Option 1: UC Merced Land Use Dataset (RECOMMENDED)**

**Best for**: Quick start, demo, hackathon submission

- **Size**: ~320 MB
- **Images**: 2,100 satellite images
- **Resolution**: 256x256 pixels
- **License**: Public domain
- **Quality**: Real satellite imagery of various land use types

**Download**:
```bash
wget http://weegee.vision.ucmerced.edu/datasets/landuse.zip
unzip landuse.zip -d satellite_data
```

**Automated** (included in `train_satellite_colab.py`):
- Script automatically downloads and processes this dataset
- Creates LR/HR pairs
- Ready for training

---

### **Option 2: WorldStrat Dataset**

**Best for**: Research, production models, best quality

- **Source**: https://github.com/worldstrat/worldstrat
- **Data**: Real Sentinel-2 (LR) + High-res (HR) pairs
- **Size**: ~50 GB
- **Quality**: Best for satellite SR training

**Download**:
```bash
git clone https://github.com/worldstrat/worldstrat
```

**Usage**:
```python
from data.dataset import WorldStratDataset

dataset = WorldStratDataset(
    root_dir='worldstrat',
    split='train',
    patch_size=64,
    scale_factor=4
)
```

---

### **Option 3: Google Earth Engine (GEE)**

**Best for**: Real-time data, custom locations, specific dates

- **Requires**: GEE API authentication
- **Advantage**: Access to entire Sentinel-2 archive
- **Flexibility**: Choose any location and time period

**Setup**:
```bash
pip install earthengine-api
```

**Authenticate**:
```python
import ee
ee.Authenticate()
ee.Initialize()
```

**Usage**:
```python
from data.dataset import GEEDataset
from data.gee_fetch import initialize_gee, fetch_patch

# Initialize
initialize_gee()

# Fetch patches from specific locations
coordinates = [
    (-122.4194, 37.7749),  # San Francisco
    (-74.0060, 40.7128),   # New York
    (2.3522, 48.8566),     # Paris
]

dataset = GEEDataset(coordinates, patch_size=64, scale_factor=4)
```

---

### **Option 4: Your Own Satellite Images**

**Best for**: Custom applications, specific regions

If you have high-resolution satellite images:

```python
from data.dataset import SyntheticSRDataset

# Creates LR by downsampling your HR images
dataset = SyntheticSRDataset(
    hr_dir='path/to/your/satellite/images',
    patch_size=64,
    scale_factor=4,
    degradation='bicubic'  # Simulates Sentinel-2 quality
)
```

**Supported formats**:
- PNG, JPEG, TIFF
- GeoTIFF (via rasterio)
- Single or multi-band images

---

## 📊 Data Preparation

### **Automatic Preparation (Recommended)**

The `train_satellite_colab.py` script handles everything:

```python
python train_satellite_colab.py
```

This will:
1. Download UC Merced dataset
2. Process images into LR/HR pairs
3. Split into train/validation
4. Start training

---

### **Manual Preparation**

If you want to prepare data manually:

```python
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Create directories
lr_dir = Path('satellite_data/lr')
hr_dir = Path('satellite_data/hr')
lr_dir.mkdir(exist_ok=True, parents=True)
hr_dir.mkdir(exist_ok=True, parents=True)

# Find all images
images = list(Path('satellite_data').rglob('*.tif'))
images += list(Path('satellite_data').rglob('*.png'))
images += list(Path('satellite_data').rglob('*.jpg'))

# Process each image
for idx, img_path in enumerate(tqdm(images)):
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Create HR (256x256)
        img_hr = img.resize((256, 256), Image.BICUBIC)
        
        # Create LR (64x64) - simulates Sentinel-2
        img_lr = img_hr.resize((64, 64), Image.BICUBIC)
        
        # Save
        img_hr.save(hr_dir / f'sat_{idx:04d}.png')
        img_lr.save(lr_dir / f'sat_{idx:04d}.png')
    except:
        continue

print(f"Created {len(list(lr_dir.glob('*.png')))} image pairs")
```

---

## 🎯 Data Quality Guidelines

### **What Makes Good Training Data**

✅ **Good**:
- Real satellite imagery
- Urban + rural + vegetation mix
- High resolution source images
- Consistent lighting/quality
- Minimal cloud cover

❌ **Avoid**:
- Synthetic geometric shapes
- Random colored patterns
- Heavily clouded images
- Low-quality scans
- Non-satellite imagery

---

## 📁 Expected Directory Structure

After data preparation:

```
satellite_data/
├── lr/                      # Low-resolution (64x64)
│   ├── sat_0000.png
│   ├── sat_0001.png
│   └── ...
├── hr/                      # High-resolution (256x256)
│   ├── sat_0000.png
│   ├── sat_0001.png
│   └── ...
└── extracted/              # Original downloaded data
    └── ...
```

---

## 🔍 Data Validation

Check your data quality:

```python
from PIL import Image
import matplotlib.pyplot as plt

# Load a sample pair
lr_img = Image.open('satellite_data/lr/sat_0000.png')
hr_img = Image.open('satellite_data/hr/sat_0000.png')

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(lr_img)
ax1.set_title('LR (64x64)')
ax2.imshow(hr_img)
ax2.set_title('HR (256x256)')
plt.show()

# Verify
print(f"LR size: {lr_img.size}")  # Should be (64, 64)
print(f"HR size: {hr_img.size}")  # Should be (256, 256)
```

**What to look for**:
- LR should show satellite features (vegetation, urban, etc.)
- HR should be sharper version of LR
- Colors should match (no extreme shifts)
- Both should be recognizable satellite imagery

---

## ⚡ Quick Data Setup Commands

### **For Google Colab**:
```bash
# Just run the training script - it handles everything!
!git clone https://github.com/Bharath-2005-07/ResolutionOf-Satellite.git
%cd ResolutionOf-Satellite
!python train_satellite_colab.py
```

### **For Local Machine**:
```bash
# Download UC Merced
wget http://weegee.vision.ucmerced.edu/datasets/landuse.zip
unzip landuse.zip -d satellite_data

# Or clone WorldStrat
git clone https://github.com/worldstrat/worldstrat

# Train
python train_satellite_colab.py
```

---

## 📊 Dataset Statistics

Recommended minimums for good results:

| Dataset Size | Training Time | Expected PSNR |
|--------------|---------------|---------------|
| 100 pairs | 30 min | 24-26 dB |
| 300 pairs | 1-2 hours | 26-28 dB |
| 1000+ pairs | 2-4 hours | 28-30 dB |

---

## 🚨 Common Issues

### **"Dataset not found"**
- Check directory structure
- Ensure LR and HR folders exist
- Verify images are .png format

### **"Images look synthetic"**
- You're using the fallback synthetic generator
- Download real satellite data instead
- Check data download completed successfully

### **"Poor results after training"**
- Ensure using real satellite imagery
- Check data quality (no corrupted images)
- Verify LR/HR pairs match correctly

---

## 📚 Additional Resources

- **UC Merced Dataset**: http://weegee.vision.ucmerced.edu/datasets/
- **WorldStrat**: https://github.com/worldstrat/worldstrat
- **Google Earth Engine**: https://earthengine.google.com/
- **Sentinel-2**: https://sentinel.esa.int/web/sentinel/missions/sentinel-2

---

## ✅ Checklist Before Training

- [ ] Data downloaded
- [ ] LR/HR pairs created
- [ ] Verified image quality (use visualization above)
- [ ] At least 100-300 image pairs
- [ ] Images are real satellite imagery (not synthetic shapes)
- [ ] Directory structure matches expected format

**Ready to train!** → Run `python train_satellite_colab.py`
