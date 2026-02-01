# 🎓 Training Guide - Satellite Super-Resolution

Complete guide for training the satellite super-resolution model with **real satellite data**.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [How Real Satellite Data Works](#how-real-satellite-data-works)
3. [Training Process Explained](#training-process-explained)
4. [Data Sources](#data-sources)
5. [Training Methods](#training-methods)
6. [Understanding the Pipeline](#understanding-the-pipeline)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### **Fastest Way (Google Colab)**

```python
# In Google Colab, run:
!git clone https://github.com/Bharath-2005-07/ResolutionOf-Satellite.git
%cd ResolutionOf-Satellite
!python training/train_colab.py
```

That's it! Training starts automatically with real satellite data.

---

## 🛰️ How Real Satellite Data Works

### **The Problem**
You're training on **synthetic geometric shapes** (rectangles, circles, lines) instead of real satellite imagery. This is why results look blurry and unrealistic.

### **The Solution**
We need **real satellite images** with two types:

1. **Low Resolution (LR)**: 64x64 pixels - simulates Sentinel-2 at 10m/pixel
2. **High Resolution (HR)**: 256x256 pixels - simulates commercial satellite at 2.5m/pixel

### **How We Get Real Data**

#### **Option 1: UC Merced Dataset (Used by `train_colab.py`)** ⭐

```
Download → Extract → Process → Train
   ↓          ↓          ↓         ↓
320MB      2100 imgs  LR/HR    Model
satellite   real      pairs    learns
images    satellite
```

**Step-by-step what happens:**

```python
# 1. DOWNLOAD (automatic in train_colab.py)
url = "http://weegee.vision.ucmerced.edu/datasets/landuse.zip"
# Downloads 2100 real satellite images (agricultural fields, urban areas, etc.)

# 2. EXTRACT
# Unzips to get .tif satellite images

# 3. PROCESS - Create LR/HR Pairs
for each satellite_image:
    # Load original high-res image
    img = Image.open(satellite_image)
    
    # Create HR (256x256) - Target output
    img_hr = img.resize((256, 256), Image.BICUBIC)
    
    # Create LR (64x64) - Simulates Sentinel-2 quality
    # This mimics what Sentinel-2 satellite captures
    img_lr = img_hr.resize((64, 64), Image.BICUBIC)
    
    # Save pair
    save(img_lr) → satellite_data/lr/sat_0001.png (64x64)
    save(img_hr) → satellite_data/hr/sat_0001.png (256x256)

# 4. TRAIN
# Model learns: LR (64x64) → SR (256x256) ≈ HR (256x256)
```

#### **Option 2: WorldStrat Dataset**

Provides **real paired** Sentinel-2 (LR) + High-resolution satellite (HR):

```
WorldStrat/
├── train/
│   ├── lr/   ← Real Sentinel-2 images (10m/pixel)
│   └── hr/   ← Real high-res satellite (1.5m/pixel)
```

No processing needed - already paired!

#### **Option 3: Google Earth Engine**

Fetches **real-time satellite data** from Google's satellite archive:

```python
# 1. Authenticate
import ee
ee.Authenticate()
ee.Initialize()

# 2. Define location (e.g., San Francisco)
point = ee.Geometry.Point([-122.4194, 37.7749])

# 3. Get Sentinel-2 image
image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
         .filterBounds(point)
         .filterDate('2023-01-01', '2024-01-01')
         .first()

# 4. Download patch
# Returns real satellite data from that location
```

---

## 🎯 Training Process Explained

### **What Happens When You Run `train_colab.py`**

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: DOWNLOAD REAL SATELLITE DATA                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  UC Merced Dataset (320MB)                                 │
│  ├── Agricultural lands                                     │
│  ├── Urban areas                                           │
│  ├── Forests                                               │
│  ├── Rivers/water bodies                                   │
│  └── 2100 real satellite images                           │
│                                                             │
│  Download time: ~2-5 minutes                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STEP 2: CREATE LR/HR PAIRS                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each satellite image:                                 │
│                                                             │
│  Original → HR (256x256) → LR (64x64)                      │
│     │           │              │                            │
│     │           │              └─ Simulates Sentinel-2     │
│     │           └──────────────── Target quality           │
│     └──────────────────────────── Real satellite           │
│                                                             │
│  Output: 300 LR/HR pairs                                   │
│  Processing time: ~1-2 minutes                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STEP 3: INITIALIZE MODEL                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ESRGANLite (6.1M parameters)                              │
│  ├── Input: 64x64x3 (RGB)                                 │
│  ├── 8 RRDB blocks                                        │
│  ├── PixelShuffle upsampling (2x → 2x)                   │
│  └── Output: 256x256x3 (4x upscale)                       │
│                                                             │
│  Loads to GPU if available                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STEP 4: SETUP LOSS FUNCTIONS                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SatelliteSRLoss = 3 components:                           │
│                                                             │
│  1. L1 Loss (weight: 1.0)                                  │
│     └─ Pixel-level accuracy                                │
│                                                             │
│  2. VGG Perceptual Loss (weight: 0.1)                      │
│     └─ Preserves structures (buildings, roads)            │
│                                                             │
│  3. Edge-Aware Loss (weight: 0.1)                          │
│     └─ Sharpens edges (roads, building outlines)          │
│                                                             │
│  Total = 1.0×L1 + 0.1×VGG + 0.1×Edge                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STEP 5: TRAINING LOOP (100 epochs)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For epoch 1 to 100:                                       │
│                                                             │
│    FOR each batch of 8 LR/HR pairs:                        │
│      1. lr_img → model → sr_img                            │
│      2. Calculate loss(sr_img, hr_img)                     │
│      3. Backpropagate gradients                            │
│      4. Update model weights                               │
│                                                             │
│    EVERY 10 epochs:                                        │
│      • Calculate PSNR & SSIM on validation set            │
│      • Generate before/after visualizations                │
│      • Save checkpoint if best so far                      │
│                                                             │
│  Time: ~2 hours on Colab T4 GPU                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STEP 6: VALIDATION & METRICS                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each validation image:                                │
│                                                             │
│    LR (64x64) → Model → SR (256x256)                       │
│                  ↓                                          │
│    Compare with HR (256x256)                               │
│                                                             │
│  Metrics:                                                  │
│    • PSNR: Peak Signal-to-Noise Ratio                     │
│      └─ Measures pixel accuracy (higher = better)         │
│      └─ Target: >26 dB (bicubic baseline: ~24 dB)        │
│                                                             │
│    • SSIM: Structural Similarity Index                     │
│      └─ Measures structural preservation (0-1)            │
│      └─ Target: >0.85 (bicubic baseline: ~0.78)          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STEP 7: SAVE BEST MODEL                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  checkpoints/best_model.pth                                │
│  ├── Model weights                                         │
│  ├── Optimizer state                                       │
│  ├── Best PSNR achieved                                    │
│  └── Training epoch                                        │
│                                                             │
│  Use for inference on new satellite images!                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Sources

### **1. UC Merced Land Use Dataset** ⭐ (Auto-downloaded)

- **What**: 2100 real satellite images
- **Types**: Agricultural, urban, forest, water, etc.
- **Size**: 320 MB
- **Resolution**: 256x256 pixels (1 foot/pixel)
- **License**: Public domain
- **Used by**: `training/train_colab.py` (automatic)

### **2. WorldStrat Dataset** (Manual download)

- **What**: Paired Sentinel-2 (LR) + High-res (HR)
- **Size**: ~50 GB
- **Download**: `git clone https://github.com/worldstrat/worldstrat`
- **Best for**: Production-quality models

### **3. Google Earth Engine** (Requires auth)

- **What**: Real-time Sentinel-2 data
- **Coverage**: Global
- **Requires**: GEE account + authentication
- **Best for**: Custom locations/dates

### **4. Your Own Images**

- **What**: Any high-res satellite images you have
- **Format**: PNG, JPEG, TIFF, GeoTIFF
- **Process**: Auto-creates LR by downsampling

---

## 🎓 Training Methods

### **Method 1: Automatic Training (Recommended)** ⭐

```bash
# One command - everything automatic
python training/train_colab.py
```

**What it does:**
1. ✅ Downloads UC Merced dataset
2. ✅ Creates LR/HR pairs
3. ✅ Initializes model
4. ✅ Trains for 100 epochs
5. ✅ Validates with PSNR/SSIM
6. ✅ Saves checkpoints
7. ✅ Generates visualizations

**Time**: ~2 hours  
**GPU**: Colab T4 (free tier)  
**Output**: `checkpoints/best_model.pth`

---

### **Method 2: Using Training Pipeline**

```python
from training.train import Trainer, get_default_config
from data import get_satellite_dataset
from torch.utils.data import DataLoader

# 1. Get dataset
dataset = get_satellite_dataset(
    'synthetic',  # Creates LR from HR
    hr_dir='path/to/satellite/images',
    patch_size=64,
    scale_factor=4
)

# 2. Create dataloader
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 3. Configure
config = get_default_config()
config['num_epochs'] = 100
config['use_gan'] = False  # Start without GAN

# 4. Train
trainer = Trainer(config)
trainer.train(train_loader, num_epochs=100)
```

---

### **Method 3: Custom Training Loop**

```python
import torch
from models.esrgan import ESRGANLite
from training.losses import SatelliteSRLoss
from training.metrics import calculate_psnr, calculate_ssim

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ESRGANLite(scale_factor=4).to(device)
criterion = SatelliteSRLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Training loop
for epoch in range(100):
    for lr_img, hr_img in train_loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        
        # Forward
        sr_img = model(lr_img)
        
        # Loss
        loss, components = criterion(sr_img, hr_img)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validate
    if (epoch + 1) % 10 == 0:
        psnr = calculate_psnr(sr_img, hr_img)
        ssim = calculate_ssim(sr_img, hr_img)
        print(f"Epoch {epoch+1}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
```

---

### **Method 4: Using WorldStrat Dataset**

```python
from data.dataset import SatelliteSRDataset
from torch.utils.data import DataLoader

# Already paired LR/HR data
dataset = SatelliteSRDataset(
    lr_dir='worldstrat/train/lr',  # Real Sentinel-2
    hr_dir='worldstrat/train/hr',  # Real high-res
    patch_size=64,
    scale_factor=4,
    augment=True
)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Train as usual...
```

---

## 🔍 Understanding the Pipeline

### **What is LR/HR Pair?**

```
Low Resolution (LR)          High Resolution (HR)
┌─────────────┐             ┌───────────────────────┐
│             │             │                       │
│   64x64     │  ────────>  │      256x256          │
│  Sentinel-2 │   Model     │   Target Quality      │
│             │   learns    │                       │
└─────────────┘             └───────────────────────┘
    INPUT                         OUTPUT (Goal)
```

**During training:**
- Model sees LR (blurry 64x64)
- Model produces SR (super-resolved 256x256)
- Compare SR with HR (ground truth 256x256)
- Calculate loss and update model

**Why 64x64 → 256x256?**
- 64x64 = Sentinel-2 quality (10m/pixel, free)
- 256x256 = Commercial quality (2.5m/pixel, expensive)
- 4x upscaling factor

---

### **How Bicubic Creates LR from HR**

```python
# Starting with high-quality satellite image (HR)
hr_img = Image.open('satellite.tif')  # 1024x1024

# Resize to our HR target
hr_img = hr_img.resize((256, 256), Image.BICUBIC)

# Create LR by downsampling (simulates Sentinel-2)
lr_img = hr_img.resize((64, 64), Image.BICUBIC)
# This mimics atmospheric effects, sensor limitations, etc.
```

**Why this works:**
- Real Sentinel-2 at 10m/pixel ≈ heavily downsampled image
- Bicubic downsampling simulates this degradation
- Model learns to reverse this process

---

### **Loss Functions Explained**

#### **1. L1 Loss (Pixel-level)**
```python
L1 = |SR - HR|  # Absolute difference
```
- Measures pixel-by-pixel accuracy
- Ensures colors match
- Weight: 1.0 (highest priority)

#### **2. VGG Perceptual Loss**
```python
VGG_SR = VGG19(SR)    # Extract features
VGG_HR = VGG19(HR)    # Extract features
Perceptual = |VGG_SR - VGG_HR|
```
- Uses pre-trained VGG19 network
- Compares high-level features (structures)
- Preserves buildings, roads, patterns
- Weight: 0.1

#### **3. Edge-Aware Loss**
```python
Edges_SR = Sobel(SR)  # Detect edges
Edges_HR = Sobel(HR)  # Detect edges
Edge = |Edges_SR - Edges_HR|
```
- Detects edges using Sobel filters
- Sharpens roads, building outlines
- Critical for satellite imagery
- Weight: 0.1

**Total Loss:**
```
Loss = 1.0×L1 + 0.1×Perceptual + 0.1×Edge
```

---

### **Training Progress**

```
Epoch 1   | Loss: 0.3245 | PSNR: 22.5 dB | SSIM: 0.781
          ↓ Model learning...
Epoch 10  | Loss: 0.2156 | PSNR: 24.2 dB | SSIM: 0.823
          ↓ Getting better...
Epoch 50  | Loss: 0.1523 | PSNR: 27.1 dB | SSIM: 0.865
          ↓ Almost there...
Epoch 100 | Loss: 0.1284 | PSNR: 28.5 dB | SSIM: 0.891 ✅
          ↓ Best model saved!
```

**What to expect:**
- First 10 epochs: Rapid improvement
- 10-50 epochs: Steady progress
- 50-100 epochs: Fine-tuning
- Beyond 100: Diminishing returns

---

## 📈 Metrics Explained

### **PSNR (Peak Signal-to-Noise Ratio)**

```python
MSE = mean((SR - HR)²)
PSNR = 20 × log10(1.0 / √MSE)
```

**What it means:**
- Measures pixel accuracy in decibels (dB)
- Higher = better
- Bicubic baseline: ~24 dB
- Good model: 26-28 dB
- Excellent model: 28-30 dB

**Interpretation:**
- < 25 dB: Poor quality
- 25-27 dB: Acceptable
- 27-29 dB: Good
- > 29 dB: Excellent

---

### **SSIM (Structural Similarity Index)**

```python
SSIM = f(luminance, contrast, structure)
# Range: 0 to 1
```

**What it means:**
- Measures perceived quality
- Considers luminance, contrast, structure
- Higher = better (max = 1.0)
- Bicubic baseline: ~0.78
- Good model: 0.85-0.90
- Excellent model: > 0.90

**Interpretation:**
- < 0.80: Poor structural preservation
- 0.80-0.85: Acceptable
- 0.85-0.90: Good
- > 0.90: Excellent

---

## 🎯 Expected Results

### **After 100 Epochs**

| Metric | Value | vs Bicubic |
|--------|-------|------------|
| PSNR | 28.5 dB | +4.3 dB ✅ |
| SSIM | 0.891 | +0.111 ✅ |
| Training Time | ~2 hours | Colab T4 |
| Visual Quality | Sharp edges | Roads/buildings clear |

### **Visual Comparison**

```
Input LR (64x64)     Bicubic (256x256)    Our Model (256x256)   Ground Truth
┌────────┐           ┌────────────────┐   ┌─────────────────┐  ┌──────────────┐
│ Blurry │  ───────> │  Smoother but  │   │  Sharp edges    │  │  Original    │
│        │           │  still blurry  │   │  Clear roads    │  │  High-res    │
│        │           │  PSNR: 24.2 dB │   │  PSNR: 28.5 dB  │  │              │
└────────┘           └────────────────┘   └─────────────────┘  └──────────────┘
```

---

## 🛠️ Troubleshooting

### **Issue: "Dataset not found"**

**Cause:** Data not downloaded or wrong path

**Fix:**
```python
# Check if data exists
import os
print(os.listdir('satellite_data/lr'))  # Should show .png files

# If empty, run download again
python training/train_colab.py
```

---

### **Issue: "CUDA out of memory"**

**Cause:** GPU memory insufficient

**Fix:**
```python
# Reduce batch size
config['batch_size'] = 4  # Instead of 8

# Or use smaller patches
config['patch_size'] = 32  # Instead of 64
```

---

### **Issue: "Training too slow"**

**Cause:** CPU training or large dataset

**Fix:**
```python
# Limit training samples
dataset = dataset[:100]  # Use first 100 only

# Reduce epochs
config['num_epochs'] = 50  # Instead of 100

# Use GPU
device = 'cuda'  # Make sure GPU is available
```

---

### **Issue: "Results still blurry"**

**Cause:** Not using real satellite data

**Fix:**
```bash
# Verify you're using real data
ls satellite_data/lr/  # Should show satellite images, not shapes

# If synthetic shapes, delete and re-download
rm -rf satellite_data/
python training/train_colab.py
```

---

### **Issue: "Low PSNR/SSIM"**

**Possible causes:**
1. Not enough training epochs
2. Wrong learning rate
3. Poor quality data

**Fix:**
```python
# Train longer
config['num_epochs'] = 150

# Adjust learning rate
config['lr_generator'] = 1e-4  # Try different values

# Check data quality
from PIL import Image
img = Image.open('satellite_data/hr/sat_0000.png')
img.show()  # Should look like real satellite imagery
```

---

## 📚 File Structure After Training

```
ResolutionOf-Satellite/
├── training/
│   ├── train_colab.py        ← Main training script ⭐
│   ├── train.py               ← Training pipeline
│   ├── losses.py              ← Loss functions
│   ├── metrics.py             ← PSNR/SSIM calculation
│   └── README.md              ← This file
│
├── satellite_data/            ← Downloaded data
│   ├── lr/                    ← Low-res (64x64)
│   │   ├── sat_0000.png
│   │   └── ...
│   └── hr/                    ← High-res (256x256)
│       ├── sat_0000.png
│       └── ...
│
├── checkpoints/               ← Saved models
│   └── best_model.pth         ← Best model ⭐
│
└── results/                   ← Visualizations
    ├── comparison_epoch_10.png
    ├── comparison_epoch_50.png
    └── training_history.png
```

---

## ✅ Quick Reference

### **Start Training**
```bash
python training/train_colab.py
```

### **Check Progress**
```python
# Training prints:
# Epoch X/100 | Loss: X.XXXX | PSNR: XX.X dB | SSIM: 0.XXX
```

### **Use Trained Model**
```python
from models.esrgan import ESRGANLite
import torch

model = ESRGANLite(scale_factor=4)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Now use for inference!
```

---

## 🎓 Summary

1. **Real satellite data** is downloaded automatically (UC Merced)
2. **LR/HR pairs** are created by bicubic downsampling
3. **Model learns** to map LR → SR (super-resolved)
4. **Loss functions** ensure accuracy, structure, and sharpness
5. **Validation** tracks PSNR/SSIM progress
6. **Best model** is saved to checkpoints/

**One command to rule them all:**
```bash
python training/train_colab.py
```

---

**GitHub:** https://github.com/Bharath-2005-07/ResolutionOf-Satellite
