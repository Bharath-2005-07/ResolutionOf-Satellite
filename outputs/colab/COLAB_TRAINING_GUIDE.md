# 🛰️ EuroSAT Super-Resolution Training Guide

This guide explains the complete training process for satellite image super-resolution using the EuroSAT dataset on Google Colab.

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | EuroSAT RGB |
| **Categories** | AnnualCrop, Forest, HerbaceousVegetation, Highway, Pasture, PermanentCrop, River, SeaLake |
| **Images per Category** | 200 |
| **Total Images** | 1,600 |
| **Training Split** | 90% (1,440 images) |
| **Validation Split** | 10% (160 images) |
| **Epochs** | 40 |
| **Batch Size** | 16 |
| **Model** | ESRGAN-Lite |
| **Parameters** | 6,128,195 |
| **GPU** | Tesla T4 (Google Colab) |
| **Transfer Learning** | Yes (loads pretrained model if available) |

## 🚀 Training Steps

### STEP 1: Extract EuroSAT Data

The training script automatically:
- Downloads the EuroSAT RGB dataset (if not present)
- Extracts `EuroSAT_RGB.zip` to `eurosat_data/` folder
- Identifies 10 land-use categories from the dataset

**Categories Found:**
- Industrial
- River
- Pasture
- SeaLake
- Forest
- HerbaceousVegetation
- PermanentCrop
- Highway
- AnnualCrop
- Residential

### STEP 2: Prepare LR/HR Pairs

**Process:**
1. Selects 8 specific categories (excluding Industrial and Residential)
2. Takes 200 images from each category
3. Creates paired data:
   - **HR (High Resolution)**: 256×256 pixels (original)
   - **LR (Low Resolution)**: 64×64 pixels (downsampled from HR)
4. Total pairs created: **1,600**

**Output Structure:**
```
outputs/colab/
├── eurosat_lr/  # 64×64 low-resolution images
└── eurosat_hr/  # 256×256 high-resolution images
```

### STEP 3: Training Process

#### Pre-Training Setup
- **Device**: CUDA (GPU acceleration)
- **VGG19 Download**: For perceptual loss computation
- **Pretrained Model Check**: Looks for `checkpoints/best_model.pth`
  - If found: Loads weights for transfer learning
  - If not found: Trains from scratch

#### Training Loop (40 Epochs)

**Loss Functions:**
- L1 Loss (pixel-wise)
- VGG Perceptual Loss
- Edge-aware Loss

**Metrics Tracked:**
- **PSNR** (Peak Signal-to-Noise Ratio) - Higher is better
- **SSIM** (Structural Similarity Index) - Closer to 1 is better
- **Training Loss** - Lower is better

#### Epoch-by-Epoch Progress

| Epoch | Train Loss | Val PSNR | Val SSIM | Status |
|-------|-----------|----------|----------|---------|
| 1 | 0.0983 | 25.00 dB | 0.8203 | ✅ SAVED |
| 2 | 0.0390 | 29.35 dB | 0.9307 | ✅ SAVED |
| 5 | 0.0191 | 34.91 dB | 0.9765 | ✅ SAVED |
| 10 | 0.0112 | 42.14 dB | 0.9912 | ✅ SAVED |
| 16 | 0.0087 | 45.23 dB | 0.9941 | ✅ SAVED |
| 22 | 0.0069 | 47.44 dB | 0.9955 | ✅ SAVED |
| 26 | 0.0075 | 48.18 dB | 0.9960 | ✅ SAVED |
| 32 | 0.0069 | 48.37 dB | 0.9963 | ✅ SAVED |
| 35 | 0.0068 | 48.69 dB | 0.9964 | ✅ SAVED |
| 37 | 0.0053 | 50.04 dB | 0.9966 | ✅ SAVED |
| **40** | **0.0067** | **50.25 dB** | **0.9967** | ✅ **BEST** |

**Final Results:**
- 🎯 **Best PSNR**: 50.25 dB (excellent quality!)
- 🎯 **Best SSIM**: 0.9967 (near-perfect structural similarity)
- ⏱️ **Time per Epoch**: ~100 seconds (~1min 40sec)
- 📁 **Model Saved**: `outputs/colab/best_model.pth`

### STEP 4: Testing on Sample Images

**Process:**
1. Loads the trained model (50.25 dB PSNR)
2. Selects 10 diverse test images from EuroSAT
3. Generates super-resolution outputs
4. Creates comparison images (LR vs SR vs HR)

**Test Categories:**
- Forest (2 images)
- River (2 images)
- Highway (2 images)
- AnnualCrop (2 images)
- SeaLake (2 images)

**Outputs Generated:**

For each test image, two files are created:

1. **Super-Resolution Result**: `sr_colab_XX_category.png`
   - Upscaled image (256×256)
   
2. **Comparison View**: `comparison_colab_XX_category.png`
   - Side-by-side: LR (64×64) | SR (256×256) | HR (256×256)
   - Shows PSNR and SSIM metrics
   - Visual quality assessment

**Example Outputs:**
```
outputs/colab/
├── sr_colab_01_forest.png
├── comparison_colab_01_forest.png
├── sr_colab_02_forest.png
├── comparison_colab_02_forest.png
├── ...
├── sr_colab_10_sealake.png
└── comparison_colab_10_sealake.png
```

## 📈 Performance Analysis

### Quality Metrics

**PSNR (Peak Signal-to-Noise Ratio):**
- **Starting**: 25.00 dB (Epoch 1)
- **Final**: 50.25 dB (Epoch 40)
- **Improvement**: +25.25 dB (101% improvement!)

**SSIM (Structural Similarity Index):**
- **Starting**: 0.8203 (Epoch 1)
- **Final**: 0.9967 (Epoch 40)
- **Improvement**: +0.1764 (21.5% improvement)

### Training Efficiency

- **Total Training Time**: ~67 minutes (40 epochs × 100 seconds)
- **GPU Utilization**: Tesla T4 (Google Colab free tier)
- **Convergence**: Stable after epoch 30
- **Best Model Selection**: Automatically saves when PSNR improves

## 🎯 Key Takeaways

1. **Transfer Learning**: If you have a pretrained model, place it at `checkpoints/best_model.pth` before training
2. **Dataset Quality**: EuroSAT provides diverse satellite imagery (8 land-use categories)
3. **4x Upscaling**: Successfully upscales 64×64 to 256×256 with minimal artifacts
4. **High PSNR**: 50+ dB is excellent for satellite imagery
5. **Near-Perfect SSIM**: 0.9967 indicates structure preservation

## 📁 Output Files

After training completes, you'll find:

```
outputs/colab/
├── COLAB_TRAINING_GUIDE.md          # This guide
├── best_model.pth                    # Trained model weights (50.25 dB)
├── eurosat_lr/                       # LR training data (64×64)
├── eurosat_hr/                       # HR training data (256×256)
├── sr_colab_01_forest.png           # Test result 1
├── comparison_colab_01_forest.png   # Comparison 1
├── sr_colab_02_forest.png           # Test result 2
├── ...                               # More test results
└── README.md                         # Folder description
```

## 🔧 How to Run

### On Google Colab

```bash
# 1. Clone the repository
!git clone https://github.com/Bharath-2005-07/ResolutionOf-Satellite.git
%cd ResolutionOf-Satellite

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Run the training script
!python train_eurosat_colab.py
```

### On Local Machine

```bash
# 1. Clone the repository
git clone https://github.com/Bharath-2005-07/ResolutionOf-Satellite.git
cd ResolutionOf-Satellite

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download EuroSAT dataset
# Place EuroSAT_RGB.zip in the project root

# 4. Run training
python train_eurosat_colab.py
```

## 🌟 Next Steps

1. **View Results**: Check `outputs/colab/` for comparison images
2. **Use Model**: Load `outputs/colab/best_model.pth` for inference
3. **Streamlit App**: Run `streamlit run app/app.py` to test interactively
4. **Fine-tune**: Adjust hyperparameters in `train_eurosat_colab.py`

## 🐛 Common Issues

**Issue**: UserWarning about missing glyphs (satellite emoji)
- **Solution**: This is just a matplotlib font warning, safe to ignore

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size in the script (default: 16)

**Issue**: Training taking too long
- **Solution**: Use Google Colab with GPU runtime (Runtime > Change runtime type > GPU)

## 📊 Comparison with Original Training

| Metric | Original (README) | EuroSAT (Colab) | Change |
|--------|------------------|------------------|--------|
| **Final PSNR** | 26.83 dB | **50.25 dB** | **+23.42 dB** ✅ |
| **Final SSIM** | 0.8939 | **0.9967** | **+0.1028** ✅ |
| **Epochs** | 15 | 40 | +25 |
| **Training Data** | 900 images | 1,440 images | +60% |
| **Categories** | 4 | 8 | +100% |

**Why Better Results?**
- More training epochs (40 vs 15)
- Larger dataset (1,600 vs 1,000 images)
- More diverse categories (8 vs 4)
- Better GPU (Tesla T4 vs CPU)

---

**Made with ❤️ for the ML Track Hackathon**
