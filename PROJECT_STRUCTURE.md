# 📁 Project Structure & Module Documentation

> Complete guide to the Satellite Super-Resolution codebase

---

## 🗂️ Folder Structure

```
ResolutionOf-Satellite/
│
├── 📂 app/                          # Web Application
│   └── app.py                       # Streamlit interface for demo
│
├── 📂 models/                       # Neural Network Architectures
│   ├── edsr.py                      # EDSR (Enhanced Deep SR)
│   ├── esrgan.py                    # ESRGAN-Lite (our main model)
│   └── __init__.py                  # Module exports
│
├── 📂 training/                     # Training Pipeline
│   ├── train.py                     # Original training script
│   ├── train_colab.py               # Complete Colab training script
│   ├── losses.py                    # Loss functions (L1, Perceptual, Edge)
│   ├── metrics.py                   # PSNR, SSIM evaluation metrics
│   ├── README.md                    # Training documentation
│   └── QUICKSTART.md                # Quick training guide
│
├── 📂 inference/                    # Inference & Testing
│   ├── infer_patch.py               # Single patch inference
│   ├── stitch.py                    # Tiled inference for large images
│   └── __init__.py                  # Module exports
│
├── 📂 utils/                        # Utility Functions
│   ├── tiling.py                    # Image tiling & stitching
│   ├── guards.py                    # Hallucination guardrails
│   ├── preprocessing.py             # Data normalization & augmentation
│   ├── verify_setup.py              # Setup verification script
│   └── __init__.py                  # Module exports
│
├── 📂 data/                         # Data Handling
│   ├── dataset.py                   # PyTorch Dataset classes
│   ├── gee_fetch.py                 # Google Earth Engine integration
│   ├── README.md                    # Data documentation
│   └── satellite_raw/               # Raw satellite image storage
│
├── 📂 notebooks/                    # Jupyter Notebooks
│   └── Complete_Satellite_Training.ipynb  # Colab training notebook
│
├── 📂 checkpoints/                  # Saved Models
│   └── best_model.pth               # Trained model (PSNR: 26.83dB)
│
├── 📂 outputs/                      # Test Results
│   ├── sr_*.png                     # Super-resolution outputs
│   └── comparison_*.png             # LR vs SR vs HR comparisons
│
├── 📂 results/                      # Training Visualizations
│   └── comparison_epoch_*.png       # Progress at epochs 1, 5, 10, 15
│
├── 📂 satellite_data/               # Training Data
│   └── processed/                   # Preprocessed LR/HR pairs
│       ├── lr/                      # Low-resolution (64×64)
│       └── hr/                      # High-resolution (256×256)
│
├── 📜 README.md                     # Main project documentation
├── 📜 TRAINING_LOG.md               # Detailed training results
├── 📜 PROJECT_STRUCTURE.md          # This file
├── 📜 requirements.txt              # Python dependencies
├── 📜 test_locations.py             # Test script for categories
├── 📜 prepare_manual_data.py        # Data preparation script
└── 📜 training_history.png          # Training curves graph
```

---

## 📦 Module Details

### 🎨 `models/` - Neural Network Architectures

| File | Purpose | Key Components |
|------|---------|----------------|
| **esrgan.py** | Main SR model | `ESRGANLite` - 6.1M params, 8 RRDB blocks |
| **edsr.py** | Alternative model | `EDSR` - baseline architecture |

**ESRGANLite Architecture:**
```
Input (64×64×3) → Conv2d (64) → 8×RRDB Blocks → Upscale 2× → Upscale 2× → Output (256×256×3)
```

---

### 🏋️ `training/` - Training Pipeline

| File | Purpose | Key Functions |
|------|---------|---------------|
| **train_colab.py** | Complete training | Downloads data, trains model, saves checkpoints |
| **losses.py** | Loss functions | `L1Loss`, `PerceptualLoss`, `EdgeLoss` |
| **metrics.py** | Evaluation | `calculate_psnr()`, `calculate_ssim()` |

**Loss Formulation:**
```python
Total_Loss = 1.0×L1 + 0.1×VGG_Perceptual + 0.1×Edge
```

---

### 🔍 `inference/` - Inference Module

| File | Purpose | Key Functions |
|------|---------|---------------|
| **infer_patch.py** | Single image | `inference_single()` |
| **stitch.py** | Tiled inference | `process_large_image()` for images > 256×256 |

---

### 🛠️ `utils/` - Utility Functions

| File | Purpose | Key Functions |
|------|---------|---------------|
| **tiling.py** | Image tiling | `extract_tiles()`, `stitch_tiles()` |
| **guards.py** | Hallucination prevention | `apply_guardrail()` - semantic check |
| **preprocessing.py** | Data processing | `normalize()`, `denormalize()`, `augment()` |

**Guardrail Checks:**
1. Semantic Consistency - Downscaled SR ≈ LR
2. Edge Preservation - SR edges align with LR
3. Color Distribution - No extreme shifts
4. Structure Integrity - No phantom features

---

### 📊 `data/` - Data Handling

| File | Purpose | Key Classes |
|------|---------|-------------|
| **dataset.py** | Data loading | `SatelliteDataset` - PyTorch Dataset |
| **gee_fetch.py** | GEE integration | `fetch_patch()` - download satellite tiles |

---

### 💻 `app/` - Web Interface

| File | Purpose | Features |
|------|---------|----------|
| **app.py** | Streamlit app | Image upload, SR processing, comparison slider |

**Run with:**
```bash
streamlit run app/app.py
```

---

## 📁 Output Folders

### `outputs/` - Test Results
Contains super-resolution results from `test_locations.py`:
- `sr_01_agricultural.png` - Agricultural land SR
- `sr_03_buildings.png` - Urban buildings SR
- `sr_06_forest.png` - Forest area SR
- `sr_08_freeway.png` - Highway/freeway SR
- `comparison_*.png` - Side-by-side LR|SR|HR comparisons

### `results/` - Training Visualizations
Contains training progress visualizations:
- `comparison_epoch_1.png` - Initial model quality
- `comparison_epoch_5.png` - Early training
- `comparison_epoch_10.png` - Mid training
- `comparison_epoch_15.png` - Final quality

### `checkpoints/` - Model Weights
Contains trained model checkpoints:
- `best_model.pth` - Best model (PSNR: 26.83dB, SSIM: 0.8939)

---

## 🔄 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  satellite_data/          models/           checkpoints/        │
│  ┌─────────┐             ┌─────────┐        ┌─────────┐        │
│  │ HR(256) │──┬──────────│ ESRGAN  │───────▶│ .pth    │        │
│  │ LR(64)  │──┘          │  Lite   │        │ weights │        │
│  └─────────┘             └─────────┘        └─────────┘        │
│       │                       │                                 │
│       │                       │                                 │
│  training/               training/            results/          │
│  ┌─────────┐             ┌─────────┐        ┌─────────┐        │
│  │ losses  │◀────────────│ train_  │───────▶│ visual  │        │
│  │ metrics │             │ colab.py│        │ compare │        │
│  └─────────┘             └─────────┘        └─────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Image           checkpoints/           outputs/          │
│  ┌─────────┐          ┌─────────┐           ┌─────────┐        │
│  │ LR.png  │──────────│ model   │──────────▶│ SR.png  │        │
│  │ 64×64   │          │  .pth   │           │ 256×256 │        │
│  └─────────┘          └─────────┘           └─────────┘        │
│       │                    │                     │              │
│       │               inference/                 │              │
│       │              ┌─────────┐                │              │
│       └──────────────│ stitch  │────────────────┘              │
│                      │ infer   │                               │
│                      └─────────┘                               │
│                           │                                    │
│                      utils/guards                              │
│                     (hallucination                             │
│                       check)                                   │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Commands

```bash
# Train model
python training/train_colab.py

# Test on categories
python test_locations.py

# Run web app
streamlit run app/app.py

# Single inference
python inference/stitch.py --input image.png --output sr_image.png
```

---

*Last Updated: February 2, 2026*
