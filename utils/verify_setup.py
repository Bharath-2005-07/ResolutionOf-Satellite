"""
Test Script - Verify All Fixes Are Working

Run this to check that everything is set up correctly
"""

import sys
import os
from pathlib import Path

print("="*80)
print("SATELLITE SUPER-RESOLUTION - VERIFICATION TEST")
print("="*80)

# Test 1: Check project structure
print("\n1. Checking project structure...")
required_dirs = ['models', 'data', 'training', 'inference', 'utils', 'app', 'notebooks']
missing_dirs = []

for dir_name in required_dirs:
    if not Path(dir_name).exists():
        missing_dirs.append(dir_name)
        print(f"  ❌ Missing: {dir_name}/")
    else:
        print(f"  ✅ Found: {dir_name}/")

if missing_dirs:
    print(f"\n⚠️  Missing directories: {missing_dirs}")
    print("   Make sure you're in the ResolutionOf-Satellite directory")
else:
    print("\n✅ All directories present!")

# Test 2: Check imports
print("\n2. Checking imports...")
try:
    import torch
    print(f"  ✅ PyTorch: {torch.__version__}")
except ImportError:
    print("  ❌ PyTorch not installed: pip install torch torchvision")

try:
    import torchvision
    print(f"  ✅ TorchVision: {torchvision.__version__}")
except ImportError:
    print("  ❌ TorchVision not installed")

try:
    import numpy as np
    print(f"  ✅ NumPy: {np.__version__}")
except ImportError:
    print("  ❌ NumPy not installed: pip install numpy")

try:
    from PIL import Image
    print(f"  ✅ Pillow installed")
except ImportError:
    print("  ❌ Pillow not installed: pip install Pillow")

try:
    import cv2
    print(f"  ✅ OpenCV: {cv2.__version__}")
except ImportError:
    print("  ❌ OpenCV not installed: pip install opencv-python-headless")

# Test 3: Check models
print("\n3. Checking models...")
try:
    from models.esrgan import ESRGANLite
    model = ESRGANLite(scale_factor=4)
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ ESRGANLite: {params:,} parameters")
except Exception as e:
    print(f"  ❌ ESRGANLite import failed: {e}")

try:
    from models.edsr import EDSR
    model = EDSR(scale_factor=4)
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ EDSR: {params:,} parameters")
except Exception as e:
    print(f"  ❌ EDSR import failed: {e}")

# Test 4: Check datasets
print("\n4. Checking datasets...")
try:
    from data.dataset import SatelliteSRDataset, SyntheticSRDataset
    print("  ✅ SatelliteSRDataset")
    print("  ✅ SyntheticSRDataset")
except Exception as e:
    print(f"  ❌ Dataset imports failed: {e}")

try:
    from data import get_satellite_dataset
    print("  ✅ get_satellite_dataset() helper")
except Exception as e:
    print(f"  ❌ Helper function failed: {e}")

# Test 5: Check losses
print("\n5. Checking loss functions...")
try:
    from training.losses import SatelliteSRLoss
    loss = SatelliteSRLoss()
    print("  ✅ SatelliteSRLoss")
except Exception as e:
    print(f"  ❌ SatelliteSRLoss failed: {e}")

try:
    from training.losses import VGGPerceptualLoss
    loss = VGGPerceptualLoss()
    print("  ✅ VGGPerceptualLoss")
except Exception as e:
    print(f"  ❌ VGGPerceptualLoss failed: {e}")

# Test 6: Check metrics
print("\n6. Checking metrics...")
try:
    from training.metrics import calculate_psnr, calculate_ssim
    print("  ✅ PSNR calculation")
    print("  ✅ SSIM calculation")
except Exception as e:
    print(f"  ❌ Metrics failed: {e}")

# Test 7: Check training pipeline
print("\n7. Checking training pipeline...")
try:
    from training.train import Trainer, get_default_config
    config = get_default_config()
    print(f"  ✅ Trainer class")
    print(f"  ✅ Default config: {len(config)} parameters")
except Exception as e:
    print(f"  ❌ Training pipeline failed: {e}")

# Test 8: Check inference
print("\n8. Checking inference pipeline...")
try:
    from inference.stitch import SatelliteSRInference
    print("  ✅ SatelliteSRInference")
except Exception as e:
    print(f"  ❌ Inference failed: {e}")

# Test 9: Check new scripts
print("\n9. Checking new training scripts...")
if Path('train_satellite_colab.py').exists():
    print("  ✅ train_satellite_colab.py")
else:
    print("  ❌ train_satellite_colab.py not found")

if Path('notebooks/Complete_Satellite_Training.ipynb').exists():
    print("  ✅ Complete_Satellite_Training.ipynb")
else:
    print("  ❌ Complete_Satellite_Training.ipynb not found")

# Test 10: Quick model test
print("\n10. Running quick model test...")
try:
    import torch
    from models.esrgan import ESRGANLite
    
    model = ESRGANLite(scale_factor=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Test input
    test_input = torch.randn(1, 3, 64, 64).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"  ✅ Model test passed")
    print(f"     Input shape: {test_input.shape}")
    print(f"     Output shape: {output.shape}")
    print(f"     Expected: (1, 3, 256, 256)")
    
    if output.shape == (1, 3, 256, 256):
        print(f"  ✅ Output shape correct!")
    else:
        print(f"  ⚠️  Output shape mismatch")
    
except Exception as e:
    print(f"  ❌ Model test failed: {e}")

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

print("\n✅ If all checks passed, you're ready to train!")
print("\nQuick start commands:")
print("\n  Google Colab:")
print("    !git clone https://github.com/Bharath-2005-07/ResolutionOf-Satellite.git")
print("    %cd ResolutionOf-Satellite")
print("    !python train_satellite_colab.py")
print("\n  Local:")
print("    python train_satellite_colab.py")

print("\n" + "="*80)
print("GitHub: https://github.com/Bharath-2005-07/ResolutionOf-Satellite")
print("="*80)
