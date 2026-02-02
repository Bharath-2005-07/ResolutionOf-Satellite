"""
Prepare manually downloaded satellite images for training
Use this after downloading dataset from Kaggle or other sources

Usage:
    python prepare_manual_data.py
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def prepare_satellite_data(
    raw_dir='data/satellite_raw',
    output_dir='satellite_data/processed',
    scale_factor=4,
    patch_size=64
):
    """
    Process manually downloaded satellite images into LR/HR pairs
    
    Args:
        raw_dir: Where you extracted the downloaded images
        output_dir: Where to save processed LR/HR pairs
        scale_factor: Upscaling factor (default: 4x)
        patch_size: Low-resolution patch size (default: 64)
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    lr_dir = output_path / 'lr'
    hr_dir = output_path / 'hr'
    
    lr_dir.mkdir(exist_ok=True, parents=True)
    hr_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all images
    image_files = []
    for ext in ['*.tif', '*.png', '*.jpg', '*.jpeg', '*.TIF', '*.PNG', '*.JPG']:
        image_files.extend(list(raw_path.rglob(ext)))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {raw_dir}")
        print(f"   Make sure you extracted the dataset to: {raw_dir}")
        print(f"   Looking for: *.tif, *.png, *.jpg, *.jpeg")
        return
    
    print(f"✅ Found {len(image_files)} images in {raw_dir}")
    print(f"📊 Processing images into LR/HR pairs...")
    print(f"   HR size: {patch_size * scale_factor}x{patch_size * scale_factor}")
    print(f"   LR size: {patch_size}x{patch_size}")
    print()
    
    hr_size = patch_size * scale_factor  # 256 for 4x
    success_count = 0
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing")):
        try:
            # Load and convert to RGB
            img = Image.open(img_path).convert('RGB')
            
            # Create HR (target quality)
            img_hr = img.resize((hr_size, hr_size), Image.BICUBIC)
            
            # Create LR (simulates Sentinel-2 quality)
            img_lr = img_hr.resize((patch_size, patch_size), Image.BICUBIC)
            
            # Save both
            hr_path = hr_dir / f'sat_{idx:04d}.png'
            lr_path = lr_dir / f'sat_{idx:04d}.png'
            
            img_hr.save(hr_path)
            img_lr.save(lr_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"⚠️  Skipped {img_path.name}: {e}")
            continue
    
    print()
    print("="*80)
    print(f"✅ SUCCESS! Created {success_count} LR/HR image pairs")
    print(f"   📂 LR images (64x64):   {lr_dir}")
    print(f"   📂 HR images (256x256): {hr_dir}")
    print()
    print("🎯 You can now run training:")
    print("   python training/train_colab.py")
    print("="*80)


if __name__ == '__main__':
    import sys
    
    # Default paths
    raw_dir = 'data/satellite_raw'
    
    # Check if user provided custom path
    if len(sys.argv) > 1:
        raw_dir = sys.argv[1]
    
    # For quick testing: Use only one folder
    # Uncomment the line below to process only buildings folder
    # raw_dir = 'data/satellite_raw/images/buildings'
    
    # Check if raw data exists
    if not Path(raw_dir).exists():
        print("="*80)
        print("📥 MANUAL DOWNLOAD REQUIRED")
        print("="*80)
        print()
        print("Download satellite dataset from one of these sources:")
        print()
        print("1. Kaggle (Recommended):")
        print("   https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification")
        print("   → Download archive.zip")
        print()
        print("2. Roboflow:")
        print("   https://universe.roboflow.com/satellite-imagery/satellite-imagery-pqdhj")
        print("   → Download in 'Folder Structure' format")
        print()
        print("3. WorldStrat (Best quality but larger):")
        print("   https://zenodo.org/record/6810792")
        print("   → Download worldstrat_train.zip")
        print()
        print(f"Then extract to: {os.path.abspath(raw_dir)}")
        print()
        print("After extracting, run this script again:")
        print(f"   python {sys.argv[0]}")
        print("="*80)
        sys.exit(1)
    
    prepare_satellite_data(raw_dir=raw_dir)
