"""
Test Model on Different Location Types
Run: python test_locations.py

Tests on 4 satellite image categories (2 images each = 10 total):
1. Agricultural (farms, crops)
2. Buildings (residential)
3. Forest (trees, vegetation)
4. Freeway (roads, highways)

Saves only the 10 SUPER-RESOLVED (HR) images!
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import os
import sys

sys.path.insert(0, '.')
from models.esrgan import ESRGANLite

def main():
    # Create output folder
    os.makedirs('outputs', exist_ok=True)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = ESRGANLite(scale_factor=4)
    
    # Check if model exists
    model_path = 'checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Make sure training is complete!")
        return
    
    # Load checkpoint (handle both formats)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded! (PSNR: {checkpoint.get('psnr', 'N/A')}, SSIM: {checkpoint.get('ssim', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model loaded!")
    
    model.to(device)
    model.eval()
    
    # Correct paths based on your folder structure
    categories = {
        'Agricultural': 'data/satellite_raw/images/agricultural',
        'Buildings': 'data/satellite_raw/images/buildings',
        'Forest': 'data/satellite_raw/images/forest',
        'Freeway': 'data/satellite_raw/images/freeway'
    }
    
    print("="*60)
    print("🧪 TESTING ON 4 CATEGORIES (2 images each = 10 total)")
    print("="*60)
    
    saved_count = 0
    
    for cat_name, cat_path in categories.items():
        path = Path(cat_path)
        
        if not path.exists():
            print(f"⚠️  {cat_name}: folder not found at {cat_path}")
            continue
        
        # Get images from this category
        images = list(path.glob('*.png')) + list(path.glob('*.jpg'))
        
        if not images:
            print(f"⚠️  {cat_name}: no images found")
            continue
        
        print(f"\n📍 {cat_name}: found {len(images)} images")
        
        # Process 2-3 images from each category to get 10 total
        # Agricultural: 2, Buildings: 3, Forest: 2, Freeway: 3 = 10
        if cat_name == 'Buildings' or cat_name == 'Freeway':
            num_to_process = 3
        else:
            num_to_process = 2
        
        for i, img_path in enumerate(images[:num_to_process]):
            # Load HR image (original is already high-res)
            hr_img = Image.open(img_path).convert('RGB')
            
            # Resize to standard size if needed
            if hr_img.size != (256, 256):
                hr_img = hr_img.resize((256, 256), Image.BICUBIC)
            
            # Create LR (simulate Sentinel-2 at 10m)
            lr_img = hr_img.resize((64, 64), Image.BICUBIC)
            
            # Super-resolve with model
            lr = np.array(lr_img) / 255.0
            lr_tensor = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                sr_tensor = model(lr_tensor)
            
            sr = sr_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            sr = np.clip(sr * 255, 0, 255).astype(np.uint8)
            sr_img = Image.fromarray(sr)
            
            # Save ONLY the super-resolved image
            saved_count += 1
            output_name = f"sr_{saved_count:02d}_{cat_name.lower()}.png"
            sr_img.save(f'outputs/{output_name}')
            
            # Also save a comparison image (LR vs SR side-by-side)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(np.array(lr_img)); axes[0].set_title(f'Input LR\n64×64', fontsize=12); axes[0].axis('off')
            axes[1].imshow(np.array(sr_img)); axes[1].set_title(f'Super-Resolved\n256×256', fontsize=12); axes[1].axis('off')
            axes[2].imshow(np.array(hr_img)); axes[2].set_title(f'Original HR\n256×256', fontsize=12); axes[2].axis('off')
            plt.suptitle(f'🛰️ {cat_name} - Satellite Super-Resolution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'outputs/comparison_{saved_count:02d}_{cat_name.lower()}.png', dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: outputs/{output_name}")
    
    print("\n" + "="*60)
    print(f"🎉 DONE! Saved {saved_count} super-resolved images to outputs/")
    print("="*60)
    
    # List saved files
    print("\n📁 Saved files:")
    for f in sorted(os.listdir('outputs')):
        if f.endswith('.png'):
            print(f"   - outputs/{f}")
    
    print("\n📋 Next steps:")
    print("   git add outputs/ checkpoints/")
    print('   git commit -m "Add 10 super-resolved satellite images"')
    print("   git push origin main")

if __name__ == '__main__':
    main()
