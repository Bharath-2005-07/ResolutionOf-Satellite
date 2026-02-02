"""
🛰️ EuroSAT Training Script for Google Colab
============================================
Train super-resolution model on EuroSAT categories (excluding buildings)

Usage in Colab:
1. Clone repo: !git clone https://github.com/Bharath-2005-07/ResolutionOf-Satellite.git
2. cd into it: %cd ResolutionOf-Satellite
3. Upload your EuroSAT_RGB.zip to this folder
4. Run: !python train_eurosat_colab.py
5. Results are pushed directly to your GitHub repo!

Results saved to: outputs/colab/
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2
import zipfile
import random
import subprocess

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Categories to use (NO buildings - excluded Residential, Industrial)
    'categories': [
        'AnnualCrop',
        'Forest', 
        'HerbaceousVegetation',
        'Highway',
        'Pasture',
        'PermanentCrop',
        'River',
        'SeaLake'
    ],
    'images_per_category': 200,  # Total: 200 × 8 = 1600 images
    'epochs': 40,                # Increase to 50-100 for better results
    'batch_size': 16,
    'learning_rate': 1e-4,
    'val_split': 0.1,
    'scale': 4,
    'hr_size': 256,
    'lr_size': 64,
    
    # ========================================
    # 🔥 TRANSFER LEARNING - USE EXISTING MODEL
    # ========================================
    'use_pretrained': True,      # Uses your existing trained model!
    'pretrained_path': 'checkpoints/best_model.pth',
}

# Paths
EUROSAT_ZIP = 'EuroSAT_RGB.zip'  # Your uploaded zip file
OUTPUT_DIR = Path('outputs/colab')
DATA_DIR = Path('satellite_data/processed')

# =============================================================================
# STEP 1: EXTRACT EUROSAT DATA
# =============================================================================

def extract_eurosat():
    """Extract EuroSAT zip file"""
    print("\n" + "="*60)
    print("STEP 1: EXTRACT EUROSAT DATA")
    print("="*60)
    
    # Find zip file
    zip_files = list(Path('.').glob('*EuroSAT*.zip')) + list(Path('.').glob('*eurosat*.zip'))
    if not zip_files:
        print("❌ No EuroSAT zip file found!")
        print("   Please upload EuroSAT_RGB.zip to the current directory")
        sys.exit(1)
    
    zip_path = zip_files[0]
    print(f"📦 Found: {zip_path}")
    
    # Extract
    extract_dir = Path('eurosat_data')
    if not extract_dir.exists():
        print(f"📦 Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        print("✅ Extraction complete!")
    else:
        print("✅ Already extracted!")
    
    # Find the actual data directory
    eurosat_dir = None
    for candidate in [
        extract_dir / 'EuroSAT_RGB',
        extract_dir / '2750',
        extract_dir,
    ]:
        if candidate.exists():
            subdirs = [d for d in candidate.iterdir() if d.is_dir()]
            if any(d.name in CONFIG['categories'] for d in subdirs):
                eurosat_dir = candidate
                break
    
    if eurosat_dir is None:
        # Search recursively
        for root, dirs, files in os.walk(extract_dir):
            if 'Forest' in dirs or 'AnnualCrop' in dirs:
                eurosat_dir = Path(root)
                break
    
    if eurosat_dir is None:
        print("❌ Could not find EuroSAT category folders!")
        sys.exit(1)
    
    categories = [d.name for d in eurosat_dir.iterdir() if d.is_dir()]
    print(f"📂 Found categories: {categories}")
    
    return eurosat_dir

# =============================================================================
# STEP 2: PREPARE LR/HR PAIRS
# =============================================================================

def prepare_data(eurosat_dir):
    """Create LR/HR pairs from EuroSAT images"""
    print("\n" + "="*60)
    print("STEP 2: PREPARE LR/HR PAIRS")
    print("="*60)
    
    lr_dir = DATA_DIR / 'lr'
    hr_dir = DATA_DIR / 'hr'
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already prepared
    existing = len(list(lr_dir.glob('*.png')))
    if existing >= len(CONFIG['categories']) * CONFIG['images_per_category'] * 0.8:
        print(f"✅ Already prepared {existing} image pairs!")
        return existing
    
    all_images = []
    
    for cat in CONFIG['categories']:
        cat_dir = eurosat_dir / cat
        if not cat_dir.exists():
            print(f"⚠️ Category not found: {cat}")
            continue
        
        images = list(cat_dir.glob('*.jpg')) + list(cat_dir.glob('*.tif')) + list(cat_dir.glob('*.png'))
        random.shuffle(images)
        selected = images[:CONFIG['images_per_category']]
        all_images.extend([(img, cat) for img in selected])
        print(f"📂 {cat}: {len(selected)} images")
    
    print(f"\n📊 Total selected: {len(all_images)} images")
    random.shuffle(all_images)
    
    # Process images
    hr_size = CONFIG['hr_size']
    lr_size = CONFIG['lr_size']
    
    print(f"🔄 Creating pairs (HR: {hr_size}×{hr_size}, LR: {lr_size}×{lr_size})...")
    
    for idx, (img_path, cat) in enumerate(tqdm(all_images, desc="Processing")):
        try:
            img = Image.open(img_path).convert('RGB')
            hr_img = img.resize((hr_size, hr_size), Image.BICUBIC)
            lr_img = hr_img.resize((lr_size, lr_size), Image.BICUBIC)
            
            filename = f"{cat}_{idx:05d}.png"
            hr_img.save(hr_dir / filename)
            lr_img.save(lr_dir / filename)
        except Exception as e:
            pass
    
    num_images = len(list(lr_dir.glob('*.png')))
    print(f"\n✅ Prepared {num_images} LR/HR pairs!")
    return num_images

# =============================================================================
# MODEL DEFINITION
# =============================================================================

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2*gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4*gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)
        
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class ESRGANLite(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=64, nb=8, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc=32) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf*4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf*4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        
    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk
        fea = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
        fea = self.lrelu(self.pixel_shuffle(self.upconv2(fea)))
        fea = self.lrelu(self.conv_hr(fea))
        return self.conv_last(fea)

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
    def forward(self, sr, hr):
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        return F.l1_loss(self.vgg(sr), self.vgg(hr))

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', sobel_x.repeat(3,1,1,1))
        self.register_buffer('sobel_y', sobel_y.repeat(3,1,1,1))
        
    def forward(self, sr, hr):
        sr_x = F.conv2d(sr, self.sobel_x, padding=1, groups=3)
        sr_y = F.conv2d(sr, self.sobel_y, padding=1, groups=3)
        hr_x = F.conv2d(hr, self.sobel_x, padding=1, groups=3)
        hr_y = F.conv2d(hr, self.sobel_y, padding=1, groups=3)
        return F.l1_loss(sr_x, hr_x) + F.l1_loss(sr_y, hr_y)

# =============================================================================
# DATASET
# =============================================================================

class SatelliteDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.images = sorted(list(self.lr_dir.glob('*.png')))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        lr_path = self.images[idx]
        hr_path = self.hr_dir / lr_path.name
        
        lr = np.array(Image.open(lr_path).convert('RGB')).astype(np.float32) / 255.0
        hr = np.array(Image.open(hr_path).convert('RGB')).astype(np.float32) / 255.0
        
        lr = torch.from_numpy(lr).permute(2, 0, 1)
        hr = torch.from_numpy(hr).permute(2, 0, 1)
        
        return lr, hr

# =============================================================================
# METRICS
# =============================================================================

def calc_psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calc_ssim(sr, hr):
    C1, C2 = 0.01**2, 0.03**2
    kernel = torch.ones(1, 1, 11, 11).to(sr.device) / 121
    
    mu_sr = F.conv2d(sr, kernel.expand(3,-1,-1,-1), padding=5, groups=3)
    mu_hr = F.conv2d(hr, kernel.expand(3,-1,-1,-1), padding=5, groups=3)
    
    sigma_sr = F.conv2d(sr*sr, kernel.expand(3,-1,-1,-1), padding=5, groups=3) - mu_sr**2
    sigma_hr = F.conv2d(hr*hr, kernel.expand(3,-1,-1,-1), padding=5, groups=3) - mu_hr**2
    sigma_sr_hr = F.conv2d(sr*hr, kernel.expand(3,-1,-1,-1), padding=5, groups=3) - mu_sr*mu_hr
    
    ssim = ((2*mu_sr*mu_hr + C1) * (2*sigma_sr_hr + C2)) / \
           ((mu_sr**2 + mu_hr**2 + C1) * (sigma_sr + sigma_hr + C2))
    return ssim.mean()

# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("\n" + "="*60)
    print("STEP 3: TRAINING")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    full_dataset = SatelliteDataset(DATA_DIR / 'lr', DATA_DIR / 'hr')
    val_size = int(len(full_dataset) * CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=2)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = ESRGANLite(scale=CONFIG['scale']).to(device)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========================================
    # 🔥 TRANSFER LEARNING: Load existing model
    # ========================================
    best_psnr = 0
    if CONFIG['use_pretrained'] and os.path.exists(CONFIG['pretrained_path']):
        print(f"\n🔥 TRANSFER LEARNING: Loading pretrained model...")
        print(f"   📂 From: {CONFIG['pretrained_path']}")
        try:
            checkpoint = torch.load(CONFIG['pretrained_path'], map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                prev_psnr = checkpoint.get('psnr', 0)
                prev_ssim = checkpoint.get('ssim', 0)
                print(f"   ✅ Loaded! Previous PSNR: {prev_psnr:.2f}dB, SSIM: {prev_ssim:.4f}")
                print(f"   ℹ️  Model will continue learning from this checkpoint!")
                best_psnr = prev_psnr  # Start from previous best
            else:
                model.load_state_dict(checkpoint)
                print(f"   ✅ Loaded model weights!")
        except Exception as e:
            print(f"   ⚠️ Could not load pretrained model: {e}")
            print(f"   ℹ️  Training from scratch instead...")
    elif CONFIG['use_pretrained']:
        print(f"\n⚠️ Pretrained model not found at: {CONFIG['pretrained_path']}")
        print(f"   ℹ️  Training from scratch...")
    else:
        print(f"\n📝 Training from scratch (use_pretrained=False)")
    
    # Losses
    l1_loss = nn.L1Loss()
    perceptual_loss = VGGPerceptualLoss().to(device)
    edge_loss = EdgeLoss().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    history = {'loss': [], 'psnr': [], 'ssim': []}
    
    print("\n🚀 Starting training...")
    
    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for lr_img, hr_img in pbar:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            
            optimizer.zero_grad()
            sr = model(lr_img)
            
            loss = (l1_loss(sr, hr_img) + 
                    0.1 * perceptual_loss(sr, hr_img) + 
                    0.1 * edge_loss(sr, hr_img))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_psnr, val_ssim = 0, 0
        
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                sr = torch.clamp(model(lr_img), 0, 1)
                
                val_psnr += calc_psnr(sr, hr_img).item()
                val_ssim += calc_ssim(sr, hr_img).item()
        
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        
        history['loss'].append(epoch_loss)
        history['psnr'].append(val_psnr)
        history['ssim'].append(val_ssim)
        
        # Print results
        status = ""
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'model_state_dict': model.state_dict(),
                'psnr': val_psnr,
                'ssim': val_ssim,
                'epoch': epoch + 1,
                'categories': CONFIG['categories']
            }, OUTPUT_DIR / 'best_model_colab.pth')
            status = "✅ SAVED"
        
        print(f"  Loss: {epoch_loss:.4f} | PSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.4f} {status}")
    
    print(f"\n✅ Training complete! Best PSNR: {best_psnr:.2f}dB")
    
    # Save training history plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['loss']); axes[0].set_title('Loss'); axes[0].grid(True)
    axes[1].plot(history['psnr']); axes[1].set_title('PSNR (dB)'); axes[1].grid(True)
    axes[2].plot(history['ssim']); axes[2].set_title('SSIM'); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history_colab.png', dpi=150)
    plt.close()
    
    return model, val_dataset, device

# =============================================================================
# TEST ON 10 IMAGES (Using EuroSAT categories)
# =============================================================================

def test_on_images(model, device, num_samples=10):
    """Test model on 10 images from EuroSAT categories and save with 'colab' name"""
    print("\n" + "="*60)
    print("STEP 4: TEST ON 10 EUROSAT IMAGES")
    print("="*60)
    
    # Load best model from colab training
    model_path = OUTPUT_DIR / 'best_model_colab.pth'
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded trained model (PSNR: {checkpoint.get('psnr', 'N/A'):.2f}dB)")
    
    model.eval()
    
    # Find EuroSAT images
    eurosat_dir = None
    for candidate in [
        Path('eurosat_data/EuroSAT_RGB'),
        Path('eurosat_data/2750'),
        Path('eurosat_data'),
    ]:
        if candidate.exists():
            subdirs = [d for d in candidate.iterdir() if d.is_dir()]
            if any(d.name in CONFIG['categories'] for d in subdirs):
                eurosat_dir = candidate
                break
    
    if eurosat_dir is None:
        print("⚠️ EuroSAT directory not found, using validation data instead")
        return generate_comparisons_fallback(model, device, num_samples)
    
    # Test categories (subset for 10 images)
    test_categories = ['Forest', 'River', 'Highway', 'AnnualCrop', 'SeaLake']
    images_per_cat = 2  # 5 categories × 2 images = 10 total
    
    saved_count = 0
    print(f"📸 Testing on {num_samples} images from EuroSAT categories...")
    
    for cat_name in test_categories:
        cat_dir = eurosat_dir / cat_name
        if not cat_dir.exists():
            continue
        
        images = list(cat_dir.glob('*.jpg')) + list(cat_dir.glob('*.tif')) + list(cat_dir.glob('*.png'))
        if not images:
            continue
        
        # Pick random images
        import random
        random.shuffle(images)
        
        for img_path in images[:images_per_cat]:
            if saved_count >= num_samples:
                break
            
            try:
                # Load and prepare image
                hr_img = Image.open(img_path).convert('RGB')
                hr_img = hr_img.resize((256, 256), Image.BICUBIC)
                
                # Create LR (simulate low-res)
                lr_img = hr_img.resize((64, 64), Image.BICUBIC)
                
                # Super-resolve with model
                lr = np.array(lr_img) / 255.0
                lr_tensor = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    sr_tensor = model(lr_tensor)
                    sr_tensor = torch.clamp(sr_tensor, 0, 1)
                
                sr = sr_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                sr = (sr * 255).astype(np.uint8)
                sr_img = Image.fromarray(sr)
                
                # Bicubic upscale for comparison
                lr_up = lr_img.resize((256, 256), Image.BICUBIC)
                
                saved_count += 1
                
                # Save SR image with 'colab' in name
                sr_name = f"sr_colab_{saved_count:02d}_{cat_name.lower()}.png"
                sr_img.save(OUTPUT_DIR / sr_name)
                
                # Save comparison with 'colab' in name
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(np.array(lr_img))
                axes[0].set_title('LR Input\n64×64', fontsize=11)
                axes[0].axis('off')
                
                axes[1].imshow(np.array(lr_up))
                axes[1].set_title('Bicubic\n256×256', fontsize=11)
                axes[1].axis('off')
                
                axes[2].imshow(sr)
                axes[2].set_title('Our Model (SR)\n256×256', fontsize=11)
                axes[2].axis('off')
                
                axes[3].imshow(np.array(hr_img))
                axes[3].set_title('Ground Truth\n256×256', fontsize=11)
                axes[3].axis('off')
                
                plt.suptitle(f'🛰️ {cat_name} - Colab Super-Resolution', fontsize=13, fontweight='bold')
                plt.tight_layout()
                comp_name = f"comparison_colab_{saved_count:02d}_{cat_name.lower()}.png"
                plt.savefig(OUTPUT_DIR / comp_name, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ {saved_count}/{num_samples}: {sr_name}")
                
            except Exception as e:
                print(f"   ⚠️ Error processing {img_path.name}: {e}")
        
        if saved_count >= num_samples:
            break
    
    print(f"\n✅ Saved {saved_count} images to {OUTPUT_DIR}/")
    print(f"   - sr_colab_01_{test_categories[0].lower()}.png ... sr_colab_{saved_count:02d}_*.png")
    print(f"   - comparison_colab_01_*.png ... comparison_colab_{saved_count:02d}_*.png")


def generate_comparisons_fallback(model, device, num_samples=10):
    """Fallback: Generate comparisons from validation dataset"""
    print("Using validation dataset for testing...")
    
    # Load validation data
    full_dataset = SatelliteDataset(DATA_DIR / 'lr', DATA_DIR / 'hr')
    val_size = int(len(full_dataset) * CONFIG['val_split'])
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - val_size, val_size]
    )
    
    indices = torch.randperm(len(val_dataset))[:num_samples]
    
    for i, idx in enumerate(indices):
        lr, hr = val_dataset[idx]
        
        with torch.no_grad():
            sr = torch.clamp(model(lr.unsqueeze(0).to(device)), 0, 1)[0].cpu()
        
        lr_np = lr.permute(1, 2, 0).numpy()
        sr_np = sr.permute(1, 2, 0).numpy()
        hr_np = hr.permute(1, 2, 0).numpy()
        lr_up = cv2.resize(lr_np, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        # Save SR image with 'colab' name
        Image.fromarray((sr_np * 255).astype(np.uint8)).save(
            OUTPUT_DIR / f'sr_colab_{i+1:02d}.png'
        )
        
        # Save comparison with 'colab' name
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(lr_np); axes[0].set_title('LR Input (64×64)'); axes[0].axis('off')
        axes[1].imshow(lr_up); axes[1].set_title('Bicubic Upscale'); axes[1].axis('off')
        axes[2].imshow(sr_np); axes[2].set_title('Our Model (SR)'); axes[2].axis('off')
        axes[3].imshow(hr_np); axes[3].set_title('Ground Truth (HR)'); axes[3].axis('off')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'comparison_colab_{i+1:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved {num_samples} images to {OUTPUT_DIR}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("🛰️ EUROSAT SUPER-RESOLUTION TRAINING (with Pretrained Model)")
    print("="*60)
    print(f"Categories: {CONFIG['categories']}")
    print(f"Images per category: {CONFIG['images_per_category']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Transfer Learning: {CONFIG['use_pretrained']} (uses your existing model!)")
    print(f"Output: {OUTPUT_DIR}/")
    
    # Step 1: Extract EuroSAT
    eurosat_dir = extract_eurosat()
    
    # Step 2: Prepare data
    prepare_data(eurosat_dir)
    
    # Step 3: Train (uses pretrained model!)
    model, val_dataset, device = train()
    
    # Step 4: Test on 10 images and save with 'colab' name
    test_on_images(model, device, num_samples=10)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📂 Results saved to: {OUTPUT_DIR}/")
    print("\n📁 Output files:")
    print(f"   - sr_colab_01_*.png to sr_colab_10_*.png")
    print(f"   - comparison_colab_01_*.png to comparison_colab_10_*.png")
    print(f"   - best_model_colab.pth")
    print(f"   - training_history_colab.png")
    print("\n📤 Now run the git commands in the next cell to push to GitHub!")

if __name__ == '__main__':
    main()
