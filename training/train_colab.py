"""
COMPLETE TRAINING SCRIPT FOR GOOGLE COLAB
Uses REAL satellite imagery with proper losses and validation

Instructions:
1. Upload this to Colab
2. Run all cells
3. Downloads UC Merced satellite dataset automatically
4. Trains with proper satellite-optimized losses
5. Validates with PSNR/SSIM metrics
6. Saves checkpoints and visualizations

GitHub: https://github.com/Bharath-2005-07/ResolutionOf-Satellite
"""

import sys
import os
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
import urllib.request
import zipfile
from pathlib import Path
import cv2

# Add project to path
if '/content/ResolutionOf-Satellite' in sys.path:
    pass
elif os.path.exists('/content/ResolutionOf-Satellite'):
    sys.path.insert(0, '/content/ResolutionOf-Satellite')

# =============================================================================
# STEP 1: DOWNLOAD REAL SATELLITE DATA
# =============================================================================

def download_satellite_dataset(data_dir='satellite_data'):
    """Download UC Merced Land Use Dataset (real satellite imagery)"""
    data_dir = Path(data_dir)
    
    if (data_dir / 'processed' / 'lr').exists():
        num_images = len(list((data_dir / 'processed' / 'lr').glob('*.png')))
        print(f"✅ Dataset already downloaded! Found {num_images} image pairs")
        print("   ℹ️  No need to re-download (cached)")
        return data_dir
    
    print("="*80)
    print("STEP 1: DOWNLOAD REAL SATELLITE DATA")
    print("="*80)
    print("📥 Downloading UC Merced Land Use Dataset (satellite imagery)...")
    print("   Size: ~320MB, Source: UC Merced (public domain)")
    print("   This only happens ONCE - data is cached for future runs!\n")
    
    # Multiple mirror URLs with retry logic
    urls = [
        "http://weegee.vision.ucmerced.edu/datasets/landuse.zip",
    ]
    
    zip_path = data_dir / "dataset.zip"
    data_dir.mkdir(exist_ok=True, parents=True)
    
    for attempt, url in enumerate(urls, 1):
        try:
            print(f"🔄 Attempt {attempt}: Connecting to server...")
            
            # Download with longer timeout and better progress
            import socket
            import time
            socket.setdefaulttimeout(60)  # 60 second timeout
            
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded / total_size * 100, 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    bar_length = 40
                    filled = int(bar_length * downloaded / total_size)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r  📊 [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f}MB)", end='', flush=True)
            
            # Retry download 3 times before giving up
            max_retries = 3
            for retry in range(max_retries):
                try:
                    urllib.request.urlretrieve(url, zip_path, reporthook=report_progress)
                    print("\n✅ Download complete!\n")
                    break
                except Exception as retry_e:
                    if retry < max_retries - 1:
                        print(f"\n⚠️  Connection interrupted, retrying in 5 seconds... ({retry+1}/{max_retries})")
                        time.sleep(5)
                    else:
                        raise retry_e
            
            # Extract with progress
            print("📦 Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                for i, member in enumerate(members, 1):
                    zip_ref.extract(member, data_dir / 'extracted')
                    if i % 100 == 0 or i == len(members):
                        print(f"\r  Extracted {i}/{len(members)} files...", end='', flush=True)
            
            print("\n✅ Extraction complete!\n")
            zip_path.unlink()  # Remove zip file
            
            return data_dir
            
        except Exception as e:
            print(f"\n❌ Download failed: {str(e)[:100]}")
            print("\n⚠️  IMPORTANT: UC Merced server may be down/slow")
            print("   Creating synthetic satellite-like dataset instead...")
            print("   (This will work but won't have real satellite features)\n")
            create_synthetic_satellite_data(data_dir)
            return data_dir


def create_synthetic_satellite_data(data_dir, num_images=300):
    """Fallback: Create synthetic satellite-like imagery"""
    print("="*80)
    print("⚠️  USING SYNTHETIC DATA (Not real satellite imagery)")
    print("="*80)
    print(f"🎨 Generating {num_images} synthetic satellite images...")
    print("   This is a FALLBACK - results may be less accurate than real data\n")
    
    processed_dir = data_dir / 'processed'
    lr_dir = processed_dir / 'lr'
    hr_dir = processed_dir / 'hr'
    lr_dir.mkdir(exist_ok=True, parents=True)
    hr_dir.mkdir(exist_ok=True, parents=True)
    
    from PIL import Image, ImageDraw
    
    for idx in tqdm(range(num_images)):
        # Generate HR image (256x256)
        img = Image.new('RGB', (256, 256))
        draw = ImageDraw.Draw(img)
        
        # Base terrain color (vegetation-like)
        base_r = int(50 + np.random.rand() * 80)
        base_g = int(100 + np.random.rand() * 100)
        base_b = int(50 + np.random.rand() * 60)
        draw.rectangle([0, 0, 256, 256], fill=(base_r, base_g, base_b))
        
        # Add fields/patches (simulate agricultural areas)
        for _ in range(np.random.randint(8, 20)):
            x1 = np.random.randint(0, 200)
            y1 = np.random.randint(0, 200)
            x2 = x1 + np.random.randint(20, 80)
            y2 = y1 + np.random.randint(20, 80)
            
            r = int(30 + np.random.rand() * 150)
            g = int(50 + np.random.rand() * 150)
            b = int(30 + np.random.rand() * 100)
            draw.rectangle([x1, y1, x2, y2], fill=(r, g, b))
        
        # Add roads (linear features)
        for _ in range(np.random.randint(1, 4)):
            if np.random.rand() > 0.5:
                y = np.random.randint(0, 256)
                draw.line([0, y, 256, y], fill=(80, 80, 80), width=np.random.randint(3, 8))
            else:
                x = np.random.randint(0, 256)
                draw.line([x, 0, x, 256], fill=(80, 80, 80), width=np.random.randint(3, 8))
        
        # Add buildings (small rectangles)
        for _ in range(np.random.randint(0, 10)):
            x = np.random.randint(0, 240)
            y = np.random.randint(0, 240)
            w = np.random.randint(5, 15)
            h = np.random.randint(5, 15)
            gray = int(100 + np.random.rand() * 100)
            draw.rectangle([x, y, x+w, y+h], fill=(gray, gray, gray))
        
        # Add natural texture
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Save HR
        hr_path = hr_dir / f'sat_{idx:04d}.png'
        img.save(hr_path)
        
        # Create LR (bicubic downsampling - simulates Sentinel-2)
        lr_img = img.resize((64, 64), Image.BICUBIC)
        lr_path = lr_dir / f'sat_{idx:04d}.png'
        lr_img.save(lr_path)
    
    print(f"✅ Created {num_images} synthetic satellite image pairs!")


def prepare_real_satellite_data(data_dir, scale_factor=4, patch_size=64):
    """Process downloaded satellite images into LR/HR pairs"""
    extract_dir = data_dir / 'extracted'
    processed_dir = data_dir / 'processed'
    lr_dir = processed_dir / 'lr'
    hr_dir = processed_dir / 'hr'
    
    if lr_dir.exists() and len(list(lr_dir.glob('*.png'))) > 0:
        num_pairs = len(list(lr_dir.glob('*.png')))
        print(f"✅ Data already processed! {num_pairs} LR/HR pairs ready")
        return processed_dir
    
    print("="*80)
    print("STEP 2: PROCESS SATELLITE IMAGES INTO LR/HR PAIRS")
    print("="*80)
    
    lr_dir.mkdir(exist_ok=True, parents=True)
    hr_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all images
    image_files = []
    for ext in ['*.tif', '*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(extract_dir.rglob(ext)))
    
    if len(image_files) == 0:
        print("⚠️  No images found, creating synthetic data...")
        create_synthetic_satellite_data(data_dir)
        return processed_dir
    
    print(f"📸 Processing {len(image_files)} satellite images...")
    
    hr_size = patch_size * scale_factor  # 256 for 4x upscaling
    
    for idx, img_path in enumerate(tqdm(image_files[:500])):  # Limit to 500
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Resize to HR size
            img_hr = img.resize((hr_size, hr_size), Image.BICUBIC)
            
            # Create LR (simulate Sentinel-2 resolution)
            img_lr = img_hr.resize((patch_size, patch_size), Image.BICUBIC)
            
            # Save
            img_hr.save(hr_dir / f'sat_{idx:04d}.png')
            img_lr.save(lr_dir / f'sat_{idx:04d}.png')
            
        except Exception as e:
            continue
    
    num_created = len(list(lr_dir.glob('*.png')))
    print(f"✅ Created {num_created} LR/HR pairs!")
    return processed_dir


# =============================================================================
# STEP 2: REAL SATELLITE DATASET
# =============================================================================

class RealSatelliteDataset(Dataset):
    """Dataset using REAL satellite imagery"""
    
    def __init__(self, lr_dir, hr_dir, augment=True):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.augment = augment
        
        self.lr_files = sorted(self.lr_dir.glob('*.png'))
        self.hr_files = sorted(self.hr_dir.glob('*.png'))
        
        assert len(self.lr_files) == len(self.hr_files), \
            f"Mismatch: {len(self.lr_files)} LR vs {len(self.hr_files)} HR"
        
        print(f"📂 Loaded {len(self.lr_files)} real satellite image pairs")
    
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        # Load images
        lr_img = np.array(Image.open(self.lr_files[idx]).convert('RGB')).astype(np.float32) / 255.0
        hr_img = np.array(Image.open(self.hr_files[idx]).convert('RGB')).astype(np.float32) / 255.0
        
        # Augmentation
        if self.augment and np.random.rand() > 0.5:
            # Horizontal flip
            lr_img = np.fliplr(lr_img).copy()
            hr_img = np.fliplr(hr_img).copy()
        
        if self.augment and np.random.rand() > 0.5:
            # Vertical flip
            lr_img = np.flipud(lr_img).copy()
            hr_img = np.flipud(hr_img).copy()
        
        if self.augment:
            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            lr_img = np.rot90(lr_img, k).copy()
            hr_img = np.rot90(hr_img, k).copy()
        
        # Convert to tensors (C, H, W)
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float()
        
        return lr_tensor, hr_tensor


# =============================================================================
# STEP 3: SATELLITE-OPTIMIZED LOSS FUNCTIONS
# =============================================================================

class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, sr, hr):
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        
        return F.l1_loss(sr_features, hr_features)


class EdgeAwareLoss(nn.Module):
    """Edge-aware loss for sharp roads and buildings"""
    
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def get_edges(self, img):
        edge_x = F.conv2d(img, self.sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(img, self.sobel_y, padding=1, groups=3)
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
    
    def forward(self, sr, hr):
        sr_edges = self.get_edges(sr)
        hr_edges = self.get_edges(hr)
        return F.l1_loss(sr_edges, hr_edges)


class SatelliteSRLoss(nn.Module):
    """Complete loss optimized for satellite imagery"""
    
    def __init__(self, pixel_weight=1.0, perceptual_weight=0.1, edge_weight=0.1):
        super().__init__()
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.edge_loss = EdgeAwareLoss()
        
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
    
    def forward(self, sr, hr):
        loss_pixel = self.pixel_loss(sr, hr)
        loss_perceptual = self.perceptual_loss(sr, hr)
        loss_edge = self.edge_loss(sr, hr)
        
        total_loss = (self.pixel_weight * loss_pixel +
                     self.perceptual_weight * loss_perceptual +
                     self.edge_weight * loss_edge)
        
        return total_loss, {
            'pixel': loss_pixel.item(),
            'perceptual': loss_perceptual.item(),
            'edge': loss_edge.item(),
            'total': total_loss.item()
        }


# =============================================================================
# STEP 4: METRICS (PSNR, SSIM)
# =============================================================================

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim(img1, img2, window_size=11):
    """Calculate SSIM between two images"""
    # Simplified SSIM implementation
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, 1, window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


# =============================================================================
# STEP 5: VISUALIZATION
# =============================================================================

def visualize_results(model, val_loader, device, epoch, save_dir='results'):
    """Visualize SR results"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        lr, hr = next(iter(val_loader))
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        
        # Convert to numpy
        lr_np = lr[0].cpu().permute(1, 2, 0).numpy()
        hr_np = hr[0].cpu().permute(1, 2, 0).numpy()
        sr_np = sr[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        # Create bicubic baseline
        bicubic = F.interpolate(lr[0:1], scale_factor=4, mode='bicubic', align_corners=False)
        bicubic_np = bicubic[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        # Calculate metrics
        psnr_bicubic = calculate_psnr(bicubic[0:1], hr[0:1]).item()
        psnr_sr = calculate_psnr(sr[0:1], hr[0:1]).item()
        ssim_bicubic = calculate_ssim(bicubic[0:1], hr[0:1]).item()
        ssim_sr = calculate_ssim(sr[0:1], hr[0:1]).item()
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(lr_np)
        axes[0, 0].set_title('Input LR (64x64)', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(bicubic_np)
        axes[0, 1].set_title(f'Bicubic Baseline\nPSNR: {psnr_bicubic:.2f}dB, SSIM: {ssim_bicubic:.4f}', fontsize=14)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(sr_np)
        axes[1, 0].set_title(f'Our Model (ESRGANLite)\nPSNR: {psnr_sr:.2f}dB, SSIM: {ssim_sr:.4f}', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(hr_np)
        axes[1, 1].set_title('Ground Truth HR (256x256)', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'comparison_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Metrics Comparison:")
        print(f"   Bicubic   - PSNR: {psnr_bicubic:.2f}dB, SSIM: {ssim_bicubic:.4f}")
        print(f"   Our Model - PSNR: {psnr_sr:.2f}dB, SSIM: {ssim_sr:.4f}")
        print(f"   Improvement: +{psnr_sr - psnr_bicubic:.2f}dB PSNR\n")


# =============================================================================
# STEP 6: MAIN TRAINING FUNCTION
# =============================================================================

def train_satellite_sr(
    data_dir='satellite_data',
    num_epochs=100,
    batch_size=8,
    learning_rate=2e-4,
    device=None,
    visualize_every=10
):
    """
    Complete training pipeline for satellite super-resolution
    
    Args:
        data_dir: Directory for satellite data
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
        visualize_every: Visualize every N epochs
    """
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Download and prepare data
    print("\n" + "="*80)
    print("STEP 1: PREPARE REAL SATELLITE DATA")
    print("="*80)
    data_dir = Path(data_dir)
    data_dir = download_satellite_dataset(data_dir)
    processed_dir = prepare_real_satellite_data(data_dir, scale_factor=4, patch_size=64)
    
    # Create datasets
    print("\n" + "="*80)
    print("STEP 2: CREATE DATALOADERS")
    print("="*80)
    
    lr_dir = processed_dir / 'lr'
    hr_dir = processed_dir / 'hr'
    
    all_files = list(lr_dir.glob('*.png'))
    train_size = int(0.9 * len(all_files))
    
    # Split into train/val
    train_lr = [str(f) for f in all_files[:train_size]]
    val_lr = [str(f) for f in all_files[train_size:]]
    
    # Create temp directories for split
    train_lr_dir = processed_dir / 'train_lr'
    train_hr_dir = processed_dir / 'train_hr'
    val_lr_dir = processed_dir / 'val_lr'
    val_hr_dir = processed_dir / 'val_hr'
    
    for d in [train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir]:
        d.mkdir(exist_ok=True)
    
    # Create symlinks or copy
    import shutil
    for i, lr_file in enumerate(all_files):
        hr_file = hr_dir / lr_file.name
        if i < train_size:
            shutil.copy(lr_file, train_lr_dir / lr_file.name)
            shutil.copy(hr_file, train_hr_dir / hr_file.name)
        else:
            shutil.copy(lr_file, val_lr_dir / lr_file.name)
            shutil.copy(hr_file, val_hr_dir / hr_file.name)
    
    train_dataset = RealSatelliteDataset(train_lr_dir, train_hr_dir, augment=True)
    val_dataset = RealSatelliteDataset(val_lr_dir, val_hr_dir, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Load model
    print("\n" + "="*80)
    print("STEP 3: INITIALIZE MODEL")
    print("="*80)
    
    try:
        from models.esrgan import ESRGANLite
        model = ESRGANLite(scale_factor=4).to(device)
    except:
        print("⚠️  Using inline model definition")
        # Inline model definition (simplified)
        from models import esrgan
        model = esrgan.ESRGANLite(scale_factor=4).to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = SatelliteSRLoss(pixel_weight=1.0, perceptual_weight=0.1, edge_weight=0.1)
    criterion = criterion.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    print("\n" + "="*80)
    print("STEP 4: START TRAINING")
    print("="*80)
    
    best_psnr = 0
    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]', 
                    ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for lr_img, hr_img in pbar:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            
            optimizer.zero_grad()
            sr = model(lr_img)
            
            loss, components = criterion(sr, hr_img)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Pixel': f"{components['pixel']:.3f}",
                'Edge': f"{components['edge']:.3f}"
            })
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_psnr_sum = 0
        val_ssim_sum = 0
        
        print(f"  📊 Validating... ", end='', flush=True)
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                sr = model(lr_img)
                
                for i in range(sr.shape[0]):
                    psnr = calculate_psnr(sr[i:i+1], hr_img[i:i+1]).item()
                    ssim = calculate_ssim(sr[i:i+1], hr_img[i:i+1]).item()
                    val_psnr_sum += psnr
                    val_ssim_sum += ssim
        
        avg_psnr = val_psnr_sum / len(val_dataset)
        avg_ssim = val_ssim_sum / len(val_dataset)
        
        history['val_psnr'].append(avg_psnr)
        history['val_ssim'].append(avg_ssim)
        
        scheduler.step()
        
        # Progress indicator with emoji
        progress_pct = (epoch + 1) / num_epochs * 100
        progress_emoji = "🟢" if avg_psnr > best_psnr else "🔵"
        
        print(f"{progress_emoji} Epoch {epoch+1}/{num_epochs} ({progress_pct:.0f}% Complete)")
        print(f"  📉 Train Loss: {avg_train_loss:.4f}")
        print(f"  📈 Val PSNR: {avg_psnr:.2f}dB  |  SSIM: {avg_ssim:.4f}")
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'ssim': avg_ssim
            }, 'checkpoints/best_model.pth')
            print(f"  ✅ New best model saved! PSNR: {best_psnr:.2f}dB")
        
        # Visualize
        if (epoch + 1) % visualize_every == 0 or epoch == 0:
            visualize_results(model, val_loader, device, epoch+1)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"✅ Best PSNR: {best_psnr:.2f}dB")
    print(f"💾 Model saved to: checkpoints/best_model.pth")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_psnr'])
    plt.title('Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_ssim'])
    plt.title('Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    
    return model, history


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                 SATELLITE SUPER-RESOLUTION TRAINING                      ║
    ║                   Using REAL Satellite Imagery                           ║
    ║                                                                          ║
    ║  GitHub: https://github.com/Bharath-2005-07/ResolutionOf-Satellite      ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Train the model
    model, history = train_satellite_sr(
        data_dir='satellite_data',
        num_epochs=100,
        batch_size=8,
        learning_rate=2e-4,
        visualize_every=10
    )
    
    print("\n✅ Training complete! Check 'checkpoints/best_model.pth' for the trained model.")
    print("📊 Visualizations saved to 'results/' folder")
    print("📈 Training history saved to 'training_history.png'")
