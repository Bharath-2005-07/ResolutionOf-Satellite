"""
Complete Training Pipeline for Satellite Super-Resolution
Supports: EDSR, ESRGAN with GAN training
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import EDSR, ESRGANLite, VGGStyleDiscriminator, get_model
from training.losses import get_loss_function, SatelliteSRLoss, AdversarialLoss
from training.metrics import calculate_psnr, calculate_ssim
from data.dataset import DemoDataset, SyntheticSRDataset, get_dataloader


class Trainer:
    """
    Complete training pipeline for satellite super-resolution
    """
    def __init__(self, config: dict):
        """
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.scale_factor = config.get('scale_factor', 4)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize losses
        self._init_losses()
        
        # Training state
        self.epoch = 0
        self.best_psnr = 0
        self.history = {'train_loss': [], 'val_psnr': [], 'val_ssim': []}
        
    def _init_models(self):
        """Initialize generator and discriminator"""
        model_type = self.config.get('model_type', 'esrgan_lite')
        
        # Generator
        self.generator = get_model(model_type, self.scale_factor)
        self.generator.to(self.device)
        
        # Discriminator (for GAN training)
        self.use_gan = self.config.get('use_gan', False)
        if self.use_gan:
            self.discriminator = VGGStyleDiscriminator()
            self.discriminator.to(self.device)
        else:
            self.discriminator = None
            
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        if self.discriminator:
            print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def _init_optimizers(self):
        """Initialize optimizers"""
        lr_g = self.config.get('lr_generator', 1e-4)
        lr_d = self.config.get('lr_discriminator', 1e-4)
        
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(0.9, 0.999)
        )
        
        if self.use_gan:
            self.optimizer_d = optim.Adam(
                self.discriminator.parameters(),
                lr=lr_d,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer_d = None
            
        # Learning rate scheduler
        self.scheduler_g = optim.lr_scheduler.StepLR(
            self.optimizer_g, 
            step_size=self.config.get('lr_decay_step', 50),
            gamma=self.config.get('lr_decay_gamma', 0.5)
        )
    
    def _init_losses(self):
        """Initialize loss functions"""
        loss_type = self.config.get('loss_type', 'satellite')
        self.criterion = get_loss_function(loss_type)
        
        if self.use_gan:
            self.criterion_adv = AdversarialLoss()
    
    def train_one_epoch(self, train_loader: DataLoader) -> dict:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary with epoch metrics
        """
        self.generator.train()
        if self.discriminator:
            self.discriminator.train()
        
        running_loss_g = 0.0
        running_loss_d = 0.0
        loss_components = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        max_steps = self.config.get('max_steps_per_epoch', float('inf'))
        
        for i, (lr, hr) in enumerate(pbar):
            if i >= max_steps:
                break
                
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            
            # ==================
            # Train Generator
            # ==================
            self.optimizer_g.zero_grad()
            
            # Forward pass
            sr = self.generator(lr)
            
            # Calculate generator loss
            if self.use_gan and self.discriminator:
                # Get discriminator prediction on fake
                pred_fake = self.discriminator(sr)
                loss_g, components = self.criterion(sr, hr, pred_fake)
            else:
                loss_g, components = self.criterion(sr, hr)
            
            # Backward pass
            loss_g.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            
            self.optimizer_g.step()
            
            running_loss_g += loss_g.item()
            
            # Accumulate loss components
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0) + v
            
            # ==================
            # Train Discriminator
            # ==================
            if self.use_gan and self.discriminator:
                self.optimizer_d.zero_grad()
                
                # Real loss
                pred_real = self.discriminator(hr)
                loss_real = self.criterion_adv(pred_real, target_is_real=True)
                
                # Fake loss
                pred_fake = self.discriminator(sr.detach())
                loss_fake = self.criterion_adv(pred_fake, target_is_real=False)
                
                loss_d = (loss_real + loss_fake) / 2
                loss_d.backward()
                self.optimizer_d.step()
                
                running_loss_d += loss_d.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{loss_g.item():.4f}',
                'D_loss': f'{running_loss_d/(i+1):.4f}' if self.use_gan else 'N/A'
            })
        
        # Average losses
        n_batches = min(len(train_loader), max_steps)
        avg_loss_g = running_loss_g / n_batches
        avg_loss_d = running_loss_d / n_batches if self.use_gan else 0
        
        for k in loss_components:
            loss_components[k] /= n_batches
        
        return {
            'loss_g': avg_loss_g,
            'loss_d': avg_loss_d,
            **loss_components
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with validation metrics
        """
        self.generator.eval()
        
        psnr_sum = 0.0
        ssim_sum = 0.0
        n_samples = 0
        
        max_val_samples = self.config.get('max_val_samples', 50)
        
        for i, (lr, hr) in enumerate(val_loader):
            if i >= max_val_samples:
                break
                
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            
            sr = self.generator(lr)
            
            # Calculate metrics for each sample in batch
            for j in range(sr.shape[0]):
                psnr = calculate_psnr(sr[j:j+1], hr[j:j+1])
                ssim = calculate_ssim(sr[j:j+1], hr[j:j+1])
                
                psnr_sum += psnr
                ssim_sum += ssim
                n_samples += 1
        
        avg_psnr = psnr_sum / max(n_samples, 1)
        avg_ssim = ssim_sum / max(n_samples, 1)
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              num_epochs: int = 100):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_one_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss_g'])
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.history['val_psnr'].append(val_metrics['psnr'])
                self.history['val_ssim'].append(val_metrics['ssim'])
                
                print(f"Epoch {epoch}: Loss={train_metrics['loss_g']:.4f}, "
                      f"PSNR={val_metrics['psnr']:.2f}dB, SSIM={val_metrics['ssim']:.4f}")
                
                # Save best model
                if val_metrics['psnr'] > self.best_psnr:
                    self.best_psnr = val_metrics['psnr']
                    self.save_checkpoint('best_model.pth')
            else:
                print(f"Epoch {epoch}: Loss={train_metrics['loss_g']:.4f}")
            
            # Learning rate decay
            self.scheduler_g.step()
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        self.save_history()
        
        print(f"Training complete! Best PSNR: {self.best_psnr:.2f}dB")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer_g.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        
        if self.discriminator:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['optimizer_d_state_dict'] = self.optimizer_d.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.best_psnr = checkpoint.get('best_psnr', 0)
        
        if self.discriminator and 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def save_history(self):
        """Save training history"""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def get_default_config():
    """Get default training configuration"""
    return {
        'model_type': 'esrgan_lite',
        'scale_factor': 4,
        'use_gan': False,  # Start with just generator, enable later
        'loss_type': 'satellite',
        'lr_generator': 1e-4,
        'lr_discriminator': 1e-4,
        'batch_size': 8,
        'num_epochs': 100,
        'max_steps_per_epoch': 100,  # Limit for demo
        'max_val_samples': 20,
        'lr_decay_step': 30,
        'lr_decay_gamma': 0.5,
        'save_freq': 10,
        'checkpoint_dir': 'checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }


def train_demo():
    """Demo training with synthetic data"""
    config = get_default_config()
    config['num_epochs'] = 5  # Quick demo
    config['max_steps_per_epoch'] = 20
    
    # Create demo dataset
    train_dataset = DemoDataset(num_samples=200, patch_size=64, scale_factor=4)
    val_dataset = DemoDataset(num_samples=50, patch_size=64, scale_factor=4)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=0)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train(train_loader, val_loader, num_epochs=config['num_epochs'])
    
    return trainer


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train satellite super-resolution model')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor')
    parser.add_argument('--model', type=str, default='esrgan_lite', help='Model type')
    parser.add_argument('--gan', action='store_true', help='Enable GAN training')
    parser.add_argument('--demo', action='store_true', help='Run demo training')
    
    args = parser.parse_args()
    
    if args.demo:
        print("Running demo training with synthetic data...")
        train_demo()
    else:
        config = get_default_config()
        config['num_epochs'] = args.epochs
        config['batch_size'] = args.batch_size
        config['lr_generator'] = args.lr
        config['scale_factor'] = args.scale
        config['model_type'] = args.model
        config['use_gan'] = args.gan
        
        if args.data_dir:
            # Use real data
            train_loader = get_dataloader(
                dataset_type='synthetic',
                hr_dir=args.data_dir,
                batch_size=args.batch_size
            )
            val_loader = None  # Add validation data as needed
        else:
            # Use demo data
            train_dataset = DemoDataset(num_samples=500)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                     shuffle=True, num_workers=0)
            val_loader = None
        
        trainer = Trainer(config)
        trainer.train(train_loader, val_loader, num_epochs=args.epochs)
