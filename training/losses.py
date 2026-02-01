"""
Loss Functions for Satellite Super-Resolution
Includes: L1, Perceptual (VGG), Adversarial, and Combined Losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG19 features
    Extracts features from conv5_4 layer for perceptual similarity
    Critical for satellite images - preserves structural details!
    """
    def __init__(self, feature_layer=35, use_input_norm=True):
        super().__init__()
        # Load pretrained VGG19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:feature_layer].eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg = vgg
        self.use_input_norm = use_input_norm
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, sr, hr):
        if self.use_input_norm:
            sr = (sr - self.mean) / self.std
            hr = (hr - self.mean) / self.std
            
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        
        return F.l1_loss(sr_features, hr_features)


class AdversarialLoss(nn.Module):
    """
    Adversarial Loss for GAN training
    Supports vanilla GAN and WGAN-GP
    """
    def __init__(self, gan_type='vanilla'):
        super().__init__()
        self.gan_type = gan_type
        
        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'wgan':
            self.loss = None  # No explicit loss for WGAN
        else:
            self.loss = nn.BCEWithLogitsLoss()
            
    def forward(self, pred, target_is_real):
        if self.gan_type == 'wgan':
            return -pred.mean() if target_is_real else pred.mean()
        
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


class CombinedLoss(nn.Module):
    """
    Combined Loss for ESRGAN Training
    Total Loss = λ_pixel * L1 + λ_perceptual * VGG + λ_adv * Adversarial
    
    Recommended weights for satellite imagery:
    - pixel_weight: 1.0 (preserve low-level details)
    - perceptual_weight: 0.1 (structural similarity)
    - adversarial_weight: 0.005 (sharpness without artifacts)
    """
    def __init__(self, pixel_weight=1.0, perceptual_weight=0.1, adversarial_weight=0.005):
        super().__init__()
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.adversarial_loss = AdversarialLoss()
        
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        
    def forward(self, sr, hr, discriminator_pred=None):
        # Pixel loss (L1)
        loss_pixel = self.pixel_loss(sr, hr)
        
        # Perceptual loss (VGG)
        loss_perceptual = self.perceptual_loss(sr, hr)
        
        # Total generator loss
        total_loss = (self.pixel_weight * loss_pixel + 
                     self.perceptual_weight * loss_perceptual)
        
        # Add adversarial loss if discriminator prediction is provided
        if discriminator_pred is not None:
            loss_adv = self.adversarial_loss(discriminator_pred, target_is_real=True)
            total_loss += self.adversarial_weight * loss_adv
            return total_loss, {
                'pixel': loss_pixel.item(),
                'perceptual': loss_perceptual.item(),
                'adversarial': loss_adv.item()
            }
        
        return total_loss, {
            'pixel': loss_pixel.item(),
            'perceptual': loss_perceptual.item()
        }


class EdgeAwareLoss(nn.Module):
    """
    Edge-Aware Loss for satellite imagery
    Preserves sharp edges for roads, buildings, and urban features
    """
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
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
    """
    Complete loss function optimized for satellite super-resolution
    Combines pixel, perceptual, edge, and optional adversarial losses
    """
    def __init__(self, 
                 pixel_weight=1.0,
                 perceptual_weight=0.1,
                 edge_weight=0.1,
                 adversarial_weight=0.005,
                 use_edge_loss=True):
        super().__init__()
        
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.edge_loss = EdgeAwareLoss() if use_edge_loss else None
        self.adversarial_loss = AdversarialLoss()
        
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.adversarial_weight = adversarial_weight
        self.use_edge_loss = use_edge_loss
        
    def forward(self, sr, hr, discriminator_pred=None):
        losses = {}
        
        # Pixel loss
        loss_pixel = self.pixel_loss(sr, hr)
        losses['pixel'] = loss_pixel.item()
        total_loss = self.pixel_weight * loss_pixel
        
        # Perceptual loss
        loss_perceptual = self.perceptual_loss(sr, hr)
        losses['perceptual'] = loss_perceptual.item()
        total_loss += self.perceptual_weight * loss_perceptual
        
        # Edge loss
        if self.use_edge_loss and self.edge_loss is not None:
            loss_edge = self.edge_loss(sr, hr)
            losses['edge'] = loss_edge.item()
            total_loss += self.edge_weight * loss_edge
        
        # Adversarial loss
        if discriminator_pred is not None:
            loss_adv = self.adversarial_loss(discriminator_pred, target_is_real=True)
            losses['adversarial'] = loss_adv.item()
            total_loss += self.adversarial_weight * loss_adv
            
        return total_loss, losses


def get_loss_function(loss_type='l1'):
    """
    Factory function to get loss functions
    
    Args:
        loss_type: 'l1', 'mse', 'perceptual', 'combined', 'satellite'
    """
    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'perceptual':
        return VGGPerceptualLoss()
    elif loss_type == 'combined':
        return CombinedLoss()
    elif loss_type == 'satellite':
        return SatelliteSRLoss()
    else:
        return nn.L1Loss()