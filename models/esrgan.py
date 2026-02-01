"""
ESRGAN: Enhanced Super-Resolution Generative Adversarial Network
For satellite image super-resolution (4x/8x upscaling)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for RRDB"""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
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


class ESRGANGenerator(nn.Module):
    """
    ESRGAN Generator Network
    - Uses RRDB blocks for better feature extraction
    - Supports 4x and 8x upscaling for satellite imagery
    """
    def __init__(self, in_channels=3, out_channels=3, nf=64, nb=23, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # First conv
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)
        
        # RRDB trunk
        self.trunk = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # Upsampling layers
        if scale_factor == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif scale_factor == 8:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.upconv3 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        
        # High-resolution conv
        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(fea))
        fea = fea + trunk
        
        # Upsampling
        fea = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
        fea = self.lrelu(self.pixel_shuffle(self.upconv2(fea)))
        
        if self.scale_factor == 8:
            fea = self.lrelu(self.pixel_shuffle(self.upconv3(fea)))
        
        fea = self.conv_last(self.lrelu(self.hr_conv(fea)))
        return fea


class VGGStyleDiscriminator(nn.Module):
    """
    VGG-style Discriminator for ESRGAN
    Used for adversarial training to improve perceptual quality
    """
    def __init__(self, in_channels=3, nf=64):
        super().__init__()
        
        self.features = nn.Sequential(
            # Input: 128x128
            nn.Conv2d(in_channels, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf, nf, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf * 2, nf * 2, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf * 4, nf * 4, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf * 4, nf * 8, 3, 1, 1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(nf * 8, nf * 8, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 8, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )
        
    def forward(self, x):
        fea = self.features(x)
        out = self.classifier(fea)
        return out


class ESRGANLite(nn.Module):
    """
    Lightweight ESRGAN for faster training on free-tier GPUs (Colab T4)
    Reduced RRDB blocks for memory efficiency
    """
    def __init__(self, in_channels=3, out_channels=3, nf=64, nb=8, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)
        self.trunk = nn.Sequential(*[RRDB(nf, gc=32) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # Upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        if scale_factor == 8:
            self.upconv3 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        
        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(fea))
        fea = fea + trunk
        
        fea = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
        fea = self.lrelu(self.pixel_shuffle(self.upconv2(fea)))
        
        if self.scale_factor == 8:
            fea = self.lrelu(self.pixel_shuffle(self.upconv3(fea)))
        
        return self.conv_last(self.lrelu(self.hr_conv(fea)))


def get_model(model_name='esrgan_lite', scale_factor=4):
    """Factory function to get the appropriate model"""
    models = {
        'esrgan': ESRGANGenerator(scale_factor=scale_factor),
        'esrgan_lite': ESRGANLite(scale_factor=scale_factor),
    }
    return models.get(model_name, ESRGANLite(scale_factor=scale_factor))


if __name__ == '__main__':
    # Quick test
    model = ESRGANLite(scale_factor=4)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
