import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, num_features):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.body(x)

class EDSR(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, num_res_blocks=16, num_features=64):
        super(EDSR, self).__init__()
        
        # 1. Head (Initial extraction)
        self.head = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        
        # 2. Body (Residual Blocks)
        body = [ResBlock(num_features) for _ in range(num_res_blocks)]
        self.body = nn.Sequential(*body)
        
        # 3. Tail (Upsampling)
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x