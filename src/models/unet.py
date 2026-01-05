"""U-Net and U-Net++ architectures for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BatchNorm -> ReLU) * 2"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        # Use bilinear upsampling or transposed convolutions
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle different input sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for semantic segmentation.
    
    Original paper: https://arxiv.org/abs/1505.04597
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        bilinear: Use bilinear upsampling (True) or transposed convolutions (False)
        base_channels: Number of channels in the first layer (default: 64)
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True, base_channels: int = 64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class NestedDoubleConv(nn.Module):
    """Nested double convolution for U-Net++"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """U-Net++ (Nested U-Net) architecture.
    
    Paper: https://arxiv.org/abs/1807.10165
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        deep_supervision: Use deep supervision for training
        base_channels: Number of channels in the first layer
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 1, deep_supervision: bool = False, base_channels: int = 32):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Level 0
        self.conv0_0 = NestedDoubleConv(n_channels, base_channels)
        self.conv1_0 = NestedDoubleConv(base_channels, base_channels * 2)
        self.conv2_0 = NestedDoubleConv(base_channels * 2, base_channels * 4)
        self.conv3_0 = NestedDoubleConv(base_channels * 4, base_channels * 8)
        self.conv4_0 = NestedDoubleConv(base_channels * 8, base_channels * 16)
        
        # Level 1
        self.conv0_1 = NestedDoubleConv(base_channels + base_channels * 2, base_channels)
        self.conv1_1 = NestedDoubleConv(base_channels * 2 + base_channels * 4, base_channels * 2)
        self.conv2_1 = NestedDoubleConv(base_channels * 4 + base_channels * 8, base_channels * 4)
        self.conv3_1 = NestedDoubleConv(base_channels * 8 + base_channels * 16, base_channels * 8)
        
        # Level 2
        self.conv0_2 = NestedDoubleConv(base_channels * 2 + base_channels * 2, base_channels)
        self.conv1_2 = NestedDoubleConv(base_channels * 4 + base_channels * 4, base_channels * 2)
        self.conv2_2 = NestedDoubleConv(base_channels * 8 + base_channels * 8, base_channels * 4)
        
        # Level 3
        self.conv0_3 = NestedDoubleConv(base_channels * 3 + base_channels * 2, base_channels)
        self.conv1_3 = NestedDoubleConv(base_channels * 6 + base_channels * 4, base_channels * 2)
        
        # Level 4
        self.conv0_4 = NestedDoubleConv(base_channels * 4 + base_channels * 2, base_channels)
        
        if self.deep_supervision:
            self.final1 = nn.Conv2d(base_channels, n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(base_channels, n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(base_channels, n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(base_channels, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


if __name__ == '__main__':
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test U-Net
    model = UNet(n_channels=3, n_classes=1).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    output = model(x)
    print(f"U-Net output shape: {output.shape}")
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test U-Net++
    model_pp = UNetPlusPlus(n_channels=3, n_classes=1).to(device)
    output_pp = model_pp(x)
    print(f"U-Net++ output shape: {output_pp.shape}")
    print(f"U-Net++ parameters: {sum(p.numel() for p in model_pp.parameters()):,}")
