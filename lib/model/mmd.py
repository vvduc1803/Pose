import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfgs

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # Global average pooling to reduce spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # MLP for channel attention: reduce channels by reduction factor, then restore
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        # Global average pooling: [B, C, 1, 1]
        y = self.avg_pool(x).view(b, c)  # [B, C]
        # MLP processing: [B, C]
        y = self.fc(y)
        # Sigmoid activation: [B, C]
        y = self.sigmoid(y).view(b, c, 1, 1)  # [B, C, 1, 1]
        # Element-wise multiplication with input
        return x * y  # [B, C, H, W]

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Average pooling across channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 7x7 convolution to generate spatial attention map
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        # Average pooling across channels: [B, 1, H, W]
        y = self.avg_pool(x).mean(dim=1, keepdim=True)
        # Convolution to generate spatial attention map: [B, 1, H, W]
        y = self.conv(y)
        # Sigmoid activation: [B, 1, H, W]
        y = self.sigmoid(y)
        # Element-wise multiplication with input
        return x * y  # [B, C, H, W]

class MultiModalDistillation(nn.Module):
    def __init__(self):
        super(MultiModalDistillation, self).__init__()
        # Total input channels after concatenation
        total_channels = cfgs.in_channels_s + cfgs.in_channels_d + cfgs.in_channels_v
        self.reduction = cfgs.reduction
        self.spatial_kernel_size = cfgs.spatial_kernel_size
        # Channel attention block
        self.channel_attention = ChannelAttention(total_channels, self.reduction)
        # Spatial attention block
        self.spatial_attention = SpatialAttention(self.spatial_kernel_size)

    def forward(self, f_s, f_d, f_v):
        """
        Inputs:
            f_s: Semantic segmentation features [B, C_s, H, W]
            f_d: Depth estimation features [B, C_d, H, W]
            f_v: Vector-field prediction features [B, C_v, H, W]
        Output:
            f_e: Enhanced feature map [B, C_s + C_d + C_v, H, W]
        """
        # Step 1: Concatenate features along the channel dimension
        f_fus = torch.cat((f_s, f_d, f_v), dim=1)  # [B, C_s + C_d + C_v, H, W]

        # Step 2: Apply channel attention
        f_fus_v = self.channel_attention(f_fus)  # [B, C_s + C_d + C_v, H, W]

        # Step 3: Apply spatial attention
        f_e = self.spatial_attention(f_fus_v)  # [B, C_s + C_d + C_v, H, W]

        return f_e

# Example usage
if __name__ == "__main__":

    # Create dummy input tensors
    f_s = torch.randn(1, 2, 480, 640)  # Semantic segmentation features
    f_d = torch.randn(1, 80, 480, 640)  # Depth estimation features
    f_v = torch.randn(1, 18, 480, 640)

    model = MultiModalDistillation()
    a = model(f_s, f_d, f_v)
    print(a.shape)