import torch
import torch.nn as nn
import torch.nn.functional as F


class CAM1D(nn.Module):
    """Channel Attention Module for 1D"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        reduced_channels_num = max(1, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, reduced_channels_num, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(reduced_channels_num, in_channels, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        # Global pooling (along sequence length)
        max_pool = F.adaptive_max_pool1d(x, 1)  # (B, C, 1)
        avg_pool = F.adaptive_avg_pool1d(x, 1)  # (B, C, 1)

        # Shared MLP
        max_out = self.mlp(max_pool)
        avg_out = self.mlp(avg_pool)

        attn = self.sigmoid(max_out + avg_out)  # (B, C, 1)
        return x * attn  # broadcast over length


class SAM1D(nn.Module):
    """Spatial Attention Module for 1D"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2  # SAME padding
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        # Pool along channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, L)
        avg_pool = torch.mean(x, dim=1, keepdim=True)    # (B, 1, L)

        concat = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2, L)
        attn = self.sigmoid(self.conv(concat))  # (B, 1, L)
        return x * attn  # broadcast along channels


class CBAM1D(nn.Module):
    """Convolutional Block Attention Module for 1D"""

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = CAM1D(in_channels, reduction_ratio)
        self.spatial_attention = SAM1D(kernel_size)

    def forward(self, x):
        # Apply Channel Attention
        x = self.channel_attention(x)
        # Apply Spatial Attention
        x = self.spatial_attention(x)
        return x


if __name__ == "__main__":
    B, C, L = 8, 64, 128  
    x = torch.randn(B, C, L)

    cbam = CBAM1D(in_channels=C)
    out = cbam(x)
    print(out.shape) 
