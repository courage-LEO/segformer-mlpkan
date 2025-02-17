import torch
import torch.nn as nn
from einops import rearrange
from LayerNorm2d import LayerNorm2d


class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=kernel_size // 2)
        self.ln = LayerNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.ln(x)
        return x