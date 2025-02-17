import torch
import torch.nn as nn
from einops import rearrange
from LayerNorm2d import LayerNorm2d


class MixFFN(nn.Module):

    def __init__(self, in_channels, explanation_ratio):
        super().__init__()
        self.ln = LayerNorm2d(in_channels)
        self.linear1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.linear2 = nn.Conv2d(in_channels * explanation_ratio, in_channels, kernel_size=1)
        self.conv = nn.Conv2d(in_channels, in_channels * explanation_ratio, kernel_size=3, padding="same")
        self.bn = nn.BatchNorm2d(in_channels * explanation_ratio)
        self.gelu = nn.GELU()

    def forward(self, x):
        _x = x
        x = self.ln(x)
        x = self.linear1(x)
        x = self.conv(x)
        x = self.gelu(x)
        x = self.bn(x)
        x = self.linear2(x)
        x = x + _x
        return x
