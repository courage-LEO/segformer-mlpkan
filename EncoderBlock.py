import torch
import torch.nn as nn
from einops import rearrange
from LayerNorm2d import LayerNorm2d
from OverlapPatchMerging import OverlapPatchMerging
from MultiHeadAttention import MultiHeadAttention
from MixFFN import MixFFN


class EncoderBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 input_dim,
                 head_num,
                 reduction_ratio,
                 explanation_ratio,
                 CALayer_num):
        super().__init__()
        self.layer_num = CALayer_num
        self.CALayer = nn.ModuleList([nn.Sequential(
            MultiHeadAttention(in_channels, input_dim, head_num, reduction_ratio=8),
            MixFFN(in_channels=in_channels, explanation_ratio=explanation_ratio)
        )
            for _ in range(CALayer_num)])
        self.OLM = OverlapPatchMerging(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.CALayer[i](x)
        x = self.OLM(x)
        return x
