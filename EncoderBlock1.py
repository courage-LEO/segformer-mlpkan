import torch
import torch.nn as nn
from einops import rearrange
from LayerNorm2d import LayerNorm2d
from OverlapPatchMerging import OverlapPatchMerging
from MultiHeadAttention import MultiHeadAttention
from MixFFN import MixFFN


class EncoderBlock1(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 input_dim,
                 head_num,
                 reduction_ratio,
                 explanation_ratio,
                 CALayer_numb):
        super().__init__()
        self.layer_num = CALayer_numb
        self.OLM = OverlapPatchMerging(in_channels, out_channels, kernel_size, stride)
        self.CALayer = nn.ModuleList([nn.Sequential(
            MultiHeadAttention(out_channels, input_dim, head_num, reduction_ratio=8),
            MixFFN(out_channels, explanation_ratio)
        )
            for _ in range(CALayer_numb)])

    def forward(self, x):
        x = self.OLM(x)
        for i in range(self.layer_num):
            x = self.CALayer[i](x)
        return x
