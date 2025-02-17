import torch
import torch.nn as nn
from einops import rearrange
from LayerNorm2d import LayerNorm2d
from OverlapPatchMerging import OverlapPatchMerging
from MultiHeadAttention import MultiHeadAttention
from MixFFN import MixFFN


class AllMLPDecoder(nn.Module):

    def __init__(self, l1_channels, l2_channels, l3_channels, l4_channels, class_num):
        super().__init__()
        self.dec_layer1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )
        self.dec_layer2 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        )
        self.dec_layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        )
        self.dec_layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=True)
        )
        self.linear1 = nn.Conv2d(64 * 4, 64, kernel_size=1)
        self.linear2 = nn.Conv2d(64, class_num, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x1, x2, x3, x4):
        x1 = self.dec_layer1(x1)
        x2 = self.dec_layer2(x2)
        x3 = self.dec_layer3(x3)
        x4 = self.dec_layer4(x4)
        x = torch.concat([x1, x2, x3, x4], dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.linear2(x)
        return x
