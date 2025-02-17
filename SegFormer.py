# import torch
import torch.nn as nn
# from einops import rearrange
# from LayerNorm2d import LayerNorm2d
# from OverlapPatchMerging import OverlapPatchMerging
# from MultiHeadAttention import MultiHeadAttention
# from MixFFN import MixFFN
from MLP import AllMLPDecoder
from EncoderBlock1 import EncoderBlock1
from EncoderBlock import EncoderBlock


class SegFormer(nn.Module):

    def __init__(self, input_height, class_num):
        super().__init__()
        self.EncBlock1 = EncoderBlock1(in_channels=3,
                                       out_channels=4,
                                       kernel_size=7,
                                       stride=4,
                                       input_dim=(input_height // 4) ** 2,
                                       head_num=2,
                                       reduction_ratio=8,
                                       explanation_ratio=1,
                                       CALayer_numb=2)
        self.EncBlock2 = EncoderBlock(in_channels=4,
                                      out_channels=8,
                                      kernel_size=3,
                                      stride=2,
                                      input_dim=(input_height // 4) ** 2,
                                      head_num=2,
                                      reduction_ratio=4,
                                      explanation_ratio=1,
                                      CALayer_num=2)
        self.EncBlock3 = EncoderBlock(in_channels=8,
                                      out_channels=16,
                                      kernel_size=3,
                                      stride=2,
                                      input_dim=(input_height // 8) ** 2,
                                      head_num=2,
                                      reduction_ratio=2,
                                      explanation_ratio=1,
                                      CALayer_num=2)
        self.EncBlock4 = EncoderBlock(in_channels=16,
                                      out_channels=32,
                                      kernel_size=3,
                                      stride=2,
                                      input_dim=(input_height // 16) ** 2,
                                      head_num=2,
                                      reduction_ratio=1,
                                      explanation_ratio=1,
                                      CALayer_num=2)
        self.Dec = AllMLPDecoder(4, 8, 16, 32, class_num)

    def forward(self, x):
        x1 = self.EncBlock1(x)
        x2 = self.EncBlock2(x1)
        x3 = self.EncBlock3(x2)
        x4 = self.EncBlock4(x3)
        x = self.Dec(x1, x2, x3, x4)  # Pass individual outputs to the decoder
        #x_combined = torch.cat([x1, x2, x3, x4], dim=1)  # 在通道维度拼接
        #x = self.Dec(x_combined)  # 传递合并后的输入到 KANLayer
        return x
