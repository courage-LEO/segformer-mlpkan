import torch
import torch.nn as nn
from einops import rearrange


class LayerNorm2d(nn.Module):

    def __init__(self,
                 channels
                 ):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=channels)
        self.norm = nn.LayerNorm(normalized_shape=channels)

    def forward(self, x):
        x_perm = rearrange(x, "n c h w -> n h w c")
        x_perm = self.ln(x_perm)  # 这里 normalized_shape=channels => LN针对最后一维= c
        # 2) 重排回 (N,C,H,W)
        x = rearrange(x_perm, "n h w c -> n c h w")

        # 第二次 LN（若只需一次，可删除以下代码）
        x_perm = rearrange(x, "n c h w -> n h w c")
        x_perm = self.norm(x_perm)
        x = rearrange(x_perm, "n h w c -> n c h w")

        return x
