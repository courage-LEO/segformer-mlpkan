import torch
import torch.nn as nn
from einops import rearrange


class LayerNorm2d(nn.Module):

    def __init__(self,
                 channels
                 ):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        x = rearrange(x, "a b c d -> a c d b")
        x = self.ln(x)
        x = rearrange(x, "a c d b -> a b c d")
        return x
