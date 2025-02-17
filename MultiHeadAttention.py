import torch
import torch.nn as nn
from einops import rearrange
from LayerNorm2d import LayerNorm2d


class MultiHeadAttention(nn.Module):
        def __init__(self, channels, dim, head_num, reduction_ratio, dropout=0.1):
            super().__init__()
            self.dim = dim
            self.head_num = head_num
            self.r = reduction_ratio
            self.ln1 = LayerNorm2d(channels)
            self.ln2 = nn.LayerNorm(channels)
            self.linear_reduceK = nn.Linear(channels * reduction_ratio, channels, bias=False)
            self.linear_reduceV = nn.Linear(channels * reduction_ratio, channels, bias=False)
            self.linear_q = nn.Linear(dim, dim, bias=False)
            self.linear_k = nn.Linear(dim // reduction_ratio, dim // reduction_ratio, bias=False)
            self.linear_v = nn.Linear(dim // reduction_ratio, dim // reduction_ratio, bias=False)
            self.linear = nn.Linear(dim, dim, bias=False)
            self.soft = nn.Softmax(dim=3)
            self.dropout = nn.Dropout(dropout)


        def split_head(self, x):
            x = torch.tensor_split(x, self.head_num, dim=2)
            x = torch.stack(x, dim=1)
            return x


        def concat_head(self, x):
            x = torch.tensor_split(x, x.size()[1], dim=1)
            x = torch.concat(x, dim=3).squeeze(dim=1)
            return x


        def forward(self, x):
            _x = x
            x = self.ln1(x)
            x = rearrange(x, "a b c d -> a (c d) b")
            q = k = v = x
            k = rearrange(k, "a (cd r) b -> a cd (b r)", r=self.r)
            v = rearrange(v, "a (cd r) b -> a cd (b r)", r=self.r)

            k = self.linear_reduceK(k)
            k = self.ln2(k)
            v = self.linear_reduceV(v)
            v = self.ln2(v)
            q = rearrange(q, "a cd br -> a br cd")
            k = rearrange(k, "a cd br -> a br cd")
            v = rearrange(v, "a cd br -> a br cd")

            q = self.linear_q(q)
            k = self.linear_k(k)
            v = self.linear_v(v)

            q = self.split_head(q)
            k = self.split_head(k)
            v = self.split_head(v)

            q = rearrange(q, "a h br cd -> a h cd br")
            k = rearrange(k, "a h br cd -> a h cd br")
            v = rearrange(v, "a h br cd -> a h cd br")

            qk = torch.matmul(q, torch.transpose(k, 3, 2))
            qk = qk / ((self.dim // self.head_num) ** 0.5)

            softmax_qk = self.soft(qk)
            softmax_qk = self.dropout(softmax_qk)

            qkv = torch.matmul(softmax_qk, v)

            qkv = rearrange(qkv, "a h br cd -> a h cd br")
            qkv = self.concat_head(qkv)
            qkv = self.linear(qkv)

            qkv = rearrange(qkv, "a b (c d) -> a b c d", c=int(self.dim ** 0.5))
            qkv = qkv + _x
            return qkv
