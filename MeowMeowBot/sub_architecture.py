import torch
import torch.nn as nn
import torch.nn.functional as F

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RoPE2D(nn.Module):
    """(B, dim, H, W) -> (B, H*W, dim) with 2D rotary applied to the tokens.

    First half of the channels is rotated by row index, second half by column
    index. dim must be divisible by 4.
    """

    def __init__(self, dim, base=10000.0):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4"
        axis_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, axis_dim, 2).float() / axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, feat):
        B, C, H, W = feat.shape
        x = feat.flatten(2).transpose(1, 2)  # (B, N, C)

        ys = torch.arange(H, device=x.device).float()
        xs = torch.arange(W, device=x.device).float()
        fy = torch.outer(ys, self.inv_freq)[:, None, :].expand(H, W, -1)
        fx = torch.outer(xs, self.inv_freq)[None, :, :].expand(H, W, -1)
        freqs = torch.cat([fy, fx], dim=-1).reshape(H * W, -1)
        emb = torch.cat([freqs, freqs], dim=-1)  # (N, C)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)

        return x * cos + rotate_half(x) * sin  # (B, N, C)


class Attention(nn.Module):
    """Standard multi-head self-attention: (B, N, dim) -> (B, N, dim)."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class EncoderBlock(nn.Module):
    """Pre-norm block: x + Attn(LN(x)), then x + FFN(LN(x)).  (B, N, dim) -> (B, N, dim)"""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x