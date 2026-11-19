"""Encoder/decoder yapi taslari -- ConvNeXt-3D, transformer (SDPA), mm tabanli sin-cos.

Tasarim kisitlari (gerekceli, degistirilmeden once oku):
  * **BatchNorm YOK.** Yama batch'i 1-2; batch istatistigi coper ve cikarimda
    sessizce baska sonuc verir. Her yerde LayerNorm (kanal ekseninde).
  * **Sabit konum gomulmesi YOK.** Konum sin-cos ile ve **mm** cinsinden uretilir;
    boylece 48^3 on egitim agirligi 128^3 ince ayarda ayni anlami tasir ve
    anizotropik spacing sessizce "ayni" saniliyor olmaz.
  * Dikkat yalnizca `F.scaled_dot_product_attention` (flash-attn / xFormers /
    mamba_ssm yerel ortamda yok; torch 2.2.2, Intel Mac).
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(x, trainable_only: bool = True) -> int:
    """Parametre sayisi. `x` bir nn.Module ya da profil cfg'si olabilir."""
    if isinstance(x, Mapping):
        from src.jepa.encoder import LiverEncoder      # gec import: dongu olmasin
        x = LiverEncoder(x)
    return sum(p.numel() for p in x.parameters() if p.requires_grad or not trainable_only)


class LayerNorm3d(nn.Module):
    """Kanal ekseninde LayerNorm, (B, C, D, H, W) duzeninde."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]


class DropPath(nn.Module):
    """Stokastik derinlik. eval() modunda kimlik -- determinizm testi bunu bekler."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class ConvNeXt3dBlock(nn.Module):
    """7^3 depthwise -> LN -> 4x MLP -> layer scale -> DropPath + artik baglanti."""

    def __init__(self, dim: int, drop_path: float = 0.0, mlp_ratio: int = 4,
                 layer_scale: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, mlp_ratio * dim)
        self.fc2 = nn.Linear(mlp_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x).permute(0, 2, 3, 4, 1)            # -> (B, D, H, W, C)
        h = self.fc2(F.gelu(self.fc1(self.norm(h)))) * self.gamma
        return x + self.drop_path(h.permute(0, 4, 1, 2, 3))


class Downsample3d(nn.Module):
    """LN -> 2^3 stride-2 konvolusyon (ConvNeXt indirgeme deseni)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm = LayerNorm3d(cin)
        self.conv = nn.Conv3d(cin, cout, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class TransformerBlock(nn.Module):
    """Pre-LN, SDPA dikkat + 4x MLP. Girdi (B, N, D)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} heads {heads} ile bolunmuyor")
        self.heads = heads
        self.n1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, mlp_ratio * dim)
        self.fc2 = nn.Linear(mlp_ratio * dim, dim)
        self.drop_path = DropPath(drop_path)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(o.transpose(1, 2).reshape(b, n, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self._attn(self.n1(x)))
        return x + self.drop_path(self.fc2(F.gelu(self.fc1(self.n2(x)))))


class CrossBlock(nn.Module):
    """Yalniz capraz-dikkat: sorgu maskeli konumlar, anahtar/deger baglam token'lari."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} heads {heads} ile bolunmuyor")
        self.heads = heads
        self.nq = nn.LayerNorm(dim, eps=1e-6)
        self.nk = nn.LayerNorm(dim, eps=1e-6)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, mlp_ratio * dim)
        self.fc2 = nn.Linear(mlp_ratio * dim, dim)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        b, n, d = q.shape
        m = ctx.shape[1]
        h = self.heads
        qq = self.q(self.nq(q)).reshape(b, n, h, d // h).transpose(1, 2)
        kv = self.kv(self.nk(ctx)).reshape(b, m, 2, h, d // h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qq, kv[0], kv[1])
        q = q + self.proj(o.transpose(1, 2).reshape(b, n, d))
        return q + self.fc2(F.gelu(self.fc1(self.n2(q))))


def sincos_pos_3d(coords_mm: torch.Tensor, dim: int) -> torch.Tensor:
    """(..., 3) mm koordinat -> (..., dim) sin-cos gomme.

    **mm** cinsinden: voksel indeksi kullanilsa 1 mm ve 5 mm spacing ayni gorunur ve
    ag, hedefleme hatasini mm olarak ogrenemez. dim 6'nin kati degilse artan kanallar
    sifir birakilir (kanal sayisini 6'ya zorlamak profil genisliklerini bozardi).
    """
    if coords_mm.shape[-1] != 3:
        raise ValueError(f"coords_mm son ekseni 3 olmali: {tuple(coords_mm.shape)}")
    if dim < 6:
        raise ValueError(f"pos dim en az 6 olmali: {dim}")
    per = dim // 6                                    # eksen basina sin+cos cifti
    i = torch.arange(per, dtype=torch.float32, device=coords_mm.device)
    # dalga boyu 2 mm ... ~1000 mm (bir abdomen CT'sinin tam boyu)
    freq = 1.0 / (500.0 ** (i / max(1, per - 1))) * torch.pi
    out = []
    for a in range(3):
        ang = coords_mm[..., a:a + 1].float() * freq
        out += [torch.sin(ang), torch.cos(ang)]
    pos = torch.cat(out, dim=-1)
    if pos.shape[-1] < dim:                           # 6'ya bolunmeyen artik -> sifir
        pad = torch.zeros(*pos.shape[:-1], dim - pos.shape[-1], device=pos.device)
        pos = torch.cat([pos, pad], dim=-1)
    return pos


def grid_coords_mm(shape, stride: int, spacing, device=None) -> torch.Tensor:
    """Izgara merkezlerinin mm koordinatlari -> (prod(shape), 3)."""
    ax = [(torch.arange(n, dtype=torch.float32, device=device) + 0.5) * stride * float(s)
          for n, s in zip(shape, spacing)]
    g = torch.meshgrid(*ax, indexing="ij")
    return torch.stack([t.reshape(-1) for t in g], dim=-1)
