"""Donmus Rad-JEPA token'lari + kucuk egitilebilir CNN -> tumor segmentasyonu.

Mimari (pencere 32 x 256 x 256, z-y-x):
  goruntu yolu (egitilebilir): L0 (16ch, 32x256x256) -> L1 (32, 16x128x128) -> L2 (48, 8x64x64)
                               -> L3 (64, 4x32x32)
  JEPA yolu (donmus token): 4., 8., son katman (3 x 384) -> 1x1 conv -> 128ch @ 4x16x16
  cozucu: token -> xy x2 -> +L3 -> x2 -> +L2 -> x2 -> +L1 -> x2 -> +L0 -> 1 logit
use_jepa=False: JEPA yolu YOK (ayni CNN, kontrol kolu) -> "iyilesme JEPA'dan mi" sorusu.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_GRID = (4, 16, 16)
TOKEN_KEYS = ("l4", "l8", "out")


def tokens_to_grid(tok: torch.Tensor) -> torch.Tensor:
    """(B, 1024, C) -> (B, C, 4, 16, 16); Rad-JEPA raster sirasi (gd, gh, gw)."""
    b, n, c = tok.shape
    assert n == TOKEN_GRID[0] * TOKEN_GRID[1] * TOKEN_GRID[2], n
    return tok.transpose(1, 2).reshape(b, c, *TOKEN_GRID)


def _block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True),
                         nn.LeakyReLU(0.01, inplace=True),
                         nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True),
                         nn.LeakyReLU(0.01, inplace=True))


class JepaSegHead(nn.Module):
    def __init__(self, use_jepa: bool = True, tok_dim: int = 384, width=(16, 32, 48, 64), tok_ch: int = 128):
        super().__init__()
        self.use_jepa = use_jepa
        w0, w1, w2, w3 = width
        self.e0 = _block(1, w0)
        self.e1 = _block(w0, w1)          # /(2,2,2)
        self.e2 = _block(w1, w2)          # /(2,2,2)
        self.e3 = _block(w2, w3)          # /(2,2,2) -> 4x32x32
        self.bott = _block(w3, tok_ch)    # /(1,2,2) -> 4x16x16
        if use_jepa:
            self.tok = nn.Sequential(nn.Conv3d(len(TOKEN_KEYS) * tok_dim, tok_ch, 1),
                                     nn.InstanceNorm3d(tok_ch, affine=True), nn.LeakyReLU(0.01, inplace=True))
            self.fuse = _block(2 * tok_ch, tok_ch)
        self.d3 = _block(tok_ch + w3, w3)
        self.d2 = _block(w3 + w2, w2)
        self.d1 = _block(w2 + w1, w1)
        self.d0 = _block(w1 + w0, w0)
        self.head = nn.Conv3d(w0, 1, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, img: torch.Tensor, tok: dict | None = None) -> torch.Tensor:
        x0 = self.e0(img)
        x1 = self.e1(F.max_pool3d(x0, 2))
        x2 = self.e2(F.max_pool3d(x1, 2))
        x3 = self.e3(F.max_pool3d(x2, 2))
        b = self.bott(F.max_pool3d(x3, (1, 2, 2)))
        if self.use_jepa:
            t = torch.cat([tokens_to_grid(tok[k].float()) for k in TOKEN_KEYS], dim=1)
            b = self.fuse(torch.cat([b, self.tok(t)], dim=1))
        y = self.d3(torch.cat([self._up(b, x3), x3], 1))
        y = self.d2(torch.cat([self._up(y, x2), x2], 1))
        y = self.d1(torch.cat([self._up(y, x1), x1], 1))
        y = self.d0(torch.cat([self._up(y, x0), x0], 1))
        return self.head(y)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """BCE + yumusak Dice (batch uzerinden toplu; bos maskede sonlu)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)
    return bce + dice


def make_folds(ids: list[str], has_tumor: dict[str, bool], k: int = 5, seed: int = 0) -> list[dict]:
    """Hasta duzeyi k-kat; tumorlu / tumorsuz ayri karistirilip sirayla dagitilir (tabakali)."""
    rng = random.Random(seed)
    # her grup karistirilir ve en az dolu kattan baslayarak sirayla dagitilir
    buckets = [[] for _ in range(k)]
    for grp in (sorted(c for c in ids if has_tumor[c]), sorted(c for c in ids if not has_tumor[c])):
        g = list(grp); rng.shuffle(g)
        order = sorted(range(k), key=lambda j: (len(buckets[j]), j))
        for i, c in enumerate(g):
            buckets[order[i % k]].append(c)
    return [{"fold": j, "test": sorted(buckets[j]), "train": sorted(c for c in ids if c not in set(buckets[j]))}
            for j in range(k)]
