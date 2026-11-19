"""GPU uzerinde veri cogaltma (augmentasyon) -- ince ayar icin.

NEDEN: MVE kosusunda (21 Eyl 2026) ince ayar HIC augmentasyon kullanmiyordu. 104 egitim
vakasi + on binlerce adim = ezber. nnU-Net'in LiTS'te calisan tarifinin yarisi budur
(donme, olcek, ayna, gama, gurultu, bulaniklik, dusuk cozunurluk benzetimi).

Tasarim:
  * Hepsi torch ile, yama GPU'ya ciktiktan SONRA -> CPU darbogazi yok.
  * Uzamsal donusum goruntu ve etikete AYNI izgarayla uygulanir; goruntu trilineer,
    ETIKET EN YAKIN KOMSU (ara deger = bozuk etiket; `data.resample` ile ayni kural).
  * Goruntu [-1, 1] araligindadir ve -1 = havadir (HU penceresinin alt ucu). Donme ile
    disaridan gelen bolge HAVA ile dolar, 0.0 (= 25 HU, yumusak doku) ile DEGIL.
  * Yogunluk donusumleri yalniz goruntuye; sonunda [-1, 1]'e kirpilir.
  * `rng` bir **CPU** torch.Generator: kararlar ondan cekilir (GPU esitlemesi yok);
    ayni tohum -> ayni karar dizisi. GPU'da yalniz gurultu kuresel tohumdan gelir.

`anizotropi` benzetimi onemli: LiTS'in ve MSD Task08'in bir kismi 5 mm kesitli;
1 mm'ye orneklenince z ekseninde bulanik. Ince kesitli vakalarda bunu taklit etmek
modeli kalin kesite dayanikli yapar.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEFAULTS = {
    "p_affine": 0.35, "rot_deg": 20.0, "scale": (0.8, 1.25),
    "p_flip": 0.5,
    "p_noise": 0.15, "noise_sigma": 0.05,
    "p_blur": 0.15, "blur_sigma": (0.5, 1.2),
    "p_contrast": 0.2, "contrast": (0.75, 1.3),
    "p_bright": 0.2, "bright": 0.1,
    "p_gamma": 0.3, "gamma": (0.7, 1.5),
    "p_lowres": 0.2, "lowres_z": (2, 4),
}


# Rastgele KARARLAR her zaman CPU ureticisinden cekilir: CUDA ureticisinden skaler cekmek
# her seferinde GPU-CPU esitlemesi (sync) demek -- ornek basina ~15, adim basina ~60 kez.
def _u(rng, lo: float, hi: float, device=None) -> float:
    return float(lo + (hi - lo) * torch.rand((), generator=rng))


def _p(rng, p: float, device=None) -> bool:
    return bool(torch.rand((), generator=rng) < p)


def _rot_matrix(ax: float, ay: float, az: float) -> torch.Tensor:
    cx, sx, cy, sy, cz, sz = (math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay),
                              math.cos(az), math.sin(az))
    rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
    return rz @ ry @ rx


def affine_pair(x: torch.Tensor, y: torch.Tensor, rot_deg: float, scale, rng) -> tuple:
    """Tek ornek (1,C,D,H,W): ayni rastgele donme+olcek goruntu ve etikete."""
    dev = x.device
    r = math.radians(rot_deg)
    m = _rot_matrix(_u(rng, -r, r, dev), _u(rng, -r, r, dev), _u(rng, -r, r, dev))
    m = m * _u(rng, 1.0 / scale[1], 1.0 / scale[0], dev)      # izgara olcegi = 1/yakinlastirma
    theta = torch.cat([m, torch.zeros(3, 1)], 1)[None].to(dev)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # goruntu: hava (-1) ile dolsun diye +1 kaydirip sifir dolguyla ornekle
    xo = F.grid_sample(x + 1.0, grid, mode="bilinear", padding_mode="zeros",
                       align_corners=False) - 1.0
    yo = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return xo, yo


def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = max(3, int(2 * round(3 * sigma) + 1))
    i = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    g = torch.exp(-(i ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    for d in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + d] = k
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = k // 2
        x = F.conv3d(F.pad(x, pad, mode="replicate"), g.reshape(shape))
    return x


def augment(x: torch.Tensor, y: torch.Tensor, rng: torch.Generator | None = None,
            cfg: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (B,1,D,H,W) [-1,1] · y: (B,C,D,H,W) {0,1}. Ornek basina bagimsiz cogaltma."""
    if x.ndim != 5 or y.ndim != 5 or x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[2:]:
        raise ValueError(f"sekil uyusmuyor: x {tuple(x.shape)} y {tuple(y.shape)}")
    c = dict(DEFAULTS, **(cfg or {}))
    dev = x.device
    xs, ys = [], []
    for b in range(x.shape[0]):
        xi, yi = x[b:b + 1].float(), y[b:b + 1].float()
        if _p(rng, c["p_affine"], dev):
            xi, yi = affine_pair(xi, yi, c["rot_deg"], c["scale"], rng)
        for d in (2, 3, 4):
            if _p(rng, c["p_flip"], dev):
                xi, yi = xi.flip(d), yi.flip(d)
        if _p(rng, c["p_lowres"], dev):
            f = int(torch.randint(c["lowres_z"][0], c["lowres_z"][1] + 1, (), generator=rng))
            ax = int(torch.randint(0, 3, (), generator=rng))
            full = list(xi.shape[2:])
            low = list(full)
            low[ax] = max(1, full[ax] // f)
            xi = F.interpolate(F.interpolate(xi, size=low, mode="nearest"), size=full,
                               mode="trilinear", align_corners=False)
        if _p(rng, c["p_blur"], dev):
            xi = _gauss_blur(xi, _u(rng, *c["blur_sigma"], dev))
        if _p(rng, c["p_noise"], dev):
            s = _u(rng, 0.0, c["noise_sigma"], dev)
            noise = torch.randn(xi.shape, generator=rng, dtype=xi.dtype) if dev.type == "cpu" \
                else torch.randn(xi.shape, device=dev, dtype=xi.dtype)   # GPU: kuresel tohum
            xi = xi + s * noise
        if _p(rng, c["p_bright"], dev):
            xi = xi + _u(rng, -c["bright"], c["bright"], dev)
        if _p(rng, c["p_contrast"], dev):
            mu = xi.mean()
            xi = (xi - mu) * _u(rng, *c["contrast"], dev) + mu
        if _p(rng, c["p_gamma"], dev):
            u = ((xi.clamp(-1, 1) + 1) / 2).clamp_min(1e-6)
            xi = u.pow(_u(rng, *c["gamma"], dev)) * 2 - 1
        xs.append(xi.clamp(-1.0, 1.0))
        ys.append((yi > 0.5).to(y.dtype))
    return torch.cat(xs).to(x.dtype), torch.cat(ys)
