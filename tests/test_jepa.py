"""JEPA kolu testleri -- saf PyTorch Mamba (mamba_ssm yerine).

Neden: Rad-JEPA 3D `from mamba_ssm import Mamba` ister; mamba_ssm derlenmis CUDA cekirdegi
gerektirir (Kaggle torch 2.10'da hazir paket olmayabilir, Mac'te hic yok). Ayni matematigi saf
PyTorch ile yazan yedek modul, AYNI agirlik adlariyla yuklenmeli ve AYNI ciktiyi vermeli.

Hedeflenen hatalar:
  - Agirlik adi/boyutu farkli -> strict=False yukleme SESSIZCE rastgele agirlikla calisir.
  - Tarama (selective scan) yanlis -> encoder ciktilari anlamsiz ama sekli dogru.
  - Nedensellik bozuk (gelecek token gecmisi etkiliyor) -> Mamba degil.
"""
import os

import pytest
import torch

from src.jepa.mamba_ref import Mamba

ENC = os.environ.get("RADJEPA_ENCODER", "")


def test_agirlik_adlari_ve_boyutlari_rad_jepa_ile_ayni():
    """encoder.pt blocks.0.mamba.* (19 Eyl 2026'da dosyadan okundu)."""
    m = Mamba(d_model=384, d_state=16, d_conv=4, expand=2)
    got = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert got == {
        "A_log": (768, 16), "D": (768,),
        "in_proj.weight": (1536, 384),
        "conv1d.weight": (768, 1, 4), "conv1d.bias": (768,),
        "x_proj.weight": (56, 768),
        "dt_proj.weight": (768, 24), "dt_proj.bias": (768,),
        "out_proj.weight": (384, 768),
    }


def _naive(m: Mamba, x: torch.Tensor) -> torch.Tensor:
    """Bagimsiz, dongulu referans: her ornek / kanal / zaman adimi tek tek (Mamba v1, Gu & Dao 2023)."""
    B, L, _ = x.shape
    di, ds, r = m.d_inner, m.d_state, m.dt_rank
    xz = x @ m.in_proj.weight.T
    xi, z = xz[..., :di], xz[..., di:]
    w, b = m.conv1d.weight[:, 0, :], m.conv1d.bias          # (di, K)
    K = w.shape[1]
    u = torch.zeros_like(xi)
    for t in range(L):
        acc = b.clone()
        for k in range(K):                                  # nedensel: t-K+1 .. t
            s = t - (K - 1) + k
            if s >= 0:
                acc = acc + w[:, k] * xi[:, s, :]
        u[:, t, :] = acc
    u = torch.nn.functional.silu(u)
    dbl = u @ m.x_proj.weight.T
    dt, Bm, Cm = dbl[..., :r], dbl[..., r:r + ds], dbl[..., r + ds:]
    dt = torch.nn.functional.softplus(dt @ m.dt_proj.weight.T + m.dt_proj.bias)
    A = -torch.exp(m.A_log)
    y = torch.zeros_like(u)
    for bi in range(B):
        h = torch.zeros(di, ds)
        for t in range(L):
            h = torch.exp(dt[bi, t, :, None] * A) * h + (dt[bi, t, :, None] * Bm[bi, t, None, :]) * u[bi, t, :, None]
            y[bi, t] = (h * Cm[bi, t, None, :]).sum(-1) + m.D * u[bi, t]
    y = y * torch.nn.functional.silu(z)
    return y @ m.out_proj.weight.T


def test_bagimsiz_dongulu_referansla_ayni():
    torch.manual_seed(0)
    m = Mamba(d_model=16, d_state=4, d_conv=4, expand=2)
    with torch.no_grad():                   # varsayilan ilklendirme cok kucuk: farklari gorunur yap
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    x = torch.randn(2, 11, 16)
    with torch.no_grad():
        assert torch.allclose(m(x), _naive(m, x), atol=1e-5)


def test_nedensel_gelecek_gecmisi_etkilemez():
    torch.manual_seed(1)
    m = Mamba(d_model=16, d_state=4, d_conv=4, expand=2)
    x = torch.randn(1, 20, 16)
    x2 = x.clone(); x2[:, 12:] += 5.0
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6)
    assert not torch.allclose(a[:, 12:], b[:, 12:])


@pytest.mark.skipif(not os.path.exists(ENC), reason="RADJEPA_ENCODER=<encoder.pt> ve RADJEPA_REPO verilmedi")
def test_gercek_rad_jepa_agirliklari_eksiksiz_yuklenir_ve_calisir():
    from src.jepa.radjepa import load_radjepa_encoder
    enc = load_radjepa_encoder(ENC, os.environ["RADJEPA_REPO"], device="cpu")
    with torch.no_grad():
        tok = enc(torch.rand(1, 1, 32, 256, 256), indices=None)
    assert tuple(tok.shape) == (1, 1024, 384) and torch.isfinite(tok).all()
