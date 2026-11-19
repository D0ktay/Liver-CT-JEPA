"""JEPA ozellik cikarma testleri -- sahte kucuk encoder ile (Rad-JEPA gerektirmez).

Hedeflenen hatalar:
  - Ara katman kancalari yanlis katmani yakalar / kaldirilmaz (sonraki cagrilar kirlenir).
  - uint8 goruntu [0,1]'e cevrilmeden encoder'a gider (model [0,1] bekler).
  - Toplu (batch) islemde pencere sirasi karisir.
"""
import numpy as np
import torch
import torch.nn as nn

from src.jepa.extract import encode_tiles, spacing_for


class _Blk(nn.Module):
    def __init__(self, k):
        super().__init__(); self.k = k
    def forward(self, t):
        return t + self.k


class _FakeEnc(nn.Module):
    """(B,1,32,256,256) -> (B,1024,384); token = pencere ortalamasi; blok i +i ekler."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Blk(float(i + 1)) for i in range(12)])
        self.seen = []
    def forward(self, x, indices=None):
        self.seen.append((float(x.min()), float(x.max())))
        t = x.mean(dim=(1, 2, 3, 4))[:, None, None].expand(-1, 1024, 384).clone()
        for b in self.blocks:
            t = b(t)
        return t * 10


def test_katmanlar_sira_ve_olcek():
    enc = _FakeEnc()
    img = np.stack([np.full((32, 256, 256), v, np.uint8) for v in (0, 51, 255, 102, 204)])
    out = encode_tiles(enc, img, device="cpu", batch=2, layers=(3, 7))
    assert set(out) == {"l4", "l8", "out"}
    means = img.reshape(5, -1).mean(1) / 255.0
    # blok 0..3 -> +1+2+3+4 = 10 ; blok 0..7 -> 36 ; son: (mean + 78) * 10
    # float16 saklama: bagil hassasiyet ~1e-3 (36 civarinda adim 0.03, 780 civarinda 0.5)
    assert np.allclose(out["l4"][:, 0, 0], means + 10, atol=0.01)
    assert np.allclose(out["l8"][:, 0, 0], means + 36, atol=0.04)
    assert np.allclose(out["out"][:, 0, 0], (means + 78) * 10, atol=0.5)
    assert np.all(np.diff(out["out"][:, 0, 0].astype(float)[[0, 3, 1]]) != 0)   # sira korundu
    assert all(v.dtype == np.float16 and v.shape == (5, 1024, 384) for v in out.values())
    assert max(m for _, m in enc.seen) <= 1.0          # [0,1] olcekli gitti


def test_kancalar_kaldirilir():
    enc = _FakeEnc()
    encode_tiles(enc, np.zeros((1, 32, 256, 256), np.uint8), device="cpu", batch=1)
    assert all(len(b._forward_hooks) == 0 for b in enc.blocks)


def test_spacing_basliktan_ya_da_tablodan():
    table = {"lits-30": {"spacing": (0.625, 0.625, 1.25)}}
    assert spacing_for("lits-30", (1.0, 1.0, 1.0), table) == (0.625, 0.625, 1.25)
    assert spacing_for("lits-5", (0.7, 0.7, 5.0), table) == (0.7, 0.7, 5.0)
    import pytest
    with pytest.raises(ValueError):          # tabloda olmayan sahte 1x1x1 -> sessizce kullanilmaz
        spacing_for("lits-99", (1.0, 1.0, 1.0), table)
