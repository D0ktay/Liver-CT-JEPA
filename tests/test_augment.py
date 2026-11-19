"""GPU augmentasyonu: etiket ikili kalir, hiza korunur, aralik tasmaz, tohum tekrarlanir."""
import torch

from src.jepa.augment import DEFAULTS, affine_pair, augment

ALWAYS = {k: 1.0 for k in DEFAULTS if k.startswith("p_")}
NEVER = {k: 0.0 for k in DEFAULTS if k.startswith("p_")}


def _xy():
    x = -torch.ones(2, 1, 32, 32, 32)
    y = torch.zeros(2, 2, 32, 32, 32)
    x[:, :, 8:20, 10:22, 12:24] = 0.5          # "organ"
    y[:, 0, 8:20, 10:22, 12:24] = 1
    y[:, 1, 12:16, 14:18, 16:20] = 1            # "tumor" organin icinde
    return x, y


def test_hepsi_kapaliyken_kimlik():
    x, y = _xy()
    xa, ya = augment(x, y, torch.Generator().manual_seed(0), NEVER)
    assert torch.equal(xa, x) and torch.equal(ya, y)


def test_etiket_ikili_aralik_tasmaz_sekil_korunur():
    x, y = _xy()
    xa, ya = augment(x, y, torch.Generator().manual_seed(1), ALWAYS)
    assert xa.shape == x.shape and ya.shape == y.shape
    assert set(ya.unique().tolist()) <= {0.0, 1.0}
    assert float(xa.min()) >= -1.0 and float(xa.max()) <= 1.0
    assert torch.isfinite(xa).all()


def test_goruntu_ve_etiket_AYNI_donusumu_goruyor():
    """Donmeden sonra 'organ' yogunlugu hala organ etiketinin altinda olmali."""
    x, y = _xy()
    g = torch.Generator().manual_seed(3)
    xa, ya = affine_pair(x[:1], y[:1], 25.0, (0.8, 1.25), g)
    ic = xa[0, 0][ya[0, 0] > 0.5]
    dis = xa[0, 0][ya[0, 0] < 0.5]
    assert ic.numel() > 100
    assert float(ic.mean()) > 0.3            # organ ici ~0.5
    assert float(dis.mean()) < -0.8          # disi ~hava
    # tumor karacigerin ICINDE kalir (bolge iliskisi bozulmaz)
    assert float((ya[0, 1] * (1 - ya[0, 0])).sum()) == 0.0


def test_disaridan_gelen_bolge_HAVA_ile_dolar():
    x = torch.full((1, 1, 16, 16, 16), 0.5)
    y = torch.ones(1, 1, 16, 16, 16)
    xa, _ = affine_pair(x, y, 0.0, (0.5, 0.5), torch.Generator().manual_seed(0))  # uzaklastir
    assert float(xa.min()) < -0.95           # kenarlar -1 (hava), 0.0 (25 HU) DEGIL


def test_ayni_tohum_ayni_sonuc():
    x, y = _xy()
    a = augment(x, y, torch.Generator().manual_seed(7), ALWAYS)
    b = augment(x, y, torch.Generator().manual_seed(7), ALWAYS)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
