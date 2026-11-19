"""On egitim veri hatti (geometri katmani) -- VERI OLMADAN calisir.

Her test, gercek veride ancak mm cinsinden sonuclar yanlis cikinca fark edilecek
bir hatayi hedefliyor: kaybolan spacing, maskede ara deger, yon karisikligi,
kirpilan pencere.
"""
import numpy as np
import pytest

from src.jepa.data import (
    GeometryError,
    achieved_spacing,
    canonical_ras,
    hu_window,
    pad_to_patch,
    patch_starts,
    resample,
    sample_patch_origin,
)


# ------------------------------------------------------------------- spacing
def test_yeniden_ornekleme_gercek_spacingi_dondurur():
    """Sekil yuvarlandigi icin ULASILAN spacing istenen spacing DEGILDIR.
    Istenen degeri dondurmek mm hatasini sessizce buyutur."""
    a = np.zeros((10, 10, 7), np.float32)
    out, sp = resample(a, (1.0, 1.0, 3.0), (1.0, 1.0, 2.0), order=1)
    # 7 kesit x 3.0 mm = 21 mm; 2.0 mm istenince 10.5 kesit cikar -> 10'a yuvarlanir
    assert out.shape == (10, 10, 10)
    assert sp[2] == pytest.approx(21.0 / 10)         # 2.1 mm -- istenen 2.0 DEGIL (%5 sapma)
    assert sp[2] != 2.0


def test_achieved_spacing_hacmi_korur():
    for n, s, t in [(7, 3.0, 2.0), (131, 0.7, 1.5), (1, 5.0, 1.0)]:
        m = max(1, int(round(n * s / t)))
        assert achieved_spacing(n, s, m) == pytest.approx(n * s / m)


def test_ayni_spacing_kopya_yapmaz():
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    out, sp = resample(a, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), order=1)
    assert np.array_equal(out, a) and sp == (1.0, 2.0, 3.0)


def test_maske_yeniden_ornekleme_ara_deger_uretmez():
    """order>0 ile maske ornekleme 0.5 gibi degerler uretir -> etiket bozulur."""
    m = np.zeros((8, 8, 8), np.uint8)
    m[2:6, 2:6, 2:6] = 1
    out, _ = resample(m, (1.0, 1.0, 1.0), (0.7, 0.7, 0.7), order=0)
    assert set(np.unique(out)).issubset({0, 1})


def test_maske_order_sifirdan_buyukse_firlatir():
    m = np.zeros((4, 4, 4), np.uint8)
    with pytest.raises(GeometryError, match="maske"):
        resample(m, (1.0, 1.0, 1.0), (0.5, 0.5, 0.5), order=1, is_mask=True)


def test_sifir_veya_negatif_spacing_firlatir():
    a = np.zeros((4, 4, 4), np.float32)
    with pytest.raises(GeometryError):
        resample(a, (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), order=1)


# ----------------------------------------------------------------------- HU
def test_hu_penceresi_sifir_bir_araliginda():
    ct = np.array([-1000, -200, 25, 250, 3000], np.float32)
    out = hu_window(ct, (-200, 250))
    assert out.min() == 0.0 and out.max() == 1.0
    assert out[2] == pytest.approx((25 + 200) / 450)


def test_hu_penceresi_ters_verilirse_firlatir():
    with pytest.raises(GeometryError):
        hu_window(np.zeros(3, np.float32), (250, -200))


# ---------------------------------------------------------------------- yon
def test_ras_donusumu_spacingi_birlikte_tasir():
    """Eksenler yer degistirince spacing de yer degistirmeli; yoksa mm yanlis."""
    arr = np.zeros((4, 5, 6), np.float32)
    aff = np.diag([-1.0, -1.0, 1.0, 1.0])          # LPS -> RAS'a donecek
    out, sp, ornt = canonical_ras(arr, aff, (0.7, 0.8, 2.5))
    assert out.shape == (4, 5, 6)
    assert sp == (0.7, 0.8, 2.5)


def test_eksen_takasinda_spacing_de_takas_olur():
    arr = np.zeros((4, 5, 6), np.float32)
    aff = np.array([[0, 1.0, 0, 0], [1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    out, sp, _ = canonical_ras(arr, aff, (0.7, 0.8, 2.5))
    assert out.shape == (5, 4, 6)
    assert sp == (0.8, 0.7, 2.5)


# -------------------------------------------------------------------- yamalar
def test_patch_starts_tum_ekseni_kapsar():
    st = patch_starts(100, 32, 24)
    assert st[0] == 0 and st[-1] == 100 - 32
    assert all(b - a <= 24 for a, b in zip(st, st[1:]))


def test_patch_starts_hacim_yamadan_kucukse_tek_pencere():
    assert patch_starts(10, 32, 24) == [0]


def test_pad_to_patch_sekli_buyutur_ve_dolguyu_bildirir():
    a = np.ones((10, 40, 40), np.float32)
    out, pad = pad_to_patch(a, (32, 32, 32))
    assert out.shape == (32, 40, 40)
    assert pad == ((0, 22), (0, 0), (0, 0))
    assert out[10:].sum() == 0


def test_sample_patch_origin_sinirlari_asmaz():
    rng = np.random.default_rng(0)
    for _ in range(50):
        o = sample_patch_origin((40, 50, 60), (32, 32, 32), rng)
        assert all(0 <= s and s + p <= n for s, p, n in zip(o, (32, 32, 32), (40, 50, 60)))


def test_sample_patch_origin_odakli_ornekleme_odagi_icerir():
    """Tumorlu pencere olasiligi: odak verildiginde yama odagi ICERMELI."""
    rng = np.random.default_rng(0)
    focus = (5, 45, 55)
    for _ in range(50):
        o = sample_patch_origin((40, 50, 60), (32, 32, 32), rng, focus=focus)
        assert all(s <= f < s + p for s, f, p in zip(o, focus, (32, 32, 32)))


def test_sample_patch_origin_hacim_kucukse_firlatir():
    rng = np.random.default_rng(0)
    with pytest.raises(GeometryError, match="dolgu"):
        sample_patch_origin((10, 50, 60), (32, 32, 32), rng)


def test_pad_to_patch_dolgu_degeri_secilebiliyor():
    """Dolgu `hu_window`'dan SONRA yapiliyor: normalize uzayda 0, (-1,1) araliginda
    25 HU (yumusak doku) demektir. Hava/pencere alti dolgu icin `fill` gerekir."""
    a = np.ones((10, 40, 40), np.float32)
    out, pad = pad_to_patch(a, (32, 32, 32), fill=-1.0)
    assert out.shape == (32, 40, 40) and pad == ((0, 22), (0, 0), (0, 0))
    assert float(out[10:].min()) == float(out[10:].max()) == -1.0
    assert float(out[:10].min()) == 1.0                    # veri bozulmadi
    esiz, _ = pad_to_patch(a, (32, 32, 32))                # varsayilan degismedi
    assert float(esiz[10:].max()) == 0.0
