"""JEPA veri hazirligi testleri -- VERI OLMADAN.

Hedeflenen hatalar:
  - Hazirla -> geri koy gidis-donusu maskeyi kaydirir/bozar (JEPA sonucu TS ile ayni cetvelde olculemez).
  - z pencereleri karacigerin bir kismini atlar.
  - Karaciger kutusu mm marjini voksel sayisi sanir (anizotropik vokselde yanlis).
  - Normalizasyon [0,1] disina tasar.
"""
import numpy as np
import pytest

from src.jepa.prep import (TARGET_SPACING, WIN, liver_bbox, normalize, prepare_case,
                           reconstruct, z_windows)


def test_z_pencereleri_her_kesiti_kapsar_ve_sinirda_kalir():
    for nz in (5, 32, 33, 57, 100, 131):
        starts = z_windows(nz, win=32, stride=24)
        covered = np.zeros(max(nz, 32), bool)
        for s in starts:
            covered[s:s + 32] = True
            assert s >= 0 and s + 32 <= max(nz, 32)
        assert covered[:nz].all(), nz


def test_karaciger_kutusu_mm_marji_anizotropik():
    seg = np.zeros((100, 100, 40), np.uint8); seg[40:60, 40:60, 20:22] = 1
    sl = liver_bbox(seg, spacing=(1.0, 1.0, 5.0), margin_mm=10.0)
    assert (sl[0].start, sl[0].stop) == (30, 70)            # 10 voksel
    assert (sl[2].start, sl[2].stop) == (18, 24)            # 10 mm / 5 mm = 2 kesit
    with pytest.raises(ValueError):
        liver_bbox(np.zeros((5, 5, 5), np.uint8), (1, 1, 1))


def test_normalizasyon_araligi():
    x = normalize(np.array([-3000, -160, 40, 240, 3000], np.int16))
    assert x.min() == 0.0 and x.max() == 1.0 and x[2] == pytest.approx(0.5)


def _case(shape=(90, 80, 30), spacing=(0.8, 0.8, 2.5), seed=0):
    rng = np.random.default_rng(seed)
    ct = rng.integers(-200, 300, shape).astype(np.int16)
    seg = np.zeros(shape, np.uint8)
    seg[10:80, 10:70, 3:27] = 1
    seg[30:40, 30:42, 10:16] = 2
    seg[60:63, 20:23, 20:22] = 2
    return ct, seg, spacing


def test_gidis_donus_hedef_spacingde_birebir():
    """Spacing zaten hedefse yeniden ornekleme yok: geri koyulan maske GT'nin TAM kendisi."""
    ct, seg, sp = _case(spacing=TARGET_SPACING)
    c = prepare_case(ct, seg, sp)
    assert c["img"].shape[1:] == (WIN, 256, 256) and c["img"].dtype == np.uint8
    back = reconstruct(c["lab"].astype(np.float32), c["meta"]) > 0.5
    assert back.shape == seg.shape
    assert np.array_equal(back, seg == 2)


def test_gidis_donus_farkli_spacingde_hacim_ve_konum_korunur():
    ct, seg, sp = _case(shape=(120, 110, 60), spacing=(0.6, 0.6, 1.25))
    c = prepare_case(ct, seg, sp)
    back = reconstruct(c["lab"].astype(np.float32), c["meta"]) > 0.5
    gt = seg == 2
    inter = (back & gt).sum()
    assert 2 * inter / (back.sum() + gt.sum()) > 0.85        # yeniden ornekleme kaybi sinirli
    cg = np.argwhere(gt).mean(0); cb = np.argwhere(back).mean(0)
    assert np.all(np.abs(cg - cb) * np.array(sp) < 1.0)      # merkez < 1 mm kaymaz


def test_karaciger_disi_tahmin_sifir_ve_ortusen_pencereler_ortalanir():
    ct, seg, sp = _case(shape=(130, 120, 100), spacing=TARGET_SPACING)
    seg[:] = 0
    seg[30:90, 30:80, 10:84] = 1                               # 74 kesit -> birden fazla pencere
    seg[50:60, 50:60, 40:46] = 2
    c = prepare_case(ct, seg, sp)
    assert c["img"].shape[0] >= 3
    ones = np.ones_like(c["lab"], np.float32)
    back = reconstruct(ones, c["meta"])
    box = c["meta"]["bbox"]
    assert back.max() == pytest.approx(1.0)                   # ortusmede toplam degil ortalama
    outside = np.ones(seg.shape, bool); outside[tuple(slice(a, b) for a, b in box)] = False
    assert outside.sum() > 0.3 * outside.size                 # test gercekten kutu disini sinar
    assert back[outside].max() == 0.0


# ---------------------------------------------------------------- yon (20 Eyl 2026)
# LiTS'te 3 yon var: LAS 33, RAS 35, LPS 63 vaka. Ham dizi kullanilirsa ayni anatomi
# hastadan hastaya ters doner (JEPA on egitimine gore de ters). Hepsi RAS'a, sonra radyolojik gorunume.
def _reoriented(ct, seg, codes, spacing=(0.8, 0.8, 2.5)):
    """RAS dizisini verilen yon koduna cevir; ayni dunya konumunu tasiyan (dizi, affine) dondur."""
    import nibabel as nib
    from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, ornt_transform
    t = ornt_transform(axcodes2ornt("RAS"), axcodes2ornt(codes))
    aff = np.diag(list(spacing) + [1.0]) @ inv_ornt_aff(t, ct.shape)
    return apply_orientation(ct, t), apply_orientation(seg, t), aff


def test_farkli_yonlu_ayni_tarama_ayni_pencereleri_verir():
    import nibabel as nib
    ct, seg, sp = _case(shape=(90, 80, 30), spacing=TARGET_SPACING)
    ref = prepare_case(ct, seg, sp, affine=np.diag([0.8, 0.8, 2.5, 1.0]))
    for codes in ("LAS", "LPS"):
        c2, s2, aff = _reoriented(ct, seg, codes)
        assert "".join(nib.aff2axcodes(aff)) == codes
        other = prepare_case(c2, s2, sp, affine=aff)
        assert np.array_equal(other["img"], ref["img"]), codes
        assert np.array_equal(other["lab"], ref["lab"]), codes


def test_yonlu_gidis_donus_orijinal_diziye_birebir():
    ct, seg, sp = _case(shape=(90, 80, 30), spacing=TARGET_SPACING)
    for codes in ("RAS", "LAS", "LPS"):
        c2, s2, aff = _reoriented(ct, seg, codes)
        c = prepare_case(c2, s2, sp, affine=aff)
        back = reconstruct(c["lab"].astype(np.float32), c["meta"]) > 0.5
        assert back.shape == s2.shape and np.array_equal(back, s2 == 2), codes


def test_radyolojik_gorunum_onu_ustte_hastanin_sagi_solda():
    """RAS dizisinde x+ = hastanin sagi, y+ = on. Radyolojik ekran: hastanin sagi SOLDA, onu USTTE."""
    ct = np.zeros((40, 40, 12), np.int16) - 1000
    seg = np.zeros(ct.shape, np.uint8); seg[5:35, 5:35, 2:10] = 1
    ct[5:8, 30:35, :] = 240          # x kucuk = hastanin SOLU (RAS'ta x+ sag), y buyuk = ON
    c = prepare_case(ct, seg, TARGET_SPACING, affine=np.diag([0.8, 0.8, 2.5, 1.0]))
    img = c["img"][0]
    z = c["meta"]["res_shape"][2] // 2           # dolgu degil, gercek kesit
    ys, xs = np.nonzero(img[z] > 200)
    assert len(ys) > 0
    rows, cols = c["meta"]["res_shape"][1], c["meta"]["res_shape"][0]
    assert ys.mean() < rows / 2        # on -> ust
    assert xs.mean() > cols / 2        # hastanin solu -> ekranin sagi


@pytest.mark.parametrize("codes", ["ARS", "SRA"])     # ARS: x<->y takasi; SRA: dongusel (kendi tersi DEGIL)
def test_eksen_degisimli_yon_spacing_ve_geri_donus(codes):
    """LiTS'te yalniz ters cevirme var; baska veri setinde eksen DEGISIMI olabilir.
    Anizotropik spacing'le: spacing yanlis eksene giderse pencereler, ters yon yanlissa geri donus bozulur."""
    import nibabel as nib
    from nibabel.orientations import apply_orientation, axcodes2ornt, ornt_transform
    ct, seg, _ = _case(shape=(90, 80, 30))
    sp = (0.6, 0.8, 2.5)
    ref = prepare_case(ct, seg, sp, affine=np.diag(list(sp) + [1.0]))
    c2, s2, aff = _reoriented(ct, seg, codes, spacing=sp)
    assert "".join(nib.aff2axcodes(aff)) == codes
    sp2 = tuple(float(z) for z in np.sqrt((aff[:3, :3] ** 2).sum(0)))        # dizinin kendi spacing'i
    other = prepare_case(c2, s2, sp2, affine=aff)
    assert np.array_equal(other["img"], ref["img"]) and np.array_equal(other["lab"], ref["lab"])
    back = reconstruct(other["lab"].astype(np.float32), other["meta"]) > 0.5
    ref_back = reconstruct(ref["lab"].astype(np.float32), ref["meta"]) > 0.5
    expect = apply_orientation(ref_back, ornt_transform(axcodes2ornt("RAS"), axcodes2ornt(codes)))
    assert back.shape == s2.shape and np.array_equal(back, expect)
