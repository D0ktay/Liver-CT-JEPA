"""LiTS 28-47 baslik duzeltmesi testleri -- VERI OLMADAN.

LiTS 28-47'nin (= IRCADb) basliginda spacing sahte 1x1x1. Duzeltilmis kopyalar:
  - goruntu verisine DOKUNMAZ (bayt bayt ayni dizi),
  - yonu korur (affine sutunlari yalniz olceklenir),
  - iki kez uygulanamaz (zaten gercek spacing'li dosyayi 'duzeltmek' sessizce bozar),
  - sekil beklenenle uyusmazsa (yanlis vaka) FIRLATIR.
"""
import numpy as np
import pytest

from src.baseline.respacing import load_spacing_table, reheader


def _save(p, arr, aff):
    import nibabel as nib
    img = nib.Nifti1Image(arr, aff)
    img.header.set_sform(aff, code=2)
    img.header.set_qform(None, code=0)
    nib.save(img, str(p))


def test_tablo_okunur_yorum_satirlari_atlanir():
    t = load_spacing_table("configs/lits_2847_true_spacing.csv")
    assert set(t) == {f"lits-{i}" for i in range(28, 48)}
    assert t["lits-28"]["spacing"] == pytest.approx((0.57, 0.57, 1.6))
    assert t["lits-46"]["spacing"][2] == pytest.approx(4.0) and t["lits-28"]["shape"] == (512, 512, 129)


def test_veri_ayni_spacing_dogru_yon_korunur(tmp_path):
    import nibabel as nib
    arr = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    aff = np.diag([-1.0, 1.0, 1.0, 1.0]); aff[:3, 3] = [10, 20, 30]      # x ters (L), orijin var
    _save(tmp_path / "a.nii", arr, aff)
    reheader(tmp_path / "a.nii", tmp_path / "b.nii.gz", (0.7, 0.7, 2.5), expect_shape=(4, 5, 6))
    im = nib.load(str(tmp_path / "b.nii.gz"))
    assert np.array_equal(np.asarray(im.dataobj), arr) and im.get_data_dtype() == np.int16
    assert im.header.get_zooms()[:3] == pytest.approx((0.7, 0.7, 2.5))
    assert nib.aff2axcodes(im.affine) == nib.aff2axcodes(aff)
    assert np.allclose(im.affine[:3, 3], [10, 20, 30])
    assert np.allclose(np.diag(im.affine)[:3], [-0.7, 0.7, 2.5])


def test_zaten_gercek_spacingli_dosya_ikinci_kez_duzeltilmez(tmp_path):
    arr = np.zeros((4, 5, 6), np.uint8)
    _save(tmp_path / "a.nii", arr, np.diag([0.8, 0.8, 2.0, 1.0]))
    with pytest.raises(ValueError, match="1x1x1"):
        reheader(tmp_path / "a.nii", tmp_path / "b.nii.gz", (0.7, 0.7, 2.5), expect_shape=(4, 5, 6))


def test_sekil_uyusmazsa_firlatir(tmp_path):
    _save(tmp_path / "a.nii", np.zeros((4, 5, 7), np.uint8), np.eye(4))
    with pytest.raises(ValueError, match="sekil"):
        reheader(tmp_path / "a.nii", tmp_path / "b.nii.gz", (0.7, 0.7, 2.5), expect_shape=(4, 5, 6))


def test_mm_metrigi_duzeltilmis_baslikla_degisir(tmp_path):
    """Asil amac: ayni maske, dogru spacing -> farkli mm. evaluate spacing'i basliktan okur."""
    from src.metrics.evaluate import load_mask
    m = np.zeros((20, 20, 20), np.uint8); m[5:10, 5:10, 5:10] = 1
    _save(tmp_path / "a.nii", m, np.eye(4))
    reheader(tmp_path / "a.nii", tmp_path / "b.nii.gz", (0.5, 0.5, 3.0), expect_shape=(20, 20, 20))
    assert load_mask(tmp_path / "a.nii")[1] == pytest.approx((1, 1, 1))
    assert load_mask(tmp_path / "b.nii.gz")[1] == pytest.approx((0.5, 0.5, 3.0))
