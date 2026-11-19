"""IRCADb <-> LiTS eslestirme testleri -- VERI OLMADAN calisir.

Hedeflenen hata: IRCADb DICOM'dan cevrilince eksenler LiTS'e gore ters/yer
degistirmis olabilir. Yone duyarli imza bu durumda kopyayi SESSIZCE kacirir
ve 'IRCADb dis test' yanilgisi dogrulanmis gibi gorunur.
"""
import numpy as np
import pandas as pd

from src.provenance.match_ircadb_lits import (hu_histogram, histogram_distance,
                                              match_table, orientation_variants,
                                              oriented_signatures)

RNG = np.random.default_rng(0)


def _ct(shape=(40, 48, 20), seed=0):
    """Rastgele ama yapili (yumusak) sahte CT, HU araliginda."""
    from scipy import ndimage
    r = np.random.default_rng(seed).normal(size=shape)
    return (ndimage.gaussian_filter(r, 2.0) * 3000).clip(-1000, 1500).astype(np.int16)


def test_orientation_variants_16_farkli_yon_uretir():
    v = _ct()
    variants = orientation_variants(v)
    assert len(variants) == 16
    keys = {a.tobytes() + str(a.shape).encode() for a in variants}
    assert len(keys) == 16


def test_ters_cevrilmis_kopya_yakalanir():
    a = _ct(seed=1)
    b = np.flip(a, axis=(0, 2))          # x ve z ters
    sa = oriented_signatures(a)[0]       # a'nin kendi yonu
    sb = oriented_signatures(b)          # b'nin 16 yonu
    assert float(np.max(sb @ sa)) > 0.999


def test_xy_yer_degistirmis_kopya_yakalanir():
    a = _ct(shape=(40, 48, 20), seed=2)
    b = np.transpose(a, (1, 0, 2))[::-1]
    assert float(np.max(oriented_signatures(b) @ oriented_signatures(a)[0])) > 0.999


def test_farkli_hasta_eslesmez():
    a, b = _ct(seed=3), _ct(seed=4)
    assert float(np.max(oriented_signatures(b) @ oriented_signatures(a)[0])) < 0.9


def test_histogram_yonden_bagimsiz_ve_farkli_hastayi_ayirir():
    a = _ct(seed=5)
    assert histogram_distance(hu_histogram(a), hu_histogram(np.flip(a, 1).T)) == 0.0
    assert histogram_distance(hu_histogram(a), hu_histogram(_ct(seed=6))) > 0.1


def test_match_table_en_iyi_esi_ve_beklenen_kumeyi_raporlar():
    lits = {f"lits-{i}": _ct(seed=100 + i) for i in range(6)}
    ircadb = {"3Dircadb1.1": np.flip(lits["lits-3"], 0), "3Dircadb1.2": lits["lits-5"],
              "3Dircadb1.3": _ct(seed=999)}
    df = match_table(ircadb, lits, expected_lits={"lits-3", "lits-5"})
    assert isinstance(df, pd.DataFrame)
    got = dict(zip(df.ircadb_id, df.best_lits_id))
    assert got["3Dircadb1.1"] == "lits-3" and got["3Dircadb1.2"] == "lits-5"
    row = df.set_index("ircadb_id").loc
    assert bool(row["3Dircadb1.1", "is_match"]) and bool(row["3Dircadb1.2", "is_match"])
    assert not bool(row["3Dircadb1.3", "is_match"])
    # beklenen kume disinda eslesme olmadigi icin uyari sutunu temiz
    assert row["3Dircadb1.1", "in_expected"] and row["3Dircadb1.2", "in_expected"]


# ---------------------------------------------------------------- DICOM -> NIfTI
from pathlib import Path

import pytest

from src.data.ircadb_to_nifti import fraction_inside, is_tumor_mask, numeric_sorted


def test_dilimler_sayisal_siralanir_sozluk_sirasiyla_degil():
    """image_10 image_2'den once gelirse CT ve maske dilimleri kayar."""
    fs = [Path(f"d/image_{i}") for i in (10, 2, 0, 1)]
    assert [f.name for f in numeric_sorted(fs)] == ["image_0", "image_1", "image_2", "image_10"]


def test_indekssiz_veya_tekrarli_dilim_firlatir():
    with pytest.raises(ValueError):
        numeric_sorted([Path("d/image_1"), Path("d/DICOMDIR")])
    with pytest.raises(ValueError):
        numeric_sorted([Path("d/image_1"), Path("d/a_1")])


def test_tumor_maskesi_secimi():
    assert all(map(is_tumor_mask, ["livertumor", "livertumor01", "livertumor1", "livertumors",
                                   "liverTumor_2"]))
    assert not any(map(is_tumor_mask, ["liver", "portalvein", "livercyst", "venacava",
                                       "metastasectomie"]))


def test_karaciger_disi_tumorler_karaciger_gt_sine_girmez():
    """IRCADb 1.5 bobrekustu tumoru, 1.7 'tumor' karaciger disi kitle. 'tumor' alt
    dizgisi kurali ikisini de karaciger GT'sine katiyordu."""
    assert not any(map(is_tumor_mask, ["leftsurretumor", "rightsurretumor", "tumor"]))


def test_karaciger_ici_orani_ve_bos_maske_nan():
    liver = np.zeros((10, 10, 10), bool); liver[2:8, 2:8, 2:8] = True
    t = np.zeros_like(liver); t[3:5, 3:5, 3:5] = True
    assert fraction_inside(t, liver) == 1.0
    t[0, 0, 0] = True
    assert 0.8 < fraction_inside(t, liver) < 1.0
    assert np.isnan(fraction_inside(np.zeros_like(liver), liver))


# ---------------------------------------------------------------- gorsel
from src.viz.overlay import pick_slices, radiological


def test_radyolojik_yon_hastanin_sagi_ekranin_solunda_onu_yukarida():
    """RAS dizisi: x+ = hastanin sagi, y+ = on. Radyolojik ekran: hastanin sagi SOLDA, onu USTTE."""
    a = np.zeros((4, 5))
    a[0, 4] = 1           # x=0 -> hastanin en SOLU; y=4 -> en ON
    s = radiological(a)
    assert s.shape == (5, 4)
    assert s[0, -1] == 1  # ust satir (on), en sag sutun (hastanin solu)


def test_pick_slices_kacirilan_ve_uydurulani_gosterir_ayni_kesitte_birlestirir():
    gt = np.zeros((20, 20, 30), bool); pr = np.zeros_like(gt)
    gt[2:8, 2:8, 2:8] = True; pr[2:8, 2:8, 2:8] = True        # bulunan, buyuk
    gt[12:14, 12:14, 20:22] = True                            # kacirilan = en kucuk, z~20
    pr[15:19, 2:6, 26:29] = True                              # uydurulan, z=27
    picks = dict((z, t) for t, z, _ in pick_slices(gt, pr, (1, 1, 1)))
    assert "EN BUYUK" in picks[2]
    z_miss = [z for z, t in picks.items() if "KACIRILAN" in t]
    assert len(z_miss) == 1 and z_miss[0] in (20, 21) and "EN KUCUK" in picks[z_miss[0]]
    assert "UYDURULAN" in picks[27]


def test_yakinlastirma_kutusu_mm_ile_ve_sinirda_kirpilir():
    from src.viz.overlay import crop_box
    assert crop_box((100, 100), (512, 512, 50), (0.5, 0.5, 2.0), half_mm=10) == (80, 120, 80, 120)
    assert crop_box((5, 500), (512, 512, 50), (1.0, 1.0, 1.0), half_mm=20) == (0, 25, 480, 512)


# ---------------------------------------------------------------- GT envanteri
def test_gt_envanteri_tumorsuz_vakayi_sayimdan_dusurmez():
    from src.data.gt_lesion_stats import lesions_of, summarize
    m = np.zeros((20, 20, 20), bool); m[1:4, 1:4, 1:4] = True; m[10:18, 10:18, 10:18] = True
    rows = lesions_of(m, (1, 1, 1), "a") + lesions_of(np.zeros_like(m), (1, 1, 1), "b")
    df = pd.DataFrame(rows)
    assert len(df[df.lesion > 0]) == 2
    s = summarize(df)
    assert "2 vaka | tumorlu 1 | tumorsuz 1" in s and "2 lezyon" in s


def test_overlay_ham_panel_ve_marj_konturu_gercek_dosyadan(tmp_path):
    """Uydurma piksel yok: PNG uc NIfTI'den uretilir; ham panel + kontur paneli + marj plani."""
    import nibabel as nib
    import numpy as np
    from src.viz.overlay import render_case
    ct = np.full((40, 40, 20), 60.0, np.float32); gt = np.zeros((40, 40, 20), np.uint8); pr = gt.copy()
    gt[15:22, 15:22, 8:12] = 2; pr[16:23, 16:23, 8:12] = 1
    aff = np.diag([1.0, 1.0, 2.5, 1.0])
    for n, a in (("ct", ct), ("gt", gt), ("pr", pr)):
        nib.save(nib.Nifti1Image(a, aff), tmp_path / f"{n}.nii.gz")
    bad = nib.Nifti1Image(ct, np.eye(4))                     # sahte 1x1x1 baslikli CT (LiTS 28-47 gibi)
    nib.save(bad, tmp_path / "ct_bad.nii.gz")
    out = render_case(tmp_path / "ct.nii.gz", tmp_path / "gt.nii.gz", tmp_path / "pr.nii.gz",
                      tmp_path / "a.png", "t", 2, 1, pred_name="LiverJEPA", raw_panel=True, margin_mm=5.0)
    assert out.exists() and out.stat().st_size > 1000
    import pytest
    with pytest.raises(ValueError):
        render_case(tmp_path / "ct_bad.nii.gz", tmp_path / "gt.nii.gz", tmp_path / "pr.nii.gz", tmp_path / "b.png", "t", 2, 1)
    render_case(tmp_path / "ct_bad.nii.gz", tmp_path / "gt.nii.gz", tmp_path / "pr.nii.gz", tmp_path / "c.png",
                "t", 2, 1, ct_header_from_gt=True)


def test_overlay_plan_yalniz_gt_kesisen_bilesenleri_icerir_ve_mm_marj_spacingle():
    import numpy as np
    from src.viz.overlay import plan_mask
    gt = np.zeros((40, 40, 20), bool); gt[10:16, 10:16, 5:8] = True
    pr = np.zeros((40, 40, 20), bool); pr[11:15, 11:15, 5:8] = True      # GT ile kesisen
    pr[30:34, 30:34, 15:18] = True                                        # salt-FP, uzak
    plan = plan_mask(gt, pr, (1.0, 1.0, 2.5), 5.0)
    assert not plan[28:36, 28:36, 13:20].any()                            # FP plana girmez (kural 9)
    zs = np.flatnonzero(plan.any(axis=(0, 1)))
    assert zs.min() == 5 - 2 and zs.max() == 7 + 2                        # 5 mm / 2.5 mm = 2 kesit (kural 7)
    assert plan[11 - 5, 12, 6] and not plan[11 - 6, 12, 6]                # x'te 5 voksel


def test_overlay_pixdim_affine_celisirse_firlatir(tmp_path):
    import nibabel as nib, numpy as np, pytest
    from src.viz.overlay import render_case
    ct = np.full((20, 20, 10), 60.0, np.float32); m = np.zeros((20, 20, 10), np.uint8); m[8:12, 8:12, 4:6] = 1
    aff = np.diag([1.0, 1.0, 2.5, 1.0])
    for n, a in (("gt", m * 2), ("pr", m)):
        nib.save(nib.Nifti1Image(a, aff), tmp_path / f"{n}.nii.gz")
    img = nib.Nifti1Image(ct, aff); img.header.set_zooms((1.0, 1.0, 1.0))  # pixdim sahte, sform gercek
    nib.save(img, tmp_path / "ct.nii.gz")
    with pytest.raises(ValueError):
        render_case(tmp_path / "ct.nii.gz", tmp_path / "gt.nii.gz", tmp_path / "pr.nii.gz", tmp_path / "x.png", "t", 2, 1)
