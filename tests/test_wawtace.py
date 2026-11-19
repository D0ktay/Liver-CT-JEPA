"""WAW-TACE (Zenodo 12741586) -> NIfTI donusturucu testleri. Veri gerektirmez (sentetik NRRD)."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from src.data.wawtace_to_nifti import (  # noqa: E402
    TUMOR_LABEL,
    WawTaceError,
    build_plan,
    convert,
    merge_masks,
    parse_mask_name,
)

SPACING = (0.7, 0.7, 2.5)
DIRECTION = (1, 0, 0, 0, -1, 0, 0, 0, 1)      # gercek veride olculen yon (LPS)


def _img(arr, spacing=SPACING, origin=(-180.0, 179.0, -400.0), direction=DIRECTION):
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing(spacing)
    im.SetOrigin(origin)
    im.SetDirection(direction)
    return im


def _dataset(tmp_path, *, meta_rows, masks, cts, clinical=None):
    """masks: {dosya_adi: (z,y,x) kutu dilimi}, cts: [ct_file_name]."""
    ct_dir, m_dir = tmp_path / "ct", tmp_path / "masks"
    ct_dir.mkdir(), m_dir.mkdir()
    for name in cts:
        sitk.WriteImage(_img(np.full((6, 8, 8), -50, np.int16)), str(ct_dir / f"{name}.nrrd"))
    for fn, sl in masks.items():
        a = np.zeros((6, 8, 8), np.int16)
        a[sl] = 1
        (m_dir / fn.split("_")[0]).mkdir(exist_ok=True)
        sitk.WriteImage(_img(a), str(m_dir / fn.split("_")[0] / fn))
    meta = tmp_path / "meta.csv"
    with meta.open("w", newline="") as f:
        w = csv.DictWriter(f, ["patient_id", "ct_phase", "ct_file_name", "tumor_count", "slice_thickness"])
        w.writeheader()
        w.writerows(meta_rows)
    clin = None
    if clinical is not None:
        clin = tmp_path / "clin.csv"
        with clin.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["PATPRI", "lesions_number"])
            w.writerows(clinical.items())
    return ct_dir, m_dir, meta, clin


def _row(pid, ph, n):
    return {"patient_id": pid, "ct_phase": ph, "ct_file_name": f"{pid}_{ph}", "tumor_count": n,
            "slice_thickness": 2.5}


# ---------------------------------------------------------------- ad kurali
def test_parse_mask_name():
    assert parse_mask_name("10_2_0_tumor_seg.nrrd") == (10, 2, 0)
    assert parse_mask_name("104_1_3_tumor_seg.nrrd") == (104, 1, 3)


@pytest.mark.parametrize("bad", ["63_1_0.nrrd", "10_2_tumor_seg.nrrd", "x_1_0_tumor_seg.nrrd"])
def test_parse_mask_name_kurala_uymayan_ad_firlatir(bad):
    with pytest.raises(WawTaceError):
        parse_mask_name(bad)


# ---------------------------------------------------------------- birlesim
def test_merge_masks_birlesim_ve_etiket_degeri():
    a = np.zeros((3, 4, 4), np.int16); a[0, 0, 0] = 1
    b = np.zeros((3, 4, 4), np.int16); b[2, 3, 3] = 1; b[0, 0, 0] = 1     # ortusme cift sayilmaz
    out = merge_masks([a, b])
    assert out.dtype == np.uint8
    assert int((out == TUMOR_LABEL).sum()) == 2 and set(np.unique(out)) == {0, TUMOR_LABEL}


def test_merge_masks_sekil_uyusmazligi_firlatir():
    with pytest.raises(WawTaceError, match="sekil"):
        merge_masks([np.ones((3, 4, 4), np.int16), np.ones((3, 4, 5), np.int16)])


def test_merge_masks_bos_maske_firlatir():
    """Bos tumor maskesi = bozuk dosya; sessizce 'tumorsuz vaka' uretilmez."""
    with pytest.raises(WawTaceError, match="bos"):
        merge_masks([np.zeros((3, 4, 4), np.int16)])


def test_merge_masks_beklenmeyen_deger_firlatir():
    a = np.zeros((3, 4, 4), np.int16); a[0, 0, 0] = 3
    with pytest.raises(WawTaceError, match="beklenmeyen"):
        merge_masks([a])


def test_merge_masks_segment_indeksi_2_tumor_sayilir():
    """Gercek veri: 377 maskenin 9'unda deger 2 var (7'si yalniz {0,2}, 2'si {0,1,2}); hepsi tek bilesen.
    65_2_0'daki 2-degerli bolge (287 voksel) = 65_2_1'in tamami -> deger tumor SIRA numarasi (Slicer segmenti)."""
    a = np.zeros((3, 4, 4), np.int16); a[0, 0, 0] = 1; a[1, 1, 1] = 2
    b = np.zeros((3, 4, 4), np.int16); b[1, 1, 1] = 2
    out = merge_masks([a, b])
    assert int((out == TUMOR_LABEL).sum()) == 2 and set(np.unique(out)) == {0, TUMOR_LABEL}


# ---------------------------------------------------------------- plan
def test_build_plan_yalniz_maskeli_fazi_secer_ve_tamligi_isaretler(tmp_path):
    ct, m, meta, clin = _dataset(
        tmp_path,
        meta_rows=[_row(10, 0, 0), _row(10, 2, 2), _row(11, 1, 1)],
        masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3], "10_2_1_tumor_seg.nrrd": np.s_[4:5, 5:7, 5:7],
               "11_1_0_tumor_seg.nrrd": np.s_[2:4, 2:4, 2:4]},
        cts=["10_0", "10_2", "11_1"], clinical={10: 2, 11: 3})
    plan = build_plan(meta, m, clin)
    assert [(c.patient_id, c.phase, len(c.masks)) for c in plan.cases] == [(10, 2, 2), (11, 1, 1)]
    by = {c.patient_id: c for c in plan.cases}
    assert by[10].complete is True          # 2 maske >= klinik 2 lezyon
    assert by[11].complete is False         # 1 maske <  klinik 3 lezyon -> negatif bolge guvenilmez
    assert plan.unlabeled_ct == ["10_0"]    # etiketsiz fazlar AYRI listelenir, etiketliymis gibi yazilmaz


def test_build_plan_klinik_tablo_yoksa_tamlik_bilinmiyor(tmp_path):
    _, m, meta, _ = _dataset(tmp_path, meta_rows=[_row(10, 2, 1)],
                             masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3]}, cts=["10_2"])
    assert build_plan(meta, m, None).cases[0].complete is None


def test_build_plan_meta_ile_maske_sayisi_uyusmazsa_firlatir(tmp_path):
    _, m, meta, _ = _dataset(tmp_path, meta_rows=[_row(10, 2, 2)],
                             masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3]}, cts=["10_2"])
    with pytest.raises(WawTaceError, match="tumor_count"):
        build_plan(meta, m, None)


def test_build_plan_metada_olmayan_maske_firlatir(tmp_path):
    _, m, meta, _ = _dataset(tmp_path, meta_rows=[_row(10, 2, 1)],
                             masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3],
                                    "99_1_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3]}, cts=["10_2"])
    with pytest.raises(WawTaceError, match="meta"):
        build_plan(meta, m, None)


def test_build_plan_kurala_uymayan_dosya_raporlanir_ama_kullanilmaz(tmp_path):
    """Gercek zip'te 1 artik dosya var (63_1_0.nrrd). Sessizce yutulmaz: plan.ignored'a yazilir."""
    _, m, meta, _ = _dataset(tmp_path, meta_rows=[_row(10, 2, 1)],
                             masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3]}, cts=["10_2"])
    sitk.WriteImage(_img(np.zeros((6, 8, 8), np.int16)), str(m / "10" / "63_1_0.nrrd"))
    plan = build_plan(meta, m, None)
    assert plan.ignored == ["63_1_0.nrrd"] and len(plan.cases) == 1


# ---------------------------------------------------------------- donusum
def test_convert_uctan_uca(tmp_path):
    ct, m, meta, clin = _dataset(
        tmp_path, meta_rows=[_row(10, 0, 0), _row(10, 2, 2)],
        masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3], "10_2_1_tumor_seg.nrrd": np.s_[4:5, 5:7, 5:7]},
        cts=["10_0", "10_2"], clinical={10: 2})
    out = tmp_path / "out"
    rows = convert(ct, m, meta, out, clinical=clin)
    assert len(rows) == 1 and rows[0]["case_id"] == "wawtace-10-2"
    img = sitk.ReadImage(str(out / "images" / "wawtace-10-2.nii.gz"))
    lab = sitk.ReadImage(str(out / "labels" / "wawtace-10-2.nii.gz"))
    assert lab.GetSize() == img.GetSize() and lab.GetSpacing() == pytest.approx(img.GetSpacing())
    assert lab.GetOrigin() == pytest.approx(img.GetOrigin()) and lab.GetDirection() == pytest.approx(img.GetDirection())
    a = sitk.GetArrayFromImage(lab)
    assert a.dtype == np.uint8 and int((a == TUMOR_LABEL).sum()) == 8 + 4
    assert not (out / "images" / "wawtace-10-0.nii.gz").exists()        # etiketsiz faz yazilmaz
    man = list(csv.DictReader((out / "manifest.csv").open()))
    assert man[0]["n_lesion_masks"] == "2" and man[0]["complete"] == "True" and man[0]["has_liver_label"] == "False"
    rep = json.loads((out / "report.json").read_text())
    assert rep["n_cases"] == 1 and rep["n_unlabeled_ct"] == 1


def test_convert_geometri_uyusmazsa_firlatir(tmp_path):
    """Maske CT izgarasinda degilse yeniden orneklemek YOK -- firlat (mm hatasi sessizce girmesin)."""
    ct, m, meta, _ = _dataset(tmp_path, meta_rows=[_row(10, 2, 1)],
                              masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3]}, cts=["10_2"])
    sitk.WriteImage(_img(np.full((6, 8, 8), -50, np.int16), spacing=(0.7, 0.7, 5.0)), str(ct / "10_2.nrrd"))
    with pytest.raises(WawTaceError, match="spacing"):
        convert(ct, m, meta, tmp_path / "out")


def test_convert_ct_bulunamazsa_firlatir(tmp_path):
    ct, m, meta, _ = _dataset(tmp_path, meta_rows=[_row(10, 2, 1)],
                              masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3]}, cts=[])
    with pytest.raises(WawTaceError, match="CT bulunamadi"):
        convert(ct, m, meta, tmp_path / "out")


def test_convert_gercek_zip_adlandirmasi_scan_sonekli_nifti(tmp_path):
    """Gercek CT zip'leri (21 Eyl 2026, Lambda): <hasta>/<hasta>_<faz>_scan.nii.gz; meta ct_file_name = '<hasta>_<faz>'."""
    ct, m, meta, _ = _dataset(tmp_path, meta_rows=[_row(10, 2, 1)],
                              masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3]}, cts=[])
    (ct / "10").mkdir()
    sitk.WriteImage(_img(np.full((6, 8, 8), -50, np.int16)), str(ct / "10" / "10_2_scan.nii.gz"))
    rows = convert(ct, m, meta, tmp_path / "out")
    assert rows[0]["case_id"] == "wawtace-10-2" and rows[0]["tumor_voxels"] == 8


def test_convert_ayni_ct_iki_adla_varsa_firlatir(tmp_path):
    ct, m, meta, _ = _dataset(tmp_path, meta_rows=[_row(10, 2, 1)],
                              masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3]}, cts=["10_2"])
    sitk.WriteImage(_img(np.full((6, 8, 8), -50, np.int16)), str(ct / "10_2_scan.nii.gz"))
    with pytest.raises(WawTaceError, match="birden fazla"):
        convert(ct, m, meta, tmp_path / "out")


def test_convert_yarim_voksel_alti_origin_farki_ayni_izgara_sayilir(tmp_path):
    """Gercek veri (hasta 86, 382): ayni boyut, origin farki 0.03-0.28 mm, spacing 0.8 vs 0.801 -> ayni izgara.
    Etiket CT'nin geometrisini alir (kayma birikmez)."""
    ct, m, meta, _ = _dataset(tmp_path, meta_rows=[_row(10, 2, 1)],
                              masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3]}, cts=[])
    sitk.WriteImage(_img(np.full((6, 8, 8), -50, np.int16), spacing=(0.7, 0.7, 2.502),
                         origin=(-180.0, 179.0, -400.28)), str(ct / "10_2_scan.nii.gz"))
    out = tmp_path / "out"
    rows = convert(ct, m, meta, out)
    assert len(rows) == 1
    lab = sitk.ReadImage(str(out / "labels" / "wawtace-10-2.nii.gz"))
    assert lab.GetOrigin() == pytest.approx((-180.0, 179.0, -400.28)) and lab.GetSpacing()[2] == pytest.approx(2.502)


def test_convert_exclude_modu_hastayi_TUMUYLE_disarida_birakir_ve_raporlar(tmp_path):
    """Gercek veri: 17 hastada en az bir maske baska seride cizilmis (farkli kesit sayisi/kalinligi).
    Tek maskeyi atip kalaniyla devam etmek YANLIS NEGATIF bolge uretir -> hasta tumuyle dislanir."""
    ct, m, meta, _ = _dataset(
        tmp_path, meta_rows=[_row(10, 2, 2), _row(11, 1, 1)],
        masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3], "10_2_1_tumor_seg.nrrd": np.s_[4:5, 5:7, 5:7],
               "11_1_0_tumor_seg.nrrd": np.s_[2:4, 2:4, 2:4]}, cts=["10_2", "11_1"])
    bad = np.zeros((9, 8, 8), np.int16); bad[1:3, 1:3, 1:3] = 1          # baska seri: 9 kesit
    sitk.WriteImage(_img(bad, spacing=(0.7, 0.7, 1.25)), str(m / "10" / "10_2_1_tumor_seg.nrrd"))
    out = tmp_path / "out"
    rows = convert(ct, m, meta, out, on_mismatch="exclude")
    assert [r["case_id"] for r in rows] == ["wawtace-11-1"]
    assert not (out / "labels" / "wawtace-10-2.nii.gz").exists()
    rep = json.loads((out / "report.json").read_text())
    assert rep["n_excluded"] == 1 and rep["excluded"][0]["case_id"] == "wawtace-10-2"
    assert "10_2_1_tumor_seg.nrrd" in rep["excluded"][0]["reason"]


def test_convert_varsayilan_mod_hala_firlatir(tmp_path):
    ct, m, meta, _ = _dataset(tmp_path, meta_rows=[_row(10, 2, 1)],
                              masks={"10_2_0_tumor_seg.nrrd": np.s_[1:3, 1:3, 1:3]}, cts=["10_2"])
    sitk.WriteImage(_img(np.zeros((9, 8, 8), np.int16)), str(m / "10" / "10_2_0_tumor_seg.nrrd"))
    with pytest.raises(WawTaceError, match="sekil"):
        convert(ct, m, meta, tmp_path / "out")
