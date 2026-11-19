"""Pipeline testleri -- VERI OLMADAN calisir.

Her test, gercek veride ancak sonuclar bozulduktan sonra fark edilecek bir
hatayi hedefliyor: ezilen dosyalar, bosalan tahminler, kacan kopyalar.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.case_id import assert_unique_ids, case_id_from_path
from src.data.make_splits import LeakageError, apply_ircadb_rule, assert_no_leakage, make
from src.metrics.evaluate import (_json_safe, aggregate, binarize, build_per_lesion,
                                  evaluate_arrays, evaluate_pair, load_metric_config)
from src.metrics.lesionwise import matches_to_records
from src.metrics.targeting import results_to_records
from src.metrics.make_pairs import make_pairs
from src.provenance.dedup_hash import is_mask_path, signature_from_array, similar_pairs
from src.provenance.resolve_canonical import confirmed_pairs, resolve

SP = (1.0, 1.0, 1.0)


def _cube(shape, center, half):
    a = np.zeros(shape, dtype=np.uint8)
    z, y, x = center
    a[z-half:z+half+1, y-half:y+half+1, x-half:x+half+1] = 1
    return a


# ---------------------------------------------------------------- vaka kimligi
def test_ircadb_jenerik_dosya_adi_klasorden_kimlik_alir():
    """IRCADb'de her dosya 'ct.nii.gz'. Dosya adindan kimlik -> tum vakalar
    ayni ciktiya yazilir, 20 vaka 1'e iner."""
    a = case_id_from_path("data/raw/3Dircadb1/3Dircadb1.1/ct.nii.gz")
    b = case_id_from_path("data/raw/3Dircadb1/3Dircadb1.2/ct.nii.gz")
    assert a == "3Dircadb1.1" and b == "3Dircadb1.2"


def test_lits_ct_ve_gt_ayni_kimligi_alir():
    assert case_id_from_path("x/volume-12.nii") == "lits-12"
    assert case_id_from_path("x/segmentation-12.nii") == "lits-12"
    assert case_id_from_path("out/lits-12.nii.gz") == "lits-12"


def test_bdmap_yolun_herhangi_yerinden_okunur():
    assert case_id_from_path("m/BDMAP_00000091/segmentations/liver.nii.gz") == "BDMAP_00000091"


def test_kimlik_cakismasi_sessizce_ezilmez():
    with pytest.raises(ValueError, match="cakisiyor"):
        assert_unique_ids([Path("kopya1/vaka7/ct.nii.gz"), Path("kopya2/vaka7/ct.nii.gz")])


def test_make_pairs_lits_gt_ile_ts_ciktisini_eslestirir():
    df, gt_only, pr_only = make_pairs(
        [Path("L/segmentation-1.nii"), Path("L/segmentation-2.nii")],
        [Path("o/lits-1.nii.gz"), Path("o/lits-9.nii.gz")])
    assert df["case_id"].tolist() == ["lits-1"]
    assert gt_only == ["lits-2"] and pr_only == ["lits-9"]


# ---------------------------------------------------------------- dedup + kanonik
def test_dedup_imzasi_ayni_taramayi_yakalar_farkliyi_ayirir():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 100, (40, 40, 40))
    b = a + rng.normal(0, 1, a.shape)          # ayni tarama, hafif gurultu
    c = rng.normal(0, 100, (40, 40, 40))       # baska tarama
    sigs = np.stack([signature_from_array(x) for x in (a, b, c)])
    pairs = similar_pairs(sigs, 0.99, block=2)  # blok siniri ciftleri bolmemeli
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_dedup_mask_modu_etiket_degerinden_bagimsiz():
    """Ayni tarama, farkli etiket semasi (1 vs 2) -> ayni imza."""
    m = _cube((40, 40, 40), (20, 20, 20), 8)
    s1 = signature_from_array(m, kind="mask")
    s2 = signature_from_array(m * 2, kind="mask")
    assert float(s1 @ s2) == pytest.approx(1.0, abs=1e-4)  # float32


def _prov():
    return pd.DataFrame({
        "volume_id": ["BDMAP_1", "BDMAP_2", "BDMAP_3", "BDMAP_4"],
        "canonical_candidate": ["LITS|MSD_TASK03::38", "LITS|MSD_TASK03::38",
                                "FLARE23::7", "KITS23::1"],
        "source_family": ["LITS|MSD_TASK03", "LITS|MSD_TASK03", "FLARE23", "KITS23"],
        "source_case": ["liver_38", "liver_38", "7", "1"],
    })


def test_onayli_icerik_kopyasi_farkli_kaynaklari_birlestirir():
    """FLARE23::7 isimden LiTS 38 oldugu anlasilmiyor; dedup yakaladi, elle onaylandi."""
    dups = pd.DataFrame({"a": ["x/BDMAP_1/ct.nii.gz", "x/BDMAP_4/ct.nii.gz"],
                         "b": ["x/BDMAP_3/ct.nii.gz", "x/BDMAP_2/ct.nii.gz"],
                         "confirmed": [1, 0]})
    out = resolve(_prov(), confirmed_pairs(dups))
    cid = dict(zip(out["volume_id"], out["canonical_id"]))
    assert cid["BDMAP_1"] == cid["BDMAP_2"] == cid["BDMAP_3"]
    assert cid["BDMAP_4"] != cid["BDMAP_1"]      # confirmed=0 birlestirmez


def test_onaysiz_dedup_kullanilmaz():
    with pytest.raises(SystemExit, match="confirmed"):
        confirmed_pairs(pd.DataFrame({"a": ["x"], "b": ["y"], "cosine": [0.999]}))


def test_k3_ircadb_kopyasini_kanonik_duzeyde_cikarir():
    """LiTS 38 = IRCADb. Baska isimle (FLARE23::7) gelen kopyasi da egitimden cikmali."""
    prov = resolve(_prov(), [("BDMAP_1", "BDMAP_3")])
    prov["source_case_num"] = [38, 38, 7, 1]
    prov["split"] = "train"
    cfg = {"leakage": {"lits_ircadb_indices": list(range(28, 48))}}
    out = apply_ircadb_rule(prov, cfg)
    excl = set(out.loc[out["split"] == "excluded_ircadb_overlap", "volume_id"])
    assert excl == {"BDMAP_1", "BDMAP_2", "BDMAP_3"}


def test_k1_ayni_kanonik_iki_splitte_ise_firlatir():
    df = pd.DataFrame({"canonical_id": ["a", "a"], "split": ["train", "val"],
                       "source_family": ["S", "S"]})
    with pytest.raises(LeakageError, match="K1"):
        assert_no_leakage(df)


# ---------------------------------------------------------------- degerlendirme
def test_yanlis_etiket_uyari_verir(capsys):
    """TotalSegmentator lezyon=1, LiTS tumor=2. Tahmine 2 uygulanirsa maske bosalir."""
    pred = _cube((20, 20, 20), (10, 10, 10), 2)
    b = binarize(pred, 2, "PRED", "c1")
    assert not b.any()
    assert "etiket 2 YOK" in capsys.readouterr().out


def test_tumorsuz_vaka_dicea_katilmaz_fp_olarak_sayilir():
    shape = (40, 40, 40)
    empty = np.zeros(shape, np.uint8)
    fp_pred = _cube(shape, (20, 20, 20), 2)
    tumor = _cube(shape, (20, 20, 20), 4)

    m1, t1, c1 = evaluate_arrays(tumor, tumor, SP, "tumorlu")
    m2, t2, c2 = evaluate_arrays(empty, fp_pred, SP, "tumorsuz")
    assert np.isnan(c2["dice"]) and c2["fp_case"] and c2["gt_empty"]

    s = aggregate(m1 + m2, t1 + t2, [c1, c2])
    assert s["voxel"]["dice_median"] == pytest.approx(1.0)   # 0 ile kirlenmedi
    assert s["n_cases_tumor_free"] == 1
    assert s["tumor_free"]["specificity_case"] == pytest.approx(0.0)
    assert s["lesion"]["fp"] == 1 and s["lesion"]["tp"] == 1


def test_summary_gecerli_json():
    """json.dumps NaN'i 'NaN' yazar -- gecersiz JSON, baska araclar okuyamaz."""
    txt = json.dumps(_json_safe({"a": float("nan"), "b": [np.float32(1.5)], "c": np.int64(3)}),
                     allow_nan=False)
    assert json.loads(txt) == {"a": None, "b": [1.5], "c": 3}


def test_per_lesion_fp_satiri_hedefleme_almaz_ve_sema_sabit():
    shape = (40, 40, 60)
    gt = _cube(shape, (20, 20, 15), 3)
    pred = _cube(shape, (20, 20, 15), 5) | _cube(shape, (20, 20, 50), 2)   # genis TP-adayi + FP
    m, t, _ = evaluate_arrays(gt, pred, SP, "c")
    df = build_per_lesion(matches_to_records(m, "c"), results_to_records(t))
    gt_row = df[df["gt_label"].notna()].iloc[0]
    fp_row = df[df["status"] == "FP"].iloc[0]
    # tespitte FN (IoU<0.5) ama hedeflemede bulundu ve marj tuttu
    assert gt_row["status"] == "FN" and bool(gt_row["detected"]) and not bool(gt_row["margin_violation"])
    assert pd.isna(fp_row["detected"]) and pd.isna(fp_row["centroid_error_mm"])
    empty = build_per_lesion([], [])
    assert list(empty.columns) == list(df.columns)            # tumorsuz calisma: ayni sema


def test_gt_var_tahmin_bos_dice_sifir_ve_medyana_girer():
    shape = (30, 30, 30)
    gt = _cube(shape, (15, 15, 15), 3)
    m1, t1, c1 = evaluate_arrays(gt, gt, SP, "iyi")
    m2, t2, c2 = evaluate_arrays(gt, np.zeros(shape, np.uint8), SP, "kacirdi")
    assert c2["dice"] == 0.0
    s = aggregate(m1 + m2, t1 + t2, [c1, c2])
    assert s["voxel"]["dice_median"] == pytest.approx(0.5) and s["voxel"]["n_dice_nan"] == 0


def _save(path, arr, zooms=(1.0, 1.0, 1.0), flip=False):
    import nibabel as nib
    aff = np.diag([*zooms, 1.0])
    if flip:
        aff[0, 0] = -aff[0, 0]
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), aff), str(path))


@pytest.mark.parametrize("kind", ["shape", "spacing", "affine"])
def test_evaluate_pair_grid_uyusmazliginda_firlatir(tmp_path, kind):
    """Ayni sekil + farkli yonelim/spacing: maskeler sessizce kaymis karsilastirilir."""
    m = _cube((20, 20, 20), (10, 10, 10), 3)
    _save(tmp_path / "gt.nii.gz", m)
    if kind == "shape":
        _save(tmp_path / "pr.nii.gz", np.zeros((20, 20, 21)))
    elif kind == "spacing":
        _save(tmp_path / "pr.nii.gz", m, zooms=(1.0, 1.0, 2.5))
    else:
        _save(tmp_path / "pr.nii.gz", m, flip=True)
    with pytest.raises(ValueError):
        evaluate_pair(tmp_path / "gt.nii.gz", tmp_path / "pr.nii.gz", "c", gt_label=None, pred_label=None)


def test_metrik_sabitleri_configten_okunur():
    cfg = load_metric_config()
    assert cfg["strata"] == (10.0, 20.0) and cfg["margin_mm"] == 5.0 and cfg["iou"] == 0.5


# ---------------------------------------------------------------- sizinti denetcisi bulgulari
@pytest.mark.parametrize("path,is_mask", [
    ("data/raw/Task03_Liver/imagesTr/liver_3.nii.gz", False),   # MSD CT -- 'liver' iceriyor!
    ("data/raw/Task03_Liver/labelsTr/liver_3.nii.gz", True),
    ("data/raw/LiTS/volume-3.nii", False),
    ("data/raw/LiTS/segmentation-3.nii", True),
    ("data/raw/AA/BDMAP_00000001/ct.nii.gz", False),
    ("data/raw/AA/mask_only/BDMAP_00000001/segmentations/liver.nii.gz", True),
    ("data/raw/3Dircadb1/3Dircadb1.1/ct.nii.gz", False),
    ("data/raw/3Dircadb1/3Dircadb1.1/tumor.nii.gz", True),
    ("data/raw/KiTS23/case_00001/imaging.nii.gz", False),
    ("data/raw/KiTS23/case_00001/segmentation.nii.gz", True),
])
def test_maske_ayiklama_ct_yi_disarida_birakmaz(path, is_mask):
    """Eski desen 'liver' kelimesiyle MSD CT'lerini de atiyordu -> MSD<->LiTS hic dogrulanmiyordu."""
    assert is_mask_path(path) is is_mask


def test_kits_imaging_klasorden_kimlik_alir():
    assert case_id_from_path("K/case_00001/imaging.nii.gz") == "case_00001"


def test_k3_ham_lits_dosyasiyla_onaylanan_kopya_da_cikar():
    """AbdomenAtlas'ta LiTS_38 satiri YOK, ama FLARE23::7'nin ham lits-38 ile ayni
    tarama oldugu dedup'ta onaylandi. Kanonik kok 'FLARE23' secilse bile cikmali."""
    prov = _prov().iloc[2:].reset_index(drop=True)          # sadece FLARE23::7, KITS23::1
    out = resolve(prov, [("BDMAP_3", "lits-38")])
    out["source_case_num"] = [7, 1]
    out["split"] = "train"
    cfg = {"leakage": {"lits_ircadb_indices": list(range(28, 48))}}
    res = apply_ircadb_rule(out, cfg)
    assert res.loc[res["volume_id"] == "BDMAP_3", "split"].item() == "excluded_ircadb_overlap"
    assert res.loc[res["volume_id"] == "BDMAP_4", "split"].item() == "train"


def test_incelenmemis_yeni_aday_varsa_eski_onay_reddedilir():
    reviewed = pd.DataFrame({"a": ["x/BDMAP_1/ct.nii.gz"], "b": ["x/BDMAP_3/ct.nii.gz"], "confirmed": [1]})
    current = pd.DataFrame({"a": ["x/BDMAP_1/ct.nii.gz", "x/BDMAP_2/ct.nii.gz"],
                            "b": ["x/BDMAP_3/ct.nii.gz", "x/BDMAP_4/ct.nii.gz"]})
    with pytest.raises(SystemExit, match="incelenmemis"):
        confirmed_pairs(reviewed, current=current)


def test_canonical_id_yoksa_split_kurulmaz(tmp_path):
    """Sessiz geri donus = dedup onayini atlayan yol."""
    _prov().to_csv(tmp_path / "p.csv", index=False)
    with pytest.raises(LeakageError, match="canonical_id"):
        make(tmp_path / "p.csv", Path("configs/paths.yaml"), tmp_path / "s.csv")
