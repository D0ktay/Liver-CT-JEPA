"""Kirpmaya dayanikli kopya arama testleri (L6) -- VERI OLMADAN calisir.

Hedeflenen hata: TotalSegmentator egitim goruntuleri karaciger cevresine KIRPILMIS.
Sekil/aralik/histogram imzasi kirpilmis kopyayi SESSIZCE kacirir ve
"TS egitim verisinde LiTS yok" yanlis sonucu cikar.
"""
import gzip
import zipfile

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage

from src.provenance.match_crop import (ExactIndex, dihedral, extract_queries,
                                       ncc_search, probe_patches, search)


def _ct(shape=(64, 72, 30), seed=0):
    """Yumusak ama ayirt edici sahte CT, tamsayi HU."""
    r = np.random.default_rng(seed).normal(size=shape)
    return (ndimage.gaussian_filter(r, 1.5) * 2500).clip(-1000, 1500).astype(np.int16)


def _q(arr, spacing=(0.8, 0.8, 1.0), qid="q"):
    return {"id": qid, "probes": probe_patches(arr), "shape": arr.shape, "spacing": spacing}


REF_SP = (0.8, 0.8, 1.0)


def test_dihedral_8_farkli_yon():
    p = np.arange(12).reshape(3, 4)
    keys = {v.tobytes() + str(v.shape).encode() for _, v in dihedral(p)}
    assert len(keys) == 8


def test_probe_tekduze_bolgeyi_secmez():
    a = _ct()
    a[20:44, 20:52, :] = 40                      # merkez tekduze (ayirt edici degil)
    for p in probe_patches(a, size=16):
        assert p["std"] > 15


def test_exact_index_kirpilmis_ve_dondurulmus_yamayi_bulur():
    ref = _ct(seed=1)
    patch = ref[10:26, 30:46, 7]
    q = np.flip(patch.T, axis=0)                 # transpoz + ters
    hits = ExactIndex(ref).find(q)
    assert (hits[0]["x"], hits[0]["y"], hits[0]["z"]) == (10, 30, 7)
    assert hits[0]["variant"] != "T0x0y0"


def test_kirpilmis_kopya_exact_olarak_eslesir():
    ref, other = _ct(seed=1), _ct(seed=2)
    crop = ref[8:52, 5:60, 4:26]
    crop = np.flip(np.swapaxes(crop, 0, 1), axis=(0, 2))   # x/y yer degistir, x ve z ters
    df = search({"lits-1": (other, REF_SP), "lits-2": (ref, REF_SP)}, [_q(crop)])
    r = df.iloc[0]
    assert r.verdict == "exact"
    assert r.ref_id == "lits-2"
    assert r.n_exact >= 2
    assert abs(r.z_scale) == pytest.approx(1.0)


def test_yabanci_hacim_eslesmez():
    df = search({"lits-1": (_ct(seed=1), REF_SP)}, [_q(_ct(seed=3)[5:50, 5:60, 2:25])])
    assert df.iloc[0].verdict == "no_match"


def test_z_yeniden_orneklenmis_kopya_ncc_ile_yakalanir():
    ref = _ct(shape=(64, 72, 40), seed=4)
    crop = ref[6:54, 8:64, 5:35].astype(np.float32)
    res = ndimage.zoom(crop, (1, 1, 1.5), order=1)          # 1.0 mm -> ~0.67 mm
    res = np.rint(res).astype(np.int16)
    df = search({"lits-9": (ref, REF_SP), "lits-8": (_ct(shape=(64, 72, 40), seed=5), REF_SP)},
                [_q(res, spacing=(0.8, 0.8, 0.667))])
    r = df.iloc[0]
    assert r.verdict == "ncc_match", r.to_dict()
    assert r.ref_id == "lits-9"
    assert r.ncc_min >= 0.95
    assert 1.2 < abs(r.z_scale) < 1.8                        # sorgu dilimleri daha sik


def test_yogunluk_kaydirilmis_kopya_ncc_ile_yakalanir():
    ref = _ct(seed=6)
    crop = ref[4:50, 4:60, 3:25].astype(np.int32) + 1024     # orn. uint16'ya kaydirilmis
    df = search({"lits-3": (ref, REF_SP)}, [_q(crop)])
    assert df.iloc[0].verdict == "ncc_match"


def test_ncc_farkli_aralikli_adayi_atlar():
    """NCC pahali: xy araligi uymayan referans denenmez, bu raporlanir."""
    ref = _ct(seed=7)
    crop = ref[4:50, 4:60, 3:25] + 1
    df = search({"lits-4": (ref, (0.6, 0.6, 1.0))}, [_q(crop)])
    r = df.iloc[0]
    assert r.verdict == "no_match"
    assert r.ncc_candidates == 0


def test_ncc_search_konumu_bulur():
    ref = _ct(seed=8).astype(np.float32)
    t = ref[20:36, 11:27, 9] * 2 + 5
    score, x, y, z = ncc_search(ref, t)
    assert (x, y, z) == (20, 11, 9)
    assert score == pytest.approx(1.0, abs=1e-4)


def test_zipten_okuma(tmp_path):
    zp = tmp_path / "ds.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for i, sp in enumerate([(0.7, 0.7, 1.0), (0.8, 0.8, 2.5)]):
            img = nib.Nifti1Image(_ct(seed=10 + i), np.diag([*sp, 1.0]))
            zf.writestr(f"Dataset/imagesTr/case{i}_0000.nii.gz", gzip.compress(img.to_bytes()))
        zf.writestr("Dataset/labelsTr/case0.nii.gz", b"yok")
        zf.writestr("Dataset/dataset.json", b"{}")
    qs = list(extract_queries(zip_path=zp))
    assert [q["id"] for q in qs] == ["case0", "case1"]
    assert qs[1]["spacing"] == pytest.approx((0.8, 0.8, 2.5))
    assert len(qs[0]["probes"]) == 3


# ---- denetim (leakage-auditor) sonrasi: sessiz YANLIS NEGATIF yollari ----
from src.provenance.match_crop import decide_ncc, main  # noqa: E402


def test_bozuk_uye_kosuyu_durdurmaz_ve_kaydedilir(tmp_path):
    zp = tmp_path / "ds.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        img = nib.Nifti1Image(_ct(seed=10), np.eye(4))
        zf.writestr("D/imagesTr/a_0000.nii.gz", gzip.compress(img.to_bytes()))
        zf.writestr("D/imagesTr/b_0000.nii.gz", b"bozuk gzip")
        img4 = nib.Nifti1Image(np.zeros((8, 8, 8, 2), np.int16), np.eye(4))
        zf.writestr("D/imagesTr/c_0000.nii.gz", gzip.compress(img4.to_bytes()))
    out = tmp_path / "probes"
    with pytest.raises(SystemExit, match="2 hata"):
        main(["extract", "--zip", str(zp), "--out", str(out)])
    assert [f.stem for f in out.glob("*.npz")] == ["a"]
    err = (out / "extract_errors.csv").read_text()
    assert "b_0000" in err and "c_0000" in err


def _hit(score, x, y, z):
    return (score, x, y, z)


def _q3(zs=(10, 20, 30), xy=((5, 5), (5, 5), (5, 5)), spacing=(0.8, 0.8, 1.0)):
    return {"id": "q", "shape": (60, 60, 40), "spacing": spacing,
            "probes": [{"x": x, "y": y, "z": z, "std": 100.0, "patch": np.zeros((16, 16))}
                       for (x, y), z in zip(xy, zs)]}


def test_tutarsiz_on_referans_tutarli_ikinciyi_golgelemez():
    """Orta yamada en yuksek skor tutarsiz referansta; tutarli referans secilmeli."""
    q = _q3()
    res = {
        "lits-A": {"T0x0y0": [_hit(0.93, 40, 3, 7), _hit(0.97, 2, 44, 19), _hit(0.92, 30, 30, 1)]},
        "lits-B": {"T0x0y0": [_hit(0.95, 25, 25, 10), _hit(0.94, 25, 25, 20), _hit(0.96, 25, 25, 30)]},
    }
    d = decide_ncc(q, res, {"lits-A": (0.8, 0.8, 1.0), "lits-B": (0.8, 0.8, 1.0)})
    assert d["verdict"] == "ncc_match" and d["ref_id"] == "lits-B"


def test_tek_zayif_yama_dogru_eslesmeyi_bozmaz():
    q = _q3()
    res = {"lits-B": {"T0x0y0": [_hit(0.95, 25, 25, 10), _hit(0.40, 3, 50, 2), _hit(0.96, 25, 25, 30)]}}
    d = decide_ncc(q, res, {"lits-B": (0.8, 0.8, 1.0)})
    assert d["verdict"] == "ncc_match"
    assert d["n_ncc"] == 2


def test_z_olcegi_tutarsizsa_eslesme_sayilmaz():
    """Beklenen z olcegi = ref_dz / q_dz = 1; 3 yama 1 dilim araliginda -> tesaduf."""
    q = _q3()
    res = {"lits-B": {"T0x0y0": [_hit(0.95, 25, 25, 10), _hit(0.94, 25, 25, 11), _hit(0.96, 25, 25, 12)]}}
    d = decide_ncc(q, res, {"lits-B": (0.8, 0.8, 1.0)})
    assert d["verdict"] == "ncc_partial"


def test_tek_guclu_yama_kismi_olarak_raporlanir():
    q = _q3()
    res = {"lits-B": {"T0x0y0": [_hit(0.5, 1, 1, 1), _hit(0.98, 25, 25, 20), _hit(0.5, 9, 9, 9)]}}
    assert decide_ncc(q, res, {"lits-B": (0.8, 0.8, 1.0)})["verdict"] == "ncc_partial"


def test_tek_orta_skorlu_yama_gurultu_sayilir():
    """Gercek IRCADb'de olmayan tarama tek yamada 0.906 verdi -> kismi bile sayilmamali."""
    q = _q3()
    res = {"lits-B": {"T0x0y0": [_hit(0.5, 1, 1, 1), _hit(0.91, 25, 25, 20), _hit(0.5, 9, 9, 9)]}}
    assert decide_ncc(q, res, {"lits-B": (0.8, 0.8, 1.0)})["verdict"] is None


def test_eksen_etiketi_ters_referans_ras_haliyle_de_aranir():
    """Referansin affine'i 3. ekseni S degil gosteriyorsa TS'in RAS kopyasinda aksiyel eksen kayar."""
    ref = _ct(shape=(40, 64, 72), seed=11)
    aff = np.array([[0, 0, 0.8, 0], [0.8, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1]])  # i->A, j->S, k->R
    ras = np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(ref, aff)).dataobj)
    crop = ras[4:60, 3:34, 5:60]
    df = search({"lits-7": (ref, (0.8, 1.0, 0.8), aff)}, [_q(crop, spacing=(0.8, 0.8, 1.0))])
    r = df.iloc[0]
    assert r.verdict == "exact" and r.ref_id == "lits-7@ras"


def test_ncc_butcesi_kesilince_eksik_tarama_gorunur():
    ref = _ct(seed=12)
    crop = ref[4:50, 4:60, 3:25] + 1
    df = search({"lits-1": (ref, REF_SP), "lits-2": (_ct(seed=13), REF_SP)}, [_q(crop)],
                ncc_budget_s=0)
    r = df.iloc[0]
    assert not r.ncc_complete
    assert r.ncc_checked < r.ncc_candidates


def test_tekduze_yamalar_exact_kararina_girmez():
    ref = np.full((60, 60, 30), 40, np.int16)
    q = np.full((40, 40, 20), 40, np.int16)
    df = search({"lits-1": (ref, REF_SP)}, [_q(q)], ncc=False)
    assert df.iloc[0].verdict == "no_match"
