"""Marj ek payi egrisi testleri -- VERI OLMADAN calisir.

Soru: AI planina 5 mm yerine 5+e mm marj verilseydi ihlal kaca iner, bedeli kac mL
fazladan yakilan doku olur?

Hedeflenen hatalar:
  - Taramadaki kapsama, tek marjli (zaten test edilmis) targeting_errors'tan farkli.
  - Kirpilmis kutuda ablasyon hacmi eksik sayilir (genisleme kutu disina tasar).
  - Anizotropik vokselde marj yanlis (binary_dilation tuzagi).
  - Kacirilan lezyon marj buyudukce 'kurtarilmis' gorunur.
"""
import numpy as np
import pytest
from scipy import ndimage

from src.metrics.lesionwise import connected_components
from src.metrics.margin_sweep import margin_sweep, summarize_sweep
from src.metrics.targeting import targeting_errors

SP = (1.0, 1.0, 1.0)


def _cube(shape, center, half):
    a = np.zeros(shape, dtype=np.uint8)
    z, y, x = center
    a[z-half:z+half+1, y-half:y+half+1, x-half:x+half+1] = 1
    return a


def _random_case(rng, shape=(24, 40, 40)):
    gt = ndimage.binary_dilation(rng.random(shape) > 0.995, iterations=2)
    pred = ndimage.binary_dilation(rng.random(shape) > 0.995, iterations=2) \
        | (gt & (rng.random(shape) > 0.4))
    return gt, pred


def test_her_marjda_targeting_errors_ile_birebir_ayni():
    """Anizotropik rastgele spacing, cok lezyon: tarama = marj basina ayri targeting_errors."""
    rng = np.random.default_rng(11)
    margins = (5.0, 6.0, 7.5, 10.0)
    n_checked = 0
    for _ in range(5):
        sp = tuple(rng.uniform(0.5, 4.0, 3))
        gt, pred = _random_case(rng)
        rows = margin_sweep(gt, pred, sp, "c", margins=margins)
        for m in margins:
            ref = {r.gt_label: r for r in targeting_errors(gt, pred, sp, margin_mm=m)}
            got = {r["gt_label"]: r for r in rows if r["margin_mm"] == m}
            assert got.keys() == ref.keys()
            for k, r in ref.items():
                assert got[k]["detected"] == r.detected
                assert got[k]["margin_coverage"] == pytest.approx(r.margin_coverage)
                assert got[k]["margin_violation"] == r.margin_violation
                assert got[k]["stratum"] == r.stratum
                n_checked += 1
    assert n_checked > 50


def test_ablasyon_hacmi_tam_hacim_kaba_kuvvetle_ayni():
    """Performans icin kutuya kirpiliyor; genisleme kutudan tasarsa hacim eksik cikar."""
    rng = np.random.default_rng(5)
    for _ in range(4):
        sp = tuple(rng.uniform(0.6, 3.0, 3))
        gt, pred = _random_case(rng)
        rows = margin_sweep(gt, pred, sp, "c", margins=(5.0, 9.0), base_margin_mm=5.0)
        gl, _ = connected_components(gt)
        pl, _ = connected_components(pred)
        vox_ml = np.prod(sp) / 1000
        for r in rows:
            g = gl == r["gt_label"]
            ideal = (ndimage.distance_transform_edt(~g, sampling=sp) <= 5.0).sum() * vox_ml
            assert r["ideal_ablation_ml"] == pytest.approx(ideal)
            if not r["detected"]:
                continue
            plan = np.isin(pl, np.unique(pl[g & (pl > 0)]))
            abl = ndimage.distance_transform_edt(~plan, sampling=sp) <= r["margin_mm"]
            assert r["ablation_ml"] == pytest.approx(abl.sum() * vox_ml)
            # bedel = TUM tumorlerin ideal planinin disinda kalan (komsu tumor saglikli doku degil)
            ideal_all = ndimage.distance_transform_edt(~(gt > 0), sampling=sp) <= 5.0
            assert r["extra_vs_ideal_ml"] == pytest.approx((abl & ~ideal_all).sum() * vox_ml)


def test_bilinen_kayma_esigi():
    """Tahmin GT'nin 3 mm kaymis kopyasi: en uzak GT vokseli plana 3 mm.
    <3 mm marjda ihlal, >=3 mm'de degil."""
    shape = (30, 30, 40)
    gt = _cube(shape, (15, 15, 15), 4)
    pred = _cube(shape, (15, 15, 18), 4)
    rows = {r["margin_mm"]: r for r in margin_sweep(gt, pred, SP, margins=(0, 2, 2.9, 3, 4),
                                                   base_margin_mm=0)}
    assert [rows[m]["margin_violation"] for m in (0, 2, 2.9, 3, 4)] == [True] * 3 + [False] * 2


def test_anizotropik_marj_z_ekseninde_dogru():
    """z spacing 5 mm: 1 dilim kayma = 5 mm. 4 mm marj yetmez, 5 mm yeter.
    binary_dilation(iterations=1) 'bir voksel = yeter' derdi."""
    shape = (20, 20, 20)
    gt = _cube(shape, (10, 10, 10), 3)
    pred = _cube(shape, (11, 10, 10), 3)          # eksen 0 = spacing 5 mm
    rows = {r["margin_mm"]: r for r in margin_sweep(gt, pred, (5.0, 1.0, 1.0), margins=(4, 5),
                                                   base_margin_mm=5)}
    assert rows[4]["margin_violation"] and not rows[5]["margin_violation"]


def test_mukemmel_tahminde_ek_maliyet_sifir_ve_egri_monoton():
    shape = (30, 30, 30)
    gt = _cube(shape, (15, 15, 15), 3)
    rows = margin_sweep(gt, gt, SP, margins=(5, 6, 7, 8), base_margin_mm=5)
    by = {r["margin_mm"]: r for r in rows}
    assert by[5]["extra_vs_ideal_ml"] == pytest.approx(0.0)
    assert by[5]["ablation_ml"] == pytest.approx(by[5]["ideal_ablation_ml"])
    abl = [by[m]["ablation_ml"] for m in (5, 6, 7, 8)]
    assert all(a < b for a, b in zip(abl, abl[1:]))


def test_kacirilan_lezyon_hicbir_marjda_kurtarilmaz():
    """Tahmin lezyona degmiyorsa plan yok: marj buyutmek onu kapsamaz, ihlal kalir."""
    shape = (30, 30, 60)
    gt = _cube(shape, (15, 15, 10), 3)
    pred = _cube(shape, (15, 15, 22), 3)          # 6 voksel uzakta, temas yok
    rows = margin_sweep(gt, pred, SP, margins=(5, 10, 20))
    assert all(r["margin_violation"] and not r["detected"] for r in rows)
    assert all(r["ablation_ml"] == 0 and r["margin_coverage"] == 0 for r in rows)


def test_ozet_ihlal_orani_monoton_ve_katmanli():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(6):
        gt, pred = _random_case(rng)
        rows += margin_sweep(gt, pred, (0.8, 0.8, 2.5), f"c{i}", margins=(5, 6, 7, 8, 10))
    s = summarize_sweep(rows)
    rates = [s["overall"][m]["margin_violation_rate_all"] for m in (5, 6, 7, 8, 10)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    det = [s["overall"][m]["margin_violation_rate_detected"] for m in (5, 6, 7, 8, 10)]
    assert all(a >= b for a, b in zip(det, det[1:]))
    n = s["overall"][5]["n_lesions"]
    assert n == len({(r["case_id"], r["gt_label"]) for r in rows})
    assert sum(v[5]["n_lesions"] for v in s["by_size"].values()) == n


@pytest.mark.parametrize("bad", [(), (-1.0,), (float("nan"),), (5, 5)])
def test_gecersiz_marj_listesi_firlatir(bad):
    gt = _cube((20, 20, 20), (10, 10, 10), 2)
    with pytest.raises(ValueError):
        margin_sweep(gt, gt, SP, margins=bad)


def test_sekil_uyusmazligi_firlatir():
    with pytest.raises(ValueError):
        margin_sweep(np.zeros((5, 5, 5)), np.zeros((5, 5, 6)), SP)


def test_komut_satiri_uctan_uca_nifti(tmp_path, monkeypatch):
    """Kaggle'da calisacak yol: pairs.csv + NIfTI -> margin_sweep.csv/json, ardindan
    evaluate + bootstrap -> ci.csv. LiTS GT etiketi 2, TS tahmin etiketi 1."""
    import json
    import sys
    import nibabel as nib
    import pandas as pd
    from src.metrics import bootstrap, evaluate, margin_sweep as ms

    shape = (30, 30, 40)
    aff = np.diag([0.8, 0.8, 2.5, 1.0])
    pairs = []
    for i, shift in enumerate((0, 2, 4)):
        gt = _cube(shape, (15, 15, 15), 4) * 2                       # LiTS: tumor = 2
        pr = _cube(shape, (15, 15 + shift, 15), 4)                   # TS: lezyon = 1
        nib.save(nib.Nifti1Image(gt.astype(np.uint8), aff), str(tmp_path / f"gt{i}.nii.gz"))
        nib.save(nib.Nifti1Image(pr.astype(np.uint8), aff), str(tmp_path / f"pr{i}.nii.gz"))
        pairs.append({"case_id": f"c{i}", "gt_path": tmp_path / f"gt{i}.nii.gz",
                      "pred_path": tmp_path / f"pr{i}.nii.gz"})
    pd.DataFrame(pairs).to_csv(tmp_path / "pairs.csv", index=False)
    out = tmp_path / "eval"

    monkeypatch.setattr(sys, "argv", ["x", "--pairs", str(tmp_path / "pairs.csv"),
                                      "--out", str(out), "--margins", "5,7,10"])
    assert ms.main() == 0
    df = pd.read_csv(out / "margin_sweep.csv")
    assert len(df) == 3 * 3 and df.detected.all()
    s = json.loads((out / "margin_sweep.json").read_text())
    assert s["overall"]["5.0"]["n_lesions"] == 3 and s["failed_cases"] == []

    monkeypatch.setattr(sys, "argv", ["x", "--pairs", str(tmp_path / "pairs.csv"), "--out", str(out)])
    assert evaluate.main() == 0
    monkeypatch.setattr(sys, "argv", ["x", "--eval", str(out), "--n-boot", "50"])
    assert bootstrap.main() == 0
    ci = pd.read_csv(out / "ci.csv")
    summ = json.loads((out / "summary.json").read_text())
    got = ci.set_index("metric").point
    assert got["dice_median"] == pytest.approx(summ["voxel"]["dice_median"])
    assert got["margin_violation_rate_all"] == pytest.approx(
        summ["targeting"]["overall"]["margin_violation_rate_all"])


# ---------------------------------------------------------------- metrik denetcisi bulgulari
def test_paylasilan_plan_komsu_lezyonu_fazla_saymaz():
    """Tek tahmin iki tumore deger: A'nin 'fazla dokusu' B'yi ve B'nin 5 mm'sini
    icermemeli (targeting.excess_volume_ml de plan - TUM GT). Eski tanim 2.49 mL verdi,
    kopru yalniz 0.147 mL."""
    shape = (30, 30, 44)
    gt = _cube(shape, (15, 15, 15), 3) | _cube(shape, (15, 15, 25), 3)   # x: 12-18, 22-28
    pred = gt.copy()
    pred[12:19, 12:19, 19:22] = 1                                          # 7x7x3 kopru
    rows = [r for r in margin_sweep(gt, pred, SP, margins=(5,)) if r["margin_mm"] == 5]
    assert len(rows) == 2 and all(r["detected"] for r in rows)
    abl = ndimage.distance_transform_edt(~(pred > 0)) <= 5
    ideal_all = ndimage.distance_transform_edt(~(gt > 0)) <= 5
    expect = (abl & ~ideal_all).sum() / 1000
    for r in rows:
        assert r["extra_vs_ideal_ml"] == pytest.approx(expect)
        assert r["extra_vs_ideal_ml"] < 0.147          # kopru hacminden bile az


def test_sifir_marj_targeting_errors_ile_ayni():
    rng = np.random.default_rng(21)
    sp = (2.5, 0.8, 0.8)
    gt, pred = _random_case(rng)
    ref = {r.gt_label: r for r in targeting_errors(gt, pred, sp, margin_mm=0.0)}
    got = {r["gt_label"]: r for r in margin_sweep(gt, pred, sp, margins=(0.0,), base_margin_mm=0)}
    assert got.keys() == ref.keys() and len(ref) > 3
    for k, r in ref.items():
        assert got[k]["margin_coverage"] == pytest.approx(r.margin_coverage)


def test_marjlar_ve_temel_marj_configten_ve_bos_cikti_basligi_var(tmp_path, monkeypatch):
    """Tumorsuz calisma: bos CSV (yalniz '\\n') sonra okunamiyordu. Baslik yazilmali.
    Varsayilan marjlar configs/paths.yaml -> metrics.margin_sweep_mm."""
    import json
    import sys
    import nibabel as nib
    import pandas as pd
    import yaml
    from src.metrics import margin_sweep as ms

    cfg = yaml.safe_load(open("configs/paths.yaml"))["metrics"]
    assert [float(x) for x in cfg["margin_sweep_mm"]] == [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    z = np.zeros((10, 10, 10), np.uint8)
    for n in ("gt", "pr"):
        nib.save(nib.Nifti1Image(z, np.eye(4)), str(tmp_path / f"{n}.nii.gz"))
    pd.DataFrame([{"case_id": "c", "gt_path": tmp_path / "gt.nii.gz",
                   "pred_path": tmp_path / "pr.nii.gz"}]).to_csv(tmp_path / "p.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["x", "--pairs", str(tmp_path / "p.csv"),
                                      "--out", str(tmp_path), "--margin-mm", "4"])
    assert ms.main() == 0
    df = pd.read_csv(tmp_path / "margin_sweep.csv")
    assert len(df) == 0 and {"margin_mm", "extra_vs_ideal_ml"} <= set(df.columns)
    params = json.loads((tmp_path / "margin_sweep.json").read_text())["params"]
    assert params["base_margin_mm"] == 4.0 and params["margins_mm"] == [5, 6, 7, 8, 9, 10]
