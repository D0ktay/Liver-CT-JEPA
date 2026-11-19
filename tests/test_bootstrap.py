"""Guven araligi (hasta duzeyi bootstrap) testleri -- VERI OLMADAN calisir.

Hedeflenen hatalar:
  - Nokta tahmini summary.json'daki sayidan farkli (aralik baska bir sayinin etrafinda).
  - Lezyonlar hastadan bagimsiz yeniden orneklenir -> ayni hastanin 10 lezyonu 10
    bagimsiz gozlem sayilir, aralik yapay olarak daralir.
  - Tumorsuz vakalar orneklemden duser -> spesifiklik araligi hesaplanamaz.
  - Katman bazi tekrarlarda bos -> cokme ya da sessizce 0.
"""
import numpy as np
import pandas as pd
import pytest

from src.metrics.bootstrap import METRICS, bootstrap_ci, ci_table, point_estimates
from src.metrics.evaluate import aggregate, build_per_lesion, evaluate_arrays
from src.metrics.lesionwise import matches_to_records
from src.metrics.targeting import results_to_records

SP = (1.0, 1.0, 1.0)


def _cube(shape, center, half):
    a = np.zeros(shape, dtype=np.uint8)
    z, y, x = center
    a[z-half:z+half+1, y-half:y+half+1, x-half:x+half+1] = 1
    return a


def _synthetic_run():
    """evaluate'in gercekte urettigi tablolar: tumorlu, kacirilan, kaymis, FP'li, tumorsuz."""
    shape = (40, 40, 60)
    cases = {
        "iyi":      (_cube(shape, (20, 20, 15), 4) | _cube(shape, (20, 20, 45), 1),
                     _cube(shape, (20, 20, 15), 4)),                         # kucuk olan kacirildi
        "kaymis":   (_cube(shape, (20, 20, 30), 6), _cube(shape, (20, 22, 31), 6)),
        # bulundu (ortusuyor) ama 8 mm kayik -> 5 mm marj TUTMAZ: "bulunanlarda ihlal" payini sinar
        "cok_kaymis": (_cube(shape, (20, 20, 30), 6), _cube(shape, (20, 28, 30), 6)),
        "fpli":     (_cube(shape, (20, 20, 20), 3),
                     _cube(shape, (20, 20, 20), 3) | _cube(shape, (5, 5, 50), 2)),
        "kacirdi":  (_cube(shape, (20, 20, 30), 5), np.zeros(shape, np.uint8)),
        "temiz":    (np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)),
        "yanlis_alarm": (np.zeros(shape, np.uint8), _cube(shape, (20, 20, 30), 2)),
    }
    ms, ts, rows, les, targ = [], [], [], [], []
    for cid, (gt, pr) in cases.items():
        m, t, c = evaluate_arrays(gt, pr, SP, cid)
        ms += m; ts += t; rows.append(c)
        les += matches_to_records(m, cid); targ += results_to_records(t)
    return (pd.DataFrame(rows), build_per_lesion(les, targ), aggregate(ms, ts, rows))


def test_nokta_tahmini_summary_json_ile_birebir_ayni():
    """Aralik, raporladigimiz sayinin ETRAFINDA olmali; baska bir tanimin degil."""
    per_case, per_lesion, s = _synthetic_run()
    p = point_estimates(per_case, per_lesion)
    to = s["targeting"]["overall"]
    expect = {
        "dice_median": s["voxel"]["dice_median"],
        "lesion_precision": s["lesion"]["precision"],
        "lesion_recall": s["lesion"]["recall"],
        "lesion_f1": s["lesion"]["f1"],
        "specificity_case": s["tumor_free"]["specificity_case"],
        "overlap_hit_rate": to["overlap_hit_rate"],
        "margin_violation_rate_all": to["margin_violation_rate_all"],
        "margin_violation_rate_detected": to["margin_violation_rate_detected"],
        "centroid_error_mm_median": to["centroid_error_mm_median"],
    }
    for stratum, b in s["lesion"]["by_size"].items():
        expect[f"lesion_recall[{stratum}]"] = b["recall"]
    for stratum, b in s["targeting"]["by_size"].items():
        expect[f"overlap_hit_rate[{stratum}]"] = b["overlap_hit_rate"]
        expect[f"margin_violation_rate_all[{stratum}]"] = b["margin_violation_rate_all"]
    assert set(expect) <= set(p), set(expect) - set(p)
    for k, v in expect.items():
        assert p[k] == pytest.approx(v, nan_ok=True), k
    # veri gercekten ayirt edici mi: bulunan+ihlal ve kacirilan lezyon ikisi de var
    assert 0 < expect["margin_violation_rate_detected"] < expect["margin_violation_rate_all"]


def _one_lesion_per_row(violations_by_case: dict[str, list[bool]]):
    """Hedefleme sutunlari dolu, sade bir per_lesion + per_case."""
    rows = []
    for cid, vs in violations_by_case.items():
        for i, v in enumerate(vs, 1):
            rows.append({"case_id": cid, "gt_label": i, "status": "FN" if v else "TP",
                         "detected": True, "margin_violation": v, "stratum": ">20mm",
                         "centroid_error_mm": 1.0})
    per_case = pd.DataFrame({"case_id": list(violations_by_case), "gt_empty": False,
                             "dice": 0.8, "fp_case": False,
                             "n_gt_lesions": [len(v) for v in violations_by_case.values()],
                             "n_pred_lesions": [sum(not x for x in v) for v in violations_by_case.values()]})
    return per_case, pd.DataFrame(rows)


def test_lezyonlar_hastayla_birlikte_orneklenir():
    """A'nin 10 lezyonu hep ihlal, B'nin 10'u hic. Hasta duzeyinde yalniz 3 sonuc
    mumkun: 0, 0.5, 1. Lezyon duzeyi (yanlis) bootstrap 0.3, 0.45... uretirdi."""
    per_case, per_lesion = _one_lesion_per_row({"A": [True] * 10, "B": [False] * 10})
    r = bootstrap_ci(per_case, per_lesion, "margin_violation_rate_all", n_boot=300, seed=0,
                     return_replicates=True)
    assert set(np.round(r["replicates"], 6)) <= {0.0, 0.5, 1.0}
    assert r["n_cases"] == 2


def test_binom_araligi_genisligi_teoriyle_uyumlu():
    """Her hastada 1 lezyon, p=0.3, n=400 -> %95 GA yari genisligi ~1.96*sqrt(p(1-p)/n)=0.045."""
    rng = np.random.default_rng(3)
    v = rng.random(400) < 0.3
    per_case, per_lesion = _one_lesion_per_row({f"c{i}": [bool(x)] for i, x in enumerate(v)})
    r = bootstrap_ci(per_case, per_lesion, "margin_violation_rate_all", n_boot=2000, seed=0)
    half = (r["hi"] - r["lo"]) / 2
    p = v.mean()
    assert half == pytest.approx(1.96 * np.sqrt(p * (1 - p) / 400), rel=0.15)
    assert r["lo"] <= r["point"] <= r["hi"]


def test_ayni_tohum_ayni_sonuc_ve_daha_cok_vaka_daha_dar_aralik():
    rng = np.random.default_rng(4)
    small = _one_lesion_per_row({f"c{i}": [bool(x)] for i, x in enumerate(rng.random(20) < .3)})
    big = _one_lesion_per_row({f"c{i}": [bool(x)] for i, x in enumerate(rng.random(320) < .3)})
    a = bootstrap_ci(*small, "margin_violation_rate_all", n_boot=500, seed=1)
    b = bootstrap_ci(*small, "margin_violation_rate_all", n_boot=500, seed=1)
    assert a == b
    c = bootstrap_ci(*big, "margin_violation_rate_all", n_boot=500, seed=1)
    assert (c["hi"] - c["lo"]) < (a["hi"] - a["lo"]) / 2


def test_tumorsuz_vakalar_orneklemde_kalir_spesifiklik_araligi_var():
    per_case, per_lesion, _ = _synthetic_run()
    r = bootstrap_ci(per_case, per_lesion, "specificity_case", n_boot=300, seed=0)
    assert r["point"] == pytest.approx(0.5)
    assert r["n_valid"] > 0 and 0.0 <= r["lo"] <= r["hi"] <= 1.0


def test_katman_bazi_tekrarlarda_bossa_cokmez_ve_sayar():
    """Tek bir vakada <=10mm lezyon var: onu icermeyen tekrarlarda metrik tanimsiz.
    Tanimsiz tekrar 0 sayilmaz, atlanir ve n_valid ile raporlanir."""
    per_case, per_lesion, s = _synthetic_run()
    r = bootstrap_ci(per_case, per_lesion, "lesion_recall[<=10mm]", n_boot=400, seed=0)
    assert 0 < r["n_valid"] < 400                      # ~(4/6)^6 = %9 tekrarda tanimsiz
    assert r["point"] == pytest.approx(s["lesion"]["by_size"]["<=10mm"]["recall"])  # 1/2


def test_hic_tanimli_tekrar_yoksa_nan():
    per_case, per_lesion = _one_lesion_per_row({"A": [True]})
    r = bootstrap_ci(per_case, per_lesion, "lesion_recall[<=10mm]", n_boot=50, seed=0)
    assert np.isnan(r["point"]) and np.isnan(r["lo"]) and r["n_valid"] == 0


def test_per_lesionda_per_casede_olmayan_vaka_firlatir():
    """Iki tablo ayni kosudan gelmiyorsa sessizce yanlis aralik cikar."""
    per_case, per_lesion = _one_lesion_per_row({"A": [True], "B": [False]})
    with pytest.raises(ValueError, match="per_case"):
        bootstrap_ci(per_case[per_case.case_id == "A"], per_lesion, "lesion_recall", n_boot=10)


def test_tekrarlanan_vaka_kimligi_firlatir():
    per_case, per_lesion = _one_lesion_per_row({"A": [True], "B": [False]})
    with pytest.raises(ValueError, match="tekrar"):
        bootstrap_ci(pd.concat([per_case, per_case]), per_lesion, "lesion_recall", n_boot=10)


def test_bilinmeyen_metrik_firlatir():
    per_case, per_lesion = _one_lesion_per_row({"A": [True]})
    with pytest.raises(KeyError):
        bootstrap_ci(per_case, per_lesion, "dice_mean_typo", n_boot=10)


def test_ci_table_tum_temel_metrikleri_icerir():
    per_case, per_lesion, _ = _synthetic_run()
    df = ci_table(per_case, per_lesion, n_boot=100, seed=0)
    assert set(METRICS) <= set(df.metric)
    assert {"metric", "point", "lo", "hi", "n_valid", "n_boot", "n_cases"} <= set(df.columns)
    ok = df.dropna(subset=["lo"])
    assert (ok.lo <= ok.hi).all()
    # yuzdelik bootstrap'ta nokta araligin disina dusebilir (garanti yok) -> burada sinanmaz


def test_csv_gidis_donus_ayni_sayilari_verir(tmp_path):
    """Gercekte Kaggle'in CSV'leri okunur: True/False/bos hucreler metne donup
    sessizce yanlis sayilmamali (or. 'False' dizgisi bool olarak True)."""
    per_case, per_lesion, _ = _synthetic_run()
    per_case.to_csv(tmp_path / "c.csv", index=False)
    per_lesion.to_csv(tmp_path / "l.csv", index=False)
    a = point_estimates(per_case, per_lesion)
    b = point_estimates(pd.read_csv(tmp_path / "c.csv"), pd.read_csv(tmp_path / "l.csv"))
    assert a.keys() == b.keys()
    for k in a:
        assert b[k] == pytest.approx(a[k], nan_ok=True), k


# ---------------------------------------------------------------- metrik denetcisi bulgulari
def test_per_case_lezyon_sayisi_per_lesion_ile_tutarsizsa_firlatir():
    """Tumorlu vakanin per_lesion satirlari eksikse (or. yanlis birlestirilmis Kaggle
    parcalari) recall sessizce yanlis cikar."""
    per_case, per_lesion = _one_lesion_per_row({"A": [False], "B": [True, True]})
    with pytest.raises(ValueError, match="lezyon sayisi"):
        bootstrap_ci(per_case, per_lesion[per_lesion.case_id == "A"], "lesion_recall", n_boot=10)


def test_bos_girdi_firlatir():
    per_case, per_lesion = _one_lesion_per_row({"A": [True]})
    with pytest.raises(ValueError, match="bos"):
        bootstrap_ci(per_case.iloc[:0], per_lesion.iloc[:0], "lesion_recall", n_boot=10)


def test_tek_lezyonlu_katman_payda_sayisini_raporlar():
    """1 lezyonlu katman: nokta 1.0, GA [1, 1] kesinlik gibi okunur. Payda sayisi
    (1) tabloda gorunmeli ki okuyan bunu anlasin."""
    per_case, per_lesion = _one_lesion_per_row({"A": [False], "B": [True, False, True]})
    per_lesion.loc[0, "stratum"] = "<=10mm"
    r = bootstrap_ci(per_case, per_lesion, "overlap_hit_rate[<=10mm]", n_boot=200, seed=0)
    assert r["n_denominator"] == 1 and r["lo"] == r["hi"] == 1.0 and r["n_valid"] < 200
    assert bootstrap_ci(per_case, per_lesion, "lesion_recall", n_boot=10)["n_denominator"] == 4
    assert bootstrap_ci(per_case, per_lesion, "dice_median", n_boot=10)["n_denominator"] == 2
    df = ci_table(per_case, per_lesion, n_boot=20)
    assert "n_denominator" in df.columns


@pytest.mark.parametrize("bad", ["1", "yes", float("nan")])
def test_bool_sutunu_taninmayan_deger_firlatir(bad):
    """'1'/'yes' False'a, NaN gt_empty -> 'tumorlu vaka' sessizce donuyordu."""
    per_case, per_lesion = _one_lesion_per_row({"A": [True]})
    per_case["gt_empty"] = per_case["gt_empty"].astype(object)
    per_case.loc[0, "gt_empty"] = bad
    with pytest.raises(ValueError):
        point_estimates(per_case, per_lesion)


def test_nokta_tahmini_anizotropik_spacingde_de_summary_ile_ayni():
    shape = (30, 40, 40)
    sp = (2.5, 0.7, 0.7)
    gt = _cube(shape, (15, 20, 10), 3) | _cube(shape, (15, 20, 30), 1)
    pr = _cube(shape, (16, 21, 11), 3)
    ms, ts, rows, les, targ = [], [], [], [], []
    for cid, g, p in (("a", gt, pr), ("b", gt, gt), ("c", np.zeros_like(gt), pr)):
        m, t, c = evaluate_arrays(g, p, sp, cid)
        ms += m; ts += t; rows.append(c)
        les += matches_to_records(m, cid); targ += results_to_records(t)
    s = aggregate(ms, ts, rows)
    p = point_estimates(pd.DataFrame(rows), build_per_lesion(les, targ))
    to = s["targeting"]["overall"]
    assert p["centroid_error_mm_median"] == pytest.approx(to["centroid_error_mm_median"])
    assert p["margin_violation_rate_all"] == pytest.approx(to["margin_violation_rate_all"])
    assert p["dice_median"] == pytest.approx(s["voxel"]["dice_median"])
    for stratum, b in s["lesion"]["by_size"].items():
        assert p[f"lesion_recall[{stratum}]"] == pytest.approx(b["recall"], nan_ok=True)
