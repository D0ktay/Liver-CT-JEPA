"""Eslestirilmis (ayni vakalar) kol karsilastirmasi -- veri gerektirmez."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.bootstrap import bootstrap_ci
from src.metrics.compare import paired_diff, side_by_side


def _arm(dice: dict[str, float], detected: dict[str, list[bool]]):
    """Vaka basina Dice + GT lezyonlarinin yakalanip yakalanmadigi (hepsi 10-20mm katmani)."""
    pc, pl = [], []
    for cid, d in dice.items():
        det = detected[cid]
        pc.append({"case_id": cid, "gt_empty": False, "dice": d, "fp_case": False,
                   "n_gt_lesions": len(det), "n_pred_lesions": sum(det)})
        for k in det:
            pl.append({"case_id": cid, "status": "TP" if k else "FN", "stratum": "10-20mm",
                       "detected": k, "margin_violation": not k,
                       "centroid_error_mm": 1.0 if k else np.nan})
    return pd.DataFrame(pc), pd.DataFrame(pl)


IDS = [f"lits-{i}" for i in range(12)]
# vakalar arasi buyuk degiskenlik (0.1 .. 0.9): eslestirilmemis GA'lar ust uste biner
BASE = {c: 0.1 + 0.8 * i / 11 for i, c in enumerate(IDS)}


def test_ayni_kol_fark_sifir_ve_aralik_sifir():
    a = _arm(BASE, {c: [True, False] for c in IDS})
    df = paired_diff(*a, *a, n_boot=200, seed=0).set_index("metric")
    r = df.loc["lesion_recall"]
    assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0 and bool(r["covers_zero"])


def test_eslestirme_vakalar_arasi_degiskenligi_siler():
    """A her vakada B'den +0.05 Dice. Tek tek GA'lar ust uste biner, ESLESTIRILMIS fark sifiri dislar."""
    a = _arm({c: v + 0.05 for c, v in BASE.items()}, {c: [True] for c in IDS})
    b = _arm(BASE, {c: [True] for c in IDS})
    ca = bootstrap_ci(*a, "dice_median", n_boot=500, seed=0)
    cb = bootstrap_ci(*b, "dice_median", n_boot=500, seed=0)
    assert ca["lo"] < cb["hi"]                                   # eslestirilmemis: ortusuyor
    r = paired_diff(*a, *b, names=["dice_mean"], n_boot=500, seed=0).iloc[0]
    assert r["delta"] == pytest.approx(0.05) and r["lo"] > 0 and not bool(r["covers_zero"])


def test_fark_gurultu_icindeyse_sifiri_kapsar():
    rng = np.random.default_rng(1)
    a = _arm(BASE, {c: [bool(rng.integers(2)), bool(rng.integers(2))] for c in IDS})
    b = _arm(BASE, {c: [bool(rng.integers(2)), bool(rng.integers(2))] for c in IDS})
    r = paired_diff(*a, *b, names=["lesion_recall"], n_boot=500, seed=0).iloc[0]
    assert r["lo"] <= 0 <= r["hi"] and bool(r["covers_zero"])


def test_nokta_tahminleri_bootstrap_modulu_ile_ayni():
    a = _arm(BASE, {c: [True, False] for c in IDS})
    b = _arm(BASE, {c: [False, False] for c in IDS[:6]} | {c: [True, False] for c in IDS[6:]})
    r = paired_diff(*a, *b, names=["lesion_recall"], n_boot=50, seed=0).iloc[0]
    assert r["a"] == pytest.approx(0.5) and r["b"] == pytest.approx(0.25)
    assert r["delta"] == pytest.approx(0.25) and r["n_cases"] == 12


def test_farkli_vaka_kumesi_firlatir():
    a = _arm(BASE, {c: [True] for c in IDS})
    b = _arm({c: BASE[c] for c in IDS[:-1]}, {c: [True] for c in IDS[:-1]})
    with pytest.raises(ValueError, match="ayni vaka"):
        paired_diff(*a, *b, n_boot=10)


def test_vaka_sirasi_onemsiz():
    a = _arm(BASE, {c: [True, False] for c in IDS})
    b = _arm(BASE, {c: [True, True] for c in IDS[:4]} | {c: [False, False] for c in IDS[4:]})
    b_shuffled = (b[0].iloc[::-1].reset_index(drop=True), b[1].iloc[::-1].reset_index(drop=True))
    r1 = paired_diff(*a, *b, names=["lesion_recall"], n_boot=100, seed=3).iloc[0]
    r2 = paired_diff(*a, *b_shuffled, names=["lesion_recall"], n_boot=100, seed=3).iloc[0]
    assert (r1["delta"], r1["lo"], r1["hi"]) == (r2["delta"], r2["lo"], r2["hi"])


def test_side_by_side_katmanli_duyarlilik_ve_paydalar():
    a = _arm(BASE, {c: [True, False] for c in IDS})
    b = _arm(BASE, {c: [True] for c in IDS})          # side_by_side eslestirme istemez
    t = side_by_side({"A": a, "B": b}, names=["lesion_recall[10-20mm]"]).set_index("metric")
    assert t.loc["lesion_recall[10-20mm]", "A"] == pytest.approx(0.5)
    assert t.loc["lesion_recall[10-20mm]", "A_n"] == 24 and t.loc["lesion_recall[10-20mm]", "B_n"] == 12


def test_farkli_GT_lezyon_sayisi_ayni_zemin_degildir_firlatir():
    """Kollar farkli GT'ye karsi olculduyse (or. eski GT vs gt_fixed) karsilastirma yapilmaz."""
    a = _arm(BASE, {c: [True, False] for c in IDS})
    b = _arm(BASE, {c: [True] for c in IDS})
    with pytest.raises(ValueError, match="ayni GT degil"):
        paired_diff(*a, *b, n_boot=10)


# ------------------------------------------------ metrics-reviewer bulgulari (22 Eyl 2026)
def _arm_full(n=8, fp_b=0, free=2, strata=("<=10mm", ">20mm")):
    """Tumorsuz vaka + FP satiri + iki katman iceren gercekci kol."""
    pc, pl = [], []
    for i in range(n):
        cid = f"lits-{i}"
        empty = i < free
        n_gt = 0 if empty else 2
        n_fp = fp_b if i % 2 else 0
        pc.append({"case_id": cid, "gt_empty": empty, "dice": np.nan if empty else 0.3 + 0.05 * i,
                   "fp_case": bool(empty and n_fp), "n_gt_lesions": n_gt, "n_pred_lesions": (1 if n_gt else 0) + n_fp})
        for j in range(n_gt):
            hit = j == 0
            pl.append({"case_id": cid, "status": "TP" if hit else "FN", "stratum": strata[j],
                       "detected": hit, "margin_violation": not hit, "centroid_error_mm": 1.5 if hit else np.nan})
        for _ in range(n_fp):
            pl.append({"case_id": cid, "status": "FP", "stratum": np.nan, "detected": np.nan,
                       "margin_violation": np.nan, "centroid_error_mm": np.nan})
    return pd.DataFrame(pc), pd.DataFrame(pl)


def test_tumorlu_vakada_tek_kolda_nan_dice_firlatir():
    """Vaka yalniz BIR koldan duserse eslestirme sessizce bozulur (delta 0.05 yerine 0.125 cikiyordu)."""
    a = _arm({c: v + 0.05 for c, v in BASE.items()}, {c: [True] for c in IDS})
    b = _arm(dict(BASE, **{IDS[-1]: np.nan}), {c: [True] for c in IDS})
    with pytest.raises(ValueError, match="Dice"):
        paired_diff(*a, *b, names=["dice_mean"], n_boot=10)


def test_ayni_sayida_farkli_katman_GT_firlatir():
    """Eski GT vs gt_fixed: lezyon SAYISI ayni, cap katmani farkli -> ayni zemin degil."""
    a, b = _arm_full(), _arm_full(strata=(">20mm", ">20mm"))
    with pytest.raises(ValueError, match="katman"):
        paired_diff(*a, *b, n_boot=10)


def test_same_ground_int_float_ve_metin_bool_ayni_sayilir():
    a, b = _arm_full(), _arm_full()
    b[0]["n_gt_lesions"] = b[0]["n_gt_lesions"].astype(float)
    b[0]["gt_empty"] = b[0]["gt_empty"].map({True: "TRUE", False: "false"})
    assert len(paired_diff(*a, *b, names=["lesion_recall"], n_boot=10)) == 1


def test_tek_vaka_firlatir():
    a = _arm({IDS[0]: 0.5}, {IDS[0]: [True]})
    with pytest.raises(ValueError, match="en az 2 vaka"):
        paired_diff(*a, *a, n_boot=10)


def test_tum_nokta_tahminleri_point_estimates_ile_birebir_ve_precision_farki_elle():
    from src.metrics.bootstrap import point_estimates
    a, b = _arm_full(fp_b=0), _arm_full(fp_b=3)                 # B'de FP var, A'da yok
    d = paired_diff(*a, *b, n_boot=50, seed=1).set_index("metric")
    pa, pb = point_estimates(*a), point_estimates(*b)
    for m in pa:
        for col, ref in (("a", pa[m]), ("b", pb[m])):
            assert (np.isnan(d.loc[m, col]) and np.isnan(ref)) or d.loc[m, col] == pytest.approx(ref), m
    assert d.loc["lesion_precision", "a"] == 1.0 and d.loc["lesion_precision", "b"] == pytest.approx(6 / (6 + 12))
    assert d.loc["specificity_case", "a_n"] == 2                   # tumorsuz vakalar orneklemde


def test_uyarilar_kucuk_payda_kismi_tekrar_ve_coklu_karsilastirma():
    from src.metrics.compare import warnings_for
    a, b = _arm_full(fp_b=0), _arm_full(fp_b=3)
    w = " ".join(warnings_for(paired_diff(*a, *b, n_boot=200, seed=0)))
    assert "payda" in w and "specificity_case" in w               # 2 tumorsuz vaka -> guvenilmez
    assert "duzeltilmemis" in w and "birincil metrik" in w        # kural 8: coklu karsilastirma
