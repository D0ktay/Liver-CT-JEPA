"""Degerlendirme ciktisi -> panel belgesi donusturucu testleri (veri gerektirmez).

Hedeflenen hatalar:
  - Panel sayisi summary.json'dan sapar (grafikte baska, raporda baska sayi).
  - Farkli vaka kumeleri yan yana konur: TS 126 vakada, JEPA 131 vakada olculup
    "JEPA daha iyi" denir. --common-with ortak kumeye indirger.
  - Ortak kume bos/eksikken sessizce devam edilir.
  - NaN / sonsuz deger JSON'a yazilir (panel "NaN" basar).
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.viz.panel_data import margin_doc, run_doc


def _write_eval(d, case_ids, dices):
    """Kucuk ama tutarli bir degerlendirme klasoru: her vakada 1 GT + 1 tahmin lezyonu."""
    d.mkdir(parents=True, exist_ok=True)
    pc = pd.DataFrame({"case_id": case_ids, "gt_empty": False, "dice": dices,
                       "n_gt_lesions": 1, "n_pred_lesions": 1, "fp_case": False,
                       "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0})
    pl = pd.DataFrame({"case_id": case_ids, "gt_label": 1, "pred_label": 1.0, "iou": 0.8,
                       "gt_voxels": 100, "pred_voxels": 100, "gt_diameter_mm": 15.0,
                       "status": "TP", "stratum": "10-20mm", "detected": True,
                       "centroid_error_mm": 1.0, "margin_coverage": 1.0,
                       "margin_violation": False, "margin_mm": 5.0})
    pc.to_csv(d / "per_case.csv", index=False)
    pl.to_csv(d / "per_lesion.csv", index=False)
    (d / "summary.json").write_text(json.dumps(
        {"n_cases": len(case_ids), "n_failed": 0, "failed_cases": [],
         "voxel": {"dice_median": float(np.median(dices))}}))
    return d


def test_vaka_sayisi_ve_olcutler_summary_ile_tutarli(tmp_path):
    ids = [f"lits-{i}" for i in range(12)]
    ev = _write_eval(tmp_path / "ev", ids, [0.7] * 12)
    doc = run_doc(ev, model="ts", label="TS")
    assert doc["n_cases"] == 12
    assert doc["metrics"]["dice_median"]["p"] == pytest.approx(0.7)
    assert doc["model"] == "ts"


def test_ortak_kumeye_indirger(tmp_path):
    """TS 10 vaka, JEPA 12 vaka -> kiyas 10 ortak vakada yapilmali."""
    ts = _write_eval(tmp_path / "ts", [f"lits-{i}" for i in range(10)], [0.5] * 10)
    jepa_ids = [f"lits-{i}" for i in range(12)]
    # fazladan iki vaka cok yuksek Dice: filtre calismazsa ortalama yukari kayar
    jepa = _write_eval(tmp_path / "jepa", jepa_ids, [0.5] * 10 + [0.99, 0.99])
    tam = run_doc(jepa, model="jepa", label="JEPA")
    ortak = run_doc(jepa, common_with=ts, model="jepa", label="JEPA")
    assert tam["n_cases"] == 12
    assert ortak["n_cases"] == 10
    assert ortak["metrics"]["dice_median"]["p"] == pytest.approx(0.5)
    assert ortak["n_dropped"] == 2
    assert ortak["restricted_to"]


def test_ortak_kume_bossa_firlatir(tmp_path):
    a = _write_eval(tmp_path / "a", ["lits-0", "lits-1"], [0.5, 0.5])
    b = _write_eval(tmp_path / "b", ["lits-90", "lits-91"], [0.5, 0.5])
    with pytest.raises(ValueError, match="ortak vaka"):
        run_doc(a, common_with=b, model="ts", label="TS")


def test_ortak_kume_ci_csv_yerine_yeniden_hesaplar(tmp_path):
    """Onceden yazilmis ci.csv, filtrelenmis kume icin YANLIS -- kullanilmamali."""
    ts = _write_eval(tmp_path / "ts", [f"lits-{i}" for i in range(10)], [0.5] * 10)
    jepa = _write_eval(tmp_path / "jepa", [f"lits-{i}" for i in range(12)], [0.5] * 10 + [0.99, 0.99])
    pd.DataFrame({"metric": ["dice_median"], "point": [0.123], "lo": [0.1], "hi": [0.2],
                  "n_denominator": [12], "n_valid": [1], "n_boot": [1], "n_cases": [12]}
                 ).to_csv(jepa / "ci.csv", index=False)
    assert run_doc(jepa, model="jepa", label="J")["metrics"]["dice_median"]["p"] == pytest.approx(0.123)
    ortak = run_doc(jepa, common_with=ts, model="jepa", label="J")
    assert ortak["metrics"]["dice_median"]["p"] == pytest.approx(0.5)


def test_nan_degerler_none_olur(tmp_path):
    ev = _write_eval(tmp_path / "ev", [f"lits-{i}" for i in range(6)], [0.6] * 6)
    doc = run_doc(ev, model="ts", label="TS")
    json.dumps(doc)                                   # NaN kacarsa gecersiz JSON olur
    for m in doc["metrics"].values():
        for k in ("p", "lo", "hi"):
            assert m[k] is None or np.isfinite(m[k])


def test_margin_doc_noktalari_sirali_ve_sonlu(tmp_path):
    f = tmp_path / "sweep.json"
    f.write_text(json.dumps({"params": {"base_margin_mm": 5.0}, "overall": {
        "7.0": {"margin_violation_rate_all": .3, "margin_violation_rate_detected": .1,
                "ablation_to_ideal_ratio_median_detected": 1.4,
                "extra_vs_ideal_ml_median_detected": 2.7},
        "5.0": {"margin_violation_rate_all": .5, "margin_violation_rate_detected": .3,
                "ablation_to_ideal_ratio_median_detected": .9,
                "extra_vs_ideal_ml_median_detected": None}}}))
    d = margin_doc(f, run="x")
    assert [p["m"] for p in d["points"]] == [5.0, 7.0]
    assert d["base_margin_mm"] == 5.0
    assert d["points"][0]["extra_ml"] is None
