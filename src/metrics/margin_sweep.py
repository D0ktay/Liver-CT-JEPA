"""Marj ek payi egrisi: AI planina 5 mm yerine 5+e mm marj verilseydi?

Klinik soru: "AI'in cizdigi maskeye kac mm ek pay koyarsam ihlal %X'in altina iner,
bunun bedeli kac mL fazladan yakilan (saglikli) doku olur?"

Her GT lezyonu x her marj icin bir satir:
  margin_coverage / margin_violation -- targeting_errors(margin_mm=m) ile BIREBIR ayni
                                        (ayni planlama maskesi = GT'ye degen tahmin bilesenleri)
  ablation_ml       -- plan + m mm (AI'a gore yakilacak hacim)
  ideal_ablation_ml -- GT + base mm (dogru maskeyle standart plan; referans)
  extra_vs_ideal_ml -- ablasyonun TUM tumorlerin ideal plani (her GT + base mm) disinda kalan
                       kismi = bedel. Komsu tumor ve onun marji saglikli doku SAYILMAZ
                       (targeting.excess_volume_ml de plan - TUM GT).

Kacirilan lezyon (plan yok) hicbir marjda kurtarilmaz: ihlal, ablasyon 0.
Paylasilan plan (bir tahmin iki GT'ye degiyor): ablation_ml her iki lezyonun satirinda da tam
sayilir; lezyon uzerinden TOPLAM ablasyon hacmi almak bu yuzden cift sayar -- medyan kullanilir.

EDT her lezyon icin BIR kez hesaplanir (plan ve GT icin), marjlar esiktir: 6 marj ~ 1 marj maliyeti.
Varsayilan marjlar ve temel marj: configs/paths.yaml -> metrics.
    python -m src.metrics.margin_sweep --pairs outputs/pairs.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

from .lesionwise import connected_components, equivalent_diameter_mm, stratum_of
from .targeting import _union_box

COLUMNS = ["case_id", "gt_label", "gt_diameter_mm", "stratum", "detected", "gt_volume_ml",
           "base_margin_mm", "ideal_ablation_ml", "margin_mm", "margin_coverage",
           "margin_violation", "ablation_ml", "extra_vs_ideal_ml"]


def _check_margins(margins: Sequence[float], base: float) -> tuple[float, ...]:
    ms = tuple(float(m) for m in margins)
    if not ms:
        raise ValueError("margins bos")
    if not all(np.isfinite(m) and m >= 0 for m in ms + (float(base),)):
        raise ValueError(f"marjlar sonlu ve >= 0 olmali: {ms}, base={base}")
    if len(set(ms)) != len(ms):
        raise ValueError(f"tekrarlanan marj: {ms}")
    return tuple(sorted(ms))


def margin_sweep(gt: np.ndarray, pred: np.ndarray, spacing: Sequence[float],
                 case_id: str = "case", margins: Sequence[float] = (5, 6, 7, 8, 9, 10),
                 base_margin_mm: float = 5.0,
                 strata: Sequence[float] = (10.0, 20.0)) -> list[dict]:
    margins = _check_margins(margins, base_margin_mm)
    base = float(base_margin_mm)
    gt = np.asarray(gt) > 0
    pred = np.asarray(pred) > 0
    if gt.shape != pred.shape:
        raise ValueError(f"{case_id}: sekil uyusmuyor {gt.shape} vs {pred.shape}")
    spacing = tuple(float(s) for s in spacing)
    vox_ml = float(np.prod(spacing)) / 1000.0

    gt_lab, n_gt = connected_components(gt)
    pr_lab, _ = connected_components(pred)
    gt_sl = ndimage.find_objects(gt_lab)
    pr_sl = ndimage.find_objects(pr_lab)
    both = gt & pred
    gt2pr: dict[int, set[int]] = {}
    for g, p in (np.unique(np.stack([gt_lab[both], pr_lab[both]], 1), axis=0)
                 if both.any() else ()):
        gt2pr.setdefault(int(g), set()).add(int(p))

    # kutu, EN BUYUK genislemeyi kapsamali: aksi halde ablasyon hacmi kesilir
    reach = max(max(margins), base)
    pad = [int(np.ceil(reach / s)) + 1 for s in spacing]
    # TUM GT'lerin ideal plani: kutu disindaki bir tumorun marji kutuya uzanabilir ->
    # EDT, base kadar daha genis kutuda hesaplanip kirpilir
    pad_b = [int(np.ceil(base / s)) + 1 for s in spacing]

    rows: list[dict] = []
    for i in range(1, n_gt + 1):
        hits = sorted(gt2pr.get(i, ()))
        box = _union_box([gt_sl[i - 1]] + [pr_sl[j - 1] for j in hits], pad, gt.shape)
        g = gt_lab[box] == i
        n_vox = int(g.sum())
        diam = equivalent_diameter_mm(n_vox, spacing)
        ideal = ndimage.distance_transform_edt(~g, sampling=spacing) <= base
        outer = tuple(slice(max(sl.start - p, 0), min(sl.stop + p, n))
                      for sl, p, n in zip(box, pad_b, gt.shape))
        inner = tuple(slice(sl.start - o.start, sl.stop - o.start) for sl, o in zip(box, outer))
        ideal_all = (ndimage.distance_transform_edt(~gt[outer], sampling=spacing) <= base)[inner]
        common = {"case_id": case_id, "gt_label": i, "gt_diameter_mm": diam,
                  "stratum": stratum_of(diam, strata), "detected": bool(hits),
                  "gt_volume_ml": n_vox * vox_ml, "base_margin_mm": base,
                  "ideal_ablation_ml": int(ideal.sum()) * vox_ml}
        if not hits:
            rows += [{**common, "margin_mm": m, "margin_coverage": 0.0, "margin_violation": True,
                      "ablation_ml": 0.0, "extra_vs_ideal_ml": 0.0} for m in margins]
            continue
        plan = np.isin(pr_lab[box], hits)
        # m=0 -> dist<=0 = planin kendisi (targeting._dilate_mm ile ayni)
        dist = ndimage.distance_transform_edt(~plan, sampling=spacing)
        d_gt = dist[g]
        for m in margins:
            cov = float((d_gt <= m).sum() / n_vox)
            abl = dist <= m
            rows.append({**common, "margin_mm": m, "margin_coverage": cov,
                         "margin_violation": bool(cov < 1.0),
                         "ablation_ml": int(abl.sum()) * vox_ml,
                         "extra_vs_ideal_ml": int((abl & ~ideal_all).sum()) * vox_ml})
    return rows


def summarize_sweep(rows: list[dict]) -> dict:
    """Marj basina (genel + boyut katmani): ihlal orani ve bedel.
    ihlal_all: kacirilan = ihlal (tum lezyonlar). ihlal_detected: yalniz bulunanlar.
    Bedel yalniz bulunanlarda (kacirilanda ablasyon yok)."""
    df = pd.DataFrame(rows)
    out: dict = {"overall": {}, "by_size": {}}
    if df.empty:
        return out

    def agg(d: pd.DataFrame) -> dict:
        det = d[d.detected]
        ratio = det.ablation_ml / det.ideal_ablation_ml
        return {
            "n_lesions": int(len(d)),
            "n_detected": int(len(det)),
            "margin_violation_rate_all": float(d.margin_violation.mean()),
            "margin_violation_rate_detected": float(det.margin_violation.mean()) if len(det) else float("nan"),
            "ablation_to_ideal_ratio_median_detected": float(ratio.median()) if len(det) else float("nan"),
            "extra_vs_ideal_ml_median_detected": float(det.extra_vs_ideal_ml.median()) if len(det) else float("nan"),
        }

    for m, d in df.groupby("margin_mm"):
        out["overall"][float(m)] = agg(d)
    for s, ds in df.groupby("stratum"):
        out["by_size"][s] = {float(m): agg(d) for m, d in ds.groupby("margin_mm")}
    return out


def main() -> int:
    from .evaluate import DEFAULT_CONFIG, _json_safe, load_metric_config, load_pair, parse_label
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/eval"))
    ap.add_argument("--margins", default=None, help="virgulle, mm (vars.: config metrics.margin_sweep_mm)")
    ap.add_argument("--margin-mm", type=float, default=None,
                    help="temel (ideal plan) marji; evaluate ile AYNI olmali (vars.: config metrics.ablation_margin_mm)")
    ap.add_argument("--gt-label", type=parse_label, default=2)
    ap.add_argument("--pred-label", type=parse_label, default=1)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--allow-affine-mismatch", action="store_true")
    a = ap.parse_args()
    import yaml
    cfg = load_metric_config(a.config)
    margins = ([float(x) for x in a.margins.split(",")] if a.margins else
               [float(x) for x in yaml.safe_load(a.config.read_text())["metrics"]["margin_sweep_mm"]])
    base = a.margin_mm if a.margin_mm is not None else cfg["margin_mm"]

    rows, failed = [], []
    for _, r in pd.read_csv(a.pairs).iterrows():
        try:
            gt_b, pr_b, sp = load_pair(Path(r.gt_path), Path(r.pred_path), str(r.case_id),
                                       a.gt_label, a.pred_label, a.allow_affine_mismatch)
            rows += margin_sweep(gt_b, pr_b, sp, str(r.case_id), margins,
                                 base, cfg["strata"])
        except Exception as e:
            print(f"[HATA] {r.case_id}: {e}")
            failed.append(str(r.case_id))
            continue
        print(f"[sweep] {r.case_id}")

    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(a.out / "margin_sweep.csv", index=False)
    if not rows:
        print("[UYARI] hic GT lezyonu yok (tum vakalar tumorsuz?) -> bos tablo, yalniz baslik")
    s = summarize_sweep(rows)
    s["params"] = {"margins_mm": margins, "base_margin_mm": base}
    s["failed_cases"] = failed
    (a.out / "margin_sweep.json").write_text(json.dumps(_json_safe(s), indent=2))
    print(f"\n{'marj':>5} {'ihlal(tum)':>11} {'ihlal(bulunan)':>15} {'ablasyon/ideal':>15} {'fazla mL':>9}")
    for m, v in s["overall"].items():
        print(f"{m:5.1f} {v['margin_violation_rate_all']:11.3f} {v['margin_violation_rate_detected']:15.3f} "
              f"{v['ablation_to_ideal_ratio_median_detected']:15.2f} {v['extra_vs_ideal_ml_median_detected']:9.2f}")
    print(f"\n[sweep] -> {a.out}/margin_sweep.csv")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
