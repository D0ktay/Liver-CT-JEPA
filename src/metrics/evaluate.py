"""Tek komutla tam degerlendirme: voksel + lezyon + klinik metrikler.

Cikti uc dosya:
  outputs/eval/per_lesion.csv     -- her lezyon bir satir
  outputs/eval/per_case.csv       -- her vaka bir satir
  outputs/eval/summary.json       -- katmanli ozet

Ve tek sekil icin gereken sey: per_lesion.csv'den Dice ekseni vs
centroid_error_mm ekseni sacilim grafigi.

Etiketler AYRI verilir: LiTS GT'sinde tumor=2, TotalSegmentator liver_lesions
ciktisinda lezyon=1. Ikisine ayni degeri uygulamak tahmini sessizce bosaltir.

Tumorsuz vakalar (GT bos) Dice'a KATILMAZ; vaka duzeyinde yanlis pozitif
orani ile ayri raporlanir (CLAUDE.md kural 5).

per_lesion.csv'de IKI ayri "bulundu" kavrami var, bilerek:
  status    -- TESPIT (IoU >= esik): TP / FN / FP
  detected  -- HEDEFLEME (en az bir voksel ortusme, planlama maskesi)
Ayni lezyon status=FN, detected=True olabilir: tahmin tumoru genis kapliyor.

IoU esigi, marj ve boyut katmanlari varsayilan olarak configs/paths.yaml -> metrics.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .lesionwise import LesionMatch, dice, match_lesions, matches_to_records, summarize
from .targeting import (TargetingResult, results_to_records, summarize_targeting,
                        targeting_errors)

DEFAULT_CONFIG = Path("configs/paths.yaml")


def load_metric_config(path: Path = DEFAULT_CONFIG) -> dict:
    """configs/paths.yaml -> metrics. Eksik anahtar sessizce varsayilana dusmez: fırlatır."""
    m = yaml.safe_load(Path(path).read_text())["metrics"]
    strata = tuple(float(x) for x in m["size_strata_mm"])
    if len(strata) != 2 or strata[0] >= strata[1]:
        raise ValueError(f"size_strata_mm iki artan deger olmali: {strata}")
    return {"iou": float(m["lesion_match_iou"]),
            "margin_mm": float(m["ablation_margin_mm"]),
            "strata": strata}


def parse_label(s: str) -> int | None:
    """'any' -> None (maske > 0), aksi halde tam sayi etiket."""
    return None if str(s).lower() == "any" else int(s)


def binarize(arr: np.ndarray, label: int | None, name: str = "", case_id: str = "") -> np.ndarray:
    b = (arr == label) if label is not None else (arr > 0)
    if label is not None and not b.any() and np.any(arr):
        vals = np.unique(arr)
        vals = vals[vals != 0][:10].tolist()
        print(f"[UYARI] {case_id}: {name} maskesinde etiket {label} YOK ama baska degerler var "
              f"{vals}. Etiket yanlis olabilir (--gt-label / --pred-label).")
    return b


def load_mask(path: Path):
    import nibabel as nib
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, spacing, img.affine


def evaluate_arrays(gt_b: np.ndarray, pr_b: np.ndarray, spacing, case_id: str,
                    margin_mm: float = 5.0, iou: float = 0.5,
                    strata=(10.0, 20.0)):
    """Saf numpy: iki ikili maske -> (eslesmeler, hedefleme sonuclari, vaka satiri)."""
    matches = match_lesions(gt_b, pr_b, spacing, iou_threshold=iou)
    targ = targeting_errors(gt_b, pr_b, spacing, case_id=case_id,
                            margin_mm=margin_mm, strata=strata)
    s = summarize(matches, strata)
    gt_empty = not bool(gt_b.any())
    case_row = {
        "case_id": case_id,
        "gt_empty": gt_empty,
        # tumorsuz vakada Dice TANIMSIZ (pred bos degilse dice() 0 dondururdu)
        "dice": float("nan") if gt_empty else dice(gt_b, pr_b),
        "n_gt_lesions": s["tp"] + s["fn"],
        "n_pred_lesions": s["tp"] + s["fp"],
        "fp_case": bool(gt_empty and pr_b.any()),
        **{k: v for k, v in s.items() if k != "by_size"},
    }
    return matches, targ, case_row


def load_pair(gt_path: Path, pred_path: Path, case_id: str,
              gt_label: int | None = 2, pred_label: int | None = 1,
              allow_affine_mismatch: bool = False):
    """GT + tahmin -> (gt ikili, tahmin ikili, spacing). Grid uyusmazliginda FIRLATIR."""
    gt, sp, aff_g = load_mask(gt_path)
    pr, sp_p, aff_p = load_mask(pred_path)
    if gt.shape != pr.shape:
        raise ValueError(f"{case_id}: sekil uyusmuyor {gt.shape} vs {pr.shape}. "
                         "Once ayni gride resample et.")
    # spacing (pixdim) affine'den (sform) ayri tutulabilir: bypass edilse bile FIRLAT,
    # yoksa GT spacing'i tahmine sessizce uygulanir
    if not np.allclose(sp, sp_p, atol=1e-3):
        raise ValueError(f"{case_id}: spacing farkli {sp} vs {sp_p}. Ayni gride resample et.")
    if not np.allclose(aff_g, aff_p, atol=1e-2):
        # ayni sekil + farkli yonelim = maskeler sessizce ters/kaymis karsilastirilir
        msg = f"{case_id}: affine farkli (yonelim/orijin/spacing).\nGT:\n{aff_g}\nPRED:\n{aff_p}"
        if not allow_affine_mismatch:
            raise ValueError(msg + "\nAyni gride resample et ya da --allow-affine-mismatch.")
        print("[UYARI]", msg)
    return binarize(gt, gt_label, "GT", case_id), binarize(pr, pred_label, "PRED", case_id), sp


def evaluate_pair(gt_path: Path, pred_path: Path, case_id: str,
                  gt_label: int | None = 2, pred_label: int | None = 1,
                  margin_mm: float = 5.0, iou: float = 0.5,
                  allow_affine_mismatch: bool = False, strata=(10.0, 20.0)):
    gt_b, pr_b, sp = load_pair(gt_path, pred_path, case_id, gt_label, pred_label,
                               allow_affine_mismatch)
    return evaluate_arrays(gt_b, pr_b, sp, case_id, margin_mm, iou, strata)


def aggregate(all_matches, targ_results: list[TargetingResult], case_rows: list[dict],
              strata=(10.0, 20.0)) -> dict:
    tumor = [c for c in case_rows if not c["gt_empty"]]
    free = [c for c in case_rows if c["gt_empty"]]
    dices = np.array([c["dice"] for c in tumor], dtype=float)
    n_dice_nan = int((~np.isfinite(dices)).sum())
    dices = dices[np.isfinite(dices)]
    return {
        "n_cases": len(case_rows),
        "n_cases_with_tumor": len(tumor),
        "n_cases_tumor_free": len(free),
        # Dice YALNIZCA tumorlu vakalar uzerinden
        "voxel": {
            "dice_median": float(np.median(dices)) if dices.size else None,
            "dice_mean": float(dices.mean()) if dices.size else None,
            "n_dice_nan": n_dice_nan,   # tumorlu vakada tanimsiz Dice -- olmamali
        },
        "lesion": summarize(all_matches, strata),
        "tumor_free": {
            "fp_case_rate": float(np.mean([c["fp_case"] for c in free])) if free else None,
            "specificity_case": float(1 - np.mean([c["fp_case"] for c in free])) if free else None,
            "fp_lesions_mean": float(np.mean([c["fp"] for c in free])) if free else None,
        },
        "targeting": summarize_targeting(targ_results),
    }


def build_per_lesion(lesion_rows: list[dict], targ_rows: list[dict]) -> pd.DataFrame:
    """Tespit satirlari (TP/FN/FP) + hedefleme satirlari (yalniz GT lezyonlari).

    Sema her zaman ayni: bos girdi de tum sutunlari tasir. FP satirlarinin
    gt_label'i yoktur, hedefleme sutunlari bos kalir.
    """
    les_cols = ["case_id"] + [f.name for f in fields(LesionMatch)]
    targ_cols = [f.name for f in fields(TargetingResult) if f.name != "gt_diameter_mm"]
    les = pd.DataFrame(lesion_rows, columns=les_cols)
    targ = pd.DataFrame(targ_rows, columns=[f.name for f in fields(TargetingResult)])
    targ = targ[targ_cols]
    les["gt_label"] = les["gt_label"].astype("Int64")
    targ["gt_label"] = targ["gt_label"].astype("Int64")
    if targ.duplicated(["case_id", "gt_label"]).any():
        raise ValueError("hedefleme satirlarinda (case_id, gt_label) tekrar ediyor")
    return les.merge(targ, on=["case_id", "gt_label"], how="left")


def _json_safe(o):
    """NaN/inf -> null (gecerli JSON), numpy skalerleri -> python."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True,
                    help="CSV: case_id,gt_path,pred_path (src.metrics.make_pairs uretir)")
    ap.add_argument("--out", type=Path, default=Path("outputs/eval"))
    ap.add_argument("--gt-label", type=parse_label, default=2,
                    help="GT'de tumor etiketi (LiTS: 2). 'any' -> >0")
    ap.add_argument("--pred-label", type=parse_label, default=1,
                    help="tahminde lezyon etiketi (TotalSegmentator liver_lesions: 1). 'any' -> >0")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--margin-mm", type=float, default=None, help="vars.: config metrics.ablation_margin_mm")
    ap.add_argument("--iou", type=float, default=None, help="vars.: config metrics.lesion_match_iou")
    ap.add_argument("--allow-affine-mismatch", action="store_true")
    a = ap.parse_args()
    cfg = load_metric_config(a.config)
    margin = a.margin_mm if a.margin_mm is not None else cfg["margin_mm"]
    iou = a.iou if a.iou is not None else cfg["iou"]
    strata = cfg["strata"]

    pairs = pd.read_csv(a.pairs)
    all_matches, all_targ, lesion_rows, targ_rows, case_rows, failed = [], [], [], [], [], []
    for _, r in pairs.iterrows():
        try:
            m, t, c = evaluate_pair(Path(r.gt_path), Path(r.pred_path), str(r.case_id),
                                    a.gt_label, a.pred_label, margin, iou,
                                    a.allow_affine_mismatch, strata)
        except Exception as e:
            print(f"[HATA] {r.case_id}: {e}")
            failed.append(str(r.case_id))
            continue
        all_matches += m
        all_targ += t
        lesion_rows += matches_to_records(m, str(r.case_id))
        targ_rows += results_to_records(t)
        case_rows.append(c)
        print(f"[eval] {r.case_id}: dice={c['dice']:.3f} "
              f"lezyon P/R={c['precision']:.3f}/{c['recall']:.3f}")

    a.out.mkdir(parents=True, exist_ok=True)
    build_per_lesion(lesion_rows, targ_rows).to_csv(a.out / "per_lesion.csv", index=False)
    pd.DataFrame(case_rows).to_csv(a.out / "per_case.csv", index=False)

    summary = aggregate(all_matches, all_targ, case_rows, strata)
    summary["n_failed"] = len(failed)
    summary["failed_cases"] = failed
    summary["params"] = {"gt_label": a.gt_label, "pred_label": a.pred_label,
                         "margin_mm": margin, "iou": iou, "size_strata_mm": list(strata),
                         "targeting_match": "overlap (planlama maskesi)",
                         "detection_match": f"IoU >= {iou}"}
    (a.out / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    print(f"\n[eval] -> {a.out}")
    print(json.dumps(_json_safe({k: summary[k] for k in ("voxel", "tumor_free")}), indent=2))
    print(json.dumps(_json_safe(summary["targeting"]["overall"]), indent=2))
    if failed:
        print(f"\n[!] {len(failed)}/{len(pairs)} vaka BASARISIZ: {failed[:10]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
