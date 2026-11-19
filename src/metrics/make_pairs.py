"""GT ve tahmin dosyalarini vaka kimligiyle eslestirip pairs.csv uretir.

Kimlik: src.data.case_id.case_id_from_path -- baseline ciktisiyla ayni kural.
Eslesmeyen dosyalar sessizce atlanmaz: listelenir ve (--strict ise) hata verilir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.case_id import assert_unique_ids, glob_paths


def make_pairs(gt_paths: list[Path], pred_paths: list[Path]) -> tuple[pd.DataFrame, list[str], list[str]]:
    gt = assert_unique_ids(gt_paths)
    pr = assert_unique_ids(pred_paths)
    common = sorted(gt.keys() & pr.keys())
    df = pd.DataFrame({"case_id": common,
                       "gt_path": [str(gt[k]) for k in common],
                       "pred_path": [str(pr[k]) for k in common]})
    return df, sorted(gt.keys() - pr.keys()), sorted(pr.keys() - gt.keys())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-glob", required=True, help="orn: 'data/raw/LiTS/**/segmentation-*.nii*'")
    ap.add_argument("--pred-glob", required=True, help="orn: 'outputs/baseline/totalsegmentator/*.nii.gz'")
    ap.add_argument("--out", type=Path, default=Path("outputs/pairs.csv"))
    ap.add_argument("--strict", action="store_true", help="tahmini olmayan GT varsa hata ver")
    a = ap.parse_args()

    df, gt_only, pred_only = make_pairs(glob_paths(a.gt_glob), glob_paths(a.pred_glob))
    if gt_only:
        print(f"[pairs] {len(gt_only)} GT'nin tahmini yok: {gt_only[:10]}")
    if pred_only:
        print(f"[pairs] {len(pred_only)} tahminin GT'si yok: {pred_only[:10]}")
    if df.empty:
        raise SystemExit("Hic eslesen vaka yok -- glob'lari ve kimlik kuralini kontrol et.")
    if a.strict and gt_only:
        raise SystemExit("--strict: eksik tahmin var.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[pairs] {len(df)} cift -> {a.out}")
