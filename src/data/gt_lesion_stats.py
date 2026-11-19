"""GT maskelerinden lezyon envanteri: kac vakada tumor var, kac lezyon, hangi boyutta.

Tumorsuz vaka da satir alir (lesion=0, diam_mm=nan) -- sayimdan dusmesin.
Cikti CSV: case_id, lesion, voxels, diam_mm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.case_id import assert_unique_ids, glob_paths
from src.metrics.lesionwise import connected_components, equivalent_diameter_mm

SIZE_BINS = [0, 3, 5, 10, 20, np.inf]
SIZE_LABELS = ["<3mm", "3-5", "5-10", "10-20", ">20"]


def lesions_of(mask: np.ndarray, spacing, case_id: str) -> list[dict]:
    lab, n = connected_components(mask)
    if n == 0:
        return [{"case_id": case_id, "lesion": 0, "voxels": 0, "diam_mm": np.nan}]
    counts = np.bincount(lab.ravel())[1:]
    return [{"case_id": case_id, "lesion": i, "voxels": int(c),
             "diam_mm": equivalent_diameter_mm(int(c), spacing)} for i, c in enumerate(counts, 1)]


def summarize(df: pd.DataFrame) -> str:
    per_case = df.groupby("case_id").lesion.max()
    real = df[df.lesion > 0]
    bins = pd.cut(real.diam_mm, SIZE_BINS, labels=SIZE_LABELS, right=False).value_counts().sort_index()
    return "\n".join([
        f"{len(per_case)} vaka | tumorlu {int((per_case > 0).sum())} | tumorsuz {int((per_case == 0).sum())}",
        f"{len(real)} lezyon | medyan cap {real.diam_mm.median():.1f} mm",
        bins.to_string(),
        "<3 mm 'lezyon'lar birkac voksellik etiket kirintisi olabilir -- ayri raporlanmali.",
    ])


if __name__ == "__main__":
    import nibabel as nib
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seg-glob", required=True)
    ap.add_argument("--label", default="2", help="tumor etiketi; 'any' -> >0")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        df = pd.read_csv(a.out)
    else:
        rows = []
        for cid, p in sorted(assert_unique_ids(glob_paths(a.seg_glob)).items()):
            img = nib.load(str(p))
            arr = np.asarray(img.dataobj)
            m = arr > 0 if a.label == "any" else arr == int(a.label)
            rows += lesions_of(m, img.header.get_zooms()[:3], cid)
        if not rows:
            raise SystemExit(f"Maske bulunamadi: {a.seg_glob}")
        df = pd.DataFrame(rows)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
    print(summarize(df))
    print("->", a.out)
