"""Iki tahmin klasorunu voksel voksel karsilastir (hizli yol == eski yol mu?).

    python -m src.baseline.compare_masks --a outputs/baseline/ts_check_cli --b outputs/baseline/ts_check_fast

Ayni sayilmasi icin: iki tarafta da var + sekil ayni + affine ayni + tum vokseller ayni.
Hicbir dosya karsilastirilmadiysa ya da biri farkliysa cikis kodu 1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def compare_dirs(a: Path, b: Path) -> list[dict]:
    import nibabel as nib
    names = sorted({p.name for p in Path(a).glob("*.nii.gz")} | {p.name for p in Path(b).glob("*.nii.gz")})
    rows = []
    for n in names:
        cid = n[:-len(".nii.gz")]
        pa, pb = Path(a) / n, Path(b) / n
        if not (pa.exists() and pb.exists()):
            rows.append({"case_id": cid, "missing": "a" if not pa.exists() else "b",
                         "shape_equal": False, "affine_equal": False, "n_diff_voxels": None,
                         "identical": False})
            continue
        ia, ib = nib.load(str(pa)), nib.load(str(pb))
        xa, xb = np.asarray(ia.dataobj), np.asarray(ib.dataobj)
        shape_eq = xa.shape == xb.shape
        aff_eq = bool(np.allclose(ia.affine, ib.affine, atol=1e-5))
        ndiff = int((xa != xb).sum()) if shape_eq else None
        rows.append({"case_id": cid, "missing": None, "shape_equal": shape_eq,
                     "affine_equal": aff_eq, "n_diff_voxels": ndiff,
                     "identical": bool(shape_eq and aff_eq and ndiff == 0)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    a = ap.parse_args()
    rows = compare_dirs(a.a, a.b)
    for r in rows:
        print(("AYNI   " if r["identical"] else "FARKLI ") + str(r))
    if not rows:
        print(f"[HATA] karsilastirilacak dosya yok: {a.a} / {a.b}")
        return 1
    bad = [r["case_id"] for r in rows if not r["identical"]]
    print(f"\n[karsilastir] {len(rows) - len(bad)}/{len(rows)} ayni" + (f", FARKLI: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
