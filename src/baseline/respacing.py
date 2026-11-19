"""LiTS 28-47 (= IRCADb) baslik duzeltmesi: sahte 1x1x1 spacing yerine gercek deger.

Bulgu (19 Eyl 2026): LiTS 28-47'nin NIfTI basliginda spacing tam 1x1x1 mm; goruntu icerigi
eslesen IRCADb DICOM'u ile ayni (outputs/kaggle131/ircadb_lits_match.csv). Gercek spacing:
configs/lits_2847_true_spacing.csv. Duzeltme yalniz BASLIK: veri bayt bayt ayni, yon korunur
(affine'in voksel eksen sutunlari olceklenir, orijin ayni kalir).

    python -m src.baseline.respacing --src-glob 'data/raw/LiTS/volume-*.nii' --out /kaggle/tmp/fixed
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

DEFAULT_TABLE = Path("configs/lits_2847_true_spacing.csv")


def load_spacing_table(path: str | Path = DEFAULT_TABLE) -> dict[str, dict]:
    lines = [l for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    out = {}
    for r in csv.DictReader(lines):
        out[r["lits_id"]] = {"spacing": (float(r["sx"]), float(r["sy"]), float(r["sz"])),
                             "shape": (int(r["nx"]), int(r["ny"]), int(r["nz"])),
                             "ircadb_id": r["ircadb_id"]}
    return out


def reheader(src: str | Path, dst: str | Path, spacing, expect_shape) -> None:
    import nibabel as nib
    img = nib.load(str(src))
    if tuple(img.shape[:3]) != tuple(expect_shape):
        raise ValueError(f"{src}: sekil {img.shape} != beklenen {tuple(expect_shape)} (yanlis vaka?)")
    zooms = img.header.get_zooms()[:3]
    if not np.allclose(zooms, 1.0, atol=1e-4):
        raise ValueError(f"{src}: spacing {zooms}, 1x1x1 degil -> zaten duzeltilmis ya da baska dosya")
    aff = img.affine.copy()
    aff[:3, :3] = aff[:3, :3] @ np.diag(np.asarray(spacing, float))
    out = nib.Nifti1Image(np.asarray(img.dataobj), aff, header=img.header.copy())
    out.set_data_dtype(img.get_data_dtype())
    out.header.set_zooms(tuple(spacing) + tuple(img.header.get_zooms()[3:]))
    out.header.set_sform(aff, code=2)
    out.header.set_qform(aff, code=2)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))


def main() -> int:
    from src.data.case_id import case_id_from_path, glob_paths
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-glob", required=True, help="volume-*.nii, segmentation-*.nii ya da tahmin *.nii.gz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    a = ap.parse_args()
    table = load_spacing_table(a.table)
    done = []
    for p in glob_paths(a.src_glob):
        cid = case_id_from_path(p)
        if cid not in table:
            continue
        name = Path(p).name
        dst = a.out / (name if name.endswith(".gz") else name + ".gz")
        reheader(p, dst, table[cid]["spacing"], table[cid]["shape"])
        done.append(cid)
    missing = sorted(set(table) - set(done))
    print(f"[respacing] {len(done)} dosya -> {a.out}" + (f" | EKSIK: {missing}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
