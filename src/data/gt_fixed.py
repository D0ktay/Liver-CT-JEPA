"""Degerlendirme icin BASLIGI DUZELTILMIS GT kopyalari (veri bayt bayt ayni).

    python -m src.data.gt_fixed --ids outputs/internal_test.json --raw data/raw/LiTS --out data/gt_fixed

NEDEN: `src.metrics.evaluate` mm metriklerini GT BASLIGINDAKI spacing'le hesaplar ve
tahmin NIfTI'si GT'nin basligini miras alir. Iki bilinen bozukluk dondurulmus testte VAR:
  * lits-28..47 (testte 37, 38, 46, 47): hacim VE segmentasyon basligi sahte 1x1x1;
    gercek deger `configs/lits_2847_true_spacing.csv` (IRCADb DICOM'undan).
  * lits-48..52 (testte 48, 49): SEGMENTASYON basligi sahte 1x1x1, CT basligi dogru.
Duzeltilmezse bu 6 vakada hedefleme hatasi (mm) ve marj ihlali YANLIS olcekte olculur.
TS tabani 28-47'de ayni `respacing.reheader` ile olculdu -> karsilastirma tutarli kalir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.baseline.respacing import load_spacing_table


def fixed_reference(seg_path: Path, ct_path: Path, case_id: str, table: dict):
    """-> (affine, zooms, neden). Veri degismez; yalniz geometri."""
    import nibabel as nib
    seg, ct = nib.load(str(seg_path)), nib.load(str(ct_path))
    if tuple(seg.shape[:3]) != tuple(ct.shape[:3]):
        raise ValueError(f"{case_id}: GT {seg.shape} != CT {ct.shape}")
    zs = tuple(float(z) for z in seg.header.get_zooms()[:3])
    zc = tuple(float(z) for z in ct.header.get_zooms()[:3])
    if case_id in table:
        t = table[case_id]
        if tuple(seg.shape[:3]) != tuple(t["shape"]):
            raise ValueError(f"{case_id}: sekil {seg.shape} != tablo {t['shape']} (yanlis vaka?)")
        base = seg.affine.copy()
        norm = np.linalg.norm(base[:3, :3], axis=0)
        aff = base.copy()
        aff[:3, :3] = base[:3, :3] / norm * np.asarray(t["spacing"], float)
        return aff, tuple(t["spacing"]), "tablo (LiTS 28-47 gercek spacing)"
    if not np.allclose(zs, zc, rtol=0, atol=1e-3):
        return ct.affine.copy(), zc, f"CT basligi (GT {zs} != CT {zc})"
    return seg.affine.copy(), zs, "degismedi"


def main(argv=None) -> int:
    import nibabel as nib
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/LiTS"))
    ap.add_argument("--out", type=Path, default=Path("data/gt_fixed"))
    ap.add_argument("--table", type=Path, default=Path("configs/lits_2847_true_spacing.csv"))
    a = ap.parse_args(argv)
    ids = json.loads(a.ids.read_text())["case_ids"]
    table = load_spacing_table(a.table)
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for cid in ids:
        n = int(cid.split("-")[1])
        seg_p, ct_p = a.raw / f"segmentation-{n}.nii", a.raw / f"volume-{n}.nii"
        aff, zooms, why = fixed_reference(seg_p, ct_p, cid, table)
        seg = nib.load(str(seg_p))
        out = nib.Nifti1Image(np.asarray(seg.dataobj), aff, header=seg.header.copy())
        out.set_data_dtype(seg.get_data_dtype())
        out.header.set_zooms(tuple(zooms) + tuple(seg.header.get_zooms()[3:]))
        out.header.set_sform(aff, code=2)
        out.header.set_qform(aff, code=2)
        nib.save(out, str(a.out / f"segmentation-{n}.nii"))
        rep[cid] = {"zooms": [float(z) for z in zooms], "neden": why}
        if why != "degismedi":
            print(f"[gt_fixed] {cid}: {why} -> {tuple(round(float(z), 4) for z in zooms)}", flush=True)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"[gt_fixed] {len(rep)} GT -> {a.out} "
          f"({sum(1 for v in rep.values() if v['neden'] != 'degismedi')} baslik duzeltildi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
