#!/usr/bin/env python
"""Mevcut bir degerlendirmeyi DONDURULMUS ic test alt kumesine kisitla.

    python scripts/frozen_baseline.py \
        --eval outputs/kaggle/fix2847/outputs/fix2847/all131/eval \
        --out  outputs/frozen/ts_liver_lesions

NEDEN: `LiverJEPA` sonuclari dondurulmus ic test kumesinde (27 vaka) raporlanacak.
TotalSegmentator tabani ve Rad-JEPA kolu ise 126 vakada olculdu. **Farkli vaka
kumelerindeki sayilar karsilastirilamaz.** Bu betik eski vaka-bazli olcumleri ayni
ayni alt kumeye kisitlayip nokta tahmini + bootstrap GA'sini yeniden uretir.
GPU ya da ham veri GEREKTIRMEZ -- yalniz per_case.csv / per_lesion.csv okur.

Dondurulmus kume `src/data/frozen_test.py` ile AYNI fonksiyondan (seg.make_folds,
hasta duzeyi, tumorlu/tumorsuz tabakali, tohum 0, kat 0) turetilir; tumor durumu
`outputs/kaggle131/lits_gt_lesions.csv`'den okunur (voxels > 0).

UYARI: kisitlama, eski kosunun DUSEN vakalarini geri getirmez. 131'den 126'ya dusen
5 vaka (lits-48..52) dondurulmus kumedeyse burada da eksik kalir ve sayisi basilir.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.frozen_test import choose  # noqa: E402


def frozen_ids(lesion_csv: Path, k: int = 5, fold: int = 0, seed: int = 0) -> list[str]:
    d = pd.read_csv(lesion_csv)
    ids = sorted(d["case_id"].astype(str).unique(), key=lambda c: int(c.split("-")[1]))
    has = {c: bool((d.loc[d["case_id"] == c, "voxels"] > 0).any()) for c in ids}
    n_t = sum(has.values())
    print(f"[frozen] {len(ids)} vaka ({n_t} tumorlu, {len(ids) - n_t} tumorsuz)")
    return choose(ids, has, k=k, fold=fold, seed=seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, required=True, help="per_case.csv iceren klasor")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lesions", type=Path,
                    default=Path("outputs/kaggle131/lits_gt_lesions.csv"))
    ap.add_argument("--intersect", type=Path, nargs="*", default=[],
                    help="baska kollarin per_case.csv'leri: ORTAK vakalarda olc. "
                         "Farkli vaka kumelerindeki sayilar karsilastirilamaz; bir kolda "
                         "eksik vaka varsa (orn. TS kosusunda lits-48/49 yok) hepsini ayni "
                         "kumeye indirmek tek adil yoldur.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ids = frozen_ids(a.lesions, a.k, a.fold, a.seed)
    print(f"[frozen] dondurulmus ic test: {len(ids)} vaka")
    for other in a.intersect:
        o = set(pd.read_csv(other)["case_id"].astype(str))
        once = len(ids)
        ids = [c for c in ids if c in o]
        print(f"[frozen] ∩ {other}: {once} -> {len(ids)} vaka")
    if not ids:
        raise SystemExit("ortak vaka kalmadi")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frozen_ids.json").write_text(json.dumps(sorted(ids), indent=1))

    pc = pd.read_csv(a.eval / "per_case.csv")
    have = set(pc["case_id"].astype(str))
    eksik = [c for c in ids if c not in have]
    sub = pc[pc["case_id"].astype(str).isin(ids)].copy()
    print(f"[frozen] per_case: {len(pc)} -> {len(sub)} satir"
          + (f" | ⚠️ {len(eksik)} vaka bu kosuda YOK: {eksik}" if eksik else ""))
    if sub.empty:
        raise SystemExit("kesisim bos -- vaka kimlikleri uyusmuyor")
    sub.to_csv(a.out / "per_case.csv", index=False)

    pl_p = a.eval / "per_lesion.csv"
    if pl_p.exists():
        pl = pd.read_csv(pl_p)
        pls = pl[pl["case_id"].astype(str).isin(ids)]
        pls.to_csv(a.out / "per_lesion.csv", index=False)
        print(f"[frozen] per_lesion: {len(pl)} -> {len(pls)} satir")

    # Tumorsuz vakalar Dice'a GIRMEZ (kural 5) -- gt_empty ile ayiriliyor
    tumorlu = sub[~sub["gt_empty"].astype(bool)] if "gt_empty" in sub else sub
    ozet = {
        "n_frozen": len(ids), "n_bulunan": len(sub), "n_eksik": len(eksik), "eksik": eksik,
        "n_tumorlu": int(len(tumorlu)), "n_tumorsuz": int(len(sub) - len(tumorlu)),
        "dice_median": float(tumorlu["dice"].median()) if len(tumorlu) else None,
        "dice_mean": float(tumorlu["dice"].mean()) if len(tumorlu) else None,
        "lesion_tp": int(sub["tp"].sum()), "lesion_fp": int(sub["fp"].sum()),
        "lesion_fn": int(sub["fn"].sum()),
        "kaynak_eval": str(a.eval), "frozen": {"k": a.k, "fold": a.fold, "seed": a.seed},
    }
    tp, fp, fn = ozet["lesion_tp"], ozet["lesion_fp"], ozet["lesion_fn"]
    ozet["lesion_precision"] = tp / (tp + fp) if tp + fp else None
    ozet["lesion_recall"] = tp / (tp + fn) if tp + fn else None
    (a.out / "summary_frozen.json").write_text(json.dumps(ozet, indent=1, ensure_ascii=False))

    r = subprocess.run([sys.executable, "-m", "src.metrics.bootstrap", "--eval", str(a.out),
                        "--n-boot", str(a.n_boot)], cwd=ROOT)
    if r.returncode:
        print("[UYARI] bootstrap kosmadi -- nokta tahminleri yine de summary_frozen.json'da")

    print(json.dumps(ozet, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
