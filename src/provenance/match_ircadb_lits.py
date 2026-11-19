"""LiTS 28-47 <-> 3D-IRCADb eslesmesini ICERIKLE teyit eder (literature korukorune guvenme).

Neden ayri modul: dedup_hash imzasi voksel dizisinin YONUNE bagli. IRCADb DICOM'dan
cevrilince eksenler LiTS'e gore ters ya da x/y yer degistirmis olabilir; o zaman
kosinus dusuk cikar ve kopya SESSIZCE kacar.

Iki bagimsiz kanit:
  1) Yon-bagimsiz imza: dedup_hash imzasi (32^3), 16 yonde (2 x/y transpoz x 8 ters).
     Merkezleme + normlama permutasyonla degismedigi icin kucuk hacmi dondurmek,
     buyuk hacmi dondurup yeniden imzalamakla AYNI sonucu verir.
  2) HU histogrami: yonden tamamen bagimsiz. Ayni tarama -> ~0 mesafe.

Cikti: her IRCADb vakasi icin en iyi LiTS esi, ikinci en iyi (ayirt edicilik),
histogram mesafesi, dilim sayisi uyumu, is_match ve config'teki beklenen kumede mi.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.case_id import assert_unique_ids, glob_paths
from src.provenance.dedup_hash import SIG_SHAPE, signature_from_array

HU_RANGE = (-1024, 3071)


def orientation_variants(vol: np.ndarray) -> list[np.ndarray]:
    """16 yon: x/y transpoz (2) x her eksende ters (8). Ilk eleman = kendisi."""
    out = []
    for transpose in (False, True):
        base = np.swapaxes(vol, 0, 1) if transpose else vol
        for flips in itertools.product((False, True), repeat=3):
            axes = tuple(i for i, f in enumerate(flips) if f)
            out.append(np.flip(base, axes) if axes else base)
    return out


def oriented_signatures(arr: np.ndarray, hu_window=(-200, 250)) -> np.ndarray:
    """(16, 32^3) birim normlu imzalar; satir 0 = hacmin kendi yonu."""
    sig = signature_from_array(arr, "ct", hu_window).reshape(SIG_SHAPE)
    return np.stack([v.ravel() for v in orientation_variants(sig)]).astype(np.float32)


def hu_histogram(arr: np.ndarray) -> np.ndarray:
    """1 HU'luk kutularla normalize histogram. Yonden bagimsiz."""
    lo, hi = HU_RANGE
    a = np.clip(np.rint(np.asarray(arr, dtype=np.float32)), lo, hi).astype(np.int64) - lo
    h = np.bincount(a.ravel(), minlength=hi - lo + 1).astype(np.float64)
    return h / h.sum()


def histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Toplam varyasyon mesafesi, [0, 1]. 0 = ozdes dagilim."""
    return float(0.5 * np.abs(h1 - h2).sum())


def _features(x) -> dict:
    if isinstance(x, (str, Path)):
        import nibabel as nib
        arr = np.asarray(nib.load(str(x)).dataobj)
    else:
        arr = np.asarray(x)
    return {"sigs": oriented_signatures(arr), "hist": hu_histogram(arr),
            "shape": tuple(sorted(int(s) for s in arr.shape[:3]))}


def match_table(ircadb: dict, lits: dict, expected_lits: set[str] | None = None,
                threshold: float = 0.99, verbose: bool = False) -> pd.DataFrame:
    """ircadb / lits: {kimlik: dizi ya da NIfTI yolu}. Her LiTS hacmi bir kez okunur."""
    lits_ids = sorted(lits)
    lits_sig, lits_hist, lits_shape = [], [], []
    for i, k in enumerate(lits_ids, 1):
        if verbose:
            print(f"[match] LiTS {i}/{len(lits_ids)} {k}", flush=True)
        f = _features(lits[k])
        lits_sig.append(f["sigs"][0])          # LiTS kendi yonunde; IRCADb dondurulur
        lits_hist.append(f["hist"])
        lits_shape.append(f["shape"])
    lits_sig = np.stack(lits_sig)

    rows = []
    for k in sorted(ircadb):
        if verbose:
            print(f"[match] IRCADb {k}", flush=True)
        f = _features(ircadb[k])
        sim = (f["sigs"] @ lits_sig.T).max(axis=0)   # her LiTS icin en iyi yon
        order = np.argsort(sim)[::-1]
        b, s = order[0], order[1] if len(order) > 1 else None
        best_id = lits_ids[b]
        rows.append({
            "ircadb_id": k,
            "best_lits_id": best_id,
            "best_cos": round(float(sim[b]), 6),
            "second_lits_id": lits_ids[s] if s is not None else None,
            "second_cos": round(float(sim[s]), 6) if s is not None else np.nan,
            "hist_tv": round(histogram_distance(f["hist"], lits_hist[b]), 6),
            "same_shape": f["shape"] == lits_shape[b],
            "is_match": bool(sim[b] >= threshold),
            "in_expected": (best_id in expected_lits) if expected_lits is not None else None,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, expected_lits: set[str]) -> str:
    matched = set(df.loc[df.is_match, "best_lits_id"])
    lines = [f"{int(df.is_match.sum())}/{len(df)} IRCADb vakasi bir LiTS vakasiyla eslesti."]
    dup = df.loc[df.is_match, "best_lits_id"].duplicated(keep=False)
    if dup.any():
        lines.append(f"[!] Ayni LiTS'e birden fazla IRCADb eslesti: "
                     f"{sorted(df.loc[df.is_match][dup].best_lits_id.unique())}")
    if unexpected := sorted(matched - expected_lits):
        lines.append(f"[!] Beklenen kume (config) DISINDA eslesme: {unexpected}")
    if missing := sorted(expected_lits - matched):
        lines.append(f"[!] Beklenen ama eslesmeyen LiTS: {missing}")
    if not dup.any() and not unexpected and not missing and len(matched) == len(df):
        lines.append("TEYIT: IRCADb'nin tamami config'teki LiTS kumesiyle birebir eslesti.")
    return "\n".join(lines)


def _load_expected(config: Path) -> set[str]:
    import yaml
    cfg = yaml.safe_load(config.read_text())
    return {f"lits-{i}" for i in cfg["leakage"]["lits_ircadb_indices"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ircadb-glob", default="data/raw/3Dircadb1/**/ct.nii.gz")
    ap.add_argument("--lits-glob", default="data/raw/LiTS/**/volume-*.nii*")
    ap.add_argument("--config", type=Path, default=Path("configs/paths.yaml"))
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--out", type=Path, default=Path("outputs/ircadb_lits_match.csv"))
    a = ap.parse_args()

    ircadb = assert_unique_ids(glob_paths(a.ircadb_glob))
    lits = assert_unique_ids(glob_paths(a.lits_glob))
    if not ircadb or not lits:
        raise SystemExit(f"Dosya bulunamadi: IRCADb={len(ircadb)} LiTS={len(lits)}")
    expected = _load_expected(a.config)
    df = match_table(ircadb, lits, expected, a.threshold, verbose=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))
    print()
    print(summarize(df, expected))
    print(f"-> {a.out}")
