"""Icerik tabanli kopya tespiti -- oneklerin kacirdigi sizintiyi yakalar.

Provenance (dosya adi) her zaman yetmez: ayni tarama farkli isimle, farkli
resample'la, farkli kirpmayla gelmis olabilir. Bu modul goruntunun KENDISINE
bakar.

Iki asama:
  1) UCUZ ON FILTRE  -- NIfTI basligi: shape, spacing, affine. Kopyalarin
     buyuk cogunlugu burada zaten ayni cikar.
  2) ICERIK IMZASI   -- HU penceresine kirp, 32x32x32'ye indir, z-normalize et,
     vektorize et. Kosinus benzerligi > esik -> kopya adayi.

--kind ct   : CT hacimleri (HU penceresi). Varsayilan glob'dan maskeler DISLANIR.
--kind mask : Sadece maske varsa (AbdomenAtlas mask_only). Maske >0 ikilenir;
              ayni taramanin farkli annotasyonlari yine yuksek benzerlik verir.
CT ile maskeyi AYNI calismada karistirma -- imzalar karsilastirilamaz.

Cikti: candidate_duplicates.csv. Bunu ELLE gozden gecir, sonra split kur.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.case_id import assert_unique_ids, case_id_from_path, glob_paths, strip_nifti

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

SIG_SHAPE = (32, 32, 32)
# --kind ct iken maskeler disarida birakilir. Alt dizgi DEGIL, klasor adi + tam dosya
# adi: eski desen 'liver' alt dizgisiyle MSD CT'lerini (imagesTr/liver_N) de atiyordu.
_MASK_DIRS = {"labelstr", "labelsts", "segmentations", "segmentation", "masks", "mask",
              "mask_only", "labels", "label"}
_MASK_STEMS = {"segmentation", "seg", "mask", "label", "labels", "tumor", "livertumor",
               "liver", "liver_tumor", "liver_lesions", "combined_labels", "gt"}
_MASK_STEM_RE = re.compile(r"^(segmentation-\d+|.+[_-](seg|mask|label|gt))$", re.IGNORECASE)


def is_mask_path(path: str | Path) -> bool:
    p = Path(path)
    if any(part.lower() in _MASK_DIRS for part in p.parts[:-1]):
        return True
    stem = strip_nifti(p.name)
    return stem.lower() in _MASK_STEMS or bool(_MASK_STEM_RE.match(stem))


def header_fingerprint(path: Path) -> dict:
    img = nib.load(str(path))
    hdr = img.header
    return {
        "path": str(path),
        "shape": tuple(int(x) for x in img.shape[:3]),
        "spacing": tuple(round(float(z), 4) for z in hdr.get_zooms()[:3]),
        "affine_hash": hash(np.round(img.affine, 3).tobytes()),
    }


def signature_from_array(arr: np.ndarray, kind: str = "ct",
                         hu_window=(-200, 250)) -> np.ndarray:
    """32^3 dusuk cozunurluklu, kontrast-normalize, birim normlu imza."""
    from scipy import ndimage

    arr = np.asarray(arr, dtype=np.float32)
    if kind == "mask":
        arr = (arr > 0).astype(np.float32)
    elif kind == "ct":
        lo, hi = hu_window
        arr = np.clip(arr, lo, hi)
    else:
        raise ValueError(f"kind 'ct' ya da 'mask' olmali: {kind}")
    zoom = [s / a for s, a in zip(SIG_SHAPE, arr.shape[:3])]
    small = ndimage.zoom(arr, zoom, order=1)
    small = small[:SIG_SHAPE[0], :SIG_SHAPE[1], :SIG_SHAPE[2]]
    pad = [(0, s - d) for s, d in zip(SIG_SHAPE, small.shape)]
    small = np.pad(small, pad, mode="edge")
    v = small.ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    if n == 0:
        # sabit hacim (bos maske vb.) -- sifir vektor her seyle 0 benzerlik verir,
        # kopya olarak isaretlenemez. Sessiz gecme.
        print("[UYARI] sabit hacim: imza sifir, kopya tespiti bu hacim icin KOR.")
        return v
    return v / n


def content_signature(path: Path, kind: str = "ct", hu_window=(-200, 250)) -> np.ndarray:
    img = nib.load(str(path))
    return signature_from_array(np.asarray(img.dataobj), kind, hu_window)


def similar_pairs(sigs: np.ndarray, threshold: float,
                  block: int = 1024) -> list[tuple[int, int, float]]:
    """i<j ve kosinus >= esik olan ciftler. n x n matrisi hic kurmaz (bloklu)."""
    n = len(sigs)
    out = []
    for s0 in range(0, n, block):
        sim = sigs[s0:s0 + block] @ sigs.T
        ii, jj = np.nonzero(sim >= threshold)
        for i, j in zip(ii, jj):
            gi = s0 + int(i)
            if int(j) > gi:
                out.append((gi, int(j), float(sim[i, j])))
    return out


def run(volumes: list[Path], out_dir: Path, threshold: float = 0.995,
        kind: str = "ct") -> pd.DataFrame:
    if nib is None:
        raise SystemExit("nibabel kurulu degil: pip install nibabel")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dedup] {len(volumes)} hacim taraniyor (kind={kind})...")
    headers = [header_fingerprint(p) for p in volumes]
    pd.DataFrame(headers).to_csv(out_dir / "headers.csv", index=False)

    sigs = np.stack([content_signature(p, kind) for p in volumes]).astype(np.float32)
    np.save(out_dir / "signatures.npy", sigs)

    rows = []
    for i, j, s in similar_pairs(sigs, threshold):
        rows.append({
            "a": str(volumes[i]), "b": str(volumes[j]),
            "a_id": case_id_from_path(volumes[i]), "b_id": case_id_from_path(volumes[j]),
            "cosine": round(s, 6),
            "same_shape": headers[i]["shape"] == headers[j]["shape"],
            "same_spacing": headers[i]["spacing"] == headers[j]["spacing"],
            "same_affine": headers[i]["affine_hash"] == headers[j]["affine_hash"],
        })
    df = pd.DataFrame(rows).sort_values("cosine", ascending=False) if rows else pd.DataFrame(
        columns=["a", "b", "a_id", "b_id", "cosine", "same_shape", "same_spacing", "same_affine"])
    df.to_csv(out_dir / "candidate_duplicates.csv", index=False)

    print(f"[dedup] {len(df)} kopya adayi (cosine >= {threshold}) -> {out_dir/'candidate_duplicates.csv'}")
    if len(df):
        print(df.head(30).to_string(index=False))
        print("\n[!] Bunlari ELLE dogrula: 'confirmed' sutunu ekle (1/0), "
              "candidate_duplicates.reviewed.csv olarak kaydet.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", required=True,
                    help="orn: 'data/raw/**/*.nii.gz'")
    ap.add_argument("--out", type=Path, default=Path("outputs/dedup"))
    ap.add_argument("--threshold", type=float, default=0.995)
    ap.add_argument("--kind", choices=["ct", "mask"], default="ct")
    ap.add_argument("--exclude", default=None,
                    help="ek olarak bu regex'e uyan yollari atla")
    a = ap.parse_args()
    vols = glob_paths(a.glob)
    before = len(vols)
    if a.kind == "ct":
        vols = [v for v in vols if not is_mask_path(v)]
    else:
        vols = [v for v in vols if is_mask_path(v)]
    if a.exclude:
        vols = [v for v in vols if not re.search(a.exclude, str(v), flags=re.IGNORECASE)]
    print(f"[dedup] {before - len(vols)}/{before} dosya disarida birakildi (kind={a.kind}).")
    if not vols:
        raise SystemExit(f"Hicbir dosya eslesmedi: {a.glob}")
    # iki farkli dosya ayni kimligi alirsa onay/birlesme yanlis dugume gider
    assert_unique_ids(vols)
    run(vols, a.out, a.threshold, a.kind)
