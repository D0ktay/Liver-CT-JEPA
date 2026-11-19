"""Dosya yolundan vaka kimligi -- baseline, dedup ve pairs AYNI fonksiyonu kullanir.

Neden: IRCADb'de her vakanin dosyasi 'ct.nii.gz'. Dosya adindan kimlik
turetilirse tum vakalar ayni kimlige cokup birbirinin ustune yazilir.

Kurallar (sirayla):
  1) Yolun herhangi bir yerinde BDMAP_xxxxxxxx  -> o          (AbdomenAtlas)
  2) volume-N / segmentation-N                  -> lits-N     (LiTS: CT ve GT ayni kimlik)
  3) Dosya adi jenerikse (ct, image, label...)  -> ust klasorun adi
  4) Aksi halde                                 -> uzantisiz dosya adi
"""
from __future__ import annotations

import glob
import re
from collections import Counter
from pathlib import Path

_BDMAP_RE = re.compile(r"BDMAP_\d+")
_LITS_RE = re.compile(r"^(?:volume|segmentation)-(\d+)$")

# Kimlik tasimayan, sadece icerik turunu soyleyen dosya adlari
GENERIC_STEMS = {
    "ct", "image", "imaging", "img", "volume", "label", "labels", "mask", "seg",
    "segmentation", "gt", "pred", "prediction", "tumor", "livertumor",
    "liver_tumor", "liver_lesions", "liver", "combined_labels",
}


def strip_nifti(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def case_id_from_path(path: str | Path) -> str:
    p = Path(path)
    m = _BDMAP_RE.search(str(p))
    if m:
        return m.group(0)
    stem = strip_nifti(p.name)
    m = _LITS_RE.match(stem)
    if m:
        return f"lits-{int(m.group(1))}"
    if stem.lower() in GENERIC_STEMS:
        # jenerik ad: kimligi tasiyan ilk jenerik-olmayan ust klasor
        for parent in p.parents:
            if parent.name and parent.name.lower() not in GENERIC_STEMS | {"segmentations", "masks"}:
                return parent.name
        raise ValueError(f"Kimlik turetilemedi: {path}")
    return stem


def assert_unique_ids(paths: list[Path]) -> dict[str, Path]:
    """Kimlik -> yol. Cakisma varsa sessizce ezmek yerine FIRLAT."""
    ids = [case_id_from_path(p) for p in paths]
    dup = [k for k, n in Counter(ids).items() if n > 1]
    if dup:
        ex = {d: [str(p) for p, i in zip(paths, ids) if i == d][:3] for d in dup[:5]}
        raise ValueError(f"{len(dup)} vaka kimligi cakisiyor (dosyalar ezilirdi): {ex}")
    return dict(zip(ids, paths))


def glob_paths(pattern: str) -> list[Path]:
    """Goreli VE mutlak glob ('**' dahil). Path().glob mutlak deseni reddeder."""
    return sorted(Path(p) for p in glob.glob(pattern, recursive=True))
