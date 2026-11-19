"""Lezyon-bazli tespit metrikleri.

Neden var: LiTS 2017'de Dice'ta 1. olan yontem lezyon precision'inda 14. oldu
(Dice 0.702 / precision 0.156). Voksel metrigi ile tespit metrigi ayni sey degil.
Bu modul ikisini ayri ayri olcer, boyut katmanlariyla birlikte.

Tum fonksiyonlar saf numpy -- veri olmadan test edilebilir (bkz. tests/).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# temel yardimcilar
# --------------------------------------------------------------------------
def voxel_volume_mm3(spacing: Sequence[float]) -> float:
    """spacing: (sx, sy, sz) mm cinsinden."""
    sx, sy, sz = (float(s) for s in spacing)
    return sx * sy * sz


def equivalent_diameter_mm(n_voxels: int, spacing: Sequence[float]) -> float:
    """Lezyon hacmine esdeger kurenin capi (mm).

    Boyut katmanlamasi bunun uzerinden yapilir; literaturdeki '<=2 cm lezyon'
    tanimi da capa gore veriliyor.
    """
    vol = n_voxels * voxel_volume_mm3(spacing)
    if vol <= 0:
        return 0.0
    return 2.0 * (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)


def connected_components(mask: np.ndarray, connectivity: int = 3):
    """3B baglantili bilesenler. connectivity=3 -> 26-komsuluk."""
    mask = np.asarray(mask) > 0
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels, n = ndimage.label(mask, structure=structure)
    return labels, int(n)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    s = a.sum() + b.sum()
    if s == 0:
        return float("nan")      # iki taraf da bos -> Dice TANIMSIZ, 1.0 degil
    return 2.0 * np.logical_and(a, b).sum() / s


# --------------------------------------------------------------------------
# lezyon eslestirme
# --------------------------------------------------------------------------
@dataclass
class LesionMatch:
    gt_label: int | None
    pred_label: int | None
    iou: float
    gt_voxels: int
    pred_voxels: int
    gt_diameter_mm: float
    status: str          # "TP" | "FN" | "FP"


def match_lesions(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: Sequence[float],
    iou_threshold: float = 0.5,
) -> list[LesionMatch]:
    """GT ve tahmin lezyonlarini greedy olarak IoU'ya gore eslestirir.

    Greedy + azalan IoU sirasi kullaniliyor; Hungarian'a gore pratikte fark
    ihmal edilebilir ve deterministik.
    """
    gt_lab, n_gt = connected_components(gt)
    pr_lab, n_pr = connected_components(pred)

    # tek geciste boyutlar (etiket basina tam hacim taramasi yerine)
    gt_cnt = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
    pr_cnt = np.bincount(pr_lab.ravel(), minlength=n_pr + 1)
    gt_sizes = {i: int(gt_cnt[i]) for i in range(1, n_gt + 1)}
    pr_sizes = {j: int(pr_cnt[j]) for j in range(1, n_pr + 1)}

    # cakisma matrisi -- sadece gercekten kesisen ciftler
    pairs: list[tuple[float, int, int]] = []
    if n_gt and n_pr:
        overlap = np.zeros((n_gt + 1, n_pr + 1), dtype=np.int64)
        both = (gt_lab > 0) & (pr_lab > 0)
        np.add.at(overlap, (gt_lab[both], pr_lab[both]), 1)
        for i in range(1, n_gt + 1):
            for j in range(1, n_pr + 1):
                inter = int(overlap[i, j])
                if inter == 0:
                    continue
                union = gt_sizes[i] + pr_sizes[j] - inter
                pairs.append((inter / union, i, j))
    pairs.sort(reverse=True)

    used_gt: set[int] = set()
    used_pr: set[int] = set()
    matches: list[LesionMatch] = []

    for iou, i, j in pairs:
        if iou < iou_threshold or i in used_gt or j in used_pr:
            continue
        used_gt.add(i)
        used_pr.add(j)
        matches.append(LesionMatch(
            gt_label=i, pred_label=j, iou=iou,
            gt_voxels=gt_sizes[i], pred_voxels=pr_sizes[j],
            gt_diameter_mm=equivalent_diameter_mm(gt_sizes[i], spacing),
            status="TP",
        ))

    for i in range(1, n_gt + 1):
        if i not in used_gt:
            matches.append(LesionMatch(
                gt_label=i, pred_label=None, iou=0.0,
                gt_voxels=gt_sizes[i], pred_voxels=0,
                gt_diameter_mm=equivalent_diameter_mm(gt_sizes[i], spacing),
                status="FN",
            ))
    for j in range(1, n_pr + 1):
        if j not in used_pr:
            matches.append(LesionMatch(
                gt_label=None, pred_label=j, iou=0.0,
                gt_voxels=0, pred_voxels=pr_sizes[j],
                gt_diameter_mm=float("nan"),
                status="FP",
            ))
    return matches


# --------------------------------------------------------------------------
# ozetleme
# --------------------------------------------------------------------------
def stratum_of(diameter_mm: float, strata: Sequence[float] = (10.0, 20.0)) -> str:
    if not np.isfinite(diameter_mm):
        return "unknown"
    lo, hi = strata
    if diameter_mm <= lo:
        return f"<={lo:g}mm"
    if diameter_mm <= hi:
        return f"{lo:g}-{hi:g}mm"
    return f">{hi:g}mm"


def summarize(matches: Iterable[LesionMatch],
              strata: Sequence[float] = (10.0, 20.0)) -> dict:
    """Genel + boyut katmanli precision / recall / F1.

    NOT: FP'lerin GT capi yoktur, bu yuzden katman bazli precision hesaplanamaz.
    Katman kirilimi yalnizca RECALL (duyarlilik) icin anlamlidir -- literaturdeki
    '<=2cm duyarlilik %39.6' sayisi da tam olarak budur.
    """
    matches = list(matches)
    tp = sum(m.status == "TP" for m in matches)
    fn = sum(m.status == "FN" for m in matches)
    fp = sum(m.status == "FP" for m in matches)

    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (tp and np.isfinite(prec) and np.isfinite(rec)) else float("nan")

    out = {"tp": tp, "fp": fp, "fn": fn,
           "precision": prec, "recall": rec, "f1": f1,
           "by_size": {}}

    for m in matches:
        if m.status == "FP":
            continue
        s = stratum_of(m.gt_diameter_mm, strata)
        b = out["by_size"].setdefault(s, {"tp": 0, "fn": 0})
        b["tp" if m.status == "TP" else "fn"] += 1
    for s, b in out["by_size"].items():
        d = b["tp"] + b["fn"]
        b["recall"] = b["tp"] / d if d else float("nan")
        b["n_lesions"] = d
    return out


def matches_to_records(matches: Iterable[LesionMatch], case_id: str) -> list[dict]:
    return [{"case_id": case_id, **asdict(m)} for m in matches]
