"""Klinik hedefleme hatasi metrigi -- projenin novelty kolu.

Soru: "AI'in cizdigi tumor maskesine bakarak ablasyon/biyopsi planlansaydi
kac milimetre sapardik, ve kac vakada ablasyon marji tutmazdi?"

Dice 'ne kadar ortusuyor' der. Bu metrik 'klinik karar degisir mi' der.

Uc sayi uretir:
  1) centroid_error_mm  -- AI maskesinin agirlik merkezi ile GT merkezinin mesafesi
  2) margin_coverage    -- AI maskesi + N mm marj, gercek tumoru kapsiyor mu (oran)
  3) margin_violation   -- kapsama < 1.0 ise ihlal (binary)

Hepsi lezyon boyutuna gore katmanlanir.

ESLESTIRME TESPITTEN FARKLIDIR (bilerek):
  Tespit (lesionwise) IoU >= 0.5 ister. Hedefleme ister MIYOR: bir GT lezyonuyla
  en az bir voksel kesisen TUM tahmin bilesenlerinin birlesimi "planlama maskesi"dir
  -- klinisyen ablasyonu o maskeye gore planlardi. Boylece:
    - tumoru genis kaplayan tahmin (IoU dusuk)  -> ihlal DEGIL (klinikte en guvenli)
    - iki lezyonu tek blob cizen tahmin         -> ikisi de kapsanir, merkez kayar
    - parcali tahmin                            -> parcalar birlesir, n_plan_components > 1
  Kesisen bilesen yoksa lezyon kacirilmistir (detected=False, ihlal).

ASIRI SEGMENTASYON CEZASIZ KALMASIN: tum karacigeri 'tumor' cizen model marji
hep tutturur. Bu yuzden her lezyon icin planlama hacmi ve FAZLA hacim (plan
icindeki, hicbir GT lezyonuna ait olmayan voksel = saglikli doku) raporlanir.
Ihlal orani TEK BASINA yorumlanmaz; fazla hacimle birlikte okunur.

Saf numpy/scipy. Veri olmadan test edilebilir.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy import ndimage

from .lesionwise import (
    connected_components,
    equivalent_diameter_mm,
    stratum_of,
)


@dataclass
class TargetingResult:
    case_id: str
    gt_label: int
    gt_diameter_mm: float
    stratum: str
    detected: bool
    centroid_error_mm: float      # tespit edilmediyse nan
    margin_coverage: float        # GT vokselinin kac orani marjin icinde
    margin_violation: bool
    margin_mm: float
    n_plan_components: int = 0    # planlama maskesini olusturan tahmin bileseni sayisi
    plan_shared_with_n_gt: int = 0  # ayni plani paylasan GT lezyonu sayisi (1 = paylasilmiyor)
    gt_volume_ml: float = float("nan")
    plan_volume_ml: float = float("nan")
    excess_volume_ml: float = float("nan")  # plan - (plan ∩ tum GT): saglikli doku


def _centroid_mm(mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """Maskenin agirlik merkezi, mm cinsinden (voksel indeksi * spacing).

    NOT: dunya koordinati degil, voksel-indeks * spacing. Ayni hacim icindeki
    iki maskeyi karsilastirdigimiz icin affine'e gerek yok; farkli hacimler
    karsilastirilacaksa affine uygulanmali.
    """
    idx = np.argwhere(mask)
    if idx.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    return idx.mean(axis=0) * np.asarray(spacing, dtype=float)


def _dilate_mm(mask: np.ndarray, spacing: Sequence[float], margin_mm: float) -> np.ndarray:
    """Anizotropik voksellerde DOGRU mm genisletme.

    Yanlis yol: binary_dilation(iterations=N) -- eksenler farkli mm'ye karsilik
    gelir, 5mm marj z ekseninde 35mm olur.
    Dogru yol: mesafe donusumu (sampling=spacing) ile gercek oklid mesafesi.
    """
    if margin_mm <= 0:
        return np.asarray(mask) > 0
    # maske DISINDAKI her vokselin maskeye mm cinsinden uzakligi
    dist = ndimage.distance_transform_edt(~(np.asarray(mask) > 0), sampling=spacing)
    return dist <= margin_mm


def _union_box(slices: list[tuple[slice, ...]], pad: Sequence[int],
               shape: Sequence[int]) -> tuple[slice, ...]:
    lo = np.min([[sl.start for sl in b] for b in slices], axis=0) - np.asarray(pad)
    hi = np.max([[sl.stop for sl in b] for b in slices], axis=0) + np.asarray(pad)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def targeting_errors(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: Sequence[float],
    case_id: str = "case",
    margin_mm: float = 5.0,
    strata: Sequence[float] = (10.0, 20.0),
) -> list[TargetingResult]:
    """Her GT lezyonu icin hedefleme hatasi (ortusme tabanli planlama maskesi)."""
    margin_mm = float(margin_mm)
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError(f"margin_mm sonlu ve >= 0 olmali: {margin_mm}")
    gt = np.asarray(gt) > 0
    pred = np.asarray(pred) > 0
    if gt.shape != pred.shape:
        raise ValueError(f"{case_id}: sekil uyusmuyor {gt.shape} vs {pred.shape}")
    spacing = tuple(float(s) for s in spacing)
    vox_ml = float(np.prod(spacing)) / 1000.0

    gt_lab, n_gt = connected_components(gt)
    pr_lab, n_pr = connected_components(pred)
    gt_sl = ndimage.find_objects(gt_lab)
    pr_sl = ndimage.find_objects(pr_lab)
    pr_size = np.bincount(pr_lab.ravel(), minlength=n_pr + 1)

    # GT <-> tahmin bilesen komsulugu ve tahmin bileseni basina GT ile kesisim
    both = gt & pred
    pairs = np.unique(np.stack([gt_lab[both], pr_lab[both]], axis=1), axis=0) if both.any() \
        else np.zeros((0, 2), dtype=int)
    gt2pr: dict[int, set[int]] = {}
    pr2gt: dict[int, set[int]] = {}
    for g, p in pairs:
        gt2pr.setdefault(int(g), set()).add(int(p))
        pr2gt.setdefault(int(p), set()).add(int(g))
    pr_in_gt = np.bincount(pr_lab[both], minlength=n_pr + 1)

    # marj icin kirpma payi: her eksende margin_mm'yi kapsayan voksel sayisi + 1
    pad = [int(np.ceil(margin_mm / s)) + 1 for s in spacing]

    results: list[TargetingResult] = []
    for i in range(1, n_gt + 1):
        g_box = gt_sl[i - 1]
        n_vox = int((gt_lab[g_box] == i).sum())
        diam = equivalent_diameter_mm(n_vox, spacing)
        stratum = stratum_of(diam, strata)
        hits = sorted(gt2pr.get(i, ()))

        if not hits:
            results.append(TargetingResult(
                case_id=case_id, gt_label=i, gt_diameter_mm=diam,
                stratum=stratum, detected=False,
                centroid_error_mm=float("nan"),
                margin_coverage=0.0, margin_violation=True, margin_mm=margin_mm,
                n_plan_components=0, plan_shared_with_n_gt=0,
                gt_volume_ml=n_vox * vox_ml, plan_volume_ml=0.0, excess_volume_ml=0.0,
            ))
            continue

        # tum islem plan+GT kutusunda: plan vokselleri kutunun icinde oldugundan
        # EDT ve merkezler tam hacimdekiyle AYNI (kaba kuvvet testiyle dogrulandi)
        box = _union_box([g_box] + [pr_sl[j - 1] for j in hits], pad, gt.shape)
        plan = np.isin(pr_lab[box], hits)
        gt_mask = gt_lab[box] == i
        err = float(np.linalg.norm(_centroid_mm(plan, spacing) - _centroid_mm(gt_mask, spacing)))
        grown = _dilate_mm(plan, spacing, margin_mm)
        coverage = float(np.logical_and(grown, gt_mask).sum() / n_vox)

        plan_vox = int(pr_size[hits].sum())
        excess_vox = plan_vox - int(pr_in_gt[hits].sum())
        shared = len(set().union(*(pr2gt[j] for j in hits)))

        results.append(TargetingResult(
            case_id=case_id, gt_label=i, gt_diameter_mm=diam,
            stratum=stratum, detected=True,
            centroid_error_mm=err,
            margin_coverage=coverage,
            margin_violation=bool(coverage < 1.0),
            margin_mm=margin_mm,
            n_plan_components=len(hits),
            plan_shared_with_n_gt=shared,
            gt_volume_ml=n_vox * vox_ml,
            plan_volume_ml=plan_vox * vox_ml,
            excess_volume_ml=excess_vox * vox_ml,
        ))
    return results


def summarize_targeting(results: list[TargetingResult]) -> dict:
    """Katman bazli ozet.

    ONEMLI: tespit edilemeyen lezyonlarda centroid hatasi TANIMSIZ'dir.
    Bunlari 0 sayarak ortalamaya katmak metrigi yalanci yapar -- ayri raporlanir.
    """
    out: dict = {"overall": {}, "by_size": {}}

    def _agg(rs: list[TargetingResult]) -> dict:
        if not rs:
            return {}
        det = [r for r in rs if r.detected]
        errs = np.array([r.centroid_error_mm for r in det], dtype=float)
        return {
            "n_lesions": len(rs),
            "n_detected": len(det),
            # ORTUSME tabanli (>=1 voksel). lesion.recall ile KARISTIRMA: o IoU>=esik.
            "overlap_hit_rate": len(det) / len(rs),
            # asagidaki uc sayi YALNIZCA tespit edilenler uzerinden
            "centroid_error_mm_median": float(np.median(errs)) if errs.size else float("nan"),
            "centroid_error_mm_p90": float(np.percentile(errs, 90)) if errs.size else float("nan"),
            "centroid_error_mm_mean": float(errs.mean()) if errs.size else float("nan"),
            # ihlal orani TUM lezyonlar uzerinden (kacirilan = ihlal)
            "margin_violation_rate_all": float(np.mean([r.margin_violation for r in rs])),
            "margin_violation_rate_detected": float(np.mean([r.margin_violation for r in det])) if det else float("nan"),
            "margin_coverage_median_detected": float(np.median([r.margin_coverage for r in det])) if det else float("nan"),
            # asiri segmentasyon: ihlal orani bununla birlikte okunur
            "excess_volume_ml_median_detected": float(np.median([r.excess_volume_ml for r in det])) if det else float("nan"),
            "plan_to_gt_volume_ratio_median_detected": float(np.median(
                [r.plan_volume_ml / r.gt_volume_ml for r in det])) if det else float("nan"),
            "n_shared_plan": int(sum(r.plan_shared_with_n_gt > 1 for r in det)),
        }

    out["overall"] = _agg(results)
    for s in sorted({r.stratum for r in results}):
        out["by_size"][s] = _agg([r for r in results if r.stratum == s])
    return out


def results_to_records(results: list[TargetingResult]) -> list[dict]:
    return [asdict(r) for r in results]
