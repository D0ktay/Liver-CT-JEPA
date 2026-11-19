"""Hasta duzeyi bootstrap guven araliklari -- evaluate ciktilarindan, maske gerektirmez.

Neden: "marj ihlali %33" tek basina yorumlanamaz; 131 vakada araligi %25-41 de olabilir
%31-35 de. Hakemin ilk sorusu bu. CLAUDE.md kural 7 (kucuk fark iddia etme) da bunu ister.

Neden HASTA duzeyi: ayni hastanin lezyonlari bagimsiz degil (ayni tarayici, ayni kontrast,
ayni model davranisi). Lezyonlari tek tek yeniden orneklemek araligi yapay olarak daraltir.
Burada her tekrarda VAKALAR yerine koyarak cekilir; vakanin tum lezyonlari (TP/FN/FP)
onunla birlikte gelir. Tumorsuz vakalar da orneklemde kalir (spesifiklik icin).

Nokta tahmini summary.json'daki sayinin AYNISIDIR (testle sabit). Aralik: yuzdelik
yontemi (%2.5-%97.5). Bir tekrarda metrik tanimsizsa (or. o tekrarda hic <=10mm lezyon
yok) tekrar 0 sayilmaz, atlanir; kac tekrarin gecerli oldugu n_valid ile raporlanir.

Girdi: outputs/eval/per_case.csv + per_lesion.csv. Cikti: outputs/eval/ci.csv
    python -m src.metrics.bootstrap --eval outputs/eval --n-boot 2000 --seed 0
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_NAN = float("nan")


def _bool(col: pd.Series, nan_ok: np.ndarray | None = None) -> np.ndarray:
    """KATI: True/False (bool ya da 'True'/'False' metni). Baska her sey FIRLATIR.
    NaN yalniz nan_ok satirlarinda kabul edilir (or. FP satirinin 'detected'i) ve False olur."""
    out = np.zeros(len(col), bool)
    for i, v in enumerate(col.to_numpy(object)):
        if isinstance(v, (bool, np.bool_)):
            out[i] = bool(v)
        elif isinstance(v, str) and v.strip().lower() in ("true", "false"):
            out[i] = v.strip().lower() == "true"
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            if nan_ok is None or not nan_ok[i]:
                raise ValueError(f"{col.name}: {i}. satir bos (NaN) -- bu satirda tanimli olmali")
        else:
            raise ValueError(f"{col.name}: taninmayan deger {v!r} ({i}. satir); True/False beklenir")
    return out


class _Tables:
    """Metriklerin ihtiyac duydugu sutunlar, numpy olarak (tekrar basina hizli)."""

    def __init__(self, per_case: pd.DataFrame, per_lesion: pd.DataFrame):
        self.gt_empty = _bool(per_case["gt_empty"])
        self.dice = pd.to_numeric(per_case["dice"], errors="coerce").to_numpy(float)
        self.fp_case = _bool(per_case["fp_case"])
        self.status = per_lesion["status"].astype(str).to_numpy()
        is_gt = per_lesion["status"].isin(["TP", "FN"]).to_numpy(bool)
        self.is_gt = is_gt
        fp_row = ~is_gt
        self.stratum = per_lesion["stratum"].astype(object).where(
            per_lesion["stratum"].notna(), None).to_numpy() if "stratum" in per_lesion \
            else np.full(len(per_lesion), None, object)
        self.detected = _bool(per_lesion["detected"], fp_row) & is_gt
        self.violation = _bool(per_lesion["margin_violation"], fp_row) & is_gt
        self.centroid = pd.to_numeric(per_lesion["centroid_error_mm"],
                                      errors="coerce").to_numpy(float)

    def take(self, ic: np.ndarray, il: np.ndarray) -> "_Tables":
        t = object.__new__(_Tables)
        for k in ("gt_empty", "dice", "fp_case"):
            setattr(t, k, getattr(self, k)[ic])
        for k in ("status", "is_gt", "stratum", "detected", "violation", "centroid"):
            setattr(t, k, getattr(self, k)[il])
        return t


# Her metrik (deger, payda) dondurur. Payda = nokta tahminin dayandigi lezyon/vaka sayisi:
# 1 lezyonlu katmanda GA [1, 1] "kesinlik" degil, "tek gozlem" demektir.
def _ratio(num: int, den: int) -> tuple[float, int]:
    return (num / den if den else _NAN), int(den)


def _median(x: np.ndarray) -> tuple[float, int]:
    x = x[np.isfinite(x)]
    return (float(np.median(x)) if x.size else _NAN), int(x.size)


def _dice_median(t, sel):
    return _median(t.dice[~t.gt_empty])


def _specificity(t, sel):
    free = t.gt_empty
    return (float(1 - t.fp_case[free].mean()) if free.any() else _NAN), int(free.sum())


def _precision(t, sel):
    tp = int((t.status == "TP").sum())
    return _ratio(tp, tp + int((t.status == "FP").sum()))


def _recall(t, sel):
    s = t.status[sel]
    tp = int((s == "TP").sum())
    return _ratio(tp, tp + int((s == "FN").sum()))


def _f1(t, sel):
    tp = int((t.status == "TP").sum())
    (p, _), (r, n) = _precision(t, sel), _recall(t, sel)
    return (2 * p * r / (p + r) if (tp and np.isfinite(p) and np.isfinite(r)) else _NAN), n


def _hit(t, sel):
    return _ratio(int(t.detected[sel].sum()), int(sel.sum()))


def _viol_all(t, sel):
    return _ratio(int(t.violation[sel].sum()), int(sel.sum()))


def _viol_det(t, sel):
    d = sel & t.detected
    return _ratio(int(t.violation[d].sum()), int(d.sum()))


def _centroid_median(t, sel):
    return _median(t.centroid[sel & t.detected])


# ad -> (fonksiyon, katmanlanabilir mi). Tanimlar lesionwise.summarize / targeting.
# summarize_targeting / evaluate.aggregate ile AYNI (test_nokta_tahmini_... sabitler).
_DEFS: dict[str, tuple[Callable, bool]] = {
    "dice_median": (_dice_median, False),
    "lesion_precision": (_precision, False),     # FP'nin capi yok -> katmansiz
    "lesion_recall": (_recall, True),
    "lesion_f1": (_f1, False),
    "specificity_case": (_specificity, False),
    "overlap_hit_rate": (_hit, True),
    "margin_violation_rate_all": (_viol_all, True),
    "margin_violation_rate_detected": (_viol_det, True),
    "centroid_error_mm_median": (_centroid_median, True),
}
METRICS = tuple(_DEFS)
# ci_table'in katman basina raporladiklari
STRATIFIED = ("lesion_recall", "overlap_hit_rate", "margin_violation_rate_all")
_NAME_RE = re.compile(r"^([a-z0-9_]+)(?:\[(.+)\])?$")


def _metric_fn(name: str) -> Callable[[_Tables], tuple[float, int]]:
    m = _NAME_RE.match(name)
    if not m or m.group(1) not in _DEFS:
        raise KeyError(f"bilinmeyen metrik: {name!r}. Mevcut: {METRICS} (+ '[katman]')")
    fn, stratifiable = _DEFS[m.group(1)]
    stratum = m.group(2)
    if stratum is not None and not stratifiable:
        raise KeyError(f"{m.group(1)} katmanlanamaz: {name!r}")

    def run(t: _Tables) -> tuple[float, int]:
        sel = t.is_gt if stratum is None else (t.is_gt & (t.stratum == stratum))
        v, n = fn(t, sel)
        return float(v), int(n)
    return run


def _validate(per_case: pd.DataFrame, per_lesion: pd.DataFrame) -> None:
    for name, df in (("per_case", per_case), ("per_lesion", per_lesion)):
        if "case_id" not in df:
            raise ValueError(f"{name}: case_id sutunu yok")
    if per_case.empty:
        raise ValueError("per_case bos: degerlendirilecek vaka yok")
    ids = per_case["case_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"per_case'te tekrar eden vaka: {ids[ids.duplicated()].tolist()[:5]}")
    extra = set(per_lesion["case_id"].astype(str)) - set(ids)
    if extra:
        raise ValueError(f"per_lesion'daki vakalar per_case'te yok (farkli kosular?): "
                         f"{sorted(extra)[:5]}")
    # ters yon: per_case'in soyledigi lezyon sayisi per_lesion satirlariyla tutmali
    st = per_lesion["status"].astype(str)
    cid = per_lesion["case_id"].astype(str)
    n_gt = cid[st.isin(["TP", "FN"])].value_counts()
    n_pr = cid[st.isin(["TP", "FP"])].value_counts()
    for col, got in (("n_gt_lesions", n_gt), ("n_pred_lesions", n_pr)):
        if col not in per_case:
            raise ValueError(f"per_case: {col} sutunu yok (evaluate ciktisi mi?)")
        want = pd.Series(per_case[col].to_numpy(int), index=ids)
        have = got.reindex(ids, fill_value=0)
        bad = want.index[want.to_numpy() != have.to_numpy()]
        if len(bad):
            raise ValueError(f"{col}: per_case ile per_lesion lezyon sayisi tutmuyor "
                             f"(eksik/yanlis birlestirilmis satirlar?): {list(bad[:5])}")


def point_estimates(per_case: pd.DataFrame, per_lesion: pd.DataFrame,
                    names: list[str] | None = None) -> dict[str, float]:
    _validate(per_case, per_lesion)
    t = _Tables(per_case.reset_index(drop=True), per_lesion.reset_index(drop=True))
    return {n: _metric_fn(n)(t)[0] for n in (names or _default_names(per_lesion))}


def _default_names(per_lesion: pd.DataFrame) -> list[str]:
    strata = sorted(s for s in per_lesion.get("stratum", pd.Series(dtype=object)).dropna().unique()
                    if s != "unknown")
    return list(METRICS) + [f"{m}[{s}]" for m in STRATIFIED for s in strata]


def _bootstrap(per_case, per_lesion, names, n_boot, seed, alpha, return_replicates=False):
    _validate(per_case, per_lesion)
    if n_boot < 1:
        raise ValueError(f"n_boot >= 1 olmali: {n_boot}")
    per_case = per_case.reset_index(drop=True)
    per_lesion = per_lesion.reset_index(drop=True)
    base = _Tables(per_case, per_lesion)
    fns = {n: _metric_fn(n) for n in names}
    point = {n: f(base) for n, f in fns.items()}      # (deger, payda)

    case_ids = per_case["case_id"].astype(str).to_numpy()
    groups = per_lesion.groupby(per_lesion["case_id"].astype(str)).indices
    les_idx = [np.asarray(groups.get(c, ()), dtype=np.int64) for c in case_ids]
    n = len(case_ids)
    rng = np.random.default_rng(seed)
    reps = {nm: np.empty(n_boot) for nm in names}
    for b in range(n_boot):
        ic = rng.integers(0, n, n)
        il = np.concatenate([les_idx[i] for i in ic]) if n else np.zeros(0, np.int64)
        t = base.take(ic, il)
        for nm, f in fns.items():
            reps[nm][b] = f(t)[0]

    out = []
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    for nm in names:
        r = reps[nm][np.isfinite(reps[nm])]
        lo, hi = (float(v) for v in np.percentile(r, q)) if r.size else (_NAN, _NAN)
        row = {"metric": nm, "point": point[nm][0], "lo": lo, "hi": hi,
               "n_denominator": point[nm][1], "n_valid": int(r.size), "n_boot": n_boot,
               "n_cases": n}
        if return_replicates:
            row["replicates"] = reps[nm]
        out.append(row)
    return out


def bootstrap_ci(per_case: pd.DataFrame, per_lesion: pd.DataFrame, metric: str,
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                 return_replicates: bool = False) -> dict:
    """Tek metrik icin hasta duzeyi bootstrap. Ad ornekleri: 'lesion_recall',
    'margin_violation_rate_all[<=10mm]'."""
    return _bootstrap(per_case, per_lesion, [metric], n_boot, seed, alpha,
                      return_replicates)[0]


def ci_table(per_case: pd.DataFrame, per_lesion: pd.DataFrame, n_boot: int = 2000,
             seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Tum temel metrikler + katmanlilar, AYNI tekrarlarla."""
    return pd.DataFrame(_bootstrap(per_case, per_lesion, _default_names(per_lesion),
                                   n_boot, seed, alpha))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, default=Path("outputs/eval"),
                    help="per_case.csv + per_lesion.csv klasoru")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="vars.: <eval>/ci.csv")
    a = ap.parse_args()
    per_case = pd.read_csv(a.eval / "per_case.csv")
    per_lesion = pd.read_csv(a.eval / "per_lesion.csv")
    df = ci_table(per_case, per_lesion, a.n_boot, a.seed)
    out = a.out or a.eval / "ci.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 140, "display.max_rows", 100):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[ci] {len(per_case)} vaka, {a.n_boot} tekrar, tohum {a.seed} -> {out}")
    none = df[df.n_valid == 0].metric.tolist()
    if none:
        print(f"[UYARI] hic gecerli tekrar yok (aralik hesaplanamadi): {none}")
    part = df[(df.n_valid > 0) & (df.n_valid < df.n_boot)].metric.tolist()
    if part:
        print(f"[UYARI] bazi tekrarlarda tanimsiz -> aralik 'katman orneklemde varsa' kosullu: {part}")
    small = df[df.n_denominator < 10].metric.tolist()
    if small:
        print(f"[UYARI] paydasi <10 (aralik guvenilmez, sayiyla birlikte raporla): {small}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
