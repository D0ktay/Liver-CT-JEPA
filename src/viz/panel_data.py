"""Degerlendirme ciktilari -> LiTS Kiyas Paneli (claude.ai artifact) veri tabani belgeleri.

    python -m src.viz.panel_data run  --eval outputs/kaggle131/eval --key ts_run1 --model ts \\
        --label "TotalSegmentator, kosu 1" --short "TS kosu 1" --order 1 --note "..."
    python -m src.viz.panel_data margin --sweep outputs/.../margin_sweep.json --key ts_all131 --run ts_fixed

Cikti: outputs/panel/docs/<koleksiyon>/<id>.json (ArtifactData batch'e file_path olarak verilir).
Sayilar ci.csv'den (yoksa per_case/per_lesion'dan hesaplanir): nokta summary.json ile ayni (testli).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

OUT = Path("outputs/panel/docs")


def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


def run_doc(eval_dir: Path, common_with: Path | None = None, **fields) -> dict:
    """common_with: o degerlendirmenin vakalariyla ORTAK kumede yeniden hesapla.

    Farkli vaka kumelerini yan yana koymak (TS 126 / JEPA 131) kiyasi bozar; bu kipte
    hazir ci.csv KULLANILMAZ, cunku o tam kume icin hesaplanmistir."""
    from src.metrics.bootstrap import ci_table
    s = json.loads((eval_dir / "summary.json").read_text())
    extra: dict = {}
    if common_with is None:
        ci = eval_dir / "ci.csv"
        df = pd.read_csv(ci) if ci.exists() else ci_table(
            pd.read_csv(eval_dir / "per_case.csv"), pd.read_csv(eval_dir / "per_lesion.csv"))
        n_cases = int(s["n_cases"])
    else:
        pc = pd.read_csv(eval_dir / "per_case.csv")
        pl = pd.read_csv(eval_dir / "per_lesion.csv")
        keep = set(pd.read_csv(Path(common_with) / "per_case.csv")["case_id"])
        sub = pc[pc["case_id"].isin(keep)]
        if sub.empty:
            raise ValueError(f"{eval_dir} ile {common_with} arasinda ortak vaka yok")
        df = ci_table(sub, pl[pl["case_id"].isin(keep)])
        n_cases = int(len(sub))
        extra = {"restricted_to": str(common_with), "n_dropped": int(len(pc) - len(sub))}
    metrics = {r.metric: {"p": _num(r.point), "lo": _num(r.lo), "hi": _num(r.hi), "n": int(r.n_denominator)}
               for r in df.itertuples()}
    return {**fields, **extra, "n_cases": n_cases, "n_failed": s.get("n_failed", 0),
            "failed_cases": s.get("failed_cases", []), "metrics": metrics,
            "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}


def margin_doc(sweep_json: Path, **fields) -> dict:
    s = json.loads(sweep_json.read_text())
    pts = [{"m": float(m), "all": _num(v["margin_violation_rate_all"]),
            "det": _num(v["margin_violation_rate_detected"]),
            "ratio": _num(v["ablation_to_ideal_ratio_median_detected"]),
            "extra_ml": _num(v["extra_vs_ideal_ml_median_detected"])} for m, v in s["overall"].items()]
    return {**fields, "points": sorted(pts, key=lambda p: p["m"]), "base_margin_mm": s["params"]["base_margin_mm"],
            "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}


def write(collection: str, key: str, doc: dict) -> Path:
    p = OUT / collection / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--eval", type=Path, required=True)
    for f in ("key", "model", "label", "short"):
        r.add_argument(f"--{f}", required=True)
    r.add_argument("--order", type=int, default=50)
    r.add_argument("--note", default="")
    r.add_argument("--hidden", action="store_true")
    r.add_argument("--common-with", type=Path, default=None,
                   help="bu degerlendirmenin vakalariyla ortak kumede hesapla (adil kiyas)")
    m = sub.add_parser("margin")
    m.add_argument("--sweep", type=Path, required=True)
    m.add_argument("--key", required=True)
    m.add_argument("--run", required=True)
    m.add_argument("--short", default="")
    a = ap.parse_args()
    if a.cmd == "run":
        doc = run_doc(a.eval, common_with=a.common_with, model=a.model, label=a.label, short=a.short,
                      order=a.order, note=a.note, show=not a.hidden)
        print(write("runs", a.key, doc))
    else:
        print(write("margin", a.key, margin_doc(a.sweep, run=a.run, short=a.short)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
