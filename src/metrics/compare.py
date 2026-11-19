"""Iki kolun AYNI vakalarda eslestirilmis karsilastirmasi -- evaluate ciktilarindan, maske gerektirmez.

Neden: CLAUDE.md kural 8 (kucuk fark iddia etme). Iki kolun GA'lari ust uste biniyor diye "fark
yok", binmiyor diye "fark var" DENMEZ: ayni 27 vakada olculen iki kol bagimsiz degildir. Dogru
soru farkin GA'sidir. Burada her bootstrap tekrarinda AYNI vaka indeksleri iki kola birden
uygulanir (hasta duzeyi, `bootstrap.py` ile ayni yeniden ornekleme) ve delta = A - B'nin yuzdelik
araligi raporlanir. `covers_zero` True ise soylenebilecek tek sey: "fark gurultuden ayirt edilemedi".

Metrik tanimlari `bootstrap._DEFS` ile AYNI (nokta tahmini summary.json'la tutar) + `dice_mean`
(tumorlu vakalarda ortalama Dice; farki = vaka-bazli farklarin ortalamasi, eslestirmeye en duyarli).

Sessiz hata yok: vaka kumeleri ya da GT (gt_empty, n_gt_lesions, lezyon CAP KATMANLARI) iki kolda
farkliysa, ya da tumorlu bir vakada Dice yalniz bir kolda tanimliysa FIRLATIR -- "ayni zemin" degilse
karsilastirma yapilmaz. ~20 metrik birden raporlanir: `covers_zero=False` DUZELTILMEMIS'tir; iddia
yalniz onceden secilmis birincil metrikle kurulur. Medyan metriklerinde (dice_median, centroid) ayrik
degerler GA'yi sifir genislige cokertebilir -> birincil kanit `dice_mean` farkidir.

    python -m src.metrics.compare --a outputs/night/ft_none/eval --b outputs/night/ts_same_gt/strict \
        --name-a ft_none --name-b TS --out outputs/night/compare
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import bootstrap as B

_NAN = float("nan")


def _dice_mean(t, sel):
    x = t.dice[~t.gt_empty]
    x = x[np.isfinite(x)]
    return (float(x.mean()) if x.size else _NAN), int(x.size)


def _fn(name: str):
    return (lambda t: _dice_mean(t, None)) if name == "dice_mean" else B._metric_fn(name)


def _names(pl_a: pd.DataFrame, pl_b: pd.DataFrame) -> list[str]:
    na, nb = B._default_names(pl_a), B._default_names(pl_b)
    return ["dice_mean"] + [n for n in na if n in nb]


def _align(pc: pd.DataFrame, pl: pd.DataFrame):
    pc = pc.assign(case_id=pc["case_id"].astype(str)).sort_values("case_id").reset_index(drop=True)
    pl = pl.assign(case_id=pl["case_id"].astype(str)).reset_index(drop=True)
    B._validate(pc, pl)
    ids = pc["case_id"].to_numpy()
    groups = pl.groupby("case_id").indices
    return pc, pl, ids, [np.asarray(groups.get(c, ()), dtype=np.int64) for c in ids]


def _same_ground(pc_a, pl_a, pc_b, pl_b) -> None:
    """Ayni vakalar + ayni GT (bos/dolu, lezyon sayisi, CAP KATMANLARI) + Dice ayni vakalarda tanimli."""
    ia, ib = list(pc_a["case_id"]), list(pc_b["case_id"])
    if ia != ib:
        raise ValueError(f"kollar ayni vaka kumesinde degil: yalniz A {sorted(set(ia) - set(ib))[:5]}, "
                         f"yalniz B {sorted(set(ib) - set(ia))[:5]}")
    if len(ia) < 2:
        raise ValueError(f"eslestirilmis bootstrap icin en az 2 vaka gerekir: {len(ia)}")
    ids = pc_a["case_id"]
    ea, eb = B._bool(pc_a["gt_empty"]), B._bool(pc_b["gt_empty"])   # 'TRUE'/True/'false' ayni
    na = pd.to_numeric(pc_a["n_gt_lesions"]).to_numpy(float)       # 3 ile 3.0 ayni
    nb = pd.to_numeric(pc_b["n_gt_lesions"]).to_numpy(float)
    for col, bad in (("gt_empty", ids[ea != eb]), ("n_gt_lesions", ids[na != nb])):
        if len(bad):
            raise ValueError(f"{col} iki kolda farkli -> ayni GT degil: {list(bad[:5])}")

    def strata(pl):                       # vaka basina GT lezyonlarinin SIRALI katman listesi
        g = pl[pl["status"].isin(["TP", "FN"])]
        st = g["stratum"].astype(object).where(g["stratum"].notna(), "unknown").astype(str) \
            if "stratum" in g else pd.Series("unknown", index=g.index)
        return st.groupby(g["case_id"]).apply(lambda x: tuple(sorted(x))).to_dict()
    sa, sb = strata(pl_a), strata(pl_b)
    bad = [c for c in ia if sa.get(c, ()) != sb.get(c, ())]
    if bad:
        raise ValueError(f"GT lezyon cap katmanlari iki kolda farkli -> ayni GT degil (eski GT vs "
                         f"gt_fixed?): {bad[:5]}, orn. A {sa.get(bad[0])} B {sb.get(bad[0])}")
    da = np.isfinite(pd.to_numeric(pc_a["dice"], errors="coerce").to_numpy(float))
    db = np.isfinite(pd.to_numeric(pc_b["dice"], errors="coerce").to_numpy(float))
    bad = ids[(~ea) & (da != db)]
    if len(bad):
        raise ValueError(f"tumorlu vakada Dice yalniz BIR kolda tanimli -> vaka o koldan duser, "
                         f"eslestirme bozulur: {list(bad[:5])}")


def warnings_for(d: pd.DataFrame, min_den: int = 10) -> list[str]:
    """Rapora girmeden once okunmasi gerekenler (bootstrap.main ile ayni esikler + kural 8)."""
    w = []
    small = d[(d["a_n"] < min_den) | (d["b_n"] < min_den)]["metric"].tolist()
    if small:
        w.append(f"paydasi <{min_den} (aralik guvenilmez, sayiyla birlikte raporla): {small}")
    part = d[d["n_valid"] < d["n_boot"]]["metric"].tolist()
    if part:
        w.append(f"bazi tekrarlarda tanimsiz -> aralik 'iki kolda da tanimli' kosullu: {part}")
    n, k = len(d), int((~d["covers_zero"]).sum())
    w.append(f"{n} metrik, {k} tanesinde fark GA'si sifiri disliyor -- duzeltilmemis (coklu karsilastirma): "
             f"fark yokken de ~{0.05 * n:.1f} yanlis pozitif beklenir; katmanli/katmansiz metrikler ayni olayi "
             "iki kez sayar. Iddia yalniz ONCEDEN secilmis birincil metrik uzerinden kurulur (kural 8).")
    return w


def paired_diff(pc_a, pl_a, pc_b, pl_b, names: list[str] | None = None, n_boot: int = 2000,
                seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """delta = A - B; her tekrarda iki kola AYNI vaka orneklemi."""
    if n_boot < 1:
        raise ValueError(f"n_boot >= 1 olmali: {n_boot}")
    pc_a, pl_a, ids, la = _align(pc_a, pl_a)
    pc_b, pl_b, _, lb = _align(pc_b, pl_b)
    _same_ground(pc_a, pl_a, pc_b, pl_b)
    names = names or _names(pl_a, pl_b)
    ta, tb = B._Tables(pc_a, pl_a), B._Tables(pc_b, pl_b)
    fns = {n: _fn(n) for n in names}
    n = len(ids)
    rng = np.random.default_rng(seed)
    reps = {nm: np.empty(n_boot) for nm in names}
    empty = np.zeros(0, np.int64)
    for b in range(n_boot):
        ic = rng.integers(0, n, n)
        sa = ta.take(ic, np.concatenate([la[i] for i in ic]) if n else empty)
        sb = tb.take(ic, np.concatenate([lb[i] for i in ic]) if n else empty)
        for nm, f in fns.items():
            reps[nm][b] = f(sa)[0] - f(sb)[0]
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    rows = []
    for nm, f in fns.items():
        (va, da), (vb, db) = f(ta), f(tb)
        r = reps[nm][np.isfinite(reps[nm])]
        lo, hi = (float(v) for v in np.percentile(r, q)) if r.size else (_NAN, _NAN)
        rows.append({"metric": nm, "a": va, "b": vb, "delta": va - vb, "lo": lo, "hi": hi,
                     "covers_zero": bool(not r.size or lo <= 0.0 <= hi),
                     "a_n": da, "b_n": db, "n_valid": int(r.size), "n_boot": n_boot, "n_cases": n})
    return pd.DataFrame(rows)


def side_by_side(arms: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
                 names: list[str] | None = None) -> pd.DataFrame:
    """Kol basina nokta tahmini + payda (n). Eslestirme gerektirmez; tablo icin."""
    out = None
    for arm, (pc, pl) in arms.items():
        pc, pl, _, _ = _align(pc, pl)
        t = B._Tables(pc, pl)
        nm = names or (["dice_mean"] + B._default_names(pl))
        vals = [_fn(n)(t) for n in nm]
        df = pd.DataFrame({"metric": nm, arm: [v for v, _ in vals], f"{arm}_n": [d for _, d in vals]})
        out = df if out is None else out.merge(df, on="metric", how="outer", sort=False)
    return out


def _load(ev: Path):
    return pd.read_csv(ev / "per_case.csv"), pd.read_csv(ev / "per_lesion.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="evaluate cikti klasoru (A kolu)")
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    A, Bb = _load(a.a), _load(a.b)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.name_a}_vs_{a.name_b}"
    d = paired_diff(*A, *Bb, n_boot=a.n_boot, seed=a.seed)
    d.insert(0, "b_name", a.name_b)
    d.insert(0, "a_name", a.name_a)
    d.to_csv(a.out / f"paired_{tag}.csv", index=False)
    side_by_side({a.name_a: A, a.name_b: Bb}).to_csv(a.out / f"table_{tag}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 99):
        print(d[["metric", "a", "b", "delta", "lo", "hi", "covers_zero", "a_n", "b_n"]].round(3).to_string(index=False))
    for w in warnings_for(d):
        print(f"[UYARI] {w}", flush=True)
    print(f"[compare] -> {a.out}/paired_{tag}.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
