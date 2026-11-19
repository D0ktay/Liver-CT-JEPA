"""Onbellek KALITE KONTROLU: egitime girecek HER hacim olculur, bozuklar ayrilir, ornekler cizilir.

    python -m src.data.qc_cache --cache data/cache/lits data/cache/task08 ... --out outputs/night/qc

Neden: tek seferlik gece kosusunda bozuk bir kaynak (yanlis HU ofseti, ters etiket haritasi,
bos maske, yanlis spacing) egitimi SESSIZCE zehirler ve sabah fark edilir. Bu adim her .npz icin
olcer, `qc.csv` yazar, kurallari cigneyenleri `<cache>/_rejected/` altina TASIR (silmez) ve
kaynak basina rastgele vakalarin karaciger/tumor konturlu kesitlerini PNG'ye cizer (gozle kontrol).

KURALLAR (uint8 onbellek: 0 = -200 HU, 255 = +250 HU; 1 mm izotropik):
  R1 sekil: her eksen >= 32 voksel, 3B, uint8
  R2 govde: 'hava olmayan' (>40) voksel orani >= 0.03
  R3 HU olcegi: govde vokselleri medyani [60, 215] (~ -95..180 HU). Ofset hatasi -> 0 ya da 255'e yapisir
  R4 doygunluk: 255'e yapisan oran < 0.5
  R5 spacing: hedeften bagil sapma <= %0.5 (pretrain de dusurur; burada RAPORLANIR, reddedilmez)
  R6 karaciger (varsa): hacim [300, 6000] cm^3, yogunluk medyani [110, 235] (~ -5..215 HU)
  R7 tumor (varsa): karaciger bolgesinin ICINDE (bolge = karaciger ∪ tumor, kurulusu geregi) ve
     tumor hacmi karaciger bolgesinin < %95'i
  R8 etiket degerleri yalniz {0, 1}
Etiketli LiTS'te TEK red bile kosuyu DURDURUR (`--fatal-dirs`); diger kaynaklarda red orani
`--max-reject` (varsayilan %5) asilirsa durur -- kaynagin tamami supheli demektir.

YON: RAS dizisinde x buyudukce hasta SAGINA gidilir; karaciger sagdadir -> karaciger agirlik
merkezi govde merkezinin SAGINDA olmali. LiTS'te affine'i ters vakalar BILINIYOR; bu yuzden
reddedilmez, SAYILIR (ayna augmentasyonu zaten var). Bir kaynagin cogunlugu tersse UYARIR.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

AIR_U8 = 40


def measure(path: str) -> dict:
    p = Path(path)
    r: dict = {"id": p.stem, "dir": p.parent.name, "fail": []}
    try:
        z = np.load(p, allow_pickle=False)
        img = z["img"]
        meta = json.loads(str(z["meta"]))
    except Exception as e:                                   # noqa: BLE001
        r["fail"] = [f"R0 okunamadi: {type(e).__name__}: {e}"]
        return r
    r.update(shape="x".join(map(str, img.shape)), dtype=str(img.dtype),
             spacing=[round(float(s), 5) for s in meta.get("spacing", [])],
             has_liver="liver" in z, has_tumor="lab" in z, hu_shift=meta.get("hu_shift", 0.0))
    if img.ndim != 3 or img.dtype != np.uint8 or min(img.shape) < 32:
        r["fail"].append(f"R1 sekil/dtype {img.shape} {img.dtype}")
        return r
    sub = img[::2, ::2, ::2]
    body = sub > AIR_U8
    r["body_frac"] = float(body.mean())
    if r["body_frac"] < 0.03:                      # ust sinir YOK: kirpilmis FOV tamamen govde olabilir
        r["fail"].append(f"R2 govde orani {r['body_frac']:.3f}")
    r["body_median"] = float(np.median(sub[body])) if body.any() else 0.0
    if not (60 <= r["body_median"] <= 215):
        r["fail"].append(f"R3 govde medyani {r['body_median']:.0f} (HU olcegi/ofset?)")
    r["sat_frac"] = float((sub == 255).mean())
    if r["sat_frac"] >= 0.5:
        r["fail"].append(f"R4 doygunluk {r['sat_frac']:.2f}")
    tg = meta.get("target", [1.0, 1.0, 1.0])
    r["spacing_dev"] = float(max(abs(float(a) - float(b)) / float(b)
                                 for a, b in zip(meta.get("spacing", tg), tg)))
    for key in ("liver", "lab"):
        if key in z:
            a = z[key]
            if a.shape != img.shape:
                r["fail"].append(f"R1 {key} sekli {a.shape} != {img.shape}")
                return r
            if not set(np.unique(a[::2, ::2, ::2]).tolist()) <= {0, 1}:
                r["fail"].append(f"R8 {key} degerleri {np.unique(a)[:5]}")
    if "liver" in z:
        liv = z["liver"] > 0
        vox_cm3 = float(np.prod(meta.get("spacing", [1, 1, 1]))) / 1000.0
        r["liver_cm3"] = float(liv.sum()) * vox_cm3
        if not (300 <= r["liver_cm3"] <= 6000):
            r["fail"].append(f"R6 karaciger hacmi {r['liver_cm3']:.0f} cm3")
        if liv.any():
            r["liver_median"] = float(np.median(img[liv]))
            if not (110 <= r["liver_median"] <= 235):
                r["fail"].append(f"R6 karaciger yogunlugu {r['liver_median']:.0f}")
            bx = np.flatnonzero((img > AIR_U8).any(axis=(1, 2)))
            cx = float(np.average(np.arange(img.shape[0]), weights=liv.sum(axis=(1, 2))))
            r["liver_on_right"] = bool(cx > (bx[0] + bx[-1]) / 2.0) if len(bx) else None
    if "lab" in z:
        tum = z["lab"] > 0
        r["tumor_cm3"] = float(tum.sum()) * float(np.prod(meta.get("spacing", [1, 1, 1]))) / 1000.0
        if "liver" in z:
            liv = z["liver"] > 0
            r["tumor_outside_region"] = int((tum & ~liv).sum())
            if r["tumor_outside_region"]:
                r["fail"].append(f"R7 {r['tumor_outside_region']} tumor vokseli bolge DISINDA")
            if liv.any() and tum.sum() > 0.95 * liv.sum():
                r["fail"].append("R7 tumor ~ karacigerin tamami (etiket haritasi ters?)")
    return r


def montage(paths: list[Path], out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(paths)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6), squeeze=False)
    for j, p in enumerate(paths):
        z = np.load(p, allow_pickle=False)
        img = z["img"]
        liv = z["liver"] if "liver" in z else None
        tum = z["lab"] if "lab" in z else None
        ref = tum if (tum is not None and tum.any()) else (liv if liv is not None and liv.any() else None)
        k = int(np.argmax(ref.sum(axis=(0, 1)))) if ref is not None else img.shape[2] // 2
        i = int(np.argmax(ref.sum(axis=(1, 2)))) if ref is not None else img.shape[0] // 2
        for row, (sl, name) in enumerate(((np.s_[:, :, k], f"aksiyel z={k}"), (np.s_[i, :, :], f"sagital x={i}"))):
            ax = axes[row][j]
            ax.imshow(img[sl].T, cmap="gray", origin="lower", vmin=0, vmax=255)
            if liv is not None and liv[sl].any():
                ax.contour(liv[sl].T, [0.5], colors="lime", linewidths=0.8)
            if tum is not None and tum[sl].any():
                ax.contour(tum[sl].T, [0.5], colors="red", linewidths=0.8)
            ax.set_title(f"{p.stem}\n{name} {img.shape}", fontsize=7)
            ax.axis("off")
    fig.suptitle(title + "  (yesil = karaciger bolgesi, kirmizi = tumor; sol = hasta SOLU)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=70)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fatal-dirs", nargs="*", default=["lits"],
                    help="bu dizinlerde TEK red bile kosuyu durdurur")
    ap.add_argument("--max-reject", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-show", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    paths = [p for d in a.cache if d.exists() for p in sorted(d.glob("*.npz"))]
    if not paths:
        raise SystemExit("[qc] hic .npz yok")
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(measure, map(str, paths), chunksize=4))
    keys = ["id", "dir", "shape", "dtype", "spacing", "spacing_dev", "hu_shift", "body_frac",
            "body_median", "sat_frac", "has_liver", "has_tumor", "liver_cm3", "liver_median",
            "liver_on_right", "tumor_cm3", "tumor_outside_region", "fail"]
    with (a.out / "qc.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "fail": " | ".join(r["fail"])})
    rng = np.random.default_rng(a.seed)
    summary, stop = {}, []
    for d in a.cache:
        rs = [r for r in rows if r["dir"] == d.name]
        if not rs:
            continue
        bad = [r for r in rs if r["fail"]]
        lr = [r["liver_on_right"] for r in rs if r.get("liver_on_right") is not None]
        s = {"n": len(rs), "rejected": len(bad),
             "spacing_dev_gt_0.5pct": sum(1 for r in rs if r.get("spacing_dev", 0) > 0.005),
             "hu_shifted": sum(1 for r in rs if r.get("hu_shift")),
             "body_median_p5_p50_p95": [round(float(x)) for x in np.percentile(
                 [r["body_median"] for r in rs if "body_median" in r] or [0], [5, 50, 95])],
             "liver_cm3_p5_p50_p95": [round(float(x)) for x in np.percentile(
                 [r["liver_cm3"] for r in rs if "liver_cm3" in r] or [0], [5, 50, 95])],
             "liver_on_right_frac": round(float(np.mean(lr)), 3) if lr else None,
             "n_with_tumor": sum(1 for r in rs if r.get("tumor_cm3", 0) > 0),
             "examples": [f"{r['id']}: {r['fail']}" for r in bad[:8]]}
        summary[d.name] = s
        print(f"[qc] {d.name}: {json.dumps(s, ensure_ascii=False)}", flush=True)
        if s["liver_on_right_frac"] is not None and s["liver_on_right_frac"] < 0.5:
            print(f"[UYARI] {d.name}: karaciger vakalarin cogunda SOLDA gorunuyor -> yon donusumu supheli",
                  flush=True)
        if bad:
            rej = d / "_rejected"
            rej.mkdir(exist_ok=True)
            for r in bad:
                shutil.move(str(d / f"{r['id']}.npz"), str(rej / f"{r['id']}.npz"))
        if d.name in a.fatal_dirs and bad:
            stop.append(f"{d.name}: {len(bad)} red (bu dizinde sifir tolerans)")
        elif len(bad) > a.max_reject * len(rs):
            stop.append(f"{d.name}: red orani {len(bad)}/{len(rs)} > %{a.max_reject * 100:g}")
        good = [d / f"{r['id']}.npz" for r in rs if not r["fail"]]
        if good:
            pick = [good[i] for i in rng.choice(len(good), min(a.n_show, len(good)), replace=False)]
            try:
                montage(pick, a.out / f"montaj_{d.name}.png", f"{d.name} ({len(good)} hacim)")
            except Exception as e:                           # noqa: BLE001 -- cizim QC'yi dusurmesin
                print(f"[UYARI] montaj cizilemedi ({d.name}): {e}", flush=True)
    (a.out / "qc_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    if stop:
        raise SystemExit("[qc] DUR: " + "; ".join(stop) + f" -> {a.out / 'qc.csv'}")
    print(f"[qc] ✅ {len(rows)} hacim olculdu -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
