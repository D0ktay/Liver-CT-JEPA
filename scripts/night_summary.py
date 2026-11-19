"""Gece kosusu kol ozeti: dogrulama egrisi + dondurulmus test (27) + boyut kirilimi + karaciger-disi pay.

    python scripts/night_summary.py --arm outputs/night/ft_none [--ts outputs/night/ts_same_gt/strict] --out docs/x.md

Yalnizca OKUR. Boyut katmanlari CLAUDE.md'deki (10, 20 mm) ile ayni. Kural 8: kollar arasi kucuk fark
IDDIA EDILMEZ -- bu betik fark cumlesi kurmaz, sayilari yan yana koyar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd



def size_recall(per_lesion: pd.DataFrame) -> list[tuple[str, int, int]]:
    """evaluate'in yazdigi `stratum` sutunu (config size_strata_mm) kullanilir -- katman burada
    yeniden tanimlanmaz. Bulunan = status TP (IoU>=0.5 eslesme), ortusme-yakalama DEGIL."""
    gt = per_lesion[per_lesion["status"].isin(["TP", "FN"])]
    if "stratum" not in gt.columns:
        raise ValueError("per_lesion.csv'de 'stratum' sutunu yok -- evaluate ciktisi degil")
    out = []
    for name in list(dict.fromkeys(gt["stratum"].astype(str))):
        s = gt[gt["stratum"].astype(str) == name]
        out.append((name, int((s["status"] == "TP").sum()), int(len(s))))
    return out


def arm_block(arm: Path, label: str) -> list[str]:
    lines = [f"## {label} (`{arm}`)", ""]
    vp = arm / "val.jsonl"
    if vp.exists():
        rows = [json.loads(l) for l in vp.read_text().splitlines() if l.strip()]
        lines += ["Doğrulama (8 eğitim vakası, yalnız tanı; teste bakılmadı):", "",
                  "| adım | tümör Dice | ham | karaciğer (ham) | karaciğer-dışı pay |", "|---|---|---|---|---|"]
        lines += [f"| {r['step']} | {r['tumor_dice']:.3f} | {r['tumor_dice_raw']:.3f} | {r['liver_dice']:.3f} | {r['outside_liver_frac']:.3f} |"
                  for r in rows]
        lines.append("")
    dn = arm / "done.json"
    if dn.exists():
        d = json.loads(dn.read_text())
        lines.append(f"Adım {d.get('steps_done', d.get('steps'))}, kontrol noktası **{d.get('checkpoint_used')}**, "
                     f"init_key {d.get('init_key', 'encoder')}, örnekleme {d.get('sampler_counts')}")
        lines.append("")
    sp = arm / "eval" / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text()); v, le, to = s.get("voxel", {}), s.get("lesion", {}), s.get("targeting", {}).get("overall", {})
        lines += [f"**Dondurulmuş iç test, n={s.get('n_cases')}** (başarısız {s.get('n_failed', 0)}):", "",
                  f"- Dice medyan {v.get('dice_median'):.3f} · ortalama {v.get('dice_mean'):.3f} (tümörsüz vaka Dice'a katılmadı, n_nan={v.get('n_dice_nan')})",
                  f"- Marj ihlali yalnız BULUNAN lezyonlarda {to.get('margin_violation_rate_detected'):.3f} (sınır kalitesi; 'tümü' kaçırılanı da ihlal sayar)",
                  f"- Lezyon recall {le.get('recall'):.3f} · precision {le.get('precision'):.3f} · F1 {le.get('f1'):.3f} (TP {le.get('tp')} FP {le.get('fp')} FN {le.get('fn')})",
                  f"- Hedefleme: örtüşme yakalama {to.get('overlap_hit_rate'):.3f} · marj ihlali (tümü) {to.get('margin_violation_rate_all'):.3f} · "
                  f"merkez hatası medyan {to.get('centroid_error_mm_median'):.2f} mm (p90 {to.get('centroid_error_mm_p90'):.2f})",
                  f"- Tümörsüz vakada FP oranı {s.get('tumor_free', {}).get('fp_case_rate')}", ""]
    pl = arm / "eval" / "per_lesion.csv"
    if pl.exists():
        lines += ["Boyuta göre lezyon recall (IoU≥0.5 eşleşme):", "", "| katman | bulunan/toplam | recall |", "|---|---|---|"]
        for name, tp, n in size_recall(pd.read_csv(pl)):
            lines.append(f"| {name} | {tp}/{n} | {tp / n:.3f} |" if n else f"| {name} | 0/0 | - |")
        lines.append("")
    pi = arm / "pred_info.json"
    if pi.exists():
        info = json.loads(pi.read_text())
        of = [x["outside_liver_frac"] for x in info.values() if x["tumor_vox_raw"] > 0]
        nol = sum(1 for x in info.values() if x["liver_vox"] == 0)
        lines.append(f"Son işleme (test): ham tümör vokselinin karaciğer maskesi dışında kalan payı medyan "
                     f"{np.median(of):.3f} (p90 {np.percentile(of, 90):.3f}); karaciğer bulunamayan vaka {nol}")
        lines.append("")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", type=Path, nargs="+", required=True)
    ap.add_argument("--ts", type=Path, default=None, help="TS strict eval klasoru (yalniz yan yana; son isleme YOK)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    lines = ["# Gece koşusu özeti", ""]
    for arm in a.arm:
        lines += arm_block(arm, arm.name)
    if a.ts and a.ts.exists():
        lines += ["## TotalSegmentator `liver_lesions` (aynı vakalar, aynı GT; son işleme YOK)", ""]
        sj = a.ts / "summary.json"
        if sj.exists():
            s = json.loads(sj.read_text())
            lines.append(f"- n={s.get('n_cases')} · Dice medyan {s['voxel']['dice_median']:.3f} · recall {s['lesion']['recall']:.3f} · precision {s['lesion']['precision']:.3f}")
        pl = a.ts / "per_lesion.csv"
        if pl.exists():
            lines += ["", "Boyuta göre lezyon recall (IoU≥0.5):", "", "| katman | bulunan/toplam | recall |", "|---|---|---|"]
            for name, tp, n in size_recall(pd.read_csv(pl)):
                lines.append(f"| {name} | {tp}/{n} | {tp / n:.3f} |" if n else f"| {name} | 0/0 | - |")
        lines.append("")
    lines += ["⚠️ 27 vaka: güven aralıkları geniş; kollar arası küçük fark iddia edilmez (kural 8). "
              "Eşleştirilmiş fark GA'sı için `src/metrics/compare.py`."]
    txt = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
