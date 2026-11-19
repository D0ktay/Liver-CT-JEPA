"""GT (yesil) ve tahmin (kirmizi) konturlarini CT kesiti uzerinde gosterir.

Vaka basina tek PNG, en fazla 4 panel -- her panel farkli bir soruya cevap:
  EN BUYUK      GT tumor alani en buyuk kesit (buyuk tumorde sinir uyumu)
  EN KUCUK      en kucuk GT lezyonunun merkez kesiti (kucuk lezyon uc)
  KACIRILAN     tahminle hic ortusmeyen en buyuk GT lezyonu (FN)
  UYDURULAN     GT ile hic ortusmeyen en buyuk tahmin bileseni (FP)
Tespit olcutu = herhangi bir ortusme (hedefleme ozetindeki overlap_hit ile ayni).

Yon: RAS'a cevrilir, radyolojik gosterim (hastanin sagi ekranin solunda, onu yukarida).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.case_id import assert_unique_ids, glob_paths
from src.metrics.lesionwise import connected_components, equivalent_diameter_mm


def radiological(slice_xy: np.ndarray) -> np.ndarray:
    """RAS dizisinden alinmis (x, y) kesit -> ekran: satir = on->arka, sutun = sag->sol."""
    return np.asarray(slice_xy).T[::-1, ::-1]


def lesion_table(gt: np.ndarray, pred: np.ndarray, spacing) -> tuple[list[dict], list[dict]]:
    """GT ve tahmin bilesenleri: cap (mm), merkez z, diger maskeyle ortusme var mi."""
    out = []
    for mask, other in ((gt, pred), (pred, gt)):
        lab, n = connected_components(mask)
        rows = []
        for i in range(1, n + 1):
            comp = lab == i
            zs = np.nonzero(comp.any(axis=(0, 1)))[0]
            x, y, z = (int(round(v.mean())) for v in np.nonzero(comp))
            rows.append({"id": i, "diam_mm": equivalent_diameter_mm(int(comp.sum()), spacing),
                         "z": z, "xy": (x, y), "z_range": (int(zs[0]), int(zs[-1])),
                         "hit": bool((comp & (other > 0)).any())})
        out.append(rows)
    return out[0], out[1]


def pick_slices(gt: np.ndarray, pred: np.ndarray, spacing) -> list[tuple[str, int, tuple | None]]:
    """(baslik, z, odak_xy) listesi. odak_xy None -> tam kesit, degilse o lezyona yakinlastir.
    Ayni kesite dusen basliklar birlesir."""
    g, p = lesion_table(gt, pred, spacing)
    picks = []
    if gt.any():
        picks.append(("EN BUYUK tumor kesiti", int(np.argmax(gt.sum(axis=(0, 1)))), None))
        s = min(g, key=lambda r: r["diam_mm"])
        picks.append((f"EN KUCUK lezyon ({s['diam_mm']:.1f} mm)", s["z"], s["xy"]))
    if miss := [r for r in g if not r["hit"]]:
        m = max(miss, key=lambda r: r["diam_mm"])
        picks.append((f"KACIRILAN ({m['diam_mm']:.1f} mm, {len(miss)}/{len(g)} lezyon)", m["z"], m["xy"]))
    if fp := [r for r in p if not r["hit"]]:
        f = max(fp, key=lambda r: r["diam_mm"])
        picks.append((f"UYDURULAN ({f['diam_mm']:.1f} mm, {len(fp)} bilesen)", f["z"], f["xy"]))
    # ayni kesite dusen basliklar birlesir (orn. en kucuk lezyon = kacirilan lezyon)
    merged: dict[int, list] = {}
    for t, z, xy in picks:
        prev = merged.setdefault(z, [[], None])
        prev[0].append(t)
        prev[1] = prev[1] or xy
    return [(" + ".join(ts), z, xy) for z, (ts, xy) in merged.items()]


def plan_mask(gt: np.ndarray, pred: np.ndarray, spacing, margin_mm: float) -> np.ndarray:
    """Hedefleme metrigindeki PLAN ile ayni tanim (kural 9): yalniz GT ile kesisen tahmin
    bilesenleri, mm marjla EDT genisletmesi (kural 7). Salt-FP blob plana GIRMEZ."""
    from src.metrics.targeting import _dilate_mm
    lab, n = connected_components(pred)
    if n == 0:
        return np.zeros(pred.shape, bool)
    hit = np.unique(lab[(lab > 0) & (gt > 0)])
    base = np.isin(lab, hit)
    if not base.any():
        return base
    return _dilate_mm(base, spacing, float(margin_mm))


def crop_box(xy, shape, spacing, half_mm: float = 60.0):
    """Lezyon merkezinin cevresinde (x0, x1, y0, y1); dizi sinirina kirpilir."""
    hx, hy = (int(round(half_mm / float(s))) for s in spacing[:2])
    x, y = xy
    return max(0, x - hx), min(shape[0], x + hx), max(0, y - hy), min(shape[1], y + hy)


def render_case(ct_path, gt_path, pred_path, out_png: Path, case_id: str = "",
                gt_label="any", pred_label=1, hu_window=(-200, 250), pred_name: str = "TotalSegmentator",
                raw_panel: bool = False, ct_header_from_gt: bool = False, margin_mm: float = 0.0) -> Path:
    """raw_panel: her konturlu panelin SOLUNA ayni kesitin kontursuz (ham CT) hali eklenir -- goz
    kontur cizgisine degil dokuya baksin. ct_header_from_gt: CT basligi bozuksa (LiTS 28-47 sahte
    spacing) geometri GT dosyasindan alinir; sekil tutmazsa FIRLATIR. margin_mm > 0: tahminin
    mm genisletilmisi (ablasyon plani) sari kesikli kontur olarak eklenir (EDT ile, kural 7).
    Hicbir piksel uydurulmaz: uc dosya da diskten okunur, yalniz kontur cizilir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    raw = [nib.load(str(p)) for p in (ct_path, gt_path, pred_path)]
    if ct_header_from_gt:
        if raw[0].shape[:3] != raw[1].shape[:3]:
            raise ValueError(f"{case_id}: CT {raw[0].shape} != GT {raw[1].shape}; baslik GT'den alinamaz")
        raw[0] = nib.Nifti1Image(np.asarray(raw[0].dataobj), raw[1].affine, raw[1].header)
    imgs = [nib.as_closest_canonical(im) for im in raw]
    shapes = {im.shape[:3] for im in imgs}
    if len(shapes) != 1 or not all(np.allclose(imgs[0].affine, im.affine, atol=1e-3) for im in imgs):
        raise ValueError(f"{case_id}: CT/GT/tahmin geometrisi uyusmuyor: {shapes}")
    ct, gt, pr = (np.asarray(im.dataobj) for im in imgs)
    gt = (gt > 0) if gt_label == "any" else (gt == int(gt_label))
    pr = (pr > 0) if pred_label == "any" else (pr == int(pred_label))
    spacing = tuple(float(v) for v in np.sqrt((imgs[0].affine[:3, :3] ** 2).sum(0)))
    pix = tuple(float(v) for v in raw[0].header.get_zooms()[:3])
    if not np.allclose(sorted(pix), sorted(spacing), rtol=1e-3):
        raise ValueError(f"{case_id}: CT pixdim {pix} ile affine spacing'i {spacing} celisiyor -- "
                         "basligi duzelt (gt_fixed) ya da --ct-header-from-gt kullan")
    if ct_header_from_gt and np.allclose(spacing, (1.0, 1.0, 1.0)):
        raise ValueError(f"{case_id}: GT basligi da tam 1x1x1 -- sahte spacing; mm konturu anlamsiz olur")

    picks = pick_slices(gt, pr, spacing)
    if not picks:  # tumorsuz ve tahminsiz: orta kesit
        picks = [("tumor yok, tahmin yok", ct.shape[2] // 2, None)]
    lo, hi = hu_window
    plan = None
    if margin_mm > 0 and pr.any():
        plan = plan_mask(gt, pr, spacing, float(margin_mm))
    k = 2 if raw_panel else 1
    fig, axes = plt.subplots(1, k * len(picks), figsize=(4.2 * k * len(picks), 5.0), squeeze=False)
    for j, (title, z, xy) in enumerate(picks):
        x0, x1, y0, y1 = crop_box(xy, ct.shape, spacing) if xy else (0, ct.shape[0], 0, ct.shape[1])
        view = lambda a: radiological(a[x0:x1, y0:y1, z])
        panels = [(axes[0][k * j], True)] if not raw_panel else [(axes[0][2 * j], False), (axes[0][2 * j + 1], True)]
        for ax, with_contours in panels:
            ax.imshow(view(np.clip(ct, lo, hi)), cmap="gray", vmin=lo, vmax=hi,
                      aspect=spacing[1] / spacing[0])
            if with_contours:
                for m, color, ls in ((plan, "#ffdd00", "--"), (gt, "#22dd55", "-"), (pr, "#ff3344", "-")):
                    if m is None:
                        continue
                    s = view(m)
                    if s.any():
                        ax.contour(s, levels=[0.5], colors=color, linewidths=1.2, linestyles=ls)
            head = "HAM CT (kontursuz)" if not with_contours else title.replace(" + ", "\n+ ")
            ax.set_title(head + f"\nz={z}" + ("  (12 cm yakin)" if xy else ""), fontsize=9)
            ax.axis("off")
    legend = f"{case_id}   yesil = gercek (radyolog)   kirmizi = {pred_name}"
    if plan is not None:
        legend += f"   sari kesikli = plan (GT ile kesisen tahmin + {margin_mm:g} mm marj)"
    fig.suptitle(legend, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct-glob", required=True)
    ap.add_argument("--gt-glob", required=True)
    ap.add_argument("--pred-glob", required=True)
    ap.add_argument("--gt-label", default="any")
    ap.add_argument("--pred-label", default="1")
    ap.add_argument("--out", type=Path, default=Path("outputs/figures"))
    ap.add_argument("--pred-name", default="TotalSegmentator", help="baslikta kirmizi konturun adi")
    ap.add_argument("--raw-panel", action="store_true", help="her panelin yanina kontursuz ham CT")
    ap.add_argument("--ct-header-from-gt", action="store_true",
                    help="CT basligi bozuksa (LiTS 28-47) geometriyi GT dosyasindan al")
    ap.add_argument("--margin-mm", type=float, default=0.0, help=">0: tahmin + marj (plan) konturu")
    a = ap.parse_args()
    ct, gt, pr = (assert_unique_ids(glob_paths(g)) for g in (a.ct_glob, a.gt_glob, a.pred_glob))
    common = sorted(ct.keys() & gt.keys() & pr.keys())
    if not common:
        raise SystemExit("CT/GT/tahmin icin ortak vaka kimligi yok -- glob'lari kontrol et.")
    for cid in common:
        p = render_case(ct[cid], gt[cid], pr[cid], a.out / f"{cid}.png", cid, a.gt_label, a.pred_label,
                        pred_name=a.pred_name, raw_panel=a.raw_panel,
                        ct_header_from_gt=a.ct_header_from_gt, margin_mm=a.margin_mm)
        print("[viz]", p)
