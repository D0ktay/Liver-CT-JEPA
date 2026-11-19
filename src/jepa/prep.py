"""JEPA veri hazirligi: LiTS hacmi -> Rad-JEPA pencereleri -> (tahmin) -> orijinal grid.

Rad-JEPA girdisi (1, 32, 256, 256) = (z, y, x). Tum karacigeri tek bloga sikistirmak z'de ~6 mm/kesit
verirdi; bunun yerine karaciger kutusu TARGET_SPACING'e (0.8 x 0.8 x 2.5 mm) orneklenip ortusen
pencerelere bolunur (pencere ~ 80 x 205 x 205 mm, token ~ 20 x 13 x 13 mm).

Geri donus (reconstruct) hazirligin tam tersi: pencere olasiliklari ortusmede ORTALANIR, karaciger
kutusunun disi 0. Boylece JEPA tahmini TS ile AYNI degerlendirme koduyla (src.metrics.evaluate) olculur.
Karaciger kutusu GT karaciger etiketinden (1) alinir -> bu bir "karaciger ROI" varsayimidir, raporda yazilir.

Normalizasyon: HU [-160, 240] (karin penceresi) kirpilip [0, 1]. Model karti hacim basina min-max diyor;
M3D-Cap goruntuleri zaten pencerelenmis goruntulerden geldigi icin sabit karin penceresi secildi (varsayim).
Yon (20 Eyl 2026): LiTS'te 3 yon var (LAS 33, RAS 35, LPS 63 vaka). Her hacim once RAS'a cevrilir,
pencere radyolojik gorunumde: (z, satir = on->arka, sutun = hastanin sagi->solu). Boylece ayni anatomi
her hastada ayni yonde ve Rad-JEPA on egitimindeki (Radiopaedia) goruntu duzenine yakin (varsayim).
reconstruct bunlarin hepsini tersine cevirip ORIJINAL dizi yonune doner.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

TARGET_SPACING = (0.8, 0.8, 2.5)      # (x, y, z) mm
WIN = 32                              # z kesit
XY = 256
STRIDE = (192, 192, 24)               # (x, y, z) pencere adimi
HU_WINDOW = (-160.0, 240.0)


def z_windows(n: int, win: int = WIN, stride: int = 24) -> list[int]:
    """Ekseni [0, max(n, win)) araligini kapsayan pencere baslangiclari (son pencere sona dayali)."""
    if n <= win:
        return [0]
    starts = list(range(0, n - win, stride))
    return starts + [n - win]


def liver_bbox(seg: np.ndarray, spacing, margin_mm: float = 10.0) -> tuple[slice, ...]:
    idx = np.argwhere(np.asarray(seg) > 0)
    if idx.size == 0:
        raise ValueError("karaciger/tumor etiketi yok -> kutu tanimsiz")
    pad = [math.ceil(margin_mm / float(s)) for s in spacing]
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + 1 + np.asarray(pad), seg.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def normalize(ct: np.ndarray, window=HU_WINDOW) -> np.ndarray:
    lo, hi = window
    return ((np.clip(ct.astype(np.float32), lo, hi) - lo) / (hi - lo)).astype(np.float32)


def _ras_ornt(affine) -> np.ndarray:
    from nibabel.orientations import io_orientation
    return io_orientation(np.asarray(affine, float))


def _inverse_ornt(ornt: np.ndarray) -> np.ndarray:
    """apply_orientation(apply_orientation(a, o), inv) == a."""
    inv = np.zeros_like(ornt)
    for i, (t, f) in enumerate(ornt):
        inv[int(t)] = [i, f]
    return inv


def _resize(arr: np.ndarray, shape, order: int) -> np.ndarray:
    if tuple(arr.shape) == tuple(shape):
        return arr
    f = [t / s for t, s in zip(shape, arr.shape)]
    out = ndimage.zoom(arr, f, order=order, mode="nearest", grid_mode=True)
    assert out.shape == tuple(shape), (out.shape, shape)
    return out


def prepare_case(ct: np.ndarray, seg: np.ndarray, spacing, target=TARGET_SPACING,
                 margin_mm: float = 10.0, tumor_label: int = 2, affine=None) -> dict:
    """-> {"img": (n, 32, 256, 256) uint8, "lab": ayni sekil uint8 (tumor), "meta": geri donus bilgisi}.
    affine: hacmin affine'i (yon icin; None -> dizi zaten RAS varsayilir)."""
    from nibabel.orientations import apply_orientation
    if ct.shape != seg.shape:
        raise ValueError(f"CT {ct.shape} != seg {seg.shape}")
    orig_shape = list(ct.shape)
    ornt = _ras_ornt(np.eye(4) if affine is None else affine)
    ct, seg = apply_orientation(ct, ornt), apply_orientation(seg, ornt)
    sp = [0.0, 0.0, 0.0]
    for i, (t, _) in enumerate(ornt):
        sp[int(t)] = float(spacing[i])
    spacing = tuple(sp)
    bb = liver_bbox(seg, spacing, margin_mm)
    ct_c, seg_c = ct[bb], seg[bb]
    res_shape = tuple(max(1, int(round(n * s / t))) for n, s, t in zip(ct_c.shape, spacing, target))
    img = _resize(normalize(ct_c), res_shape, order=1)
    lab = _resize((seg_c == tumor_label).astype(np.uint8), res_shape, order=0)
    # RAS (x, y, z) -> radyolojik (z, on->arka, sag->sol)
    img, lab = img.transpose(2, 1, 0)[:, ::-1, ::-1], lab.transpose(2, 1, 0)[:, ::-1, ::-1]
    wins = (WIN, XY, XY)
    strides = (STRIDE[2], STRIDE[1], STRIDE[0])
    pshape = tuple(max(n, w) for n, w in zip(img.shape, wins))
    pad = [(0, p - n) for n, p in zip(img.shape, pshape)]
    img_p, lab_p = np.pad(img, pad), np.pad(lab, pad)
    tiles = [(z, y, x) for z in z_windows(pshape[0], WIN, strides[0])
             for y in z_windows(pshape[1], XY, strides[1])
             for x in z_windows(pshape[2], XY, strides[2])]
    I = np.stack([img_p[z:z + WIN, y:y + XY, x:x + XY] for z, y, x in tiles])
    L = np.stack([lab_p[z:z + WIN, y:y + XY, x:x + XY] for z, y, x in tiles])
    meta = {"orig_shape": list(ct.shape), "src_shape": orig_shape, "ornt": ornt.tolist(), "bbox": [[s.start, s.stop] for s in bb],
            "crop_shape": list(ct_c.shape), "res_shape": list(res_shape),
            "padded_zyx": list(pshape), "tiles_zyx": [list(t) for t in tiles],
            "spacing": [float(s) for s in spacing], "target": list(target)}
    return {"img": np.round(I * 255).astype(np.uint8), "lab": L.astype(np.uint8), "meta": meta}


def reconstruct(tiles: np.ndarray, meta: dict) -> np.ndarray:
    """(n, 32, 256, 256) olasilik -> orijinal (x, y, z) grid; kutu disi 0, ortusme ortalanir."""
    pz = meta["padded_zyx"]
    acc = np.zeros(pz, np.float32); cnt = np.zeros(pz, np.float32)
    for t, (z, y, x) in zip(tiles, meta["tiles_zyx"]):
        acc[z:z + WIN, y:y + XY, x:x + XY] += t
        cnt[z:z + WIN, y:y + XY, x:x + XY] += 1
    rz = [meta["res_shape"][2], meta["res_shape"][1], meta["res_shape"][0]]
    prob = (acc / np.maximum(cnt, 1))[:rz[0], :rz[1], :rz[2]]
    if "ornt" in meta:          # radyolojik -> RAS (ornt'siz eski meta: ham (z, y, x) duzeni)
        prob = prob[:, ::-1, ::-1]
    prob = _resize(prob.transpose(2, 1, 0), meta["crop_shape"], order=1)
    out = np.zeros(meta["orig_shape"], np.float32)
    out[tuple(slice(a, b) for a, b in meta["bbox"])] = prob
    if "ornt" in meta:          # RAS -> hacmin orijinal dizi yonu
        from nibabel.orientations import apply_orientation
        out = apply_orientation(out, _inverse_ornt(np.asarray(meta["ornt"], float)))
        assert list(out.shape) == meta["src_shape"], (out.shape, meta["src_shape"])
    return out
