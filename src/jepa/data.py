"""On egitim veri hatti -- geometri katmani (torch GEREKTIRMEZ, saf numpy/scipy).

Bu dosya mimariden bagimsizdir: encoder ViT de olsa CNN de olsa girdi geometrisi
buradan gecer. Amaci tek: **mm bilgisi hicbir adimda kaybolmasin.**

UC SESSIZ HATA, ucu de burada assert'e baglandi
-----------------------------------------------
1. `resample` ISTENEN spacing'i degil ULASILAN spacing'i dondurur. Sekil tam sayiya
   yuvarlandigi icin ikisi ayni degildir: 7 kesit x 3.0 mm = 21 mm'yi 2.0 mm'ye
   orneklemek 10.5 kesit ister, 10 kesit cikar ve gercek aralik 2.1 mm olur (%5
   sapma). Istenen degeri tasimak,
   hedefleme hatasini (mm) ve marj ihlalini sessizce yanlis olcer.
2. Maske trilineer ornekleme ile 0.5 gibi ara degerler uretir -> etiket bozulur.
   `is_mask=True` order>0'i REDDEDER (kural: sessiz hata yok).
3. Yon degisiminde spacing de yer degistirmeli. `canonical_ras` ikisini BIRLIKTE
   dondurur; ayri tasinirsa mm ekseni karisir.

Konum kaybi: kirpma/dolgu yapan her fonksiyon ne yaptigini geri dondurur
(`pad`, `origin`), boylece tahmin orijinal izgaraya donebilir.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class GeometryError(ValueError):
    """Geometri tutarsizligi. Sessizce duzeltme yok -- firlat."""


# ------------------------------------------------------------------- spacing
def achieved_spacing(n_in: int, spacing_in: float, n_out: int) -> float:
    """Yeniden orneklemeden SONRA gercekten elde edilen aralik.

    Fiziksel uzunluk korunur: n_in * spacing_in == n_out * spacing_out.
    """
    if n_out <= 0:
        raise GeometryError(f"gecersiz cikti boyutu: {n_out}")
    return float(n_in) * float(spacing_in) / float(n_out)


def _check_spacing(spacing) -> tuple[float, ...]:
    sp = tuple(float(s) for s in spacing)
    if any(not np.isfinite(s) or s <= 0 for s in sp):
        raise GeometryError(f"spacing pozitif ve sonlu olmali: {spacing}")
    return sp


def resample(arr: np.ndarray, spacing, target, order: int = 1,
             is_mask: bool = False) -> tuple[np.ndarray, tuple[float, ...]]:
    """Hacmi hedef aralige orneklen; (dizi, ULASILAN spacing) dondur.

    order: goruntu icin 1 (trilineer) ya da 3 (spline); maske icin 0 (en yakin).
    is_mask=True ve order>0 -> FIRLAT (maskede ara deger = bozuk hat).
    """
    if is_mask and order > 0:
        raise GeometryError(
            f"maske order={order} ile orneklenemez: ara deger uretir (0.5 gibi) ve "
            "etiketi bozar. Maske icin order=0 kullan.")
    sp = _check_spacing(spacing)
    tg = _check_spacing(target)
    if len(sp) != arr.ndim or len(tg) != arr.ndim:
        raise GeometryError(f"spacing/target boyutu dizi ile uyusmuyor: "
                            f"{arr.ndim}D dizi, {len(sp)} spacing, {len(tg)} target")
    shape_out = tuple(max(1, int(round(n * s / t))) for n, s, t in zip(arr.shape, sp, tg))
    got = tuple(achieved_spacing(n, s, m) for n, s, m in zip(arr.shape, sp, shape_out))
    if shape_out == tuple(arr.shape):
        return arr, tuple(float(s) for s in sp)
    zoom = [m / n for n, m in zip(arr.shape, shape_out)]
    out = ndimage.zoom(arr, zoom, order=order, mode="nearest", grid_mode=True)
    if out.shape != shape_out:                       # zoom yuvarlamasi kayabilir
        sl = tuple(slice(0, m) for m in shape_out)
        pad = [(0, max(0, m - n)) for n, m in zip(out.shape, shape_out)]
        out = np.pad(out[sl], pad, mode="edge")
    if is_mask:
        vals = np.unique(out)
        if not np.all(np.isin(vals, np.unique(arr))):
            raise GeometryError(f"maske orneklemesi yeni deger uretti: {vals[:10]}")
    return out, got


# ------------------------------------------------------------------------ HU
def hu_window(ct: np.ndarray, window, out_range=(0.0, 1.0)) -> np.ndarray:
    """HU penceresini kirp ve `out_range`'e getir. Egitim ve cikarimda BIREBIR ayni olmali.

    Pencere `configs/paths.yaml` -> `metrics.hu_window`. Normalizasyonun TEK yeri
    burasi: encoder [-1, 1] istiyor diye ikinci bir `normalize_hu` yazilmaz, cunku
    iki tanim egitim ile cikarimi sessizce ayirir. Encoder tarafi
    `hu_window(x, window, out_range=(-1.0, 1.0))` cagirir.
    """
    lo, hi = (float(window[0]), float(window[1]))
    if not (hi > lo):
        raise GeometryError(f"HU penceresi ters ya da sifir genislikte: ({lo}, {hi})")
    a, b = (float(out_range[0]), float(out_range[1]))
    if not (b > a):
        raise GeometryError(f"out_range artan olmali: ({a}, {b})")
    z = (np.clip(ct.astype(np.float32), lo, hi) - lo) / (hi - lo)
    return (z * (b - a) + a).astype(np.float32)


# ----------------------------------------------------------------------- yon
def canonical_ras(arr: np.ndarray, affine, spacing) -> tuple[np.ndarray, tuple[float, ...], list]:
    """Hacmi kanonik RAS dizi duzenine cevir; spacing'i BIRLIKTE tasi.

    -> (dizi, spacing, ornt). `ornt` geri donus icin saklanir
    (`nibabel.orientations.apply_orientation` + ters ornt).
    """
    from nibabel.orientations import apply_orientation, io_orientation
    sp = _check_spacing(spacing)
    ornt = io_orientation(np.asarray(affine, float))
    out = apply_orientation(arr, ornt)
    new_sp = [0.0] * len(sp)
    for i, (t, _) in enumerate(ornt):
        new_sp[int(t)] = sp[i]
    return out, tuple(new_sp), ornt.tolist()


def inverse_ornt(ornt) -> np.ndarray:
    """apply_orientation(apply_orientation(a, o), inverse_ornt(o)) == a."""
    o = np.asarray(ornt, float)
    inv = np.zeros_like(o)
    for i, (t, f) in enumerate(o):
        inv[int(t)] = [i, f]
    return inv


# -------------------------------------------------------------------- yamalar
def patch_starts(n: int, patch: int, stride: int) -> list[int]:
    """[0, n) eksenini kapsayan yama baslangiclari; son yama sona dayali."""
    if patch <= 0 or stride <= 0:
        raise GeometryError(f"patch/stride pozitif olmali: {patch}, {stride}")
    if n <= patch:
        return [0]
    st = list(range(0, n - patch, stride))
    return st + [n - patch]


def pad_to_patch(arr: np.ndarray, patch,
                 fill: float = 0.0) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Hacim yamadan kucukse `fill` ile doldur; uygulanan dolguyu da dondur.

    Dolgu geri donuste kirpilmak zorunda -- bu yuzden sessizce yapilmaz, bildirilir.

    `fill` DIKKAT: dolgu genellikle `hu_window`'dan SONRA yapilir, yani normalize
    uzaydadir. out_range=(-1, 1) ve pencere (-200, 250) icin 0.0 = 25 HU (yumusak
    doku), hava degil -- hacim kenarina doku benzeri bir serit koymak lezyon-bazli
    yanlis pozitifi sisirebilir. Cikarim tarafi `fill=out_range[0]` gecer
    (`src/jepa/decoder.py:predict_volume`). Varsayilan geriye donuk uyum icin 0.0.
    """
    if len(patch) != arr.ndim:
        raise GeometryError(f"patch boyutu dizi ile uyusmuyor: {arr.ndim}D vs {len(patch)}")
    pad = tuple((0, max(0, int(p) - int(n))) for n, p in zip(arr.shape, patch))
    if all(b == 0 for _, b in pad):
        return arr, pad
    return np.pad(arr, pad, mode="constant", constant_values=float(fill)), pad


def sample_patch_origin(shape, patch, rng: np.random.Generator,
                        focus: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """Rastgele yama baslangici. `focus` verilirse yama o vokseli ICERIR.

    focus: tumorlu/odakli pencere ornekleme icin (orn. lezyon merkezi).
    Hacim yamadan kucukse FIRLATIR -- once `pad_to_patch` cagrilmalidir, aksi
    halde sessizce kirpilmis bir pencere egitime girer.
    """
    if len(shape) != len(patch):
        raise GeometryError(f"shape/patch boyutu uyusmuyor: {len(shape)} vs {len(patch)}")
    small = [(n, p) for n, p in zip(shape, patch) if n < p]
    if small:
        raise GeometryError(
            f"hacim yamadan kucuk {small} -- once pad_to_patch cagir (sessiz dolgu yok)")
    out = []
    for i, (n, p) in enumerate(zip(shape, patch)):
        hi = n - p
        if focus is None:
            out.append(int(rng.integers(0, hi + 1)))
        else:
            f = int(focus[i])
            if not (0 <= f < n):
                raise GeometryError(f"odak eksen {i} disinda: {f} / {n}")
            lo_i = max(0, min(f - p + 1, hi))
            hi_i = max(0, min(f, hi))
            out.append(int(rng.integers(lo_i, hi_i + 1)))
    return tuple(out)
