"""Kirpilmis kopya arama: bir veri setinin goruntuleri baska bir setten mi alinmis? (L6)

Neden ayri modul: TotalSegmentator liver_lesions egitim goruntuleri karaciger cevresine
KIRPILMIS (hicbiri 512x512 degil). Sekil/aralik/histogram/32^3 imza kirpmayla bozulur;
"eslesme yok" sonucu "kopya yok" anlamina GELMEZ.

Fikir: kirpma = ayni vokseller, daha kucuk pencere. Sorgu goruntusunun 3 kesitinden
kucuk yamalar (probe) al, referans hacimde ara:
  1) exact: yama referansin bir kesitinde birebir (tamsayi HU) var mi? 8 duzlem-ici yon
     (x/y transpoz x ters). En nadir HU degeri uzerinden indeks -> spacing filtresi gerekmez.
  2) ncc: exact yoksa (z yeniden orneklenmis / yogunluk kaydirilmis kopya) normalize
     capraz korelasyon: 3 yama x 8 yon, once yarim cozunurlukte kaba, sonra tam cozunurlukte
     ince arama. Karar: >=2 guclu yama AYNI kirpma kokenine (+-2 px) isaret etmeli ve
     z olcegi header'daki aralik oranina uymali. Pahali -> yalniz xy araligi uyan ve fiziksel
     boyuta sigan adaylar; xy de yeniden orneklenmisse KORDUR ('ncc_candidates=0').

Eksen: aksiyel kesit = 2. eksen varsayilir. TS dosyalari RAS'a dondurulmus; affine'i 2. ekseni
S/I gostermeyen referans ayrica RAS haliyle ('<id>@ras') aranir. Eksen kodlari CSV'de.

Sessiz yanlis negatif yok: okunamayan sorgu extract_errors.csv'ye yazilir ve komut hata
koduyla biter; NCC butcesi kesilirse ncc_complete=False; ozet "gercekten denetlenen" sayisini verir.

Uc komut (Kaggle'da zip 38.9 GB, acmadan):
  extract : sorgu goruntulerinden probe'lar -> <out>/<id>.npz (yarida kalirsa devam eder)
  search  : probe'lari referans hacimlerde ara -> CSV (sorgu basina bir satir, ara kayitli)
  summary : CSV'den ozet
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PATCH = 32
N_PROBES = 3
MIN_STD = 15.0            # HU (pencerede); tekduze yama her yerde eslesir, karara girmez
STD_HALF_WINDOW = 250     # yapi olcusu: yama medyani +-250'ye kirpilmis std. Hava/kemik/FOV kenari
                          # baskin olmasin; medyana gore oldugu icin yogunluk kaymasindan etkilenmez
# Gercek CT'de (IRCADb, 1.6 -> 1.0 mm z) tekduze karaciger yamasi dogru konumda NCC ~0.90 verdi.
# Karar esigin yaninda kirpma kokeni ve z olcegi tutarliligina dayanir.
NCC_THRESHOLD = 0.90
NCC_POOL = 2              # kaba arama icin xy ortalama havuzlama
Z_SCALE_TOL = 0.15
# Tek guclu yama tesadufen ~0.90 verebiliyor (IRCADb'de olmayan tarama: 0.906). Tek yamalik
# 'kismi' sonuc icin daha yuksek esik; gercek kopyalarda yama skoru ~0.99.
NCC_SINGLE = 0.97
MEMBER_RE = r"images(Tr|Ts)/[^/]+_0000\.nii(\.gz)?$"
NONE_HIT = (float("nan"), -1, -1, -1)


# ---------------------------------------------------------------- yardimcilar
def dihedral(p: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """8 duzlem-ici yon. Ad: T<transpoz>x<ters>y<ters>; ilk eleman = kendisi."""
    out = []
    for t in (0, 1):
        b = p.T if t else p
        for fx, fy in itertools.product((0, 1), repeat=2):
            axes = tuple(a for a, f in zip((0, 1), (fx, fy)) if f)
            out.append((f"T{t}x{fx}y{fy}", np.flip(b, axes) if axes else b))
    return out


def _probe_origin(variant: str, x0: int, y0: int, size: int, nx: int, ny: int) -> tuple[int, int]:
    """Sorgu kesiti butunuyle 'variant' ile donduruldugunde yamanin yeni sol-ust kosesi."""
    t, fx, fy = (int(variant[i]) for i in (1, 3, 5))
    if t:
        x0, y0, nx, ny = y0, x0, ny, nx
    if fx:
        x0 = nx - x0 - size
    if fy:
        y0 = ny - y0 - size
    return x0, y0


def _as_int16(a) -> np.ndarray:
    return np.clip(np.rint(np.asarray(a, dtype=np.float32)), -32768, 32767).astype(np.int16)


def _axcodes(affine) -> str:
    import nibabel as nib
    if affine is None or not np.all(np.isfinite(affine)):
        return "???"
    return "".join(c or "?" for c in nib.aff2axcodes(affine))


def _structure(p: np.ndarray) -> float:
    m = float(np.median(p))
    return float(np.clip(p, m - STD_HALF_WINDOW, m + STD_HALF_WINDOW).std())


def probe_patches(arr: np.ndarray, n: int = N_PROBES, size: int = PATCH) -> list[dict]:
    """z'nin %25/%50/%75'inden (n=3) birer kare yama; merkez cevresindeki 7x7 izgarada en yapili
    konum. Yapi = _structure (kaydedilir; MIN_STD alti karara girmez)."""
    arr = np.asarray(arr)
    nx, ny, nz = arr.shape[:3]
    size = min(size, nx, ny)
    zs = sorted({int(np.clip(round((k + 1) * nz / (n + 1)), 0, nz - 1)) for k in range(n)})
    step = max(size // 2, 1)
    cx, cy = (nx - size) // 2, (ny - size) // 2
    out = []
    for z in zs:
        sl = np.asarray(arr[:, :, z], dtype=np.float32)
        best = None
        for ox, oy in itertools.product(range(-3, 4), repeat=2):
            x0 = int(np.clip(cx + ox * step, 0, nx - size))
            y0 = int(np.clip(cy + oy * step, 0, ny - size))
            p = sl[x0:x0 + size, y0:y0 + size]
            s = _structure(p)
            if best is None or s > best["std"]:
                best = {"z": z, "x": x0, "y": y0, "std": s, "patch": p.copy()}
        out.append(best)
    return out


# ---------------------------------------------------------------- exact arama
class ExactIndex:
    """Referans hacimde deger -> konum indeksi (radix sort). Yama aramasi en nadir degerden baslar."""

    def __init__(self, vol):
        self.v = _as_int16(vol)
        flat = self.v.ravel()
        self.order = np.argsort(flat, kind="stable")
        self.counts = np.bincount(flat.astype(np.int32) + 32768, minlength=65536)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)])

    def find(self, patch, max_hits: int = 20) -> list[dict]:
        q0 = _as_int16(patch)
        keys = q0.astype(np.int32) + 32768
        i, j = np.unravel_index(np.argmin(self.counts[keys]), q0.shape)
        k = keys[i, j]
        pos = self.order[self.starts[k]:self.starts[k + 1]]
        if pos.size == 0:
            return []
        px, py, pz = np.unravel_index(pos, self.v.shape)
        nx, ny, _ = self.v.shape
        rng = np.random.default_rng(0)
        hits = []
        for name, q in dihedral(q0):
            a, b = q.shape
            if a > nx or b > ny:
                continue
            ti, tj = _probe_origin(name, i, j, 1, *q0.shape)   # (i, j) donusumde nereye gitti
            ox, oy = px - ti, py - tj
            ok = (ox >= 0) & (oy >= 0) & (ox + a <= nx) & (oy + b <= ny)
            ox, oy, oz = ox[ok], oy[ok], pz[ok]
            for ci, cj in rng.integers(0, [a, b], size=(16, 2)):
                if ox.size == 0:
                    break
                keep = self.v[ox + ci, oy + cj, oz] == q[ci, cj]
                ox, oy, oz = ox[keep], oy[keep], oz[keep]
            for x, y, z in zip(ox, oy, oz):
                if np.array_equal(self.v[x:x + a, y:y + b, z], q):
                    hits.append({"variant": name, "x": int(x), "y": int(y), "z": int(z)})
                    if len(hits) >= max_hits:
                        return hits
        return hits


# ---------------------------------------------------------------- ncc arama
def _box_sum(c: np.ndarray, a: int, b: int) -> np.ndarray:
    """(X,Y,Z) -> her a x b pencerenin toplami, 'valid' boyutunda."""
    s = np.cumsum(np.cumsum(c, axis=0), axis=1)
    s = np.pad(s, ((1, 0), (1, 0), (0, 0)))
    return s[a:, b:] - s[:-a, b:] - s[a:, :-b] + s[:-a, :-b]


def ncc_multi(volf: np.ndarray, templates: list, zchunk: int = 32) -> list[tuple[float, int, int, int]]:
    """Ayni boyutlu 2B sablonlari tum kesitlerde ara; her biri icin (en iyi NCC, x, y, z).

    Hacim parcasinin FFT'si ve pencere istatistikleri TUM sablonlar arasinda paylasilir.
    Dairesel korelasyon 'valid' bolgede sarmaz.
    """
    import scipy.fft as sfft
    if not templates:
        return []
    nx, ny, nz = volf.shape
    a, b = np.asarray(templates[0]).shape
    if a > nx or b > ny:
        return [NONE_HIT] * len(templates)
    tf, tn = [], []
    for t in templates:
        t = np.asarray(t, dtype=np.float64)
        if t.shape != (a, b):
            raise ValueError(f"sablon boyutlari farkli: {t.shape} != {(a, b)}")
        t0 = t - t.mean()
        pad = np.zeros((nx, ny))
        pad[:a, :b] = t0
        tf.append(np.conj(sfft.rfft2(pad))[..., None].astype(np.complex64))
        tn.append(float(np.sqrt((t0 ** 2).sum())))
    n = a * b
    best = [(-np.inf, -1, -1, -1)] * len(templates)
    with sfft.set_workers(-1):
        for z0 in range(0, nz, zchunk):
            c = np.asarray(volf[:, :, z0:z0 + zchunk], dtype=np.float64)
            V = sfft.rfft2(c.astype(np.float32), axes=(0, 1))
            s1 = _box_sum(c, a, b)
            sd = np.sqrt(np.maximum(_box_sum(c * c, a, b) - s1 * s1 / n, 0))
            flat = sd <= 1e-3 * np.sqrt(n)
            for k in range(len(templates)):
                if tn[k] == 0:
                    continue
                num = sfft.irfft2(V * tf[k], s=(nx, ny), axes=(0, 1))[:nx - a + 1, :ny - b + 1]
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where(flat, 0.0, num / (sd * tn[k]))
                i = int(np.argmax(r))
                if r.flat[i] > best[k][0]:
                    x, y, z = np.unravel_index(i, r.shape)
                    best[k] = (float(r.flat[i]), int(x), int(y), int(z0 + z))
    return [bk if tn[k] > 0 else NONE_HIT for k, bk in enumerate(best)]


def ncc_search(volf: np.ndarray, t: np.ndarray, zchunk: int = 32) -> tuple[float, int, int, int]:
    """Tek sablon: (en iyi NCC, x, y, z). Yogunluk olcek/kaymasina dayanikli."""
    return ncc_multi(volf, [t], zchunk)[0]


def _pool(a: np.ndarray, f: int) -> np.ndarray:
    """xy'de f x f ortalama (kenar artigi atilir). 2B ve 3B."""
    if f == 1:
        return np.asarray(a, dtype=np.float32)
    nx, ny = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    a = np.asarray(a[:nx, :ny], dtype=np.float32)
    return a.reshape(nx // f, f, ny // f, f, *a.shape[2:]).mean(axis=(1, 3))


def _ncc_at(volf: np.ndarray, t: np.ndarray, x: int, y: int, z: int) -> float:
    a, b = t.shape
    w = np.asarray(volf[x:x + a, y:y + b, z], dtype=np.float64)
    if w.shape != t.shape:
        return float("nan")
    w0, t0 = w - w.mean(), t - t.mean()
    d = np.sqrt((w0 ** 2).sum() * (t0 ** 2).sum())
    return float((w0 * t0).sum() / d) if d > 0 else float("nan")


def _refine(volf: np.ndarray, t: np.ndarray, x: int, y: int, z: int, f: int) -> tuple:
    """Kaba konum (havuzlanmis) -> tam cozunurlukte +-f piksel, +-1 kesit icinde en iyi NCC."""
    t = np.asarray(t, dtype=np.float64)
    best = (-np.inf, -1, -1, -1)
    for zz in range(max(z - 1, 0), min(z + 2, volf.shape[2])):
        for xx in range(max(x * f - f, 0), x * f + f + 1):
            for yy in range(max(y * f - f, 0), y * f + f + 1):
                s = _ncc_at(volf, t, xx, yy, zz)
                if s > best[0]:
                    best = (s, xx, yy, zz)
    return best if best[1] >= 0 else NONE_HIT


# ---------------------------------------------------------------- girdi/cikti
def _nifti(obj) -> tuple[np.ndarray, tuple, np.ndarray]:
    import nibabel as nib
    img = obj if isinstance(obj, nib.Nifti1Image) else nib.load(str(obj))
    arr = np.asanyarray(img.dataobj)
    if arr.ndim != 3:
        raise ValueError(f"3B bekleniyordu, sekil {arr.shape}")
    return arr, tuple(float(s) for s in img.header.get_zooms()[:3]), img.affine


def _query(qid: str, arr, spacing, affine=None, source: str = "") -> dict:
    from src.provenance.match_ircadb_lits import hu_histogram
    probes = probe_patches(arr)
    return {"id": qid, "source": source, "shape": tuple(int(s) for s in arr.shape),
            "spacing": tuple(spacing), "affine": affine, "dtype": str(arr.dtype),
            "integer_valued": bool(all(np.all(p["patch"] == np.rint(p["patch"])) for p in probes)),
            "hist": hu_histogram(arr).astype(np.float32), "probes": probes}


def extract_queries(zip_path=None, glob: str | None = None, member_regex: str = MEMBER_RE,
                    skip: set[str] = frozenset(), errors: list | None = None, verbose: bool = False):
    """Zip uyelerini ACMADAN (bellekte) ya da glob ile dosyalari oku; sorgu dict'i uretir.
    Okunamayan uye durdurmaz: (kimlik, kaynak, neden) 'errors' listesine eklenir."""
    import nibabel as nib
    errors = [] if errors is None else errors

    def items():
        if zip_path is not None:
            with zipfile.ZipFile(zip_path) as zf:
                names = sorted(n for n in zf.namelist() if re.search(member_regex, n))
                for i, name in enumerate(names, 1):
                    qid = re.sub(r"_0000\.nii(\.gz)?$", "", Path(name).name)

                    def load(name=name):
                        raw = zf.read(name)
                        if name.endswith(".gz"):
                            raw = gzip.decompress(raw)
                        return _nifti(nib.Nifti1Image.from_bytes(raw))
                    yield qid, name, f"{i}/{len(names)}", load
        else:
            from src.data.case_id import assert_unique_ids, glob_paths
            paths = assert_unique_ids(glob_paths(glob))
            for i, (qid, p) in enumerate(paths.items(), 1):
                yield qid, str(p), f"{i}/{len(paths)}", (lambda p=p: _nifti(p))

    for qid, src, prog, load in items():
        if qid in skip:
            continue
        if verbose:
            print(f"[extract] {prog} {src}", flush=True)
        try:
            arr, sp, aff = load()
            q = _query(qid, arr, sp, aff, source=src)
        except Exception as e:  # noqa: BLE001 -- kaydedilir ve komut hatayla biter, sessiz degil
            errors.append((qid, src, f"{type(e).__name__}: {e}"[:300]))
            print(f"[extract] HATA {src}: {errors[-1][2]}", flush=True)
            continue
        yield q


def count_members(zip_path=None, glob: str | None = None, member_regex: str = MEMBER_RE) -> int:
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as zf:
            return sum(bool(re.search(member_regex, n)) for n in zf.namelist())
    from src.data.case_id import glob_paths
    return len(glob_paths(glob))


def save_query(q: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{q['id']}.npz"
    pr = q["probes"]
    np.savez_compressed(
        f, id=q["id"], source=q["source"], shape=q["shape"], spacing=q["spacing"],
        affine=q["affine"] if q["affine"] is not None else np.full((4, 4), np.nan),
        dtype=q["dtype"], integer_valued=q["integer_valued"], hist=q["hist"],
        patches=np.stack([p["patch"] for p in pr]),
        probe_xyz=np.array([[p["x"], p["y"], p["z"]] for p in pr]),
        probe_std=np.array([p["std"] for p in pr]))
    return f


def load_queries(probe_dir: Path) -> list[dict]:
    qs = []
    for f in sorted(Path(probe_dir).glob("*.npz")):
        d = np.load(f)
        probes = [{"x": int(x), "y": int(y), "z": int(z), "std": float(s), "patch": p}
                  for (x, y, z), s, p in zip(d["probe_xyz"], d["probe_std"], d["patches"])]
        qs.append({"id": str(d["id"]), "shape": tuple(int(s) for s in d["shape"]),
                   "spacing": tuple(float(s) for s in d["spacing"]),
                   "axcodes": _axcodes(d["affine"]),
                   "integer_valued": bool(d["integer_valued"]), "probes": probes})
    return qs


def _ref_views(rid: str, ref) -> list[tuple[str, np.ndarray, tuple, str]]:
    """Referansin aranacak halleri: kendi dizisi + (2. eksen S/I degilse) RAS hali."""
    import nibabel as nib
    if isinstance(ref, tuple):
        arr, sp = np.asarray(ref[0]), tuple(ref[1])
        aff = ref[2] if len(ref) > 2 else None
    else:
        arr, sp, aff = _nifti(ref)
        arr = np.asarray(arr)
    codes = _axcodes(aff)
    views = [(rid, arr, sp, codes)]
    if aff is not None and codes[2] not in "SI":
        can = nib.as_closest_canonical(nib.Nifti1Image(arr, aff))
        views.append((f"{rid}@ras", np.asarray(can.dataobj),
                      tuple(float(s) for s in can.header.get_zooms()[:3]), _axcodes(can.affine)))
    return views


# ---------------------------------------------------------------- karar
def _ncc_candidate(q: dict, ref_shape, ref_sp, tol: float) -> bool:
    """xy araligi ayni (transpoz serbest) ve kirpinti fiziksel olarak referansa sigiyor."""
    qs, rs = sorted(q["spacing"][:2]), sorted(ref_sp[:2])
    if any(abs(a - b) > tol for a, b in zip(qs, rs)):
        return False
    qe = sorted(n * s for n, s in zip(q["shape"][:2], q["spacing"][:2]))
    re_ = sorted(n * s for n, s in zip(ref_shape[:2], ref_sp[:2]))
    if any(a > b + 2 * max(rs) for a, b in zip(qe, re_)):
        return False
    return q["shape"][2] * q["spacing"][2] <= ref_shape[2] * ref_sp[2] + 2 * ref_sp[2]


def _crop_origin(q: dict, h: dict) -> tuple[int, int]:
    """Isabetten kirpintinin (donusturulmus sorgu kesitinin) referanstaki sol-ust kosesi."""
    p = q["probes"][h["probe"]]
    px, py = _probe_origin(h["variant"], p["x"], p["y"], p["patch"].shape[0], *q["shape"][:2])
    return h["x"] - px, h["y"] - py


def _structured(q: dict) -> set[int]:
    return {k for k, p in enumerate(q["probes"]) if p["std"] >= MIN_STD}


def _group(q: dict, hits: list[dict]) -> list[dict]:
    """Exact isabetleri kirpma kokenine (ref, yon, ox, oy) gore topla; tekduze yama karara girmez."""
    ok = _structured(q)
    groups: dict[tuple, dict] = {}
    for h in hits:
        if h["probe"] not in ok:
            continue
        key = (h["ref"], h["variant"], *_crop_origin(q, h))
        g = groups.setdefault(key, {"ref": h["ref"], "variant": h["variant"],
                                    "ox": key[2], "oy": key[3], "z": {}})
        g["z"].setdefault(h["probe"], h["z"])
    return sorted(groups.values(), key=lambda g: -len(g["z"]))


def _z_scale(q: dict, zmap: dict) -> float:
    """Sorgu kesit farki / referans kesit farki. Birebir kopya: +-1; z yeniden ornekleme: != 1."""
    if len(zmap) < 2:
        return float("nan")
    i, j = min(zmap), max(zmap)
    dr = zmap[j] - zmap[i]
    dq = q["probes"][j]["z"] - q["probes"][i]["z"]
    return float(dq / dr) if dr else float("nan")


def decide_ncc(q: dict, res: dict, ref_spacing: dict, thr: float = NCC_THRESHOLD) -> dict:
    """res: {ref: {yon: [(ncc, x, y, z) probe basina]}}. Tum referans/yonlar icinde en iyi
    TUTARLI grubu secer: guclu (>=thr, yapili) yamalar ayni kirpma kokeni (+-2 px) ve
    |z_scale| ~ ref_dz / q_dz (+-%15). >=2 tutarli -> ncc_match; z tutmazsa ya da tek yama
    >= NCC_SINGLE ise ncc_partial."""
    ok = _structured(q)
    cands = []
    for rid, by_var in res.items():
        expect = ref_spacing[rid][2] / q["spacing"][2]
        for var, hits in by_var.items():
            strong = [{"ref": rid, "probe": k, "variant": var, "x": x, "y": y, "z": z, "ncc": s}
                      for k, (s, x, y, z) in enumerate(hits)
                      if k in ok and np.isfinite(s) and s >= thr]
            grp = []
            for h in strong:
                o = _crop_origin(q, h)
                g = [s for s in strong
                     if max(abs(a - b) for a, b in zip(_crop_origin(q, s), o)) <= 2]
                if len(g) > len(grp):
                    grp = g
            if not grp or (len(grp) == 1 and grp[0]["ncc"] < NCC_SINGLE):
                continue
            zs = _z_scale(q, {h["probe"]: h["z"] for h in grp})
            z_ok = (len(grp) >= 2 and np.isfinite(zs)
                    and abs(abs(zs) - expect) <= Z_SCALE_TOL * expect)
            score = float(np.mean([h["ncc"] for h in grp]))
            cands.append((z_ok, len(grp), score, rid, var, grp, zs))
    if not cands:
        return {"verdict": None, "n_ncc": 0}
    z_ok, n, score, rid, var, grp, zs = max(cands, key=lambda c: (c[0], c[1], c[2]))
    ox, oy = np.median(np.array([_crop_origin(q, h) for h in grp]), axis=0).astype(int)
    return {"verdict": "ncc_match" if z_ok else "ncc_partial", "ref_id": rid, "variant": var,
            "crop_x": int(ox), "crop_y": int(oy), "z_scale": zs, "n_ncc": n,
            "ncc_min": round(min(h["ncc"] for h in grp), 4)}


def _row(q: dict) -> dict:
    return {"query_id": q["id"], "verdict": "no_match", "ref_id": None,
            "n_probes": len(q["probes"]), "n_structured": len(_structured(q)),
            "n_exact": 0, "n_ncc": 0, "variant": None, "crop_x": None, "crop_y": None,
            "z_scale": np.nan, "other_exact_refs": "", "ncc_min": np.nan,
            "ncc_best": np.nan, "ncc_best_ref": None, "ncc_second": np.nan, "ncc_second_ref": None,
            "ncc_candidates": 0, "ncc_checked": 0, "ncc_complete": True,
            "q_shape": "x".join(map(str, q["shape"])),
            "q_spacing": "x".join(f"{s:.4f}" for s in q["spacing"]),
            "q_axcodes": q.get("axcodes", "???"), "ref_spacing": None, "ref_axcodes": None,
            "integer_valued": q.get("integer_valued"),
            "probe_std_min": round(min(p["std"] for p in q["probes"]), 1)}


def _ref_fields(ref_meta: dict, vid: str) -> dict:
    return {"ref_spacing": "x".join(f"{s:.4f}" for s in ref_meta[vid][1]),
            "ref_axcodes": ref_meta[vid][2]}


def search(refs: dict, queries: list[dict], ncc: bool = True, ncc_threshold: float = NCC_THRESHOLD,
           spacing_tol: float = 1e-3, ncc_budget_s: float | None = None,
           checkpoint: Path | None = None, verbose: bool = False) -> pd.DataFrame:
    """refs: {kimlik: NIfTI yolu | (dizi, spacing[, affine])}. Her referans exact icin bir,
    ncc icin bir kez okunur. checkpoint verilirse ara sonuclar oraya yazilir."""
    exact: dict[str, list] = {q["id"]: [] for q in queries}
    ref_meta = {}   # gorunum kimligi -> (sekil, spacing, eksen kodlari, referans kimligi)
    for n, (rid, ref) in enumerate(sorted(refs.items()), 1):
        t0 = time.time()
        for vid, arr, sp, codes in _ref_views(rid, ref):
            ref_meta[vid] = (arr.shape, sp, codes, rid)
            idx = ExactIndex(arr)
            for q in queries:
                for k, p in enumerate(q["probes"]):
                    for h in idx.find(p["patch"]):
                        exact[q["id"]].append({**h, "ref": vid, "probe": k})
            del idx, arr
        if verbose:
            nh = sum(any(h["ref"].split("@")[0] == rid for h in v) for v in exact.values())
            print(f"[exact] {n}/{len(refs)} {rid} {time.time() - t0:.1f}s  isabetli sorgu: {nh}",
                  flush=True)

    rows, todo = {}, []
    for q in queries:
        r = rows[q["id"]] = _row(q)
        groups = _group(q, exact[q["id"]])
        if groups:
            g = groups[0]
            r.update(verdict="exact" if len(g["z"]) >= 2 else "exact_partial", ref_id=g["ref"],
                     n_exact=len(g["z"]), variant=g["variant"], crop_x=g["ox"], crop_y=g["oy"],
                     z_scale=_z_scale(q, g["z"]),
                     other_exact_refs=";".join(sorted({gg["ref"] for gg in groups[1:]} - {g["ref"]})),
                     **_ref_fields(ref_meta, g["ref"]))
        if r["verdict"] != "exact":
            todo.append(q)
    if checkpoint:
        _write(rows, checkpoint)

    if ncc and todo:
        _ncc_pass(refs, ref_meta, todo, rows, ncc_threshold, spacing_tol, ncc_budget_s,
                  checkpoint, verbose)
    return pd.DataFrame(list(rows.values()))


def _write(rows: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows.values())).to_csv(path, index=False)


def _ncc_pass(refs, ref_meta, todo, rows, thr, tol, budget_s, checkpoint, verbose):
    """Referans basina BIR kez oku; tum aday sorgularin 3 yama x 8 yon sablonu ayni FFT'yi paylasir.
    Kaba (NCC_POOL) arama + tam cozunurlukte ince ayar. Butce referans sinirinda kontrol edilir."""
    t_start = time.time()
    cands = {q["id"]: {vid for vid, (shp, sp, _, _) in ref_meta.items()
                       if _ncc_candidate(q, shp, sp, tol)} for q in todo}
    for q in todo:
        rows[q["id"]]["ncc_candidates"] = len(cands[q["id"]])
    res: dict[str, dict] = {q["id"]: {} for q in todo}       # qid -> ref -> yon -> [probe]
    best: dict[str, dict] = {q["id"]: {} for q in todo}      # qid -> ref -> en iyi NCC
    sp_of = {vid: m[1] for vid, m in ref_meta.items()}
    f = NCC_POOL
    stopped = False
    for n, (rid, ref) in enumerate(sorted(refs.items()), 1):
        wanted = {vid for q in todo for vid in cands[q["id"]] if ref_meta[vid][3] == rid}
        if not wanted:
            continue
        if budget_s is not None and time.time() - t_start >= budget_s:
            stopped = True
            break
        t0 = time.time()
        npairs = 0
        for vid, arr, _, _ in _ref_views(rid, ref):
            if vid not in wanted:
                continue
            qs = [q for q in todo if vid in cands[q["id"]]]
            npairs += len(qs)
            volf = np.asarray(arr, dtype=np.float32)
            keys, tpls = [], []
            for q in qs:
                res[q["id"]][vid] = {}
                for k, p in enumerate(q["probes"]):
                    for name, v in dihedral(p["patch"]):
                        keys.append((q, k, name, v))
                        tpls.append(_pool(v, f))
            for (q, k, name, v), (_, x, y, z) in zip(keys, ncc_multi(_pool(volf, f), tpls)):
                fine = _refine(volf, v, x, y, z, f) if x >= 0 else NONE_HIT
                slot = res[q["id"]][vid].setdefault(name, [NONE_HIT] * len(q["probes"]))
                slot[k] = fine
                if np.isfinite(fine[0]):
                    best[q["id"]][vid] = max(best[q["id"]].get(vid, -1.0), fine[0])
            for q in qs:
                rows[q["id"]]["ncc_checked"] += 1
            del volf, arr
        for q in todo:
            _apply_ncc(q, rows[q["id"]], res[q["id"]], best[q["id"]], sp_of, ref_meta, thr)
            rows[q["id"]]["ncc_complete"] = rows[q["id"]]["ncc_checked"] >= rows[q["id"]]["ncc_candidates"]
        if checkpoint:
            _write(rows, checkpoint)
        if verbose:
            print(f"[ncc] {n}/{len(refs)} {rid}: {npairs} cift, {time.time() - t0:.0f}s, "
                  f"toplam {(time.time() - t_start) / 60:.1f} dk", flush=True)
    for q in todo:
        r = rows[q["id"]]
        r["ncc_complete"] = r["ncc_checked"] >= r["ncc_candidates"]
    if stopped:
        print(f"[ncc] SURE BUTCESI DOLDU ({budget_s / 60:.0f} dk): ncc_complete=False satirlar "
              f"eksik tarandi.", flush=True)


def _apply_ncc(q, r, res_q, best_q, sp_of, ref_meta, thr):
    ranked = sorted(best_q.items(), key=lambda t: -t[1])
    if ranked:
        r.update(ncc_best=round(ranked[0][1], 4), ncc_best_ref=ranked[0][0])
    if len(ranked) > 1:
        r.update(ncc_second=round(ranked[1][1], 4), ncc_second_ref=ranked[1][0])
    d = decide_ncc(q, res_q, sp_of, thr)
    if d["verdict"] is None:
        return
    r["n_ncc"] = d["n_ncc"]
    if r["verdict"] == "exact_partial" and d["verdict"] != "ncc_match":
        return      # birebir kismi isabet, kismi NCC'den daha guclu kanit
    r.update({k: v for k, v in d.items() if k != "n_ncc"}, **_ref_fields(ref_meta, d["ref_id"]))


def summarize(df: pd.DataFrame, expected: int | None = None) -> str:
    vc = df.verdict.value_counts()
    lines = [f"{len(df)} sorgu: " + ", ".join(f"{k}={v}" for k, v in vc.items())]
    if expected is not None and len(df) != expected:
        lines.append(f"[!] Beklenen {expected} sorgu, CSV'de {len(df)} -- eksik extract?")
    m = df[df.verdict.isin(["exact", "ncc_match"])]
    if len(m):
        refs = m.ref_id.astype(str).str.split("@").str[0]
        lines.append(f"Eslesen {len(m)} sorgu -> {refs.nunique()} farkli referans: "
                     f"{sorted(refs.unique())[:40]}")
        dup = refs[refs.duplicated()].unique()
        if len(dup):
            lines.append(f"[!] Ayni referansa birden fazla sorgu: {sorted(dup)[:20]}")
    part = df[df.verdict.isin(["exact_partial", "ncc_partial"])]
    if len(part):
        lines.append(f"Kismi (elle bak): {len(part)} -> {sorted(part.query_id)[:20]}")
    rest = df[df.verdict != "exact"]          # exact olanlar NCC'ye gitmez; kapsam onlarda anlamsiz
    lines.append(
        f"Kapsam (exact disi {len(rest)} sorgu): tamsayi olmayan (exact'e kor) "
        f"{int((~rest.integer_valued.astype(bool)).sum())}, NCC adayi yok "
        f"{int((rest.ncc_candidates == 0).sum())}, NCC eksik taranan "
        f"{int((~rest.ncc_complete.astype(bool)).sum())}, yapili yama < 2 {int((rest.n_structured < 2).sum())}.")
    nm = df[df.verdict == "no_match"]
    blind = ((~nm.integer_valued.astype(bool)) & (nm.ncc_candidates == 0)) \
        | ~nm.ncc_complete.astype(bool) | (nm.n_structured < 2)
    lines.append(f"no_match {len(nm)} sorgunun GERCEKTEN denetlenen kismi: {len(nm) - int(blind.sum())} "
                 f"(yalniz bunlar 'kopya yok' kaniti; xy yeniden orneklenmis kopya yine gorulmez).")
    bad_ax = df[~df.q_axcodes.astype(str).str[2].isin(list("SI"))]
    if len(bad_ax):
        lines.append(f"[!] {len(bad_ax)} sorguda 3. eksen S/I degil (aksiyel varsayimi bozuk olabilir).")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="sorgu goruntulerinden probe cikar")
    src = e.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=Path)
    src.add_argument("--glob")
    e.add_argument("--member-regex", default=MEMBER_RE)
    e.add_argument("--out", type=Path, required=True, help="probe klasoru (<id>.npz)")
    e.add_argument("--limit", type=int, default=0, help="prova icin ilk N yeni sorgu")
    s = sub.add_parser("search", help="probe'lari referans hacimlerde ara")
    s.add_argument("--probes", type=Path, required=True)
    s.add_argument("--ref-glob", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--no-ncc", action="store_true")
    s.add_argument("--ncc-budget-min", type=float, default=None)
    s.add_argument("--limit-refs", type=int, default=0, help="prova icin ilk N referans")
    s.add_argument("--expected", type=int, default=None, help="beklenen sorgu sayisi (orn. 842)")
    m = sub.add_parser("summary", help="CSV ozeti")
    m.add_argument("csv", type=Path)
    m.add_argument("--expected", type=int, default=None)
    a = ap.parse_args(argv)

    if a.cmd == "summary":
        print(summarize(pd.read_csv(a.csv), a.expected))
        return

    if a.cmd == "extract":
        done = {f.stem for f in a.out.glob("*.npz")} if a.out.exists() else set()
        if done:
            print(f"[extract] {len(done)} sorgu zaten var, atlaniyor", flush=True)
        errors: list = []
        n = 0
        for q in extract_queries(zip_path=a.zip, glob=a.glob, member_regex=a.member_regex,
                                 skip=done, errors=errors, verbose=True):
            save_query(q, a.out)
            n += 1
            if a.limit and n >= a.limit:
                break
        a.out.mkdir(parents=True, exist_ok=True)
        with open(a.out / "extract_errors.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["query_id", "source", "error"])
            w.writerows(errors)
        total = len(list(a.out.glob("*.npz")))
        members = count_members(a.zip, a.glob, a.member_regex)
        print(f"[extract] yeni {n}, toplam {total}, hata {len(errors)}, kaynakta {members} -> {a.out}")
        if errors:
            raise SystemExit(f"[extract] {len(errors)} hata: {a.out / 'extract_errors.csv'} "
                             f"(bunlar aranmadi -- 'kopya yok' sayilamaz)")
        if total == 0:
            raise SystemExit("Hic sorgu cikmadi: --member-regex / --glob'u kontrol et")
        if not a.limit and total != members:
            raise SystemExit(f"[extract] probe {total} != kaynak {members}")
        return

    from src.data.case_id import assert_unique_ids, glob_paths
    refs = assert_unique_ids(glob_paths(a.ref_glob))
    if a.limit_refs:
        refs = dict(sorted(refs.items())[:a.limit_refs])
    queries = load_queries(a.probes)
    if not refs or not queries:
        raise SystemExit(f"Bos girdi: referans={len(refs)} sorgu={len(queries)}")
    print(f"[search] {len(queries)} sorgu x {len(refs)} referans", flush=True)
    df = search(refs, queries, ncc=not a.no_ncc,
                ncc_budget_s=a.ncc_budget_min * 60 if a.ncc_budget_min is not None else None,
                checkpoint=a.out, verbose=True)
    df.to_csv(a.out, index=False)
    text = summarize(df, a.expected)
    a.out.with_suffix(".summary.txt").write_text(text + "\n")
    print(text)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
