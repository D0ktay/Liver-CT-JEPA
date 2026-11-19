"""RAM'de hacim deposu + yama ornekleyiciler (on egitim ve ince ayar ortak kullanir).

NEDEN: MVE kosusunda (21 Eyl 2026) ornekleyici her adimda SIKISTIRILMIS .npz aciyordu
(8 hacimlik LRU). Tam hacimde bu ~0.4 sn/hacim demek -> H100 bos bekler. Burada:
  * Tum hacimler baslangicta cok is parcacigiyla RAM'e acilir (uint8: ~66 MB/hacim).
  * RAM butcesi (`ram_gb`) asilirsa rastgele bir ALT KUME yerlesik tutulur ve arka
    planda surekli tazelenir (her hacim er ya da gec gorulur) -- sessizce korpusun
    bir kismini atmak YOK; yerlesik/total sayisi log'a yazilir.
  * Yama uretimi arka plan is parcaciginda, kuyrukla -> GPU beklemez.

ORNEKLEME (ince ayar) -- 22 Eyl teshisinin duzeltmesi:
  eski: yamalarin 2/3'u tumor merkezli, hepsi KARACIGER KUTUSUNDAN -> model karaciger
  disini hic gormedi, tum govdede test edilince her yere "tumor" dedi (olculdu:
  tahmin voksellerinin cogu karaciger DISINDA).
  yeni: TAM HACIMDEN; p_tumor tumor merkezli, p_liver karaciger merkezli, kalani
  govdenin RASTGELE yeri (nnU-Net: %33 on plan + %67 rastgele).
"""
from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

from src.jepa.cache_build import CacheError, to_model_input
from src.jepa.data import pad_to_patch, sample_patch_origin

N_COORDS = 20000          # sinif basina saklanan odak vokseli (tamamini tutmak RAM israfi)


def load_case(path: Path, patch, need_labels: bool, rng: np.random.Generator,
              fill_liver_holes: bool = False) -> dict:
    """.npz -> RAM kaydi. Etiketler TEK uint8'e paketlenir: 0 arka, 1 karaciger, 2 tumor."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    img, _ = pad_to_patch(z["img"], patch)             # uint8 0 = pencere alt ucu = hava
    rec = {"id": path.stem, "img": img, "meta": meta, "nbytes": img.nbytes}
    if not need_labels:
        return rec
    has_liver, has_tumor = "liver" in z, "lab" in z
    if not (has_liver or has_tumor):
        raise CacheError(f"{path.stem}: onbellekte ne 'lab' ne 'liver' var -- cache_build --labels-glob")
    seg = np.zeros(img.shape, np.uint8)
    if has_liver:
        liv, _ = pad_to_patch(z["liver"], patch)
        liv = liv > 0
        if fill_liver_holes and not has_tumor and liv.any():
            # liver_only (TS-v2 / AMOS): kanal 0 = karaciger ∪ tumor. Organ etiketi ara sira kucuk
            # bir ic lezyonu/kisti DISLIYOR (olculdu: TS'te <=3.3 ml kapali delik) -> 3B delik
            # doldurma HACIM duzeyinde, bir kez (yama icinde yapilsa yama siniri deligi acik gosterir).
            ii = [np.flatnonzero(liv.any(axis=ax)) for ax in ((1, 2), (0, 2), (0, 1))]
            box = tuple(slice(int(i[0]), int(i[-1]) + 1) for i in ii)
            liv[box] = ndimage.binary_fill_holes(liv[box])
        seg[liv] = 1
    if has_tumor:
        tum, _ = pad_to_patch(z["lab"], patch)
        seg[tum > 0] = 2
    # UC VAKA TURU: full (LiTS: karaciger+tumor) · tumor_only (Task08) · liver_only (TS-v2, AMOS)
    group = "full" if (has_liver and has_tumor) else ("tumor_only" if has_tumor else "liver_only")
    rec.update(seg=seg, has_liver=has_liver, has_tumor=has_tumor, group=group,
               nbytes=img.nbytes + seg.nbytes)
    for name, val in (("c_tumor", 2), ("c_liver", 1)):
        idx = np.flatnonzero(seg == val)
        if len(idx) > N_COORDS:
            idx = rng.choice(idx, N_COORDS, replace=False)
        rec[name] = np.stack(np.unravel_index(idx, seg.shape), 1).astype(np.int32)
    return rec


class VolumeStore:
    """Hacimleri RAM'de tutar; butce asilirsa yerlesik alt kumeyi arka planda tazeler."""

    def __init__(self, paths: list[Path], patch, need_labels: bool, ram_gb: float = 64.0,
                 seed: int = 0, threads: int = 8, verbose: bool = True,
                 fill_liver_holes: bool = False):
        self.fill_liver_holes = bool(fill_liver_holes)
        if not paths:
            raise CacheError("VolumeStore: hacim listesi bos")
        self.paths, self.patch, self.need_labels = list(paths), tuple(int(p) for p in patch), need_labels
        self.rng = np.random.default_rng(seed)
        self.budget = float(ram_gb) * 2 ** 30
        self.lock = threading.Lock()
        self.res: dict[int, dict] = {}
        order = list(self.rng.permutation(len(self.paths)))
        used = 0

        def _ld(i):
            return i, load_case(self.paths[i], self.patch, need_labels,
                                np.random.default_rng(seed * 100003 + int(i)),
                                fill_liver_holes=self.fill_liver_holes)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for k in range(0, len(order), threads):
                if used > self.budget:
                    break
                for i, rec in ex.map(_ld, order[k:k + threads]):
                    self.res[int(i)] = rec
                    used += rec["nbytes"]
        self.partial = len(self.res) < len(self.paths)
        self._stop = False
        if verbose:
            print(f"[depo] {len(self.res)}/{len(self.paths)} hacim RAM'de ({used / 2**30:.1f} GB)"
                  + (" -- KISMI: arka planda tazeleniyor" if self.partial else ""), flush=True)
        if self.partial:
            self._t = threading.Thread(target=self._refresh, args=(seed,), daemon=True)
            self._t.start()

    def _refresh(self, seed: int) -> None:
        rng = np.random.default_rng(seed + 991)
        while not self._stop:
            out = [i for i in range(len(self.paths)) if i not in self.res]
            if not out:
                return
            i = int(rng.choice(out))
            try:
                rec = load_case(self.paths[i], self.patch, self.need_labels, rng,
                                fill_liver_holes=self.fill_liver_holes)
            except Exception as e:                       # noqa: BLE001 -- parcacik SESSIZCE olmesin
                print(f"[UYARI] depo tazeleme: {self.paths[i].name} okunamadi ({e}) -- atlandi",
                      flush=True)
                time.sleep(1.0)
                continue
            with self.lock:
                drop = int(rng.choice(list(self.res)))
                self.res.pop(drop)
                self.res[i] = rec
            time.sleep(0.2)                              # CPU'yu ornekleyiciye birak

    def pick(self, rng: np.random.Generator, group: str | None = None) -> dict:
        """Rastgele yerlesik hacim. `group` verilirse o turden (yoksa tum havuzdan)."""
        with self.lock:
            keys = list(self.res)
            if group is not None:
                sub = [k for k in keys if self.res[k].get("group") == group]
                keys = sub or keys
            return self.res[keys[int(rng.integers(0, len(keys)))]]

    def group_counts(self) -> dict:
        with self.lock:
            out: dict = {}
            for r in self.res.values():
                out[r.get("group")] = out.get(r.get("group"), 0) + 1
            return out

    def close(self) -> None:
        self._stop = True


class _Prefetch:
    """`make()` ciktisini arka planda uretip kuyruga koyar. Hata ANA is parcaciginda firlar."""

    def __init__(self, make, depth: int = 6, workers: int = 3):
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = False
        self._ts = [threading.Thread(target=self._run, args=(make, w), daemon=True)
                    for w in range(workers)]
        for t in self._ts:
            t.start()

    def _run(self, make, w: int) -> None:
        while not self._stop:
            try:
                item = make(w)
            except Exception as e:                       # noqa: BLE001 -- ana parcaciga tasi
                item = e
            while not self._stop:
                try:
                    self.q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue
            if isinstance(item, Exception):
                return

    def get(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self._stop = True


class LabeledSampler:
    """Ince ayar yamalari: x (B,1,D,H,W) float32 [-1,1] · y (B,2,...) {0,1} · valid (B,2).

    y[:,0] = karaciger BOLGESI (karaciger ∪ tumor) · y[:,1] = tumor.
    valid[:,0] = 0 -> bu vakada karaciger etiketi yok (MSD Task08) -> o kanalin kaybi kapali.
    """

    def __init__(self, store: VolumeStore, cfg: dict, batch: int, seed: int = 0,
                 p_tumor: float = 0.4, p_liver: float = 0.3, prefetch: bool = True,
                 p_primary: float | None = None, group_probs: dict | None = None):
        if not (0 <= p_tumor and 0 <= p_liver and p_tumor + p_liver <= 1.0):
            raise ValueError(f"p_tumor + p_liver <= 1 olmali: {p_tumor}, {p_liver}")
        self.store, self.cfg, self.bs = store, cfg, int(batch)
        self.p_tumor, self.p_liver = float(p_tumor), float(p_liver)
        self.patch = store.patch
        self.seed = seed
        # KAYNAK DENGESI (kod denetimi, 22 Eyl): hacim DUZGUN secilirse ~96 LiTS + 303 Task08
        # ile yamalarin ~%76'si Task08'den gelir -- karaciger etiketi YOK ve z'de karacigere
        # kirpilmis. Karaciger-disi negatifler (onceki kosunun olum nedeni) yamalarin ~%7'sine
        # duserdi ve tum tumor ciktisini kapilayan karaciger kanali ac kalirdi.
        # p_primary = karaciger etiketli (LiTS) hacimden ornekleme olasiligi.
        # group_probs = {"full": .., "tumor_only": .., "liver_only": ..}; depoda OLMAYAN gruplar
        # atilir ve kalanlar yeniden normalize edilir. (Geriye uyum: p_primary -> full vs digerleri.)
        have = store.group_counts()
        gp = dict(group_probs or {})
        if not gp and p_primary is not None and "full" in have and len(have) > 1:
            rest = [g for g in have if g != "full"]
            gp = {"full": float(p_primary), **{g: (1 - float(p_primary)) / len(rest) for g in rest}}
        gp = {g: float(v) for g, v in gp.items() if g in have and float(v) > 0}
        tot = sum(gp.values())
        self.groups = [(g, v / tot) for g, v in gp.items()] if tot > 0 and len(have) > 1 else None
        self.counts = {"tumor": 0, "liver": 0, "random": 0, "full": 0, "tumor_only": 0, "liver_only": 0}
        self._rngs: dict[int, np.random.Generator] = {}
        self._pf = _Prefetch(self._make) if prefetch else None

    def _one(self, rng):
        if self.groups is None:
            rec = self.store.pick(rng)
        else:
            u0, acc, g = rng.random(), 0.0, self.groups[-1][0]
            for name, pr in self.groups:
                acc += pr
                if u0 < acc:
                    g = name
                    break
            rec = self.store.pick(rng, group=g)
        self.counts[rec["group"]] += 1
        u, focus, mode = rng.random(), None, "random"
        if u < self.p_tumor and len(rec["c_tumor"]):
            focus, mode = rec["c_tumor"][int(rng.integers(0, len(rec["c_tumor"])))], "tumor"
        elif u < self.p_tumor + self.p_liver and len(rec["c_liver"]):
            focus, mode = rec["c_liver"][int(rng.integers(0, len(rec["c_liver"])))], "liver"
        self.counts[mode] += 1
        o = sample_patch_origin(rec["img"].shape, self.patch, rng,
                                focus=None if focus is None else tuple(int(v) for v in focus))
        sl = tuple(slice(a, a + b) for a, b in zip(o, self.patch))
        seg = rec["seg"][sl]
        y = np.stack([(seg > 0), (seg == 2)]).astype(np.float32)
        return (to_model_input(rec["img"][sl], self.cfg), y,
                np.array([1.0 if rec["has_liver"] else 0.0,
                          1.0 if rec["has_tumor"] else 0.0], np.float32))

    def _make(self, w: int = 0):
        import torch
        rng = self._rngs.setdefault(w, np.random.default_rng(self.seed * 7919 + w))
        xs, ys, vs = zip(*[self._one(rng) for _ in range(self.bs)])
        return (torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ys)),
                torch.from_numpy(np.stack(vs)))

    def batch(self):
        return self._pf.get() if self._pf is not None else self._make(0)

    def close(self) -> None:
        if self._pf is not None:
            self._pf.close()
        self.store.close()


class UnlabeledSampler:
    """On egitim yamalari. Cogu HAVA olan yama reddedilir (tam hacimde masa/hava bol)."""

    def __init__(self, store: VolumeStore, cfg: dict, batch: int, seed: int = 0,
                 min_body_frac: float = 0.35, air_u8: int = 40, prefetch: bool = True):
        self.store, self.cfg, self.bs, self.seed = store, cfg, int(batch), seed
        self.patch = store.patch
        self.min_body, self.air = float(min_body_frac), int(air_u8)
        self._rngs: dict[int, np.random.Generator] = {}
        self._pf = _Prefetch(self._make) if prefetch else None

    def _one(self, rng):
        for _ in range(20):
            rec = self.store.pick(rng)
            o = sample_patch_origin(rec["img"].shape, self.patch, rng)
            sl = tuple(slice(a, a + b) for a, b in zip(o, self.patch))
            p = rec["img"][sl]
            # seyrek ornekle: tam yamayi taramak gereksiz
            if float((p[::4, ::4, ::4] > self.air).mean()) >= self.min_body:
                break
        return to_model_input(p, self.cfg)

    def _make(self, w: int = 0):
        import torch
        rng = self._rngs.setdefault(w, np.random.default_rng(self.seed * 7919 + w))
        return torch.from_numpy(np.stack([self._one(rng) for _ in range(self.bs)]))[:, None]

    def batch(self):
        return self._pf.get() if self._pf is not None else self._make(0)

    def close(self) -> None:
        if self._pf is not None:
            self._pf.close()
        self.store.close()
