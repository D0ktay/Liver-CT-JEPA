"""LiverJEPA on egitimi: onbellekten yama ornekle, JEPA ya da MAE hedefiyle egit.

    python -m src.jepa.pretrain --objective jepa --profile kucuk \
        --manifest outputs/pretrain_manifest.csv --cache data/cache/iso1mm \
        --out outputs/liverjepa/pre_jepa --steps 20000 --resume

IKI KOL AYNI KOD YOLUNDAN. `--objective jepa|mae` yalnizca hedefi degistirir;
tohum, ornekleme, maske, optimizator, adim sayisi BIREBIR aynidir. Bu desen
`src/jepa/train.py`'den geliyor ve kontrol kolunun anlamli olmasinin tek kosulu.
`--objective none` on egitim YAPMAZ (rastgele baslatma kolu) -- cagrilirsa firlatir.

🔴 KURAL 10: manifest `src/data/pretrain_manifest.py` ile uretilmis ve denetimi
gecmis olmalidir. Bu betik manifest'i OLDUGU GIBI okur; denetimi `scripts/lambda_bootstrap.sh`
`audit` adimi yapar ve kirmiziysa buraya hic gelinmez. Burada ayrica bir kontrol daha var:
`--eval-ids` verilirse, o kimliklerden biri manifest'te GORUNURSE egitim BASLAMAZ.

COKUS (collapse) en olasi basarisizlik modudur ve SESSIZDIR: kayip duserken temsil
sabitlenir. `CollapseMonitor` her `--log-every` adimda RankMe + boyut-basi std olcer,
esik altinda ucuncu ardisik logda `collapse.json` yazip DURUR.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.jepa.cache_build import CacheError, to_model_input
from src.jepa.data import GeometryError, pad_to_patch, sample_patch_origin
from src.jepa.encoder import load_cfg, pick_config
from src.jepa.objective import CollapseMonitor, build_objective, pretrain_step


def cache_paths(manifest: Path, cache) -> list[Path]:
    """Manifest satirlarina karsilik gelen .npz yollari. Eksik olan SESSIZCE atlanmaz.

    `cache` tek dizin ya da dizin listesi olabilir (etiketli LiTS / etiketli Task08 /
    etiketsiz korpus ayri dizinlerde durur); ilk bulunan kullanilir.
    """
    import pandas as pd
    man = pd.read_csv(manifest)
    dirs = [Path(cache)] if isinstance(cache, (str, Path)) else [Path(c) for c in cache]
    out, missing = [], []
    for vid in man["volume_id"].astype(str):
        hit = next((d / f"{vid}.npz" for d in dirs if (d / f"{vid}.npz").exists()), None)
        p = hit if hit is not None else dirs[0] / f"{vid}.npz"
        (out if hit is not None else missing).append(p)
    if missing:
        raise CacheError(
            f"{len(missing)}/{len(man)} hacmin onbellegi yok (orn. {[p.name for p in missing[:5]]}). "
            "Once `scripts/lambda_bootstrap.sh --only cache`. Eksik korpusla egitmek, "
            "raporlanan korpus buyuklugunu yanlis yapar.")
    if not out:
        raise CacheError(f"{dirs} bos -- on egitim verisi yok")
    return out


def assert_cache_spacing(metas: list[dict], target, rtol: float = 0.005) -> None:
    """Onbellekteki ULASILAN spacing'lerin hedefe yakin oldugunu assert et.

    On egitim spacing'i modele ILETMEZ (bir parti farkli hacimlerden gelen yamalardan
    olusur, tek bir spacing gerekir) ve `cfg["spacing_mm"]`'i kullanir. Encoder ise
    mm tabanli sin-cos konum kodlamasi yapiyor -> onbellekteki gercek aralik hedeften
    uzaksa model SESSIZCE yanlis olculer kullanir.

    Tolerans BAGIL: sekil tam sayiya yuvarlandigi icin ulasilan aralik hedeften
    hep biraz sapar. Orn. 512 x 0.700000 mm -> 358 voksel -> 1.001117 mm (%0.11).
    Bu GERCEK LiTS geometrisidir ve reddedilmemelidir; mutlak 1e-3'luk bir esik
    tam da bunu reddederdi (metrics-reviewer bulgusu 2 ile ayni tuzak).
    """
    tg = tuple(float(t) for t in target)
    bad = []
    for m in metas:
        sp = tuple(float(x) for x in m["spacing"])
        if any(abs(a - b) > rtol * b for a, b in zip(sp, tg)):
            bad.append((m.get("id", "?"), sp))
    if bad:
        raise CacheError(
            f"{len(bad)} hacmin onbellek spacing'i hedeften (%{rtol * 100:g}) fazla sapiyor: "
            f"{bad[:5]}. On egitim tek spacing varsayar -- bu hacimleri yeniden onbellege al "
            "ya da korpustan cikar. (mm tabanli konum kodlamasi yanlis olcek kullanir.)")


def assert_no_eval_leak(manifest: Path, eval_ids: Path | None) -> None:
    """Degerlendirme kimlikleri manifest'te GORUNURSE egitim baslamaz (kural 10)."""
    if eval_ids is None:
        print("[UYARI] --eval-ids verilmedi: on egitimin degerlendirme vakalarini gormedigi "
              "BU BETIKTE dogrulanmadi (bootstrap `audit` adimina guveniliyor).")
        return
    import pandas as pd
    d = json.loads(Path(eval_ids).read_text())
    ids = set(d.get("canonical_ids", d) if isinstance(d, dict) else d)
    man = set(pd.read_csv(manifest)["canonical_id"].astype(str))
    bad = sorted(ids & man)
    if bad:
        raise CacheError(
            f"KURAL 10 IHLALI: {len(bad)} degerlendirme vakasi on egitim korpusunda "
            f"(orn. {bad[:5]}). EGITIM BASLAMAZ. Manifest'i --exclude-ids ile yeniden uret.")
    print(f"[on egitim] ✅ {len(ids)} degerlendirme kimliginin hicbiri korpusta yok")


class PatchSampler:
    """Onbellekten rastgele yama. Hacim yamadan kucukse ONCE doldurulur (sessiz kirpma yok)."""

    def __init__(self, paths: list[Path], patch, cfg: dict, seed: int = 0):
        self.paths, self.patch, self.cfg = paths, tuple(int(p) for p in patch), cfg
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, np.ndarray] = {}

    def _vol(self, i: int) -> np.ndarray:
        if i not in self._cache:
            if len(self._cache) > 8:                    # kucuk LRU: RAM patlamasin
                self._cache.pop(next(iter(self._cache)))
            z = np.load(self.paths[i], allow_pickle=False)
            v, _ = pad_to_patch(z["img"], self.patch)
            self._cache[i] = v
        return self._cache[i]

    def batch(self, n: int) -> torch.Tensor:
        xs = []
        for _ in range(n):
            v = self._vol(int(self.rng.integers(0, len(self.paths))))
            o = sample_patch_origin(v.shape, self.patch, self.rng)
            sl = tuple(slice(a, a + b) for a, b in zip(o, self.patch))
            xs.append(to_model_input(v[sl], self.cfg))
        return torch.from_numpy(np.stack(xs))[:, None]


def lr_at(step: int, total: int, cfg: dict) -> float:
    """Warmup + kosinus. `configs/train.yaml -> optim`."""
    o = cfg["optim"]
    lr, lo = float(o["lr"]), float(o["min_lr"])
    w = max(1, int(float(o["warmup_frac"]) * total))
    if step < w:
        return lr * (step + 1) / w
    t = (step - w) / max(1, total - w)
    return lo + 0.5 * (lr - lo) * (1 + math.cos(math.pi * min(1.0, t)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", choices=["jepa", "mae", "none"], required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, nargs="+", required=True,
                    help="bir ya da birden fazla onbellek dizini")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--profile", default="auto")
    ap.add_argument("--gpu-hours", type=float, default=24.0)
    ap.add_argument("--eval-ids", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=0, help="0 -> config pretrain.steps")
    ap.add_argument("--batch", type=int, default=0, help="0 -> config pretrain.batch")
    ap.add_argument("--patch", type=int, default=0,
                    help=">0: profilin yama kenarini ez (16'nin kati). Konum kodlamasi mm tabanli -> guvenli")
    ap.add_argument("--bench", type=int, default=0,
                    help=">0: N adim kosup it/s yazar ve CIKAR (encoder.pt YAZMAZ)")
    ap.add_argument("--ram-gb", type=float, default=96.0)
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help=">0: bu sure asilinca durur ve encoder.pt yazar (kota korumasi)")
    ap.add_argument("--collapse-fatal", action="store_true",
                    help="cokus esiginde DUR. Varsayilan: yalniz UYAR + collapse.json. Esikler "
                         "SEZGISEL (kaynaksiz); tek seferlik gece kosusunda yanlis alarm tum "
                         "kosuyu oldurmesin. Gercek cokus ince ayarda zaten gorunur.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    if a.objective == "none":
        raise SystemExit(
            "--objective none: on egitim YOK (rastgele baslatma kontrol kolu). "
            "Dogrudan `src.jepa.finetune_e2e --init random` cagir.")

    assert_no_eval_leak(a.manifest, a.eval_ids)
    paths = cache_paths(a.manifest, a.cache)
    profile = a.profile if a.profile != "auto" else pick_config(len(paths), a.gpu_hours, a.config)
    cfg = load_cfg(profile, a.config)
    if a.patch:
        if a.patch % 16:
            raise SystemExit("--patch 16'nin kati olmali")
        cfg["patch"] = [a.patch] * 3
    metas = []
    for q in paths:
        m = json.loads(str(np.load(q, allow_pickle=False)["meta"]))
        m["id"] = q.stem
        metas.append(m)
    # 🔴 22 Eyl (kod denetimi, olculdu): ulasilan spacing sapmasi ~0.5/n_cikti. Bir ekseni
    # <100 voksel kalan kucuk-FOV hacimler (TS-v2 kirpilmis seriler: 61 kesit @1.5 mm ->
    # 0.99457) %0.5 esigini asar. Eskiden TEK boyle hacim tum on egitimi dusuruyordu.
    # Artik: o hacimler YUKSEK SESLE dusurulur (dosyaya yazilir), kalan korpus denetlenir.
    rtol, tg = 0.005, [float(t) for t in cfg["spacing_mm"]]
    keep = [i for i, m in enumerate(metas)
            if all(abs(float(x) - t) <= rtol * t for x, t in zip(m["spacing"], tg))]
    if len(keep) < len(paths):
        dusen = [paths[i].stem for i in range(len(paths)) if i not in set(keep)]
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "dropped_spacing.json").write_text(json.dumps(dusen, indent=1))
        print(f"[UYARI] {len(dusen)}/{len(paths)} hacim spacing sapmasi >%{rtol * 100:g} -> korpustan "
              f"DUSURULDU ({a.out / 'dropped_spacing.json'}), orn. {dusen[:5]}", flush=True)
        if len(dusen) > 0.1 * len(paths):
            raise CacheError(f"korpusun %10'undan fazlasi ({len(dusen)}) spacing'den dustu -- "
                             "onbellek bozuk olabilir, egitim BASLAMAZ")
        paths, metas = [paths[i] for i in keep], [metas[i] for i in keep]
    assert_cache_spacing(metas, cfg["spacing_mm"])
    pc = cfg["pretrain"]
    batch = a.batch or int(pc["batch"])
    a.steps = a.bench or a.steps or int(pc["steps"])
    cfg["optim"] = dict(cfg["optim"], lr=float(pc["lr"]))
    print(f"[on egitim] hedef={a.objective} profil={profile} korpus={len(paths)} hacim "
          f"yama={cfg['patch']} batch={batch} adim={a.steps} cihaz={a.device}", flush=True)

    torch.manual_seed(a.seed)
    cuda = a.device.startswith("cuda")
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    bf16 = cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    model = build_objective(a.objective, cfg).to(a.device)
    from src.jepa.finetune_e2e import param_groups
    opt = torch.optim.AdamW(param_groups(model, float(cfg["optim"]["weight_decay"])),
                            lr=float(cfg["optim"]["lr"]))
    a.out.mkdir(parents=True, exist_ok=True)
    mon_dim = model.level_dims[0] if a.objective == "jepa" else model.pred_dim
    mon = CollapseMonitor(cfg, mon_dim, out_dir=a.out)
    from src.jepa.store import UnlabeledSampler, VolumeStore
    sampler = UnlabeledSampler(VolumeStore(paths, cfg["patch"], need_labels=False,
                                           ram_gb=a.ram_gb, seed=a.seed), cfg, batch, seed=a.seed)

    start = 0
    print(f"[tarif] remask_conv={model.remask_conv} drop_tokens={model.drop_tokens} mask_fill={model.mask_fill} vicreg_w={model.vicreg_w} "
          f"seviyeler={model.levels}", flush=True)
    state_p = a.out / "state.pt"
    if a.resume and state_p.exists():
        st = torch.load(state_p, map_location=a.device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
        print(f"[on egitim] {state_p} adim {start}'dan devam")
        lp = a.out / "log.jsonl"
        if lp.exists():                                   # devamda log EZILMESIN
            log0 = [json.loads(x) for x in lp.read_text().splitlines() if x.strip()]

    amp = cuda
    scaler = torch.cuda.amp.GradScaler(enabled=amp and not bf16)
    clip = float(cfg["optim"]["grad_clip"])
    log, t0, run = list(locals().get("log0", [])), time.time(), None
    model.train()
    for step in range(start, a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, cfg)
        x = sampler.batch().to(a.device, non_blocking=True)
        with torch.autocast(device_type="cuda" if amp else "cpu",
                            dtype=amp_dtype, enabled=amp):
            is_log = (step % a.log_every == 0 or step == a.steps - 1) and not a.bench
            out = pretrain_step(model, x, np.random.default_rng(a.seed * 10**6 + step),
                                step, a.steps, with_stats=is_log)
        loss = out["loss"]
        if not torch.isfinite(loss):
            raise SystemExit(f"adim {step}: kayip sonlu degil ({float(loss)}). "
                             "lr'yi yariya indir (--steps sabit) ya da AMP'yi kapat.")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt); scaler.update()
        m = model.ema_step(step, a.steps)
        run = float(loss) if run is None else 0.98 * run + 0.02 * float(loss)
        if a.bench:
            if step == min(10, a.steps // 3):
                if cuda:
                    torch.cuda.synchronize()
                tb, sb = time.time(), step
            if step == a.steps - 1:
                if cuda:
                    torch.cuda.synchronize()
                r = {"it_s": (step - sb) / max(1e-6, time.time() - tb), "batch": batch,
                     "profile": profile, "objective": a.objective,
                     "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None}
                (a.out / "bench.json").write_text(json.dumps(r))
                print("[bench] " + json.dumps(r), flush=True)
                sampler.close()
                return 0
            continue

        if step % a.log_every == 0 or step == a.steps - 1:
            st = out["stats"]
            rec = {"step": step, "loss": round(run, 5), "lr": round(opt.param_groups[0]["lr"], 7),
                   "ema": None if m is None else round(m, 5), "sec": round(time.time() - t0),
                   **{k: round(float(v), 5) for k, v in st.items()}}
            log.append(rec)
            print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            (a.out / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log))
            try:
                mon.check(st)
            except Exception as e:                       # CollapseError
                if a.collapse_fatal:
                    raise SystemExit(f"COKUS: {e}")
                print(f"[UYARI] cokus esigi asildi (DURMUYOR, --collapse-fatal kapali): {e}",
                      flush=True)
                mon.hits = 0
        if (step + 1) % a.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1},
                       state_p)
        if a.max_minutes > 0 and (time.time() - t0) / 60 > a.max_minutes:
            print(f"[on egitim] {a.max_minutes} dk asildi -> adim {step + 1}'de durdu", flush=True)
            break
    sampler.close()

    enc = a.out / "encoder.pt"
    ck = {"cfg": cfg, "encoder": model.encoder.state_dict(), "objective": a.objective,
          "steps": a.steps, "seed": a.seed, "profile": profile,
          "corpus_n": len(paths), "manifest": str(a.manifest)}
    # EK KAYIT (tarif DEGISMEZ: load_encoder yalniz "encoder" anahtarini okur). I-JEPA asagi akista
    # HEDEF (EMA) encoder'i kullanir (arXiv:2301.08243 Ek); state.pt yalniz 1000'in katlarinda
    # yazildigi icin bitisteki EMA agirliklari baska hicbir yerde saklanmiyordu.
    tgt_enc = getattr(model, "target_encoder", None)
    if tgt_enc is not None:
        ck["target_encoder"] = tgt_enc.state_dict()
    torch.save(ck, enc)
    (a.out / "done.json").write_text(json.dumps(
        {"objective": a.objective, "profile": profile, "steps": a.steps, "seed": a.seed,
         "corpus_n": len(paths), "final_loss": run, "minutes": round((time.time() - t0) / 60, 1)},
        indent=1))
    print(f"[on egitim] bitti: {enc} ({(time.time() - t0) / 60:.1f} dk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
