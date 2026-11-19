"""LiverJEPA ince ayari: encoder + decoder UCTAN UCA, karaciger bolgesi + tumor.

    python -m src.jepa.finetune_e2e --init outputs/liverjepa/pre_jepa/encoder.pt \
        --cache data/cache/iso1mm --test-ids outputs/internal_test.json \
        --out outputs/liverjepa/ft_jepa --steps 80000

    python -m src.jepa.finetune_e2e --init random ...        # (a) kontrol kolu

UC KOL AYNI KOD YOLUNDAN, tek fark `--init` (random | pre_mae/encoder.pt | pre_jepa/encoder.pt).
Ayni tohum, ayni adim sayisi, ayni ornekleme DAGILIMI, ayni takvim (dondurma asamasi dahil).

22 EYL 2026 -- MVE TESHISINDEN SONRA DEGISENLER (gerekce: src/jepa/region.py, store.py):
  1. TAM HACIMDE egitim (karaciger kutusu degil) + karaciger/tumor/rastgele ornekleme.
  2. Iki kanal: karaciger bolgesi + tumor. Tumor tahmini modelin KENDI karaciger
     tahminiyle sinirlanir (GT'ye dokunulmaz).
  3. GPU augmentasyonu (src/jepa/augment.py) -- onceki kosuda HIC yoktu.
  4. RAM deposu + arka plan ornekleme; toplu cikarim + ayna TTA.
  5. Dogrulama egrisi: EGITIM vakalarindan ayrilan kucuk bir kume (`--val-n`).
     Dondurulmus TEST'e egitim boyunca BAKILMAZ.

MODEL SECIMI YOK: test tahmini SON kontrol noktasindan uretilir (sabit adim, onceden
bildirilmis). `best.pt` ve `val.jsonl` yalniz TANI icindir -- 8 vakalik dogrulama ile
kontrol noktasi secmek gurultuye uyum saglar. Esik 0.5 sabit.

Tahmin yolu: onbellek -> predict_volume_fast -> postprocess (kendi karaciger maskesi) ->
cache_build.reconstruct (orijinal izgara, yon) -> GT'nin affine/basligiyla NIfTI. Boylece
`src.metrics.evaluate` TS tabaniyla AYNI kodla olcer (--gt-label 2 --pred-label 1 AYRI).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.jepa.augment import augment
from src.jepa.cache_build import CacheError, reconstruct, to_hu
from src.jepa.decoder import SegModel
from src.jepa.encoder import LiverEncoder, load_cfg, load_encoder
from src.jepa.region import (N_REGION, dice, postprocess, predict_volume_fast, region_ds_loss,
                             voxel_valid)
from src.jepa.store import LabeledSampler, VolumeStore


def read_ids(path: Path) -> list[str]:
    d = json.loads(Path(path).read_text())
    ids = d.get("case_ids", d) if isinstance(d, dict) else d
    if not ids:
        raise CacheError(f"{path}: vaka listesi bos")
    return [str(x) for x in ids]


def split_ids(cache: Path, test_ids: list[str]) -> tuple[list[str], list[str]]:
    """Onbellekteki vakalari egitim / dondurulmus test diye ayir.

    Test kimliklerinden biri onbellekte YOKSA firlatir: eksik bir test vakasi
    sessizce raporun disinda kalir ve sonuc oldugundan iyi gorunur.
    """
    have = sorted(p.stem for p in cache.glob("*.npz"))
    missing = [c for c in test_ids if c not in have]
    if missing:
        raise CacheError(f"{len(missing)} test vakasinin onbellegi yok: {missing[:5]}")
    train = [c for c in have if c not in set(test_ids)]
    if not train:
        raise CacheError("egitim vakasi kalmadi -- test kumesi tum onbellegi kapsiyor")
    return train, [c for c in have if c in set(test_ids)]


def build_model(init: str, cfg: dict, key: str = "encoder") -> SegModel:
    """`random` -> rastgele encoder; aksi halde ckpt'ten strict yukleme.

    strict=True bilerek: bizim encoder'imiza dis bir agirligi (orn. Rad-JEPA) kismen
    yuklemeye calismak, hangi katmanin egitimli hangisinin rastgele oldugunu belirsiz
    birakir ve karsilastirmayi anlamsiz yapar. Dis referanslar AYRI koldur.
    """
    if str(init).lower() == "random":
        return SegModel(cfg, encoder=LiverEncoder(cfg), n_classes=N_REGION)
    enc = load_encoder(init, cfg=cfg, strict=True, key=key)
    return SegModel(cfg, encoder=enc, n_classes=N_REGION)


def split_val(train_ids: list[str], n_val: int, seed: int = 1234) -> tuple[list[str], list[str]]:
    """Egitim kimliklerinden deterministik kucuk dogrulama kumesi (TEST'e dokunmaz)."""
    if n_val <= 0:
        return train_ids, []
    if n_val >= len(train_ids):
        raise CacheError(f"val_n={n_val} >= egitim vakasi {len(train_ids)}")
    ids = sorted(train_ids)
    pick = set(np.random.default_rng(seed).choice(len(ids), n_val, replace=False).tolist())
    return [c for i, c in enumerate(ids) if i not in pick], [c for i, c in enumerate(ids) if i in pick]


def lr_sawtooth(step: int, total: int, freeze: int, cfg: dict) -> float:
    """Iki asamali isinma (Wald ve ark. arXiv:2410.23132 'sawtooth'): once yalniz decoder
    0->lr, sonra TUM ag yeniden 0->lr + kosinus. On egitimli encoder'i ilk adimlarda
    rastgele decoder'in gradyaniyla bozmamak icin. Rastgele kol AYNI takvimi alir."""
    o = cfg["optim"]
    lr, lo = float(o["lr"]), float(o["min_lr"])
    if step < freeze:
        return lr * (step + 1) / max(1, freeze)
    s, n = step - freeze, max(1, total - freeze)
    w = max(1, int(float(o["warmup_frac"]) * n))
    if s < w:
        return lr * (s + 1) / w
    return lo + 0.5 * (lr - lo) * (1 + math.cos(math.pi * min(1.0, (s - w) / max(1, n - w))))


def param_groups(model, wd: float) -> list[dict]:
    """Norm/bias/1-B parametrelere agirlik curumesi UYGULANMAZ."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    return [{"params": decay, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}]


@torch.no_grad()
def validate(model, cache: Path, ids: list[str], cfg: dict, device: str, amp_dtype) -> dict:
    """Dogrulama: tam hacim, son isleme dahil, 1 mm onbellek izgarasinda (hizli tani)."""
    was = model.training
    model.eval()
    rows = []
    for cid in ids:
        z = np.load(cache / f"{cid}.npz", allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        prob = predict_volume_fast(model, to_hu(z["img"], cfg), meta["spacing"], cfg,
                                   device=device, tta=False, amp_dtype=amp_dtype)
        tum_p, info = postprocess(prob, spacing=meta["spacing"])
        rows.append({"id": cid, "tumor_dice": dice(tum_p > 0.5, z["lab"]),
                     "tumor_dice_raw": dice(prob[1] > 0.5, z["lab"]),
                     "liver_dice": dice(prob[0] > 0.5, z["liver"]) if "liver" in z else float("nan"),
                     "outside_liver_frac": info["outside_liver_frac"]})
    model.train(was)
    agg = {k: float(np.nanmean([r[k] for r in rows])) if rows else float("nan")
           for k in ("tumor_dice", "tumor_dice_raw", "liver_dice", "outside_liver_frac")}
    agg["n"] = len(rows)
    agg["cases"] = rows
    return agg


def main(argv: list[str] | None = None) -> int:
    import nibabel as nib

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="'random' ya da encoder.pt yolu")
    ap.add_argument("--init-key", default="encoder", choices=("encoder", "target_encoder"),
                    help="encoder.pt icindeki agirlik: cevrimici (varsayilan, on-kayitli tarif) ya da EMA hedef")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--extra-cache", type=Path, nargs="*", default=[],
                    help="EK etiketli egitim onbellekleri (orn. MSD Task08). Tamami EGITIME girer; "
                         "test/dogrulama yalniz --cache'ten gelir.")
    ap.add_argument("--test-ids", type=Path, required=True,
                    help="src.data.frozen_test ciktisi (DONDURULMUS ic test)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--profile", default="orta")
    ap.add_argument("--steps", type=int, default=0, help="0 -> config finetune.steps")
    ap.add_argument("--batch", type=int, default=0, help="0 -> config finetune.batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-n", type=int, default=-1, help="-1 -> config finetune.val_n")
    ap.add_argument("--val-every", type=int, default=0, help="0 -> config finetune.val_every")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help=">0: egitim bu sureyi asinca DURUR ve tahmine gecer (kota korumasi). "
                         "Kosinus takvimi o zaman yarim kalir -- done.json'a yazilir.")
    ap.add_argument("--patch", type=int, default=0,
                    help=">0: profilin yama kenarini ez (16'nin kati). Konum kodlamasi mm tabanli -> guvenli")
    ap.add_argument("--bench", type=int, default=0,
                    help=">0: N adim kosup it/s yazar ve CIKAR (kayit/tahmin yok). Butce planlamasi icin.")
    ap.add_argument("--ram-gb", type=float, default=64.0)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--predict-only", action="store_true", help="last.pt'den yalniz tahmin")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gt-dir", type=Path, default=Path("data/raw/LiTS"),
                    help="GT segmentasyonlari (affine/baslik icin)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_cfg(a.profile, a.config)
    a.patch = a.patch or int((cfg["finetune"].get("patch_by_profile") or {}).get(a.profile, 0))
    if a.patch:
        if a.patch % 16:
            raise SystemExit("--patch 16'nin kati olmali")
        cfg["patch"] = [a.patch] * 3
    fc = cfg["finetune"]
    steps = a.steps or int(fc["steps"])
    batch = a.batch or int(fc["batch"])
    val_n = int(fc["val_n"]) if a.val_n < 0 else a.val_n
    val_every = a.val_every or int(fc["val_every"])
    freeze = min(int(fc["freeze_encoder_steps"]), steps // 4)
    cfg["optim"] = dict(cfg["optim"], lr=float(fc["lr"]))
    if a.bench:
        steps, val_n, freeze = a.bench, 0, 0
    test_ids = read_ids(a.test_ids)
    train_ids, test_ids = split_ids(a.cache, test_ids)
    train_ids, val_ids = split_val(train_ids, val_n)
    paths = [a.cache / f"{c}.npz" for c in train_ids]
    n_extra = 0
    for ec in a.extra_cache:
        ep = sorted(ec.glob("*.npz"))
        if not ep:
            raise CacheError(f"--extra-cache {ec} bos")
        cak = {q.stem for q in ep} & set(test_ids + val_ids + train_ids)
        if cak:
            raise CacheError(f"--extra-cache kimlikleri ana onbellekle cakisiyor: {sorted(cak)[:5]}")
        paths += ep
        n_extra += len(ep)
    arm = "random" if str(a.init).lower() == "random" else Path(a.init).parent.name
    print(f"[ince ayar] kol={arm} profil={a.profile} egitim={len(train_ids)}+{n_extra} "
          f"dogrulama={len(val_ids)} dondurulmus test={len(test_ids)} adim={steps} batch={batch} "
          f"yama={cfg['patch']} cihaz={a.device}", flush=True)

    torch.manual_seed(a.seed)
    cuda = a.device.startswith("cuda")
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    bf16 = cuda and torch.cuda.is_bf16_supported()
    amp_dtype = (torch.bfloat16 if bf16 else torch.float16) if cuda else None
    model = build_model(a.init, cfg, key=a.init_key).to(a.device)
    opt = torch.optim.AdamW(param_groups(model, float(cfg["optim"]["weight_decay"])),
                            lr=float(cfg["optim"]["lr"]))
    a.out.mkdir(parents=True, exist_ok=True)
    state_p, last_p, best_p = a.out / "state.pt", a.out / "last.pt", a.out / "best.pt"

    start, best, vlog, log, stopped_early = 0, -1.0, [], [], False
    if a.resume and state_p.exists():
        st = torch.load(state_p, map_location=a.device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start, best = st["step"], st.get("best", -1.0)
        vlog, log = st.get("vlog", []), st.get("log", [])
        print(f"[ince ayar] adim {start}'dan devam", flush=True)

    t0, run = time.time(), None
    if not a.predict_only and start < steps:
        store = VolumeStore(paths, cfg["patch"], need_labels=True, ram_gb=a.ram_gb, seed=a.seed,
                            fill_liver_holes=bool(fc.get("liver_only_fill_holes", False)))
        sampler = LabeledSampler(store, cfg, batch, seed=a.seed,
                                 p_tumor=float(fc["p_tumor"]), p_liver=float(fc["p_liver"]),
                                 group_probs=fc["group_probs"])
        print(f"  [ornekleme] gruplar: {store.group_counts()} -> {sampler.groups}", flush=True)
        scaler = torch.cuda.amp.GradScaler(enabled=cuda and not bf16)
        clip = float(cfg["optim"]["grad_clip"])
        neg_margin = int(fc.get("liver_only_tumor_neg_margin_mm", -1))   # -1 -> kapali (eski tarif)
        print(f"  [tarif] liver_only tumor-negatif marji: {neg_margin} mm (-1 = kapali)", flush=True)
        gen = torch.Generator().manual_seed(a.seed + start)      # CPU: bkz. augment.py
        model.train()
        frozen = None
        for step in range(start, steps):
            want_frozen = step < freeze
            if want_frozen != frozen:
                for q in model.encoder.parameters():
                    q.requires_grad_(not want_frozen)
                frozen = want_frozen
                print(f"  [takvim] adim {step}: encoder {'DONDURULDU' if frozen else 'ACILDI'}",
                      flush=True)
            lr = lr_sawtooth(step, steps, freeze, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y, v = sampler.batch()
            x, y, v = (x.to(a.device, non_blocking=True), y.to(a.device, non_blocking=True),
                       v.to(a.device, non_blocking=True))
            if not a.no_aug:
                x, y = augment(x, y, gen, cfg.get("augment"))
            if neg_margin >= 0:
                v = voxel_valid(y, v, neg_margin)          # augment SONRASI: maske y ile hizali
            with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=amp_dtype,
                                enabled=cuda):
                loss = region_ds_loss(model(x), y, v, cfg)
            if not torch.isfinite(loss):
                raise SystemExit(f"adim {step}: kayip sonlu degil ({float(loss)})")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update()
            run = float(loss) if run is None else 0.98 * run + 0.02 * float(loss)
            if a.bench:
                if step == min(10, steps // 3):              # isinma (cudnn benchmark) haric
                    if cuda:
                        torch.cuda.synchronize()
                    tb, sb = time.time(), step
                if step == steps - 1:
                    if cuda:
                        torch.cuda.synchronize()
                    r = {"it_s": (step - sb) / max(1e-6, time.time() - tb), "batch": batch,
                         "patch": list(cfg["patch"]), "profile": a.profile,
                         "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None}
                    (a.out / "bench.json").write_text(json.dumps(r))
                    print("[bench] " + json.dumps(r), flush=True)
                    sampler.close()
                    return 0
                continue
            if step % a.log_every == 0 or step == steps - 1:
                dt = time.time() - t0
                rec = {"step": step, "loss": round(run, 5), "lr": round(lr, 7),
                       "sec": round(dt), "it_s": round((step - start + 1) / max(dt, 1e-6), 2)}
                log.append(rec)
                print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
                (a.out / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log))
            if val_ids and ((step + 1) % val_every == 0 or step == steps - 1):
                vr = validate(model, a.cache, val_ids, cfg, a.device, amp_dtype)
                vr["step"] = step + 1
                vlog.append(vr)
                (a.out / "val.jsonl").write_text("\n".join(json.dumps(r) for r in vlog))
                print(f"  [dogrulama] adim {step + 1}: tumor Dice {vr['tumor_dice']:.3f} "
                      f"(ham {vr['tumor_dice_raw']:.3f}) karaciger {vr['liver_dice']:.3f} "
                      f"karaciger-disi pay {vr['outside_liver_frac']:.3f}", flush=True)
                if vr["tumor_dice"] == vr["tumor_dice"] and vr["tumor_dice"] > best:
                    best = vr["tumor_dice"]
                    torch.save({"model": model.state_dict(), "step": step + 1, "val": best}, best_p)
            if (step + 1) % 1000 == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step + 1, "best": best, "vlog": vlog, "log": log}, state_p)
            if a.max_minutes > 0 and (time.time() - t0) / 60 > a.max_minutes:
                stopped_early = True
                print(f"  [sure] {a.max_minutes} dk asildi -> adim {step + 1}'de egitim durdu", flush=True)
                break
        sampler.close()
        torch.save({"model": model.state_dict(), "step": step + 1, "cfg_profile": a.profile}, last_p)
        sampler_counts = dict(sampler.counts)
    else:
        if not last_p.exists():
            raise CacheError(f"{last_p} yok -- once egit")
        model.load_state_dict(torch.load(last_p, map_location=a.device)["model"])
        sampler_counts = None
    train_min = (time.time() - t0) / 60

    # ---- tahmin: SADECE burada, tek seferde, dondurulmus test uzerinde, SON agirliklarla ----
    model.eval()
    pred_dir = a.out / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    infos = {}
    for cid in test_ids:
        z = np.load(a.cache / f"{cid}.npz", allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        prob = predict_volume_fast(model, to_hu(z["img"], cfg), meta["spacing"], cfg,
                                   device=a.device, tta=not a.no_tta, amp_dtype=amp_dtype)
        tum_p, info = postprocess(prob, a.thr, spacing=meta["spacing"])
        if info["liver_vox"] == 0:
            print(f"[UYARI] {cid}: model karaciger BULAMADI -> tumor tahmini bos", flush=True)
        infos[cid] = info
        full = reconstruct(tum_p, meta)
        n = int(cid.split("-")[1]) if cid.startswith("lits-") else None
        ref_p = a.gt_dir / (f"segmentation-{n}.nii" if n is not None else f"{cid}.nii.gz")
        if not ref_p.exists():
            raise CacheError(f"{cid}: GT dosyasi yok ({ref_p}) -- affine/baslik alinamaz")
        ref = nib.load(str(ref_p))
        if tuple(ref.shape[:3]) != full.shape:
            raise CacheError(f"{cid}: tahmin {full.shape} != GT {ref.shape}")
        img = nib.Nifti1Image((full > a.thr).astype(np.uint8), ref.affine, header=ref.header.copy())
        img.set_data_dtype(np.uint8)
        nib.save(img, str(pred_dir / f"{cid}.nii.gz"))
    (a.out / "pred_info.json").write_text(json.dumps(infos, indent=1))
    (a.out / "done.json").write_text(json.dumps(
        {"arm": arm, "init": str(a.init), "profile": a.profile, "steps": steps,
         "steps_done": (step + 1) if sampler_counts is not None else None,
         "stopped_early": stopped_early, "batch": batch, "seed": a.seed,
         "n_train": len(train_ids), "n_extra": n_extra, "val_ids": val_ids,
         "n_test": len(test_ids), "final_loss": run, "thr": a.thr, "tta": not a.no_tta,
         "best_val_tumor_dice": best, "checkpoint_used": "last", "init_key": a.init_key,
         "sampler_counts": sampler_counts, "train_minutes": round(train_min, 1),
         "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    print(f"[ince ayar] {len(test_ids)} tahmin -> {pred_dir} ({(time.time() - t0) / 60:.1f} dk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
