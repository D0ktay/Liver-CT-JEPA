"""Kol topluluğu: birden fazla kolun (none / jepa / mae) SON kontrol noktalarindan olasilik ORTALAMASI.

Neden: farkli baslangicli modellerin olasilik ortalamasi tekil modelden genelde +1-2 Dice verir
(nnU-Net 5-kat toplulugu). Burada kollar AYNI veri/tarif, farkli encoder baslangici.

Sizinti kurali: her kolun `last.pt`'si (test sonucuna bakilarak secilmis "en iyi" DEGIL), esik 0.5,
son isleme tekil kolla AYNI (postprocess: modelin kendi karaciger olasiliginin ortalamasi). Kol
secimi/agirligi YOK -- esit agirlik, onceden ilan. Ciktilar ayri klasore yazilir; kol klasorlerine
DOKUNULMAZ.

    python -m src.jepa.ensemble --arms outputs/night/ft_none outputs/night/ft_jepa \
        --cache data/cache/lits --test-ids outputs/internal_test.json --gt-dir data/gt_fixed \
        --out outputs/night/ens_none_jepa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.jepa.cache_build import CacheError, reconstruct, to_hu
from src.jepa.encoder import load_cfg
from src.jepa.finetune_e2e import build_model, read_ids
from src.jepa.region import postprocess, predict_volume_fast


def average_probs(probs: list[np.ndarray]) -> np.ndarray:
    """Esit agirlikli ortalama; sekil/kanal uyusmazsa FIRLATIR (sessiz yayin yok)."""
    if not probs:
        raise ValueError("bos kol listesi")
    shp = probs[0].shape
    for i, p in enumerate(probs):
        if p.shape != shp:
            raise ValueError(f"kol {i} sekli {p.shape} != {shp}")
    return np.mean(np.stack([np.asarray(p, np.float32) for p in probs], 0), 0)


def main(argv=None) -> int:
    import nibabel as nib

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", type=Path, nargs="+", required=True, help="ft_* klasorleri (last.pt icermeli)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--test-ids", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--profile", default="orta")
    ap.add_argument("--patch", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if len(a.arms) < 2:
        raise SystemExit("topluluk icin en az 2 kol")
    for d in a.arms:
        if not (d / "last.pt").exists():
            raise SystemExit(f"{d}/last.pt yok -- kol bitmemis (best.pt KULLANILMAZ)")
        if a.out.resolve() == d.resolve():
            raise SystemExit("cikti klasoru bir kol klasoru olamaz")

    cfg = load_cfg(a.profile, a.config)
    a.patch = a.patch or int((cfg["finetune"].get("patch_by_profile") or {}).get(a.profile, 0))
    if a.patch:
        cfg["patch"] = [a.patch] * 3
    cuda = a.device.startswith("cuda")
    amp = (torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16) if cuda else None
    models = []
    for d in a.arms:
        m = build_model("random", cfg).to(a.device)
        m.load_state_dict(torch.load(d / "last.pt", map_location=a.device)["model"])
        m.eval()
        models.append(m)
    test_ids = read_ids(a.test_ids)
    pred_dir = a.out / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    infos = {}
    for cid in test_ids:
        z = np.load(a.cache / f"{cid}.npz", allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        hu = to_hu(z["img"], cfg)
        prob = average_probs([predict_volume_fast(m, hu, meta["spacing"], cfg, device=a.device,
                                                  tta=not a.no_tta, amp_dtype=amp) for m in models])
        tum_p, info = postprocess(prob, a.thr, spacing=meta["spacing"])
        infos[cid] = info
        full = reconstruct(tum_p, meta)
        n = int(cid.split("-")[1]) if cid.startswith("lits-") else None
        ref_p = a.gt_dir / (f"segmentation-{n}.nii" if n is not None else f"{cid}.nii.gz")
        if not ref_p.exists():
            raise CacheError(f"{cid}: GT dosyasi yok ({ref_p})")
        ref = nib.load(str(ref_p))
        if tuple(ref.shape[:3]) != full.shape:
            raise CacheError(f"{cid}: tahmin {full.shape} != GT {ref.shape}")
        img = nib.Nifti1Image((full > a.thr).astype(np.uint8), ref.affine, header=ref.header.copy())
        img.set_data_dtype(np.uint8)
        nib.save(img, str(pred_dir / f"{cid}.nii.gz"))
        print(f"[ens] {cid} karaciger-disi pay {info['outside_liver_frac']:.3f}", flush=True)
    (a.out / "pred_info.json").write_text(json.dumps(infos, indent=1))
    (a.out / "done.json").write_text(json.dumps({"arms": [str(d) for d in a.arms], "weights": "esit",
                                                  "checkpoint_used": "last", "thr": a.thr,
                                                  "tta": not a.no_tta, "n_test": len(test_ids)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
