"""JEPA (ya da JEPA'siz kontrol) tumor basligini bir katta egit, test hastalarini tahmin et.

    python -m src.jepa.train --feats outputs/jepa_feats --variant jepa --fold 0 --out outputs/jepa_seg

Sabit hiperparametre, dogrulama/ayar YOK (test katina bakilmaz): AdamW 3e-4, --iters adim, batch 2,
tumorlu pencere olasiligi 2/3, esik 0.5. Iki kol AYNI tohum, adim ve ornekleme ile egitilir;
tek fark JEPA token yolu. Tahmin: pencere olasiliklari -> src.jepa.prep.reconstruct -> >0.5 ->
GT segmentasyonunun affine/basligiyla NIfTI (degerlendirme ayni kodla: src.metrics.evaluate).
Katlar: make_folds (hasta duzeyi, tumorlu/tumorsuz tabakali, tohum 0) -> <out>/folds.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.jepa.prep import reconstruct
from src.jepa.seg import TOKEN_KEYS, JepaSegHead, dice_bce_loss, make_folds


def load_case(f: Path) -> dict:
    d = np.load(f)
    return {"img": d["img"], "lab": d["lab"], "l4": d["tok_l4"], "l8": d["tok_l8"], "out": d["tok_out"],
            "meta": json.loads(str(d["meta"]))}


def case_ids(feats: Path) -> list[str]:
    return sorted((p.stem for p in feats.glob("lits-*.npz")), key=lambda c: int(c.split("-")[1]))


def folds_for(feats: Path, out: Path, k: int = 5, seed: int = 0) -> list[dict]:
    f = out / "folds.json"
    if f.exists():
        return json.loads(f.read_text())
    ids = case_ids(feats)
    has_tumor = {c: bool(np.load(feats / f"{c}.npz")["lab"].any()) for c in ids}
    folds = make_folds(ids, has_tumor, k=k, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(folds, indent=1))
    return folds


def _batch(tiles, idx, device):
    img = torch.from_numpy(np.stack([tiles["img"][i] for i in idx])).to(device).float().div_(255.0)[:, None]
    lab = torch.from_numpy(np.stack([tiles["lab"][i] for i in idx])).to(device).float()[:, None]
    tok = {k: torch.from_numpy(np.stack([tiles[k][i] for i in idx])).to(device) for k in TOKEN_KEYS}
    return img, lab, tok


def train(cases: list[dict], use_jepa: bool, iters: int, device: str, seed: int = 0,
          batch: int = 2, lr: float = 3e-4, log_every: int = 200) -> JepaSegHead:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tiles = {k: [t for c in cases for t in c[k]] for k in ("img", "lab") + TOKEN_KEYS}
    pos = [i for i, l in enumerate(tiles["lab"]) if l.any()]
    neg = [i for i, l in enumerate(tiles["lab"]) if not l.any()]
    model = JepaSegHead(use_jepa=use_jepa).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    t0, run = time.time(), 0.0
    for it in range(1, iters + 1):
        idx = [int(rng.choice(pos)) if (pos and (not neg or rng.random() < 2 / 3)) else int(rng.choice(neg))
               for _ in range(batch)]
        img, lab, tok = _batch(tiles, idx, device)
        with torch.autocast(device_type="cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
            loss = dice_bce_loss(model(img, tok).float(), lab)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        run = 0.98 * run + 0.02 * float(loss) if it > 1 else float(loss)
        if it % log_every == 0 or it == iters:
            print(f"  it {it}/{iters} kayip {run:.4f} ({time.time() - t0:.0f} sn)", flush=True)
    return model


@torch.no_grad()
def predict_case(model: JepaSegHead, case: dict, device: str) -> np.ndarray:
    model.eval()
    amp = device.startswith("cuda")
    probs = []
    for i in range(len(case["img"])):
        img, _, tok = _batch({k: case[k] for k in ("img", "lab") + TOKEN_KEYS}, [i], device)
        with torch.autocast(device_type="cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
            probs.append(torch.sigmoid(model(img, tok).float())[0, 0].cpu().numpy())
    return reconstruct(np.stack(probs), case["meta"])


def save_like(mask: np.ndarray, ref_path: Path, out_path: Path) -> None:
    import nibabel as nib
    ref = nib.load(str(ref_path))
    if tuple(ref.shape[:3]) != mask.shape:
        raise ValueError(f"{out_path.name}: tahmin {mask.shape} != GT {ref.shape}")
    img = nib.Nifti1Image(mask.astype(np.uint8), ref.affine, header=ref.header.copy())
    img.set_data_dtype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", type=Path, required=True)
    ap.add_argument("--seg-dir", type=Path, default=Path("data/raw/LiTS"), help="segmentation-N.nii (affine icin)")
    ap.add_argument("--out", type=Path, default=Path("outputs/jepa_seg"))
    ap.add_argument("--variant", choices=["jepa", "nojepa"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--label-frac", type=float, default=1.0, help="egitim hastalarinin orani (az etiket deneyi)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    folds = folds_for(a.feats, a.out, a.k)
    f = folds[a.fold]
    train_ids = list(f["train"])
    if a.label_frac < 1.0:
        rng = np.random.default_rng(a.seed + 1000)
        n = max(2, int(round(a.label_frac * len(train_ids))))
        train_ids = sorted(rng.choice(train_ids, n, replace=False).tolist())
    tag = f"{a.variant}" + (f"_frac{a.label_frac:g}" if a.label_frac < 1.0 else "")
    print(f"[train] {tag} kat {a.fold}: egitim {len(train_ids)} / test {len(f['test'])} hasta, "
          f"{a.iters} adim, {a.device}", flush=True)
    model = train([load_case(a.feats / f"{c}.npz") for c in train_ids], a.variant == "jepa",
                  a.iters, a.device, a.seed)
    pred_dir = a.out / tag / "pred"
    for c in f["test"]:
        case = load_case(a.feats / f"{c}.npz")
        mask = predict_case(model, case, a.device) > 0.5
        n = int(c.split("-")[1])
        save_like(mask, a.seg_dir / f"segmentation-{n}.nii", pred_dir / f"{c}.nii.gz")
    (a.out / tag / f"fold{a.fold}.json").write_text(json.dumps(
        {"fold": a.fold, "train": train_ids, "test": f["test"], "iters": a.iters, "seed": a.seed}, indent=1))
    print(f"[train] {tag} kat {a.fold}: {len(f['test'])} tahmin -> {pred_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
