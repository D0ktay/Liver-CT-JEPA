"""Dogrusal sonda: DONMUS encoder ozelliklerinden token sinifi (arka plan / karaciger / tumor).

Amac: on egitimli temsil lezyon bilgisini tasiyor mu -- ince ayardan BAGIMSIZ, dakikalar icinde.
Encoder donuk, yalniz tek dogrusal katman egitilir. Vaka duzeyinde egitim/sinama ayrimi;
DONDURULMUS IC TEST vakalari --exclude-ids ile DISARIDA tutulur (zorunlu arguman).

    python -m src.jepa.probe --init outputs/night/pre_jepa/encoder.pt --cache data/cache/lits \
        --exclude-ids outputs/internal_test.json --out outputs/night/probe/jepa.json
    python -m src.jepa.probe --init random ...      # taban: rastgele agirlikli ayni encoder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.jepa.encoder import LiverEncoder, load_cfg, load_encoder
from src.jepa.cache_build import to_hu
from src.jepa.data import hu_window

CLASSES = ("arka_plan", "karaciger", "tumor")


def token_labels(liver: torch.Tensor, tumor: torch.Tensor, stride: int = 4, min_frac: float = 0.5) -> torch.Tensor:
    """(B,1,D,H,W) ikili maskeler -> (B,d,h,w) token sinifi. Tumor payi >= min_frac -> 2;
    degilse karaciger∪tumor payi >= min_frac -> 1; degilse 0. Arada kalan (belirsiz) -> -1."""
    ft = F.avg_pool3d(tumor.float(), stride)[:, 0]
    fl = F.avg_pool3d(torch.clamp(liver.float() + tumor.float(), max=1.0), stride)[:, 0]
    lab = torch.full(ft.shape, -1, dtype=torch.long, device=ft.device)
    lab[fl == 0] = 0
    lab[fl >= min_frac] = 1
    lab[ft >= min_frac] = 2
    return lab


@torch.no_grad()
def token_features(enc: LiverEncoder, x: torch.Tensor, levels=("s4", "s8", "s16")) -> torch.Tensor:
    """Secilen seviyeler s4 izgarasinda (en yakin komsu) birlestirilir, (B,C,d,h,w).
    Tek seviye (orn. yalniz s16) = o seviyenin temsili tek basina ne tasiyor: s4 conv ozelligi
    HU'yu neredeyse dogrudan tasidigi icin rastgele agda bile yuksek cikar ve farki maskeler."""
    p = enc(x)
    size = p["s4"].shape[2:]
    return torch.cat([p[k].float() if k == "s4" else F.interpolate(p[k].float(), size=size, mode="nearest")
                      for k in levels], 1)


def balanced_accuracy(pred: np.ndarray, y: np.ndarray, n_cls: int = 3) -> tuple[float, list[float]]:
    rec = [float((pred[y == c] == c).mean()) if (y == c).any() else float("nan") for c in range(n_cls)]
    return float(np.nanmean(rec)), rec


def _patches(z, patch: int, n: int, rng: np.random.Generator):
    img, liv = z["img"], z["liver"] > 0
    tum = z["lab"] > 0
    idx = np.argwhere(tum if (tum.any() and rng.random() < 0.7) else liv)
    out = []
    for _ in range(n):
        c = idx[rng.integers(len(idx))]
        lo = [int(np.clip(ci - patch // 2, 0, max(0, s - patch))) for ci, s in zip(c, img.shape)]
        sl = tuple(slice(a, a + patch) for a in lo)
        def cut(a, fill):
            a = a[sl]
            return np.pad(a, [(0, patch - a.shape[i]) for i in range(3)], constant_values=fill)
        out.append((cut(img, 0), cut(liv, False), cut(tum, False)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--exclude-ids", type=Path, required=True, help="dondurulmus ic test (json: case_ids)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", default="orta")
    ap.add_argument("--levels", default="s4,s8,s16", help="virgulle: s4,s8,s16 alt kumesi")
    ap.add_argument("--key", default="encoder", choices=("encoder", "target_encoder"))
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--patches-per-case", type=int, default=6)
    ap.add_argument("--max-tokens-per-class", type=int, default=150_000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    cfg = load_cfg(a.profile, a.config)
    levels = tuple(a.levels.split(","))
    if not levels or any(k not in ("s4", "s8", "s16") for k in levels):
        raise SystemExit(f"--levels gecersiz: {a.levels}")
    torch.manual_seed(a.seed)                # rastgele taban encoder'i da tekrarlanabilir olsun
    if a.init == "random":
        enc = LiverEncoder(cfg)
    elif a.key == "encoder":
        enc = load_encoder(a.init, cfg)
    else:                                   # EMA (hedef) encoder: encoder.pt'deki EK anahtar
        ck = torch.load(a.init, map_location="cpu")
        if a.key not in ck:
            raise SystemExit(f"{a.init} icinde '{a.key}' yok (anahtarlar: {sorted(ck)})")
        enc = LiverEncoder(cfg)
        enc.load_state_dict(ck[a.key], strict=True)
    enc = enc.to(a.device).eval()
    raw = json.loads(a.exclude_ids.read_text())
    excl = set(raw["case_ids"] if isinstance(raw, dict) else raw)
    stems = {f.stem for f in a.cache.glob("*.npz")}
    if not excl or not (excl & stems):
        raise SystemExit(f"--exclude-ids onbellekteki HICBIR vakayla eslesmedi ({len(excl)} kimlik, "
                         f"{len(stems)} dosya) -- dondurulmus test sondaya SIZARDI")
    files = sorted(f for f in a.cache.glob("*.npz") if f.stem not in excl)
    if not files:
        raise SystemExit("sonda icin vaka yok")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(files))
    n_te = max(1, len(files) // 4)
    split = {"test": [files[i] for i in order[:n_te]], "train": [files[i] for i in order[n_te:]]}
    patch = int(cfg["patch"][0])
    data = {}
    for name, fs in split.items():
        X, Y = [], []
        for f in fs:
            z = np.load(f, allow_pickle=False)
            if "lab" not in z or "liver" not in z:
                continue
            for img, liv, tum in _patches(z, patch, a.patches_per_case, rng):
                # cikarimla AYNI girdi yolu: onbellek -> sozde HU -> hu_window (region.predict_volume_fast)
                xin = hu_window(to_hu(img, cfg), cfg["hu_window"], out_range=cfg["out_range"])
                x = torch.from_numpy(np.asarray(xin, np.float32))[None, None].to(a.device)
                feat = token_features(enc, x, levels)[0]
                lab = token_labels(torch.from_numpy(liv)[None, None].to(a.device),
                                   torch.from_numpy(tum)[None, None].to(a.device))[0]
                fl, ll = feat.permute(1, 2, 3, 0).reshape(-1, feat.shape[0]), lab.reshape(-1)
                for c in range(3):                         # yama basina sinif basina <= 1500 token (RAM)
                    idx = torch.nonzero(ll == c)[:, 0]
                    idx = idx[torch.randperm(len(idx), device=idx.device)[:1500]]
                    X.append(fl[idx].cpu()); Y.append(ll[idx].cpu())
        X, Y = torch.cat(X), torch.cat(Y)
        sel = torch.cat([torch.nonzero(Y == c)[:, 0][torch.randperm(int((Y == c).sum()))[:a.max_tokens_per_class]]
                         for c in range(3)])
        data[name] = (X[sel], Y[sel])
    (Xtr, Ytr), (Xte, Yte) = data["train"], data["test"]
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = ((Xtr - mu) / sd).to(a.device), ((Xte - mu) / sd).to(a.device)
    Ytr = Ytr.to(a.device)
    w = (1.0 / torch.bincount(Ytr, minlength=3).float().clamp(min=1)); w = w / w.sum() * 3
    lin = torch.nn.Linear(Xtr.shape[1], 3).to(a.device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(a.epochs):
        perm = torch.randperm(len(Xtr), device=a.device)
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]
            loss = F.cross_entropy(lin(Xtr[b]), Ytr[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = lin(Xte).argmax(1).cpu().numpy()
    ba, rec = balanced_accuracy(pred, Yte.numpy())
    tl = Yte.numpy() > 0                                   # karaciger-ici tumor/karaciger ayrimi
    ba_tl, rec_tl = balanced_accuracy(np.where(pred[tl] == 2, 1, 0), (Yte.numpy()[tl] == 2).astype(int), 2)
    res = {"init": a.init, "key": a.key, "levels": list(levels), "n_train_cases": len(split["train"]), "n_test_cases": len(split["test"]),
           "n_train_tokens": int(len(Xtr)), "n_test_tokens": int(len(Xte)), "feat_dim": int(Xtr.shape[1]),
           "balanced_acc_3sinif": ba, "recall": dict(zip(CLASSES, rec)),
           "balanced_acc_tumor_vs_karaciger": ba_tl, "recall_tumor_vs_karaciger": rec_tl,
           "not": "8-27 vakalik sinama; kucuk fark IDDIA EDILMEZ (kural 8). Dondurulmus test DISARIDA."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
