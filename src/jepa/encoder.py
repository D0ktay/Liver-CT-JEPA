"""Kendi 3D CT encoder'imiz: ConvNeXt-3D govde (stride 1/2/4) + transformer (stride 8/16).

Neden hibrit
------------
Saf ViT'te yama 8^3 olsa bile 96^3 girdide token 8 mm eder; bizim olctugumuz medyan
lezyon (esdeger cap 10.03 mm) eksen basina 1.25 token kalir -- Rad-JEPA'nin
(token 20x12.8x12.8 mm) olculmus basarisizliginin tekrari olur. Konvolusyon govdesi
ince olcegi (1-4 mm) tasiyor, transformer yalniz kaba olcekte (8-16 mm) baglam kuruyor.
Stride 1'de ConvNeXt blogu YOK: aktivasyon bellegi orada patliyor, stride 1 yalniz
decoder skip'i olarak tasiniyor.

Elenenler: Mamba (saf PyTorch referansi geri yayilim icin kullanilamaz, `mamba_ssm`
garantisi yok -- `src/jepa/finetune.py:check_mamba_impl`), saf ViT patch 8^3 (yukarida).

Kullanim:
    cfg = load_cfg(pick_config(n_vol=131, gpu_h=24))
    enc = LiverEncoder(cfg)
    piramit = enc(x)        # {"s1","s2","s4","s8","s16"} -> (B, C, D/s, H/s, W/s)

Bellek olcumu (GPU gelince):
    python -m src.jepa.encoder --probe --profile kucuk
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.jepa.blocks import (
    ConvNeXt3dBlock,
    Downsample3d,
    LayerNorm3d,
    TransformerBlock,
    grid_coords_mm,
    sincos_pos_3d,
)

DEFAULT_TRAIN_CFG = Path("configs/train.yaml")
STRIDE_KEYS = ("s1", "s2", "s4", "s8", "s16")


# ------------------------------------------------------------------- konfig
def load_train_yaml(path: Path | str = DEFAULT_TRAIN_CFG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _resolve_ref(ref: str, base: Path) -> object:
    """'configs/paths.yaml#metrics.hu_window' -> o dosyadaki deger (kopya YOK)."""
    fname, _, dotted = str(ref).partition("#")
    node = yaml.safe_load((base.parent.parent / fname if not Path(fname).exists()
                           else Path(fname)).read_text())
    for k in dotted.split("."):
        if k not in node:
            raise KeyError(f"{fname} icinde '{dotted}' yok ({k} bulunamadi)")
        node = node[k]
    return node


def load_cfg(name: str, path: Path | str = DEFAULT_TRAIN_CFG) -> dict:
    """Profil + kuresel ayarlar tek sozluk. HU penceresi paths.yaml'dan COZULUR."""
    p = Path(path)
    raw = load_train_yaml(p)
    if name not in raw["profiles"]:
        raise KeyError(f"profil yok: {name} (var olanlar: {sorted(raw['profiles'])})")
    cfg = dict(raw["profiles"][name])
    cfg["name"] = name
    cfg["strides"] = dict(raw["strides"])
    for k in ("mask", "predictor", "ema", "collapse", "optim", "inference", "decoder",
              "loss", "pick", "finetune", "pretrain"):
        cfg[k] = raw[k]
    cfg["augment"] = raw.get("augment")
    cfg["layer_scale_init"] = float(raw.get("layer_scale_init", 1e-6))
    g = raw["geometry"]
    cfg["spacing_mm"] = tuple(float(s) for s in g["spacing_mm"])
    cfg["out_range"] = tuple(float(s) for s in g["encoder_out_range"])
    cfg["hu_window"] = tuple(float(v) for v in _resolve_ref(g["hu_window_ref"], p))
    return cfg


def pick_config(n_vol: int, gpu_h: float, path: Path | str = DEFAULT_TRAIN_CFG) -> str:
    """Korpus buyuklugu + GPU saatine gore profil adi. "buyuk" ASLA donmez."""
    p = load_train_yaml(path)["pick"]
    small, large = p["small_name"], p["large_name"]
    if float(gpu_h) < float(p["gpu_h_min"]):
        return small                              # GPU yoksa profil buyutulmez
    if int(n_vol) < int(p["n_small"]):
        return small                              # 131 hacim -> kucuk
    if int(n_vol) < int(p["n_large"]):
        return large
    return large if float(gpu_h) >= float(p["gpu_h_large"]) else small


# --------------------------------------------------------------- token fizigi
def token_mm(cfg: dict, spacing) -> dict:
    """Her piramit seviyesinde bir token'in fiziksel kenar uzunlugu (mm)."""
    sp = tuple(float(s) for s in spacing)
    if len(sp) != 3:
        raise ValueError(f"spacing 3 elemanli olmali: {spacing}")
    return {k: tuple(float(v) * s for s in sp) for k, v in cfg["strides"].items()}


def tokens_per_axis(size_mm: float, spacing: float, stride: int) -> float:
    """Verilen capta bir lezyon eksen basina kac token eder."""
    return float(size_mm) / (float(spacing) * int(stride))


def lesion_tokens(volume_mm3: float, spacing: float = 1.0, stride: int = 4) -> float:
    """Hacim (mm^3) -> token sayisi. 1 token = (spacing*stride)^3 mm^3."""
    return float(volume_mm3) / (float(spacing) * int(stride)) ** 3


# ------------------------------------------------------------------- encoder
class _TransformerStage(nn.Module):
    """Indirgeme + mm tabanli sin-cos konum + N transformer blogu."""

    def __init__(self, cin: int, dim: int, depth: int, heads: int, stride: int,
                 drop_path: float = 0.0):
        super().__init__()
        self.down = Downsample3d(cin, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, drop_path=drop_path) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.dim, self.stride = dim, stride

    def forward(self, x: torch.Tensor, spacing, keep: torch.Tensor | None = None) -> torch.Tensor:
        """keep: (d,h,w) bool bu asamanin izgarasinda. Verilirse maskeli token'lar DIZIDEN
        CIKARILIR (I-JEPA / HySparK arXiv:2408.05815: dikkat yalniz gorunur token'lar arasinda),
        cikista sifirla geri yerlestirilir (yogun API korunur)."""
        x = self.down(x)
        b, c, d, h, w = x.shape
        pos = sincos_pos_3d(grid_coords_mm((d, h, w), self.stride, spacing, x.device), c)
        t = x.reshape(b, c, -1).transpose(1, 2) + pos[None]
        if keep is not None:
            if tuple(keep.shape) != (d, h, w):
                raise ValueError(f"keep {tuple(keep.shape)} != asama izgarasi {(d, h, w)}")
            idx = torch.nonzero(keep.reshape(-1).to(t.device))[:, 0]
            if idx.numel() == 0:
                raise ValueError("transformer asamasinda gorunur token kalmadi")
            t = t[:, idx]
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        if keep is not None:
            full = t.new_zeros(b, d * h * w, c)
            full[:, idx] = t
            t = full
        return t.transpose(1, 2).reshape(b, c, d, h, w)


class LiverEncoder(nn.Module):
    """Girdi (B, 1, D, H, W) -> {s1, s2, s4, s8, s16} piramidi."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        c1, c2, c3 = [int(c) for c in cfg["conv_width"]]
        n2, n4 = [int(n) for n in cfg["conv_depth"]]
        dp = float(cfg.get("drop_path", 0.0))
        self.max_stride = int(max(cfg["strides"].values()))
        self.spacing = tuple(float(s) for s in cfg["spacing_mm"])

        # stride 1: yalniz govde konvolusyonu (ConvNeXt blogu YOK -- aktivasyon patlar)
        self.stem = nn.Sequential(nn.Conv3d(1, c1, 3, padding=1), LayerNorm3d(c1))
        self.down2 = Downsample3d(c1, c2)
        ls = float(cfg.get("layer_scale_init", 1e-6))
        self.blocks2 = nn.ModuleList([ConvNeXt3dBlock(c2, dp, layer_scale=ls) for _ in range(n2)])
        self.down4 = Downsample3d(c2, c3)
        self.blocks4 = nn.ModuleList([ConvNeXt3dBlock(c3, dp, layer_scale=ls) for _ in range(n4)])
        self.stage8 = _TransformerStage(c3, int(cfg["s8"]["dim"]), int(cfg["s8"]["depth"]),
                                        int(cfg["s8"]["heads"]), int(cfg["strides"]["s8"]), dp)
        self.stage16 = _TransformerStage(int(cfg["s8"]["dim"]), int(cfg["s16"]["dim"]),
                                         int(cfg["s16"]["depth"]), int(cfg["s16"]["heads"]),
                                         int(cfg["strides"]["s16"]), dp)
        self.channels = {"s1": c1, "s2": c2, "s4": c3,
                         "s8": int(cfg["s8"]["dim"]), "s16": int(cfg["s16"]["dim"])}

    @staticmethod
    def _remask(f: torch.Tensor, keep: torch.Tensor | None, grid_stride: int, stride: int) -> torch.Tensor:
        """Ozelligi maske disinda SIFIRLA. Yogun conv maskeli bolgeye bilgi tasir ve maske deseni
        birkac katmanda erir; her bloktan sonra yeniden maskelemek seyrek conv'a sayisal esdegerdir
        (ConvNeXt V2 arXiv:2301.00808: 79.3 -> 83.7; SparK arXiv:2301.03580)."""
        if keep is None:
            return f
        r = grid_stride // stride
        m = keep
        for d in range(3):
            m = m.repeat_interleave(r, dim=d)
        return f * m.to(f.dtype)[None, None]

    def forward(self, x: torch.Tensor, spacing=None, keep_mask: torch.Tensor | None = None,
                drop_tokens: bool = False) -> dict:
        """keep_mask: (D/4, H/4, W/4) bool, s4 izgarasinda GORUNEN token'lar. Yalniz on egitimde
        (pretrain.remask_conv) verilir; None -> davranis birebir eski.
        drop_tokens: keep_mask ile birlikte, s8/s16 asamalarinda gorunur payi < 0.5 olan token'lar
        diziden cikarilir (pretrain.drop_tokens; HySparK/I-JEPA). Cikista "keep_s16" anahtari
        (d,h,w bool) predictor'in yalniz gorunur baglami almasi icin doner."""
        if keep_mask is not None:
            g = tuple(n // 4 for n in x.shape[2:])
            if keep_mask.ndim != 3 or tuple(keep_mask.shape) != g:
                raise ValueError(f"keep_mask {tuple(keep_mask.shape)} != s4 izgarasi {g}")
            keep_mask = keep_mask.to(x.device).bool()
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"girdi (B, 1, D, H, W) olmali: {tuple(x.shape)}")
        bad = [n for n in x.shape[2:] if n % self.max_stride]
        if bad:
            raise ValueError(
                f"girdi boyutu {tuple(x.shape[2:])} {self.max_stride} ile bolunmuyor "
                f"(kirpma sessizce maske hizasini kaydirir; once data.pad_to_patch)")
        sp = self.spacing if spacing is None else tuple(float(s) for s in spacing)
        km = keep_mask
        s1 = self._remask(self.stem(x), km, 4, 1)
        s2 = self._remask(self.down2(s1), km, 4, 2)
        for b in self.blocks2:
            s2 = self._remask(b(s2), km, 4, 2)
        s4 = self._remask(self.down4(s2), km, 4, 4)
        for b in self.blocks4:
            s4 = self._remask(b(s4), km, 4, 4)
        k8 = k16 = None
        if drop_tokens:
            if km is None:
                raise ValueError("drop_tokens icin keep_mask gerekli")
            kf = km.float()[None, None]
            k8 = (F.avg_pool3d(kf, 2) >= 0.5)[0, 0]
            k16 = (F.avg_pool3d(kf, 4) >= 0.5)[0, 0]
        s8 = self.stage8(s4, sp, keep=k8)
        s16 = self.stage16(s8, sp, keep=k16)
        out = {"s1": s1, "s2": s2, "s4": s4, "s8": s8, "s16": s16}
        if k16 is not None:
            out["keep_s16"] = k16
        return out


def load_encoder(ckpt: Path | str, cfg: dict | None = None, strict: bool = True,
                 key: str = "encoder") -> LiverEncoder:
    """Ckpt'ten encoder. Eksik/fazla anahtar sessizce yutulmaz (strict=True).

    key="target_encoder": JEPA'nin EMA (hedef) encoder'i (I-JEPA asagi akista bunu kullanir).
    Istenen anahtar ckpt'te yoksa FIRLATIR -- sessizce cevrimici encoder'a dusmez."""
    d = torch.load(ckpt, map_location="cpu")
    if key != "encoder":
        if key not in d:
            raise KeyError(f"{ckpt} icinde '{key}' yok (anahtarlar: {sorted(d)})")
        state = d[key]
    else:
        state = d.get("encoder", d.get("state_dict", d))
    if cfg is None:
        if "cfg" not in d:
            raise KeyError(f"{ckpt} icinde 'cfg' yok, cfg parametresi ver")
        cfg = d["cfg"]
    enc = LiverEncoder(cfg)
    enc.load_state_dict(state, strict=strict)
    return enc


def _probe(profile: str, sizes, batches) -> list[dict]:
    cfg = load_cfg(profile)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = LiverEncoder(cfg).to(dev)
    rows = []
    for n in sizes:
        for b in batches:
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            x = torch.zeros(b, 1, n, n, n, device=dev)
            try:
                enc(x)["s16"].sum().backward()
                mem = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else None
                rows.append({"patch": n, "batch": b, "peak_mib": mem, "ok": True})
            except RuntimeError as e:                    # OOM: sessizce yutma, kaydet
                rows.append({"patch": n, "batch": b, "peak_mib": None, "ok": False,
                             "err": str(e)[:120]})
            enc.zero_grad(set_to_none=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="kucuk")
    ap.add_argument("--probe", action="store_true", help="artan batch ile bellek olc")
    ap.add_argument("--sizes", type=int, nargs="+", default=[48, 64, 96])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    a = ap.parse_args()
    from src.jepa.blocks import count_params
    cfg = load_cfg(a.profile)
    out = {"profile": a.profile, "params": count_params(LiverEncoder(cfg)),
           "token_mm": token_mm(cfg, cfg["spacing_mm"]),
           "cuda": torch.cuda.is_available()}
    if a.probe:
        out["probe"] = _probe(a.profile, a.sizes, a.batches)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
