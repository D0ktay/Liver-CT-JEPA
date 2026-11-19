"""On egitim hedefleri: JEPA (gizli uzayda tahmin), MAE (voksel tahmini), none (kontrol).

Kiyasin tek gecerli bicimi
--------------------------
Uc kol da AYNI encoder'i, AYNI predictor'u, AYNI maskeyi, AYNI tohumu ve AYNI
ornekleme sirasini kullanir. Tek fark:
  jepa -> hedef = EMA hedef encoder'inin stride-4 ozellikleri (LayerNorm), kayip L1
  mae  -> hedef = blok-normalize HU voksel (4^3 = 64), kayip MSE, EMA YOK
  none -> on egitim adimi YOK (encoder rastgele baslatilir); `pretrain_step` FIRLATIR
Maske stride-4 izgarasinda kurulur: 1 token = 4 mm = 64 mm^3. Maskeyi voksel uzayinda
kaydirmak (orn. 4 yerine 3) hedefin bir kismini baglamda birakir ve gorevi sessizce
kolaylastirir -- kayip duser, kimse fark etmez. `apply_input_mask` bunu izgaraya baglar.

Cokme (collapse) izleme: kayip duserken temsil rank-1'e inebilir. Her `log_every`de
maskeli hedef vektorlerinde RankMe ve boyut-basi std olculur.
ESIKLER SEZGISELDIR -- kaynak gerekiyor; MVE kosusunda saglikli taban olculecek ve
`configs/train.yaml -> collapse` bu olcume gore guncellenecek.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.jepa.blocks import CrossBlock, grid_coords_mm, sincos_pos_3d
from src.jepa.encoder import LiverEncoder

OBJECTIVES = ("jepa", "mae", "none")


class CollapseError(RuntimeError):
    """Temsil coktu. Egitim sessizce devam etmez -- durur."""


# -------------------------------------------------------------------- maske
def _rand_block(grid, frac_lo: float, frac_hi: float, rng: np.random.Generator,
                tries: int = 50):
    """Izgaranin [frac_lo, frac_hi] kadarini kaplayan rastgele en-boy oranli blok."""
    n = int(np.prod(grid))
    for _ in range(tries):
        v = float(rng.uniform(frac_lo, frac_hi)) * n
        r = rng.uniform(0.6, 1.6, size=3)
        r = r / float(np.prod(r)) ** (1 / 3)
        dims = [int(max(1, min(g, round(v ** (1 / 3) * ri)))) for g, ri in zip(grid, r)]
        if not (frac_lo * n <= float(np.prod(dims)) <= frac_hi * n):
            continue
        m = np.zeros(grid, bool)
        o = [int(rng.integers(0, g - d + 1)) for g, d in zip(grid, dims)]
        m[o[0]:o[0] + dims[0], o[1]:o[1] + dims[1], o[2]:o[2] + dims[2]] = True
        return m
    return None


def sample_masks(cfg: dict, grid, rng: np.random.Generator):
    """(baglam, hedefler) -- ikisi AYRIK. Kosullar saglanamazsa FIRLATIR.

    baglam: (gd,gh,gw) bool, True = encoder'in gordugu token
    hedefler: (n_blocks, gd,gh,gw) bool
    """
    mc = cfg["mask"]
    grid = tuple(int(g) for g in grid)
    n = int(np.prod(grid))
    blo, bhi = [float(x) for x in mc["block_frac"]]
    ulo, uhi = [float(x) for x in mc["union_frac"]]
    clo, chi = [float(x) for x in mc["context_frac"]]
    for _ in range(int(mc["max_tries"])):
        blocks = [_rand_block(grid, blo, bhi, rng) for _ in range(int(mc["n_blocks"]))]
        if any(b is None for b in blocks):
            continue
        tgt = np.stack(blocks)
        u = tgt.any(0)
        if not (ulo <= u.sum() / n <= uhi):
            continue
        if u.sum() < int(mc["min_target_tokens"]):
            continue
        k = int(round(float(rng.uniform(clo, chi)) * n))
        sel = rng.permutation(n)[:k]
        ctx = np.zeros(n, bool)
        ctx[sel] = True
        ctx = ctx.reshape(grid) & ~u                      # AYRIK: hedef baglamda yok
        if ctx.sum() / n < float(mc["min_visible_frac"]):
            continue
        return ctx, tgt
    raise ValueError(
        f"maske uretilemedi: izgara {grid} ({n} token) icin blok {blo}-{bhi}, birlesim "
        f"{ulo}-{uhi}, en az {mc['min_target_tokens']} hedef token kosullari "
        f"{mc['max_tries']} denemede saglanamadi")


def target_union(targets: np.ndarray) -> np.ndarray:
    return np.asarray(targets).any(0)


def apply_input_mask(x: torch.Tensor, keep, stride: int, fill: float = 0.0) -> torch.Tensor:
    """Izgara maskesini voksel uzayina AYNEN yansit: token (i,j,k) -> [s*i, s*i+s).

    Kaydirma ya da yanlis stride burada yakalanir (test: maske izgaraya hizali).
    """
    k = torch.as_tensor(np.asarray(keep, bool))
    if k.ndim != 3 or x.ndim != 5:
        raise ValueError(f"maske 3D, girdi 5D olmali: {tuple(k.shape)}, {tuple(x.shape)}")
    bekle = tuple(int(g) * int(stride) for g in k.shape)
    if tuple(x.shape[2:]) != bekle:
        raise ValueError(f"izgara {tuple(k.shape)} x stride {stride} = {bekle}, "
                         f"girdi {tuple(x.shape[2:])} -- hiza kayik")
    m = k
    for d in range(3):
        m = m.repeat_interleave(int(stride), dim=d)
    return torch.where(m.to(x.device)[None, None], x, torch.full_like(x, float(fill)))


def _blocks_of(x: torch.Tensor, grid, stride: int) -> torch.Tensor:
    """(B,1,D,H,W) -> (B, prod(grid), stride^3) blok gorunumu (C sirasi)."""
    b = x.shape[0]
    g = tuple(int(v) for v in grid)
    v = x[:, 0].reshape(b, g[0], stride, g[1], stride, g[2], stride)
    return v.permute(0, 1, 3, 5, 2, 4, 6).reshape(b, g[0] * g[1] * g[2], stride ** 3)


def _select(flat: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    idx = torch.from_numpy(np.flatnonzero(np.asarray(mask).reshape(-1))).long()
    return flat[:, idx.to(flat.device)]


def mae_targets(x: torch.Tensor, mask: np.ndarray, stride: int, eps: float = 1e-6,
                norm: bool = True) -> torch.Tensor:
    """Maskeli 4^3 bloklarin voksel degerleri; `norm=True` -> blok-normalize (norm_pix_loss).

    22 Eyl 2026: MVE'de `norm=True` ile MAE HICBIR SEY OGRENMEDI (kayip 0.95; sifir
    tahmini = 1.0, var_ratio 0.01-0.08). 4 mm'lik CT blogunda ortalama ve olcek
    cikarilinca geriye kalan neredeyse saf gurultudur -- tahmin edilebilir kisim
    (blok ortalamasi = doku tipi) tam da atilan seydi. Wald ve ark. (arXiv:2410.23132)
    ham (z-skorlu) voksel hedefi kullanir, blok-normalizasyon YOK. Uretim: norm=False.
    """
    grid = np.asarray(mask).shape
    if tuple(x.shape[2:]) != tuple(int(g) * stride for g in grid):
        raise ValueError(f"maske izgarasi {grid} girdi {tuple(x.shape[2:])} ile uyusmuyor")
    t = _select(_blocks_of(x, grid, stride), mask)
    if not norm:
        return t
    mu = t.mean(-1, keepdim=True)
    sd = t.std(-1, keepdim=True, unbiased=False)
    return (t - mu) / (sd + eps)


# ------------------------------------------------------------------ kayiplar
def jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """LayerNorm'lu hedefte L1 (bkz. I-JEPA). Bos hedef sessizce 0 donmez -- firlatir."""
    if pred.shape != target.shape:
        raise ValueError(f"tahmin {tuple(pred.shape)} hedef {tuple(target.shape)} degil")
    if target.shape[1] == 0:
        raise ValueError("bos hedef: maskeli token yok -- maske ornekleme bozuk")
    return F.l1_loss(pred.float(), target.float())


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"tahmin {tuple(pred.shape)} hedef {tuple(target.shape)} degil")
    if target.shape[1] == 0:
        raise ValueError("bos hedef: maskeli token yok -- maske ornekleme bozuk")
    return F.mse_loss(pred.float(), target.float())


def vicreg_terms(z: torch.Tensor, max_tokens: int = 4096, gamma: float = 1.0) -> tuple:
    """(varyans mentesesi, kovaryans cezasi) -- z: (N, D) token ozellikleri.

    C-JEPA (arXiv:2410.19560): EMA tek basina cokusu ONLEMEZ; VICReg'in varyans +
    kovaryans terimleri kucuk agirlikla (0.001 x (25 var + 1 cov)) eklenir. MVE'de hedef
    ozelliklerinin RankMe'si 64 boyutta 6-18'di (dusuk rank). Partimiz kucuk oldugu icin
    istatistik ornek-havuzlu degil TOKEN duzeyinde alinir.
    """
    with torch.autocast(device_type=z.device.type, enabled=False):
        return _vicreg_fp32(z, max_tokens, gamma)


def _vicreg_fp32(z: torch.Tensor, max_tokens: int, gamma: float) -> tuple:
    # bf16'da z.T @ z ~3 anlamli basamak -> kovaryans cezasi gurultu olur; fp32 SART.
    z = z.float().reshape(-1, z.shape[-1])
    if z.shape[0] > max_tokens:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_tokens]]
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0, unbiased=False) + 1e-4)
    var = F.relu(gamma - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(1, n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var, (off ** 2).sum() / d


# ----------------------------------------------------------------------- EMA
def ema_momentum(step: int, total: int, cfg: dict) -> float:
    """Dogrusal takvim: start -> end (varsayilan 0.996 -> 1.0)."""
    a, b = float(cfg["ema"]["start"]), float(cfg["ema"]["end"])
    if total <= 0:
        raise ValueError(f"total pozitif olmali: {total}")
    return a + (b - a) * min(1.0, max(0.0, step / total))


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, m: float) -> None:
    for t, s in zip(target.parameters(), source.parameters()):
        t.mul_(m).add_(s.detach(), alpha=1.0 - m)
    for t, s in zip(target.buffers(), source.buffers()):
        t.copy_(s)


# --------------------------------------------------------------------- cokus
def collapse_stats(z: torch.Tensor) -> dict:
    """RankMe (spektral entropinin usteli) + boyut-basi std. Saglikli std ~ 1/sqrt(D)."""
    z = z.detach().float().reshape(-1, z.shape[-1])
    if z.shape[0] < 2:
        raise ValueError(f"cokus istatistigi icin en az 2 vektor gerekir: {tuple(z.shape)}")
    n = F.normalize(z, dim=-1, eps=1e-8)
    std_dim = float(n.std(dim=0, unbiased=False).mean())
    s = torch.linalg.svdvals(z)
    p = s / (s.sum() + 1e-12)
    ent = float(-(p * torch.log(p + 1e-12)).sum())
    return {"rankme": math.exp(ent), "std_dim": std_dim, "n": int(z.shape[0]),
            "dim": int(z.shape[1])}


class CollapseMonitor:
    """`patience` ardisik tetikte `collapse.json` yazar ve CollapseError firlatir."""

    def __init__(self, cfg: dict, dim: int, out_dir: Path | str | None = None):
        c = cfg["collapse"]
        self.dim = int(dim)
        self.rankme_min = float(c["rankme_frac"]) * self.dim
        self.std_min = float(c["std_frac"]) / math.sqrt(self.dim)
        self.patience = int(c["patience"])
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.hits = 0

    def is_collapsed(self, stats: dict) -> bool:
        return stats["rankme"] < self.rankme_min or stats["std_dim"] < self.std_min

    def check(self, stats: dict) -> bool:
        if not self.is_collapsed(stats):
            self.hits = 0
            return False
        self.hits += 1
        if self.hits >= self.patience:
            rec = dict(stats, rankme_min=self.rankme_min, std_min=self.std_min,
                       patience=self.patience)
            if self.out_dir is not None:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                (self.out_dir / "collapse.json").write_text(json.dumps(rec, indent=2))
            raise CollapseError(
                f"temsil coktu: rankme {stats['rankme']:.2f} < {self.rankme_min:.2f} ya da "
                f"std_dim {stats['std_dim']:.4f} < {self.std_min:.4f}, "
                f"{self.patience} ardisik log")
        return True


# ----------------------------------------------------------------- predictor
class Predictor(nn.Module):
    """Yalniz capraz-dikkat: sorgu = maskeli stride-4 konumlari (mm sin-cos),
    anahtar/deger = baglam encoder'inin stride-16 token'lari."""

    def __init__(self, cfg: dict, ctx_dim: int, out_dim: int):
        super().__init__()
        pc = cfg["predictor"]
        d = int(pc["dim"])
        if not bool(pc.get("cross_only", True)):
            raise ValueError("predictor.cross_only=False desteklenmiyor "
                             "(sorgular birbirini gorurse hedef sizar)")
        self.dim = d
        self.ctx_proj = nn.Linear(int(ctx_dim), d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([CrossBlock(d, int(pc["heads"]))
                                     for _ in range(int(pc["depth"]))])
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.head = nn.Linear(d, int(out_dim))

    def forward(self, ctx_tok: torch.Tensor, ctx_coords: torch.Tensor,
                q_coords: torch.Tensor) -> torch.Tensor:
        ctx = self.ctx_proj(ctx_tok) + sincos_pos_3d(ctx_coords, self.dim)[None]
        q = self.mask_token[None, None] + sincos_pos_3d(q_coords, self.dim)[None]
        q = q.expand(ctx.shape[0], -1, -1)
        for blk in self.blocks:
            q = blk(q, ctx)
        return self.head(self.norm(q))


# ------------------------------------------------------------------- modeller
class PretrainModel(nn.Module):
    """Tek govde, uc hedef. `objective` disindaki her sey ayni."""

    def __init__(self, cfg: dict, objective: str = "jepa"):
        super().__init__()
        if objective not in OBJECTIVES:
            raise ValueError(f"bilinmeyen hedef: {objective} (secenekler: {OBJECTIVES})")
        pdp = (cfg.get("pretrain") or {}).get("drop_path")
        if pdp is not None and objective != "none":
            # On egitimde cevrimici encoder DropPath'i (I-JEPA'da yok; deterministik hedefe gurultulu
            # tahmin). KOPYADA ezilir: kaydedilen cfg ince ayar profilini (drop_path 0.1) korur.
            cfg = dict(cfg, drop_path=float(pdp))
        self.cfg = cfg
        self.objective = objective
        self.encoder = LiverEncoder(cfg)
        self.stride = int(cfg["strides"]["s4"])
        self.ctx_stride = int(cfg["strides"]["s16"])
        self.spacing = tuple(float(s) for s in cfg["spacing_mm"])
        if objective == "none":
            self.pred_dim = None
            return
        pc = cfg.get("pretrain", {})
        self.mask_fill = float(pc.get("mask_fill", 0.0))
        self.remask_conv = bool(pc.get("remask_conv", False))     # varsayilan KAPALI (22 Eyl tarifi)
        self.drop_tokens = bool(pc.get("drop_tokens", False))     # s8/s16 maskeli token dusurme (B1-b)
        self.vicreg_w = float(pc.get("vicreg_weight", 0.0))
        self.mae_norm = bool(pc.get("mae_norm_target", True))
        # JEPA hedefi COK SEVIYELI: s4 (konv, yerel) + s8 + s16 (transformer, baglamsal).
        # I-JEPA ablasyonu (arXiv:2301.08243): piksel/dusuk seviye hedef 40.7 vs derin
        # ozellik hedefi 66.9. MVE'de hedef YALNIZ s4 konv ozelligiydi (~15 voksel alici
        # alan) -- "gizli uzayda tahmin" adina ragmen neredeyse piksel duzeyi.
        self.levels = tuple(pc.get("jepa_target_levels", ("s4", "s8", "s16")))
        ch = {"s4": int(cfg["conv_width"][2]), "s8": int(cfg["s8"]["dim"]),
              "s16": int(cfg["s16"]["dim"])}
        bad = [k for k in self.levels if k not in ch]
        if bad or not self.levels:
            raise ValueError(f"jepa_target_levels gecersiz: {self.levels}")
        self.level_dims = tuple(ch[k] for k in self.levels)
        self.pred_dim = sum(self.level_dims) if objective == "jepa" else self.stride ** 3
        self.predictor = Predictor(cfg, int(cfg["s16"]["dim"]), self.pred_dim)
        if objective == "jepa":
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
            self.target_encoder.eval()

    def train(self, mode: bool = True):
        """Hedef encoder HER ZAMAN eval modunda kalir.

        🔴 22 Eyl 2026 (kod denetimi, olculdu): hedef encoder train modunda kaliyordu ->
        DropPath hedefte de acikti; ayni girdide iki hedef gecisi arasinda bagil fark 0.379.
        JEPA kaybi %38 stokastik gurultuye regresyon yapiyordu -- MVE'deki JEPA kolu dahil.
        """
        super().train(mode)
        if hasattr(self, "target_encoder"):
            self.target_encoder.eval()
        return self

    # -- hedefe gore degisen TEK sey
    def _targets(self, x: torch.Tensor, union: np.ndarray, spacing) -> torch.Tensor:
        if self.objective == "mae":
            return mae_targets(x, union, self.stride, norm=self.mae_norm)
        with torch.no_grad():
            pyr = self.target_encoder(x, spacing)
            outs = []
            for k in self.levels:
                f = pyr[k].float()                                # (B,C,d,h,w)
                r = int(self.cfg["strides"][k]) // self.stride    # s4 izgarasina en yakin komsu
                for d in (2, 3, 4):
                    if r > 1:
                        f = f.repeat_interleave(r, dim=d)
                b, c = f.shape[:2]
                flat = f.reshape(b, c, -1).transpose(1, 2)        # (B,G,C)
                outs.append(F.layer_norm(_select(flat, union), (c,)))
            return torch.cat(outs, -1)

    def step(self, x: torch.Tensor, rng: np.random.Generator, step: int = 0,
             total: int = 1, spacing=None, with_stats: bool = True) -> dict:
        if self.objective == "none":
            raise ValueError(
                "objective='none': on egitim adimi YOK (kontrol kolu encoder'i rastgele "
                "baslatir). pretrain_step cagirma, dogrudan ince ayara gec.")
        sp = self.spacing if spacing is None else tuple(float(s) for s in spacing)
        grid = tuple(int(n) // self.stride for n in x.shape[2:])
        if any(int(n) % self.stride for n in x.shape[2:]):
            raise ValueError(f"girdi {tuple(x.shape[2:])} stride {self.stride} ile bolunmuyor")
        ctx_mask, targets = sample_masks(self.cfg, grid, rng)
        union = target_union(targets)
        km = torch.as_tensor(np.asarray(ctx_mask, bool)) if (self.remask_conv or self.drop_tokens) else None
        pyr = self.encoder(apply_input_mask(x, ctx_mask, self.stride, fill=self.mask_fill), sp,
                           keep_mask=km, drop_tokens=self.drop_tokens)
        ctx = pyr["s16"]
        b, c16 = ctx.shape[:2]
        ctx_tok = ctx.reshape(b, c16, -1).transpose(1, 2)
        ctx_coords = grid_coords_mm(tuple(ctx.shape[2:]), self.ctx_stride, sp, x.device)
        if "keep_s16" in pyr:
            # predictor yalniz GORUNUR baglam token'larini alir (I-JEPA ile birebir)
            k16 = pyr["keep_s16"].reshape(-1).cpu().numpy()
            ctx_tok, ctx_coords = _select(ctx_tok, k16), ctx_coords[torch.from_numpy(np.flatnonzero(k16)).long()]
        q_coords = grid_coords_mm(grid, self.stride, sp, x.device)[
            torch.from_numpy(np.flatnonzero(union.reshape(-1))).long()]
        pred = self.predictor(ctx_tok, ctx_coords, q_coords)
        tgt = self._targets(x, union, sp)
        if self.objective == "jepa":
            # seviye basina L1, sonra ORTALAMA: genis seviye (s16) kaybi domine etmesin
            parts, o = [], 0
            for dlev in self.level_dims:
                parts.append(jepa_loss(pred[..., o:o + dlev], tgt[..., o:o + dlev]))
                o += dlev
            loss = torch.stack(parts).mean()
        else:
            loss = mae_loss(pred, tgt)
        reg = None
        if self.objective == "jepa" and self.vicreg_w > 0:
            # cevrimici encoder'in GORUNEN s4 token'lari (hedef = bunlarin EMA'si)
            f4 = pyr["s4"]
            z = F.layer_norm(_select(f4.reshape(b, f4.shape[1], -1).transpose(1, 2), ctx_mask),
                             (f4.shape[1],))
            v, c = vicreg_terms(z)
            reg = self.vicreg_w * (25.0 * v + 1.0 * c)
            loss = loss + reg
        # SVD her adimda PAHALI (72000x128'de ~0.17 sn) -> yalniz log adimlarinda.
        stats = {}
        if with_stats:
            stats = collapse_stats(tgt[..., :self.level_dims[0]] if self.objective == "jepa" else tgt)
            if reg is not None:
                stats["vicreg"] = float(reg.detach())
            stats["var_ratio"] = float(pred.detach().var() / (tgt.var() + 1e-12))
            if self.objective == "jepa":
                # seviye basina kayip: yalniz toplam dusuyorsa s4'un (yerel, kolay) surukledigi gorunmez
                for lev, part in zip(self.levels, parts):
                    stats[f"loss_{lev}"] = float(part.detach())
            stats["loss"] = float(loss.detach())
        return {"loss": loss, "stats": stats, "mask": (ctx_mask, targets),
                "n_target_tokens": int(union.sum())}

    @torch.no_grad()
    def ema_step(self, step: int, total: int) -> float | None:
        if self.objective != "jepa":
            return None
        m = ema_momentum(step, total, self.cfg)
        ema_update(self.target_encoder, self.encoder, m)
        return m


def build_objective(name: str, cfg: dict) -> PretrainModel:
    return PretrainModel(cfg, name)


def pretrain_step(model: PretrainModel, x: torch.Tensor, rng: np.random.Generator,
                  step: int = 0, total: int = 1, spacing=None, with_stats: bool = True) -> dict:
    return model.step(x, rng, step=step, total=total, spacing=spacing, with_stats=with_stats)


def seg_step(model, x: torch.Tensor, y: torch.Tensor, cfg: dict, loss_fn=None) -> torch.Tensor:
    """Ince ayar adimi: derin denetimli segmentasyon kaybi (varsayilan Dice+CE)."""
    from src.jepa.decoder import deep_supervision_loss
    if y.shape != x.shape:
        raise ValueError(f"etiket {tuple(y.shape)} girdi {tuple(x.shape)} ile uyusmuyor")
    return deep_supervision_loss(model(x), y, cfg, loss_fn)
