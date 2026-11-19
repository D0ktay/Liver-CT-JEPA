"""Rad-JEPA encoder'ini egitime acma (ince ayar kolu).

Donmus kolda token'lar bir kez cikarilip .npz'ye yazilir (src.jepa.extract). Ince ayarda
token'lar her adimda YENIDEN hesaplanir, cunku encoder'in son bloklari da ogrenir.
`token_forward` bu yuzden `extract.encode_tiles`in gradyanli ikizidir: AYNI bloklar (4. ve 8.),
AYNI anahtarlar (l4, l8, out) -- ikisi ayrisirsa donmus/ince-ayar kiyasi anlamini yitirir.

Neden yalniz son N blok: 131 hasta ile 39M parametrenin tamami acilirsa ezberler.
Neden ayri ogrenme hizi: baslik sifirdan ogrenir (3e-4), encoder zaten egitilmis (1e-5).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.jepa.seg import TOKEN_KEYS

TOKEN_LAYERS = (3, 7)          # src.jepa.extract.encode_tiles varsayilani: l4, l8
_NORM_NAMES = ("norm", "norm_f", "ln_f")


def check_mamba_impl(impl: str, unfreeze: int) -> None:
    """Saf PyTorch Mamba 1024 adimi Python dongusuyle tarar; geri yayilim kullanilamaz."""
    if unfreeze > 0 and "mamba_ssm" not in impl:
        raise RuntimeError(
            f"ince ayar (--unfreeze {unfreeze}) gercek mamba_ssm CUDA cekirdegini gerektirir; "
            f"yuklu olan: {impl}. Donmus kol icin --unfreeze 0 kullan.")


def token_forward(enc, x: torch.Tensor, layers=TOKEN_LAYERS) -> dict[str, torch.Tensor]:
    """(B,1,32,256,256) -> {l4, l8, out} (B,1024,384), gradyan KOPARILMADAN."""
    grabbed: dict[str, torch.Tensor] = {}
    hooks = [enc.blocks[i].register_forward_hook(
        lambda m, a, o, i=i: grabbed.__setitem__(f"l{i + 1}", o[0] if isinstance(o, tuple) else o))
        for i in layers]
    try:
        grabbed["out"] = enc(x, indices=None)
    finally:
        for h in hooks:
            h.remove()
    if set(grabbed) != set(TOKEN_KEYS):
        raise RuntimeError(f"token anahtarlari {sorted(grabbed)} != {sorted(TOKEN_KEYS)}")
    return grabbed


def set_finetune(enc, last_n: int) -> list[nn.Parameter]:
    """Encoder'i dondur, son `last_n` blogu (+ son norm) ac. Acilan parametreleri dondurur."""
    n = len(enc.blocks)
    if last_n > n:
        raise ValueError(f"encoder'da {n} blok var, {last_n} blok acilamaz")
    for p in enc.parameters():
        p.requires_grad_(False)
    if last_n <= 0:
        enc.eval()
        return []
    opened: list[nn.Parameter] = []
    seen: set[int] = set()

    def _open(mod: nn.Module) -> None:
        for p in mod.parameters():
            p.requires_grad_(True)
            if id(p) not in seen:
                seen.add(id(p))
                opened.append(p)

    for b in list(enc.blocks)[n - last_n:]:
        _open(b)
    for name in _NORM_NAMES:
        m = getattr(enc, name, None)
        if isinstance(m, nn.Module):
            _open(m)
            break
    return opened


def param_groups(head: nn.Module, opened: list[nn.Parameter], lr_head: float = 3e-4,
                 lr_enc: float = 1e-5) -> list[dict]:
    g = [{"params": list(head.parameters()), "lr": lr_head}]
    if opened:
        g.append({"params": list(opened), "lr": lr_enc})
    return g
