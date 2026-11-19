"""Rad-JEPA 3D encoder yukleyici (github.com/huyquoctrinh/radjepa3d + HF huyquoctrinh/Rad-Jepa-3D).

Onlarin kodu bizim repoya KOPYALANMAZ (GitHub reposunda LICENSE yok); calisma aninda klonlanan
repo sys.path'e eklenir. mamba_ssm kurulu degilse (ya da force_ref=True) saf PyTorch
`src.jepa.mamba_ref.Mamba` onun yerine `mamba_ssm` adiyla enjekte edilir.

Girdi (model karti): (B, 1, 32, 256, 256) float32, hacim basina min-max ile [0, 1].
Cikti: (B, 1024, 384) token = 4 x 16 x 16 izgara (z, y, x).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


def _install_mamba(force_ref: bool) -> str:
    if not force_ref:
        try:
            import mamba_ssm  # noqa: F401
            return "mamba_ssm"
        except Exception:
            pass
    from src.jepa.mamba_ref import Mamba
    mod = types.ModuleType("mamba_ssm")
    mod.Mamba = Mamba
    sys.modules["mamba_ssm"] = mod
    return "mamba_ref (saf PyTorch)"


def load_radjepa_encoder(ckpt: str | Path, repo: str | Path, device: str = "cuda",
                         force_ref: bool = False):
    repo = Path(repo)
    if not (repo / "vjepa_model.py").exists():
        raise FileNotFoundError(f"Rad-JEPA reposu bulunamadi: {repo} (vjepa_model.py yok)")
    impl = _install_mamba(force_ref)
    for p in (repo, repo / "utils"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from load_checkpoint import load_encoder   # eksik/fazla anahtarda kendisi FIRLATIR
    enc, cfg, meta = load_encoder(str(ckpt), device=device, router_mode="layer")
    print(f"[radjepa] Mamba uygulamasi: {impl} | grid {cfg.grid_size} | dim {cfg.dim}")
    enc.mamba_impl = impl
    return enc
