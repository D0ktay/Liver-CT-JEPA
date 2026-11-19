"""Saf PyTorch Mamba (v1, Gu & Dao 2023) -- `mamba_ssm.Mamba` yerine.

Neden: Rad-JEPA 3D `from mamba_ssm import Mamba` ister. mamba_ssm derlenmis CUDA cekirdegi
gerektirir (Kaggle'in torch surumunde hazir paket olmayabilir, Mac'te hic yok). Burada ayni
hesap saf PyTorch ile: parametre adlari ve bicimleri mamba_ssm.modules.mamba_simple.Mamba ile
AYNI (in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj) -> ayni state_dict yuklenir.

Yavas (zaman ekseninde dongu) ama Rad-JEPA'da L = 1024 token; ozellik cikarma icin yeterli.
Dogrulama: tests/test_jepa.py (bagimsiz dongulu referans + nedensellik + gercek agirlik yukleme).
Kaggle'da mamba_ssm kurulabilirse ciktilar ayrica birebir karsilastirilir.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: int | str = "auto", conv_bias: bool = True, bias: bool = False,
                 **_ignored):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor, **_ignored) -> torch.Tensor:
        """(B, L, d_model) -> (B, L, d_model)."""
        B, L, _ = hidden_states.shape
        dtype = hidden_states.dtype
        x = hidden_states.float()
        xz = F.linear(x, self.in_proj.weight.float(),
                      None if self.in_proj.bias is None else self.in_proj.bias.float())
        u, z = xz.chunk(2, dim=-1)                                      # (B, L, di)
        u = F.conv1d(u.transpose(1, 2), self.conv1d.weight.float(),
                     None if self.conv1d.bias is None else self.conv1d.bias.float(),
                     padding=self.d_conv - 1, groups=self.d_inner)[..., :L]   # nedensel
        u = F.silu(u).transpose(1, 2)                                   # (B, L, di)
        dbl = F.linear(u, self.x_proj.weight.float())
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(F.linear(dt, self.dt_proj.weight.float(), self.dt_proj.bias.float()))
        A = -torch.exp(self.A_log.float())                              # (di, ds)
        dA = torch.exp(dt.unsqueeze(-1) * A)                            # (B, L, di, ds)
        dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)      # (B, L, di, ds)
        h = u.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):                                              # secici tarama
            h = dA[:, t] * h + dBu[:, t]
            ys.append(torch.einsum("bds,bs->bd", h, Cm[:, t]))
        y = torch.stack(ys, dim=1) + u * self.D.float()
        y = y * F.silu(z)
        out = F.linear(y, self.out_proj.weight.float(),
                       None if self.out_proj.bias is None else self.out_proj.bias.float())
        return out.to(dtype)
