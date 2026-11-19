"""JEPA ince ayar testleri -- VERI / GPU OLMADAN (kucuk sahte encoder).

Hedeflenen hatalar:
  - Gradyanli token cikarma, extract.encode_tiles ile AYNI katmanlari/anahtarlari vermez
    (egitim ile ozellik dosyalari tutarsizlasir, kiyas anlamsiz olur).
  - Kanca gradyani koparir (detach) -> encoder ogrenmiyor ama kimse fark etmiyor.
  - "son N blogu ac" yanlis uctan acar (ilk bloklar acilir) ya da hepsini acik birakir.
  - Encoder'a basligin ogrenme hizi verilir -> 39M parametre 3e-4 ile bozulur.
  - Kancalar hata durumunda kalir (ikinci cagri token'lari iki kez toplar).
  - Saf PyTorch Mamba (1024 adimlik Python dongusu) ile ince ayar sessizce kabul edilir.
"""
import pytest
import torch
import torch.nn as nn

from src.jepa.finetune import (TOKEN_LAYERS, check_mamba_impl, param_groups, set_finetune,
                               token_forward)
from src.jepa.seg import TOKEN_KEYS


class _Blk(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.lin(x)


class _Enc(nn.Module):
    """Rad-JEPA encoder'inin test ikizi: .blocks + enc(x, indices=None) -> (B, N, C)."""

    def __init__(self, dim=16, n_blocks=12, grid=(2, 4, 4)):
        super().__init__()
        self.patch = nn.Conv3d(1, dim, kernel_size=(8, 16, 16), stride=(8, 16, 16))
        self.blocks = nn.ModuleList([_Blk(dim) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.grid = grid

    def forward(self, x, indices=None):
        t = self.patch(x).flatten(2).transpose(1, 2)
        for b in self.blocks:
            t = b(t)
        return self.norm(t)


def _enc_and_input():
    torch.manual_seed(0)
    enc = _Enc()
    return enc, torch.randn(2, 1, 16, 64, 64)


def test_token_anahtarlari_ve_katmanlari_extract_ile_ayni():
    enc, x = _enc_and_input()
    tok = token_forward(enc, x)
    assert set(tok) == set(TOKEN_KEYS)                  # l4, l8, out
    assert TOKEN_LAYERS == (3, 7)                       # extract.encode_tiles varsayilani
    for v in tok.values():
        assert v.shape == (2, 2 * 4 * 4, 16)


def test_ara_katman_tokenlari_gercekten_o_bloklarin_ciktisi():
    enc, x = _enc_and_input()
    tok = token_forward(enc, x)
    t = enc.patch(x).flatten(2).transpose(1, 2)
    beklenen = {}
    for i, b in enumerate(enc.blocks):
        t = b(t)
        if i in TOKEN_LAYERS:
            beklenen[f"l{i + 1}"] = t
    for k, v in beklenen.items():
        assert torch.allclose(tok[k], v, atol=1e-6)
    assert torch.allclose(tok["out"], enc.norm(t), atol=1e-6)


def test_gradyan_encoder_e_kadar_akar():
    enc, x = _enc_and_input()
    set_finetune(enc, 2)
    tok = token_forward(enc, x)
    sum(v.sum() for v in tok.values()).backward()
    assert enc.blocks[-1].lin.weight.grad is not None
    assert enc.blocks[-1].lin.weight.grad.abs().sum() > 0


def test_son_n_blok_acilir_oncekiler_donuk_kalir():
    enc, _ = _enc_and_input()
    acik = set_finetune(enc, 3)
    for b in list(enc.blocks)[:-3]:
        assert all(not p.requires_grad for p in b.parameters())
    for b in list(enc.blocks)[-3:]:
        assert all(p.requires_grad for p in b.parameters())
    assert all(not p.requires_grad for p in enc.patch.parameters())
    assert all(p.requires_grad for p in enc.norm.parameters())   # son norm da acilir
    ids = {id(p) for p in acik}
    assert ids == {id(p) for p in enc.parameters() if p.requires_grad}
    assert len(ids) == len(acik)                                 # tekrar yok


def test_sifir_blok_tum_encoderi_dondurur():
    enc, _ = _enc_and_input()
    assert set_finetune(enc, 0) == []
    assert all(not p.requires_grad for p in enc.parameters())


def test_blok_sayisindan_fazla_acmak_firlatir():
    enc, _ = _enc_and_input()
    with pytest.raises(ValueError, match="12"):
        set_finetune(enc, 13)


def test_encoder_daha_dusuk_ogrenme_hizi_alir():
    enc, _ = _enc_and_input()
    head = nn.Conv3d(1, 1, 1)
    acik = set_finetune(enc, 2)
    g = param_groups(head, acik, lr_head=3e-4, lr_enc=1e-5)
    assert [x["lr"] for x in g] == [3e-4, 1e-5]
    assert {id(p) for p in g[0]["params"]} == {id(p) for p in head.parameters()}
    assert {id(p) for p in g[1]["params"]} == {id(p) for p in acik}
    torch.optim.AdamW(g, weight_decay=1e-4)              # optimizer kabul etmeli


def test_acik_parametre_yoksa_encoder_grubu_eklenmez():
    head = nn.Conv3d(1, 1, 1)
    g = param_groups(head, [], lr_head=3e-4, lr_enc=1e-5)
    assert len(g) == 1


def test_kancalar_her_cagridan_sonra_kaldirilir():
    enc, x = _enc_and_input()
    token_forward(enc, x)
    token_forward(enc, x)
    assert all(len(b._forward_hooks) == 0 for b in enc.blocks)


def test_hata_durumunda_da_kancalar_kaldirilir():
    enc, _ = _enc_and_input()
    with pytest.raises(RuntimeError):
        token_forward(enc, torch.randn(2, 1, 3, 3, 3))    # patch conv'a kucuk girdi
    assert all(len(b._forward_hooks) == 0 for b in enc.blocks)


def test_saf_pytorch_mamba_ile_ince_ayar_reddedilir():
    with pytest.raises(RuntimeError, match="mamba_ssm"):
        check_mamba_impl("mamba_ref (saf PyTorch)", unfreeze=4)
    check_mamba_impl("mamba_ref (saf PyTorch)", unfreeze=0)   # donmusta sorun yok
    check_mamba_impl("mamba_ssm", unfreeze=4)
