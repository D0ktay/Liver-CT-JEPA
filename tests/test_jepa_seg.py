"""JEPA segmentasyon basligi testleri -- VERI OLMADAN (kucuk CPU girdileri).

Hedeflenen hatalar:
  - Token -> 3B izgara sirasi yanlis (Rad-JEPA: rearrange 'b d gd gh gw -> b (gd gh gw) d').
  - 'JEPA'siz' kontrol kolu gizlice token kullanir (kiyas anlamsizlasir).
  - Katlar hasta duzeyinde ayrik degil / her vaka tam bir kez testte degil.
  - Dice kaybi bos maskede NaN / patlama.
"""
import numpy as np
import pytest
import torch

from src.jepa.seg import JepaSegHead, dice_bce_loss, make_folds, tokens_to_grid


def test_token_izgara_sirasi_rad_jepa_ile_ayni():
    from einops import rearrange
    g = torch.arange(2 * 3 * 4 * 16 * 16, dtype=torch.float32).reshape(2, 3, 4, 16, 16)  # (B,C,gd,gh,gw)
    tok = rearrange(g, "b d gd gh gw -> b (gd gh gw) d")
    assert torch.equal(tokens_to_grid(tok), g)


def _batch(b=1):
    img = torch.rand(b, 1, 32, 256, 256)
    tok = {k: torch.randn(b, 1024, 384) for k in ("l4", "l8", "out")}
    return img, tok


def test_cikti_sekli_ve_jepasiz_kol_tokeni_gormez():
    torch.manual_seed(0)
    img, tok = _batch()
    m = JepaSegHead(use_jepa=False).eval()
    with torch.no_grad():
        a = m(img, tok)
        b = m(img, {k: v * 100 + 5 for k, v in tok.items()})
    assert a.shape == (1, 1, 32, 256, 256)
    assert torch.equal(a, b)                         # token degisti, cikti ayni
    mj = JepaSegHead(use_jepa=True).eval()
    with torch.no_grad():
        c = mj(img, tok); d = mj(img, {k: v * 100 + 5 for k, v in tok.items()})
    assert not torch.allclose(c, d)                  # JEPA kolu tokeni gercekten kullaniyor


def test_kayip_bos_maskede_sonlu_ve_mukemmelde_kucuk():
    y = torch.zeros(1, 1, 4, 8, 8)
    assert torch.isfinite(dice_bce_loss(torch.full_like(y, -10.0), y))
    y[..., 2:4, 2:6, 2:6] = 1
    good = dice_bce_loss(torch.where(y > 0, 10.0, -10.0), y)
    bad = dice_bce_loss(torch.where(y > 0, -10.0, 10.0), y)
    assert good < 0.05 < 1.0 < bad


def test_katlar_hasta_duzeyinde_ayrik_kapsayici_ve_tabakali():
    ids = [f"lits-{i}" for i in range(131)]
    has_tumor = {c: (i % 10 != 0) for i, c in enumerate(ids)}          # 14 tumorsuz
    folds = make_folds(ids, has_tumor, k=5, seed=0)
    test_sets = [set(f["test"]) for f in folds]
    assert sorted(c for s in test_sets for c in s) == sorted(ids)      # her vaka tam bir kez testte
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
        assert set(f["train"]) | set(f["test"]) == set(ids)
    free = [sum(not has_tumor[c] for c in s) for s in test_sets]
    assert max(free) - min(free) <= 1                                   # tumorsuzler dengeli
    assert make_folds(ids, has_tumor, k=5, seed=0) == folds             # deterministik


def test_egitim_ve_tahmin_duman_testi(tmp_path):
    """Sahte npz (gercek sekiller) -> 2 adim egitim -> test hastasi tahmini GT basligiyla kaydedilir."""
    import json as _json
    import nibabel as nib
    from src.jepa.prep import prepare_case
    from src.jepa.train import load_case, predict_case, save_like, train
    rng = np.random.default_rng(0)
    feats = tmp_path / "f"; feats.mkdir(); segd = tmp_path / "s"; segd.mkdir()
    for n in (1, 2):
        ct = rng.integers(-200, 300, (60, 60, 20)).astype(np.int16)
        seg = np.zeros(ct.shape, np.uint8); seg[10:50, 10:50, 2:18] = 1; seg[20:30, 20:30, 6:12] = 2
        c = prepare_case(ct, seg, (0.8, 0.8, 2.5))
        tok = {k: rng.standard_normal((len(c["img"]), 1024, 384)).astype(np.float16) for k in ("l4", "l8", "out")}
        np.savez(feats / f"lits-{n}.npz", img=c["img"], lab=c["lab"], tok_l4=tok["l4"], tok_l8=tok["l8"],
                 tok_out=tok["out"], meta=_json.dumps(c["meta"]))
        aff = np.diag([-0.8, 0.8, 2.5, 1.0])
        nib.save(nib.Nifti1Image(seg, aff), str(segd / f"segmentation-{n}.nii"))
    m = train([load_case(feats / "lits-1.npz")], use_jepa=True, iters=2, device="cpu", log_every=1)
    prob = predict_case(m, load_case(feats / "lits-2.npz"), "cpu")
    assert prob.shape == (60, 60, 20) and 0 <= prob.min() and prob.max() <= 1
    save_like(prob > 0.5, segd / "segmentation-2.nii", tmp_path / "p" / "lits-2.nii.gz")
    out = nib.load(str(tmp_path / "p" / "lits-2.nii.gz"))
    assert np.allclose(out.affine, np.diag([-0.8, 0.8, 2.5, 1.0])) and out.shape == (60, 60, 20)
