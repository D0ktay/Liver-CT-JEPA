"""On egitim hedefleri (JEPA / MAE / none) -- VERI OLMADAN calisir.

Bu dosyanin varlik sebebi tek cumle: **kontrol kolu gercekten kontrol olsun.**
JEPA ile MAE ayni encoder'i, ayni predictor'u, ayni maskeyi, ayni tohumu ve ayni
ornekleme sirasini gormezse "JEPA daha iyi" cumlesi olcum degil temenni olur.

Yakalanan sessiz hatalar:
  * maskenin izgara yerine voksel uzayinda 3 kaydirilmasi -> hedefin bir kismi
    baglamda kalir, gorev sessizce kolaylasir, kayip duser, kimse fark etmez;
  * baglam ile hedefin kesismesi (ayni sey);
  * EMA hedef encoder'ina gradyan sizmasi -> temsil cokmesi;
  * cokmenin (rank 1 temsil) kayipta gorunmemesi.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.jepa.decoder import SegModel
from src.jepa.encoder import LiverEncoder, load_cfg, load_encoder
from src.jepa.objective import (
    CollapseError,
    CollapseMonitor,
    PretrainModel,
    apply_input_mask,
    build_objective,
    collapse_stats,
    ema_momentum,
    ema_update,
    jepa_loss,
    mae_targets,
    pretrain_step,
    sample_masks,
    seg_step,
    target_union,
)

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return load_cfg("cok_kucuk")


# ------------------------------------------------------------------- maske
def test_maske_blok_sayisi_ve_hacim_orani():
    cfg = _cfg()
    grid = (12, 12, 12)
    n = int(np.prod(grid))
    rng = np.random.default_rng(0)
    ctx, tgt = sample_masks(cfg, grid, rng)
    lo, hi = cfg["mask"]["block_frac"]
    assert tgt.shape == (cfg["mask"]["n_blocks"], *grid)
    for b in tgt:
        assert lo <= b.sum() / n <= hi, f"blok orani disarida: {b.sum() / n}"
    u = target_union(tgt).sum() / n
    ulo, uhi = cfg["mask"]["union_frac"]
    assert ulo <= u <= uhi, f"birlesim orani disarida: {u}"
    # ayni tohum -> ayni maske
    c2, t2 = sample_masks(cfg, grid, np.random.default_rng(0))
    assert np.array_equal(ctx, c2) and np.array_equal(tgt, t2)


def test_maske_izgaraya_hizali_ve_girdi_uzayina_dogru_yansiyor():
    """4'luk hizalama bozulursa (orn. 3 kaydirma) hedefin bir kismi baglamda kalir."""
    stride = _cfg()["strides"]["s4"]
    x = torch.arange(1, 1 + 8 * 8 * 8, dtype=torch.float32).reshape(1, 1, 8, 8, 8)
    keep = np.ones((2, 2, 2), bool)
    keep[0, 0, 0] = False                       # tek token gizlendi
    y = apply_input_mask(x, keep, stride, fill=0.0)
    assert torch.equal(y[0, 0, :stride, :stride, :stride],
                       torch.zeros(stride, stride, stride))
    gorunur = torch.ones(1, 1, 8, 8, 8, dtype=torch.bool)
    gorunur[0, 0, :stride, :stride, :stride] = False
    assert torch.equal(y[gorunur], x[gorunur]), "maskesiz voksel degistirilmis"
    assert int((y == 0).sum()) == stride ** 3, "maske izgaraya hizali degil"


def test_baglam_ve_hedef_ayrik():
    cfg = _cfg()
    grid = (12, 12, 12)
    n = int(np.prod(grid))
    for seed in range(5):
        ctx, tgt = sample_masks(cfg, grid, np.random.default_rng(seed))
        u = target_union(tgt)
        assert not np.any(ctx & u), "baglam ile hedef kesisiyor -- gorev sessizce kolaylasti"
        assert ctx.sum() / n >= cfg["mask"]["min_visible_frac"]
        assert u.sum() >= cfg["mask"]["min_target_tokens"]


def test_cok_kucuk_izgarada_firlatir():
    """Kosullar saglanamiyorsa sessizce kotu maske dondurmek yerine FIRLAT."""
    cfg = _cfg()
    with pytest.raises(ValueError, match="maske"):
        sample_masks(cfg, (4, 4, 4), np.random.default_rng(0))


def test_ayni_tohum_jepa_ve_mae_ayni_maske():
    """Kontrol kolu farkli maske gorurse kiyas anlamsiz olur."""
    cfg = _cfg()
    grid = (12, 12, 12)
    a = sample_masks(cfg, grid, np.random.default_rng(7))
    b = sample_masks(cfg, grid, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    x = torch.randn(1, 1, 48, 48, 48)
    outs = {}
    for name in ("jepa", "mae"):
        torch.manual_seed(0)
        m = build_objective(name, cfg)
        outs[name] = m.step(x, np.random.default_rng(7), step=0, total=10)["mask"]
    assert np.array_equal(outs["jepa"][0], outs["mae"][0])
    assert np.array_equal(outs["jepa"][1], outs["mae"][1])


# ------------------------------------------------------------------ kayiplar
def test_jepa_kaybi_mukemmel_tahminde_kucuk_bos_hedefte_firlatir():
    t = torch.randn(1, 20, 32)
    assert float(jepa_loss(t, t)) < 1e-5
    assert float(jepa_loss(torch.zeros_like(t), t)) > 0.1
    with pytest.raises(ValueError, match="bos"):
        jepa_loss(torch.zeros(1, 0, 32), torch.zeros(1, 0, 32))


def test_mae_hedefi_blok_normalize_ve_maskesiz_voksel_kayba_girmez():
    stride = 4
    x = torch.randn(1, 1, 16, 16, 16)
    m = np.zeros((4, 4, 4), bool)
    m[0, 0, 0] = m[1, 2, 3] = True
    t = mae_targets(x, m, stride)
    assert t.shape == (1, 2, stride ** 3)
    assert float(t.mean().abs()) < 1e-5 and float(t.std()) == pytest.approx(1.0, abs=0.1)
    x2 = x.clone()
    x2[0, 0, 8:12, 8:12, 8:12] += 100.0            # maskesiz blok bozuldu
    assert torch.allclose(t, mae_targets(x2, m, stride), atol=1e-5)


# ----------------------------------------------------------------------- EMA
def test_ema_takvimi_ve_guncelleme():
    cfg = _cfg()
    total = 100
    m0 = ema_momentum(0, total, cfg)
    mT = ema_momentum(total, total, cfg)
    assert m0 == pytest.approx(cfg["ema"]["start"]) and mT == pytest.approx(cfg["ema"]["end"])
    ms = [ema_momentum(s, total, cfg) for s in range(total + 1)]
    assert all(b >= a for a, b in zip(ms, ms[1:]))
    torch.manual_seed(0)
    src = torch.nn.Linear(3, 3)
    tgt = torch.nn.Linear(3, 3)
    onceki = tgt.weight.detach().clone()
    ema_update(tgt, src, 0.9)
    beklenen = 0.9 * onceki + 0.1 * src.weight.detach()
    assert torch.allclose(tgt.weight.detach(), beklenen, atol=1e-7)


def test_ema_hedef_encoder_gradyan_almaz():
    cfg = _cfg()
    torch.manual_seed(0)
    m = PretrainModel(cfg, "jepa")
    out = m.step(torch.randn(1, 1, 48, 48, 48), np.random.default_rng(0), step=0, total=4)
    out["loss"].backward()
    assert all(p.grad is None for p in m.target_encoder.parameters())
    assert any(p.grad is not None for p in m.encoder.parameters())


# --------------------------------------------------------------------- cokus
def test_cokus_metrikleri(tmp_path):
    cfg = _cfg()
    d = 32
    ayni = torch.ones(64, d) * torch.randn(1, d)
    s = collapse_stats(ayni)
    assert s["rankme"] == pytest.approx(1.0, abs=1e-3)
    assert s["std_dim"] == pytest.approx(0.0, abs=1e-5)
    mon = CollapseMonitor(cfg, d, out_dir=tmp_path)
    rast = torch.randn(256, d)
    sr = collapse_stats(rast)
    assert sr["rankme"] > 0.05 * d and not mon.check(sr)
    for i in range(cfg["collapse"]["patience"] - 1):
        assert mon.check(s) is True
    with pytest.raises(CollapseError):
        mon.check(s)
    assert json.loads((tmp_path / "collapse.json").read_text())["rankme"] < 1.1


def test_objective_none():
    cfg = _cfg()
    m = build_objective("none", cfg)
    assert isinstance(m.encoder, LiverEncoder)
    with pytest.raises(ValueError, match="none"):
        pretrain_step(m, torch.zeros(1, 1, 48, 48, 48), np.random.default_rng(0), 0, 1)


def test_bilinmeyen_hedef_firlatir():
    with pytest.raises(ValueError, match="hedef"):
        build_objective("simclr", _cfg())


# ------------------------------------------------------------- uctan uca duman
def _kure(shape, merkez, r):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - merkez[0]) ** 2 + (yy - merkez[1]) ** 2 + (xx - merkez[2]) ** 2) <= r * r


@pytest.mark.parametrize("hedef", ["jepa", "mae", "none"])
def test_uctan_uca_duman(hedef, tmp_path):
    """On egitim -> ckpt -> strict yukleme -> ince ayar -> tahmin -> NIfTI -> evaluate.

    Gercek veri yok, gercek egitim yok: hattin ucundan ucuna gectigini ve
    `src.metrics.evaluate`'in bu ciktiyi okuyabildigini gosterir.
    """
    nib = pytest.importorskip("nibabel")
    cfg = _cfg()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    shape = (48, 48, 48)
    vol = (rng.normal(40, 20, shape)).astype(np.float32)
    gt = np.zeros(shape, np.uint8)
    gt[_kure(shape, (24, 24, 24), 6)] = 2          # LiTS tumor etiketi 2
    vol[gt == 2] += 120.0
    x = torch.from_numpy(vol)[None, None]
    y = torch.from_numpy((gt == 2).astype(np.float32))[None, None]

    # --- 1) on egitim (2 epoch x 2 adim)
    model = build_objective(hedef, cfg)
    once = model.encoder.stem[0].weight.detach().clone()
    if hedef != "none":
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]))
        mon = CollapseMonitor(cfg, model.pred_dim, out_dir=tmp_path)
        toplam = 4
        for step in range(toplam):
            out = pretrain_step(model, x, np.random.default_rng(step), step, toplam)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            model.ema_step(step, toplam)
            assert np.isfinite(float(out["loss"])), "on egitim kaybi sonlu degil"
            assert mon.check(out["stats"]) is False, "cokus tetiklendi"
        assert not torch.equal(once, model.encoder.stem[0].weight.detach()), \
            "on egitim parametreleri degistirmedi"

    ckpt = tmp_path / "enc.pt"
    torch.save({"cfg": cfg, "encoder": model.encoder.state_dict(), "objective": hedef}, ckpt)

    # --- 2) ince ayar (strict yukleme)
    enc = load_encoder(ckpt, strict=True)
    seg = SegModel(cfg, encoder=enc)
    opt = torch.optim.AdamW(seg.parameters(), lr=1e-3)
    for _ in range(2):
        loss = seg_step(seg, x, y, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        assert np.isfinite(float(loss))

    # --- 3) tahmin -> NIfTI (GT affine'i ile)
    from src.jepa.decoder import predict_volume
    prob = predict_volume(seg.eval(), vol, cfg["spacing_mm"], cfg)
    assert prob.shape == shape
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    gt_p, pred_p = tmp_path / "gt.nii.gz", tmp_path / "pred.nii.gz"
    nib.save(nib.Nifti1Image(gt, affine), gt_p)
    nib.save(nib.Nifti1Image((prob > 0.5).astype(np.uint8), affine), pred_p)

    # --- 4) src.metrics.evaluate (--gt-label 2 / --pred-label 1 AYRI)
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(f"case_id,gt_path,pred_path\nsentetik-1,{gt_p},{pred_p}\n")
    out_dir = tmp_path / "eval"
    r = subprocess.run([sys.executable, "-m", "src.metrics.evaluate", "--pairs", str(pairs),
                        "--out", str(out_dir), "--gt-label", "2", "--pred-label", "1"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    summary = json.loads((out_dir / "summary.json").read_text())
    assert "dice_median" in summary["voxel"]
    assert summary["n_cases"] == 1
