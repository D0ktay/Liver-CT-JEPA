"""Bolge-tabanli kayip, hizli cikarim, son isleme ve ornekleyici sozlesmeleri."""
import json

import numpy as np
import pytest
import torch

from src.jepa.encoder import load_cfg
from src.jepa.region import (N_REGION, dice, liver_mask_from_prob, postprocess,
                             predict_volume_fast, region_ds_loss, region_loss)

CFG = load_cfg("cok_kucuk")


def _lt(shape=(2, 2, 8, 8, 8)):
    t = torch.zeros(shape)
    t[:, 0, 2:6, 2:6, 2:6] = 1
    t[:, 1, 3:5, 3:5, 3:5] = 1
    return t


def test_mukemmel_tahmin_kucuk_yanlis_tahmin_buyuk_kayip():
    t = _lt()
    v = torch.ones(2, 2)
    iyi = float(region_loss((t * 2 - 1) * 12, t, v))
    kotu = float(region_loss(-(t * 2 - 1) * 12, t, v))
    assert iyi < 0.01 < 1.5 < kotu


def test_gecersiz_kanal_kayba_HIC_girmiyor():
    """Task08'de karaciger etiketi yok: o kanalda ne tahmin edilirse edilsin kayip degismemeli."""
    t = _lt()
    v = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    lo = (t * 2 - 1) * 5
    a = float(region_loss(lo, t, v))
    lo2 = lo.clone()
    lo2[:, 0] = torch.randn_like(lo2[:, 0]) * 20
    assert float(region_loss(lo2, t, v)) == pytest.approx(a, abs=1e-7)
    # ve gradyan o kanala akmiyor
    lo3 = lo.clone().requires_grad_(True)
    region_loss(lo3, t, v).backward()
    assert float(lo3.grad[:, 0].abs().sum()) == 0.0 and float(lo3.grad[:, 1].abs().sum()) > 0


def test_karisik_partide_gecersiz_ornek_yalniz_kendi_kanalini_kapatir():
    t = _lt()
    lo = (t * 2 - 1) * 5
    v = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    lo2 = lo.clone()
    lo2[1, 0] = -lo2[1, 0]                        # gecersiz ornegin karaciger kanali bozuldu
    assert float(region_loss(lo2, t, v)) == pytest.approx(float(region_loss(lo, t, v)), abs=1e-7)
    lo3 = lo.clone()
    lo3[0, 0] = -lo3[0, 0]                        # GECERLI ornegin kanali bozulunca kayip artmali
    assert float(region_loss(lo3, t, v)) > float(region_loss(lo, t, v)) + 0.1


def test_bos_hedefte_sonlu_ve_hic_gecerli_kanal_yoksa_firlatir():
    t = torch.zeros(1, 2, 8, 8, 8)
    assert np.isfinite(float(region_loss(torch.zeros_like(t), t, torch.ones(1, 2))))
    with pytest.raises(ValueError, match="gecerli kanal"):
        region_loss(torch.zeros_like(t), t, torch.zeros(1, 2))


def test_kanal_sayisi_yanlissa_firlatir():
    with pytest.raises(ValueError):
        region_loss(torch.zeros(1, 1, 4, 4, 4), torch.zeros(1, 1, 4, 4, 4), torch.ones(1, 1))


def test_derin_denetim_uc_baslik():
    t = _lt((1, 2, 16, 16, 16))
    lo = [torch.zeros(1, 2, 16, 16, 16), torch.zeros(1, 2, 8, 8, 8), torch.zeros(1, 2, 4, 4, 4)]
    assert np.isfinite(float(region_ds_loss(lo, t, torch.ones(1, 2), CFG)))


class _Sahte(torch.nn.Module):
    """Girdi > 0 ise karaciger; > 0.5 ise tumor. Ayna TTA'ya gore esdegisken."""

    def forward(self, x, spacing=None):
        return [torch.cat([(x > 0).float() * 20 - 10, (x > 0.5).float() * 20 - 10], 1)]


def test_hizli_cikarim_dikissiz_ve_TTA_ile_ayni():
    hu = np.full((60, 70, 50), -1000.0, np.float32)
    hu[10:40, 10:50, 10:40] = 100.0                # pencere icinde > 0
    hu[20:30, 20:30, 20:30] = 240.0                # > 0.5
    p1 = predict_volume_fast(_Sahte().eval(), hu, (1, 1, 1), CFG, batch=3, tta=False)
    p2 = predict_volume_fast(_Sahte().eval(), hu, (1, 1, 1), CFG, batch=5, tta=True)
    assert p1.shape == (N_REGION,) + hu.shape
    assert np.allclose(p1, p2, atol=1e-5)
    assert ((p1[0] > 0.5) == (hu > 25.0)).all()
    assert (p1[1] > 0.5).sum() == 1000


def test_hizli_cikarim_train_modunda_ve_yanlis_spacingde_firlatir():
    m = _Sahte().eval()
    hu = np.zeros((48, 48, 48), np.float32)
    with pytest.raises(ValueError, match="spacing"):
        predict_volume_fast(m, hu, (1, 1, 2.5), CFG)
    m.train()
    with pytest.raises(ValueError, match="eval"):
        predict_volume_fast(m, hu, (1, 1, 1), CFG)


def test_son_isleme_karaciger_DISI_tumoru_siler_GT_kullanmadan():
    p = np.zeros((2, 40, 40, 40), np.float32)
    p[0, 5:25, 5:25, 5:25] = 0.9                   # karaciger
    p[0, 35:38, 35:38, 35:38] = 0.9                # kucuk sahte "karaciger" adacigi -> atilir
    p[1, 10:14, 10:14, 10:14] = 0.9                # karaciger ICI tumor
    p[1, 34:39, 34:39, 34:39] = 0.9                # karaciger DISI (uzak) yanlis pozitif
    out, info = postprocess(p)
    assert out[10:14, 10:14, 10:14].min() > 0.5
    assert out[34:39, 34:39, 34:39].max() == 0.0
    assert 0.5 < info["outside_liver_frac"] < 0.8


def test_son_isleme_icteki_deligi_doldurur_ve_sinirdaki_tumoru_korur():
    p = np.zeros((2, 40, 40, 40), np.float32)
    p[0, 5:30, 5:30, 5:30] = 0.9
    p[0, 12:18, 12:18, 12:18] = 0.1                # model tumoru karaciger saymadi (delik)
    p[1, 12:18, 12:18, 12:18] = 0.9
    p[1, 28:32, 10:14, 10:14] = 0.9                # sinirdan 2 voksel tasan (egzofitik) tumor
    out, _ = postprocess(p, dilate_mm=3.0)
    assert out[12:18, 12:18, 12:18].min() > 0.5
    assert out[28:32, 10:14, 10:14].min() > 0.5


def test_genisletme_MM_cinsinden_anizotropide():
    p = np.zeros((20, 20, 20), np.float32)
    p[8:12, 8:12, 8:12] = 0.9
    m = liver_mask_from_prob(p, close_vox=0, dilate_mm=5.0, spacing=(5.0, 1.0, 1.0))
    assert m[7, 10, 10] and not m[6, 10, 10]       # 5 mm eksende 1 voksel
    assert m[10, 3, 10] and not m[10, 2, 10]       # 1 mm eksende 5 voksel


def test_model_karaciger_bulamazsa_tumor_bos_ve_bildirilir():
    p = np.zeros((2, 16, 16, 16), np.float32)
    p[1, 4:8, 4:8, 4:8] = 0.9
    out, info = postprocess(p)
    assert out.max() == 0.0 and info["liver_vox"] == 0 and info["outside_liver_frac"] == 1.0


def test_dice_bos_gtde_nan():
    assert np.isnan(dice(np.ones((3, 3, 3)), np.zeros((3, 3, 3))))


# ------------------------------------------------------------------ ornekleyici
def _npz(tmp, cid, liver=True, tumor=True):
    img = np.zeros((60, 60, 60), np.uint8)
    lab = np.zeros_like(img)
    liv = np.zeros_like(img)
    liv[10:30, 10:30, 10:30] = 1
    if tumor:
        lab[15:20, 15:20, 15:20] = 1
    img[liv > 0] = 180
    arrays = {"img": img, "lab": lab, "meta": np.array(json.dumps({"spacing": [1, 1, 1]}))}
    if liver:
        arrays["liver"] = liv
    np.savez_compressed(tmp / f"{cid}.npz", **arrays)
    return tmp / f"{cid}.npz"


def test_ornekleyici_sekil_gecerlilik_ve_dagilim(tmp_path):
    from src.jepa.store import LabeledSampler, VolumeStore
    paths = [_npz(tmp_path, "a"), _npz(tmp_path, "t08", liver=False)]
    st = VolumeStore(paths, CFG["patch"], need_labels=True, verbose=False)
    sm = LabeledSampler(st, CFG, batch=4, seed=0, p_tumor=0.4, p_liver=0.3, prefetch=False)
    n_valid0 = 0
    for _ in range(40):
        x, y, v = sm.batch()
        assert x.shape == (4, 1, 48, 48, 48) and y.shape == (4, 2, 48, 48, 48)
        assert float(x.min()) >= -1 and float(x.max()) <= 1
        assert float((y[:, 1] * (1 - y[:, 0])).sum()) == 0          # tumor ⊂ bolge
        assert set(v[:, 1].tolist()) == {1.0}
        n_valid0 += int(v[:, 0].sum())
    assert 40 < n_valid0 < 120                                       # iki kaynak da orneklendi
    c = sm.counts
    assert c["random"] > 20 and c["tumor"] > 30 and c["liver"] > 10, c
    sm.close()


def test_tumorsuz_vakada_tumor_odagi_karacigere_duser(tmp_path):
    from src.jepa.store import LabeledSampler, VolumeStore
    st = VolumeStore([_npz(tmp_path, "saglam", tumor=False)], CFG["patch"], True, verbose=False)
    sm = LabeledSampler(st, CFG, batch=2, seed=1, p_tumor=1.0, p_liver=0.0, prefetch=False)
    for _ in range(5):
        _, y, _ = sm.batch()
        assert float(y[:, 1].sum()) == 0
    assert sm.counts["tumor"] == 0 and sm.counts["liver"] == 10
    sm.close()


def test_etiketsiz_ornekleyici_hava_yamasini_reddeder(tmp_path):
    from src.jepa.store import UnlabeledSampler, VolumeStore
    img = np.zeros((120, 120, 120), np.uint8)
    img[60:, :, :] = 150                                             # yarisi govde, yarisi hava
    np.savez_compressed(tmp_path / "v.npz", img=img, meta=np.array(json.dumps({"spacing": [1, 1, 1]})))
    st = VolumeStore([tmp_path / "v.npz"], CFG["patch"], need_labels=False, verbose=False)
    sm = UnlabeledSampler(st, CFG, batch=8, seed=0, prefetch=False)
    x = sm.batch()
    assert x.shape == (8, 1, 48, 48, 48)
    assert all(float((x[i] > -0.5).float().mean()) >= 0.3 for i in range(8))
    sm.close()


def test_kismi_depo_arka_planda_tazelenir(tmp_path):
    import time
    from src.jepa.store import VolumeStore
    paths = [_npz(tmp_path, f"c{i}") for i in range(6)]
    st = VolumeStore(paths, CFG["patch"], need_labels=True, ram_gb=0.0009, threads=1, verbose=False)
    assert st.partial and len(st.res) < 6
    gorulen = set()
    t0 = time.time()
    rng = np.random.default_rng(0)
    while len(gorulen) < 6 and time.time() - t0 < 20:
        gorulen.add(st.pick(rng)["id"])
        time.sleep(0.01)
    st.close()
    assert len(gorulen) == 6, f"yalniz {gorulen} goruldu -- korpusun bir kismi HIC orneklenmiyor"


# ---- voksel duzeyi gecerlilik (22 Eyl: liver_only vakada tumor kanali karaciger DISINDA negatif) ----
def test_voksel_gecerliligi_tekduze_ise_ornek_duzeyiyle_ayni_kayip():
    from src.jepa.region import region_loss
    g = torch.Generator().manual_seed(0)
    lg = torch.randn(3, 2, 8, 8, 8, generator=g)
    tg = (torch.rand(3, 2, 8, 8, 8, generator=g) > 0.7).float()
    v = torch.tensor([[1., 1.], [0., 1.], [1., 0.]])
    vm = v[:, :, None, None, None].expand_as(lg).clone()
    assert torch.allclose(region_loss(lg, tg, v), region_loss(lg, tg, vm), atol=1e-6)


def test_voxel_valid_liver_only_tumor_kanali_yalniz_karaciger_disinda_acik():
    from src.jepa.region import voxel_valid
    y = torch.zeros(3, 2, 16, 16, 16)
    y[:, 0, 6:10, 6:10, 6:10] = 1
    v = torch.tensor([[1., 0.], [1., 1.], [0., 1.]])       # liver_only, full, tumor_only
    vm = voxel_valid(y, v, margin_vox=2)
    assert vm.shape == y.shape
    assert vm[0, 1, 7, 7, 7] == 0 and vm[0, 1, 4, 7, 7] == 0      # icerisi + 2 voksel marj KAPALI
    assert vm[0, 1, 3, 7, 7] == 1 and vm[0, 1, 0, 0, 0] == 1      # disarisi ACIK (negatif denetim)
    assert vm[0, 0].min() == 1                                     # karaciger kanali degismez
    assert vm[1].min() == 1                                        # full: her yer acik
    assert vm[2, 0].max() == 0 and vm[2, 1].min() == 1             # tumor_only: karaciger kanali kapali


def test_voksel_maskeli_bolgede_gradyan_sifir_disinda_yanlis_pozitif_cezali():
    from src.jepa.region import region_loss, voxel_valid
    y = torch.zeros(1, 2, 16, 16, 16); y[:, 0, 6:10, 6:10, 6:10] = 1
    v = torch.tensor([[1., 0.]])
    lg = torch.zeros(1, 2, 16, 16, 16, requires_grad=True)
    region_loss(lg, y, voxel_valid(y, v, 1)).backward()
    g = lg.grad[0, 1]
    assert g[7, 7, 7] == 0 and g[5, 7, 7] == 0                     # karaciger ici + marj: denetim yok
    assert g[0, 0, 0] > 0                                          # uzakta tumor olasiligi asagi itilir


def test_derin_denetimde_voksel_gecerliligi_min_pool_ile_kuculur():
    from src.jepa.region import region_ds_loss, voxel_valid
    cfg = {"decoder": {"deep_supervision_weights": [1.0, 0.5, 0.25]}, "loss": {"dice_eps": 1e-5}}
    y = torch.zeros(2, 2, 16, 16, 16); y[:, 0, 4:12, 4:12, 4:12] = 1; y[1, 1, 6:8, 6:8, 6:8] = 1
    v = torch.tensor([[1., 0.], [1., 1.]])
    heads = [torch.randn(2, 2, 16 // s, 16 // s, 16 // s) for s in (1, 2, 4)]
    out = region_ds_loss(heads, y, voxel_valid(y, v, 1), cfg)
    assert torch.isfinite(out)


def test_voxel_valid_ayrisabilir_havuz_kup_havuzla_birebir_ayni():
    import torch.nn.functional as F
    from src.jepa.region import voxel_valid
    g = torch.Generator().manual_seed(3)
    y = torch.zeros(2, 2, 20, 20, 20); y[:, 0] = (torch.rand(2, 20, 20, 20, generator=g) > 0.995).float()
    v = torch.tensor([[1., 0.], [1., 0.]])
    ref = 1.0 - F.max_pool3d(y[:, 0:1], 7, stride=1, padding=3)[:, 0]
    assert torch.equal(voxel_valid(y, v, 3)[:, 1], ref)
