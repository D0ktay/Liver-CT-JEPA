"""22 Eyl kod denetimi bulgularinin regresyon testleri."""
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from src.jepa.encoder import load_cfg

CFG = load_cfg("cok_kucuk")


def test_hedef_encoder_HER_ZAMAN_eval_modunda():
    """Hedefte DropPath acik kalirsa JEPA gurultuye regresyon yapar (olculen bagil fark 0.379)."""
    from src.jepa.objective import build_objective
    cfg = dict(CFG, drop_path=0.3)
    m = build_objective("jepa", cfg)
    m.train()
    assert m.training and m.encoder.training and not m.target_encoder.training
    x = torch.randn(1, 1, 48, 48, 48)
    with torch.no_grad():
        a = m.target_encoder(x)["s16"]
        b = m.target_encoder(x)["s16"]
    assert torch.equal(a, b), "hedef encoder stokastik"


def test_istatistik_yalniz_istenince_hesaplanir():
    from src.jepa.objective import build_objective
    m = build_objective("jepa", CFG)
    x = torch.randn(1, 1, 48, 48, 48)
    out = m.step(x, np.random.default_rng(0), with_stats=False)
    assert out["stats"] == {} and torch.isfinite(out["loss"])
    out = m.step(x, np.random.default_rng(0), with_stats=True)
    assert {"rankme", "std_dim", "var_ratio", "vicreg"} <= set(out["stats"])
    assert {"loss_s4", "loss_s8", "loss_s16"} <= set(out["stats"])      # seviye basina kayip loglaniyor


def test_mae_ham_hedef_tahmin_edilebilir_kismi_TASIYOR():
    """Blok-normalize hedef blok ortalamasini atar; ham hedef tasir."""
    from src.jepa.objective import mae_targets
    x = torch.zeros(1, 1, 8, 8, 8)
    x[:, :, :4] = 0.7
    mask = np.ones((2, 2, 2), bool)
    ham = mae_targets(x, mask, 4, norm=False)
    assert float(ham[0, 0].mean()) == pytest.approx(0.7) and float(ham[0, -1].mean()) == 0.0
    nrm = mae_targets(x, mask, 4, norm=True)
    # normalize hedefte 0.7'lik blok ile 0.0'lik blok AYIRT EDILEMEZ (ortalama atildi);
    # kalan ~0.06 yalniz kayan nokta artigi / eps
    assert abs(float(nrm[0, 0].mean()) - float(nrm[0, -1].mean())) < 0.1 < 0.7


def _npz(tmp, cid, liver):
    img = np.full((60, 60, 60), 150, np.uint8)
    lab = np.zeros_like(img); lab[20:25, 20:25, 20:25] = 1
    arr = {"img": img, "lab": lab, "meta": np.array(json.dumps({"spacing": [1, 1, 1]}))}
    if liver:
        liv = np.zeros_like(img); liv[10:40, 10:40, 10:40] = 1
        arr["liver"] = liv
    np.savez_compressed(tmp / f"{cid}.npz", **arr)
    return tmp / f"{cid}.npz"


def test_kaynak_dengesi_karaciger_etiketli_hacim_azinlikta_olsa_bile(tmp_path):
    from src.jepa.store import LabeledSampler, VolumeStore
    paths = [_npz(tmp_path, "lits", True)] + [_npz(tmp_path, f"t{i}", False) for i in range(6)]
    st = VolumeStore(paths, CFG["patch"], need_labels=True, verbose=False)
    sm = LabeledSampler(st, CFG, batch=4, seed=0, prefetch=False, p_primary=0.65)
    assert st.group_counts() == {'full': 1, 'tumor_only': 6}
    for _ in range(100):
        sm.batch()
    pay = sm.counts["full"] / (sm.counts["full"] + sm.counts["tumor_only"])
    assert 0.55 < pay < 0.75, f"karaciger etiketli pay {pay:.2f} (duzgun secimde 1/7=0.14 olurdu)"
    sm.close()


def _nii(path, arr, zooms):
    nib.save(nib.Nifti1Image(arr, np.diag(list(zooms) + [1.0])), str(path))


def test_gt_fixed_uc_durum(tmp_path):
    from src.data.gt_fixed import fixed_reference
    seg = np.zeros((8, 8, 6), np.uint8)
    ct = np.zeros((8, 8, 6), np.int16)
    # (a) tabloda: ikisi de sahte 1x1x1 -> tablo degeri
    _nii(tmp_path / "s.nii", seg, (1, 1, 1)); _nii(tmp_path / "c.nii", ct, (1, 1, 1))
    tab = {"lits-37": {"spacing": (0.7, 0.7, 1.6), "shape": (8, 8, 6)}}
    aff, z, why = fixed_reference(tmp_path / "s.nii", tmp_path / "c.nii", "lits-37", tab)
    assert z == (0.7, 0.7, 1.6) and "tablo" in why
    assert np.allclose(np.linalg.norm(aff[:3, :3], axis=0), (0.7, 0.7, 1.6))
    # (b) GT basligi sahte, CT dogru (lits-48..52)
    _nii(tmp_path / "c2.nii", ct, (0.98, 0.98, 2.5))
    _, z, why = fixed_reference(tmp_path / "s.nii", tmp_path / "c2.nii", "lits-48", tab)
    assert z == pytest.approx((0.98, 0.98, 2.5)) and "CT" in why
    # (c) tutarli -> dokunulmaz
    _nii(tmp_path / "s3.nii", seg, (0.98, 0.98, 2.5))
    _, z, why = fixed_reference(tmp_path / "s3.nii", tmp_path / "c2.nii", "lits-5", tab)
    assert why == "degismedi"
    # sekil tutmuyorsa FIRLAT
    with pytest.raises(ValueError, match="tablo"):
        fixed_reference(tmp_path / "s.nii", tmp_path / "c.nii", "lits-37",
                        {"lits-37": {"spacing": (1, 1, 1), "shape": (9, 9, 9)}})


def test_onbellek_sahte_basligi_gercek_spacingle_orneklEr(tmp_path):
    from src.jepa.cache_build import CacheError, build_one
    ct = np.random.default_rng(0).normal(40, 10, (40, 40, 30)).astype(np.float32)
    _nii(tmp_path / "v.nii", ct, (1, 1, 1))
    d0 = build_one(tmp_path / "v.nii")
    assert d0["img"].shape == (40, 40, 30)                       # sahte baslik: hic orneklenmez
    d1 = build_one(tmp_path / "v.nii", true_spacing=((0.5, 0.5, 2.0), (40, 40, 30)))
    assert d1["img"].shape == (20, 20, 60)                       # gercek fiziksel boyut
    assert d1["meta"]["src_spacing_ras"] == [0.5, 0.5, 2.0]
    with pytest.raises(CacheError, match="sekil"):
        build_one(tmp_path / "v.nii", true_spacing=((0.5, 0.5, 2.0), (41, 40, 30)))
    _nii(tmp_path / "w.nii", ct, (0.9, 0.9, 3.0))                # ne sahte ne tablo
    with pytest.raises(CacheError, match="ne 1x1x1"):
        build_one(tmp_path / "w.nii", true_spacing=((0.5, 0.5, 2.0), (40, 40, 30)))


def test_cikarim_egitimle_AYNI_konum_kafesini_kullanir():
    """predict_volume_fast modele spacing VERMEZ (egitim de vermiyor)."""
    from src.jepa.region import predict_volume_fast

    class M(torch.nn.Module):
        def forward(self, x, spacing=None):
            assert spacing is None, "cikarim egitimde gorulmemis bir kafes veriyor"
            return [torch.zeros(x.shape[0], 2, *x.shape[2:])]
    predict_volume_fast(M().eval(), np.zeros((48, 48, 48), np.float32), (1.0011, 1, 1), CFG, tta=False)


def test_uc_vaka_turu_gecerlilik_vektoru_ve_grup_olasiliklari(tmp_path):
    """liver_only vakada tumor kanali KAPALI olmali: 'tumor yok' diye ogretilmez."""
    from src.jepa.store import LabeledSampler, VolumeStore
    img = np.full((60, 60, 60), 150, np.uint8)
    liv = np.zeros_like(img); liv[10:40, 10:40, 10:40] = 1
    np.savez_compressed(tmp_path / "ts.npz", img=img, liver=liv,
                        meta=np.array(json.dumps({"spacing": [1, 1, 1]})))
    paths = [_npz(tmp_path, "lits", True), _npz(tmp_path, "t08", False), tmp_path / "ts.npz"]
    st = VolumeStore(paths, CFG["patch"], need_labels=True, verbose=False)
    assert st.group_counts() == {"full": 1, "tumor_only": 1, "liver_only": 1}
    sm = LabeledSampler(st, CFG, batch=6, seed=0, prefetch=False,
                        group_probs={"full": 0.5, "tumor_only": 0.25, "liver_only": 0.25})
    gorulen = set()
    for _ in range(60):
        _, y, v = sm.batch()
        for i in range(6):
            gorulen.add(tuple(v[i].tolist()))
            if v[i, 1] == 0:
                assert float(y[i, 1].sum()) == 0
    assert gorulen == {(1.0, 1.0), (0.0, 1.0), (1.0, 0.0)}
    n = sum(sm.counts[g] for g in ("full", "tumor_only", "liver_only"))
    assert 0.4 < sm.counts["full"] / n < 0.6 and 0.15 < sm.counts["liver_only"] / n < 0.35
    sm.close()


def test_onbellek_tumor_etiketsiz_kaynakta_lab_YAZMAZ(tmp_path):
    from src.jepa.cache_build import build_one
    ct = np.random.default_rng(0).normal(40, 10, (40, 40, 30)).astype(np.float32)
    seg = np.zeros((40, 40, 30), np.uint8); seg[10:30, 10:30, 5:25] = 6      # AMOS: 6 = karaciger
    _nii(tmp_path / "v.nii", ct, (1, 1, 1)); _nii(tmp_path / "s.nii", seg, (1, 1, 1))
    d = build_one(tmp_path / "v.nii", label_path=tmp_path / "s.nii", tumor_label=-1, liver_label=6)
    assert "lab" not in d and int(d["liver"].sum()) == 20 * 20 * 20
    assert d["meta"]["tumor_label"] is None and d["meta"]["liver_label"] == 6


def test_etiket_basligi_bozuksa_CT_yonu_uygulanir_maske_AYNALANMAZ(tmp_path):
    """lits-51 (QC yakaladi): etiket kendi affine'iyle cevrilince karaciger maskesi mideye dusuyordu."""
    from src.jepa.cache_build import build_one
    ct = np.full((40, 40, 30), -1000.0, np.float32)
    seg = np.zeros((40, 40, 30), np.uint8)
    ct[5:15, 10:30, 5:25] = 100.0                       # "karaciger" dizinin x-BASINDA
    seg[5:15, 10:30, 5:25] = 1
    seg[8:11, 15:20, 10:15] = 2
    ct_aff = np.diag([-1.0, -1.0, 1.0, 1.0])            # CT: LPS benzeri (x, y ters)
    nib.save(nib.Nifti1Image(ct, ct_aff), str(tmp_path / "v.nii"))
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(tmp_path / "s.nii"))   # etiket basligi BOZUK (kimlik)
    d = build_one(tmp_path / "v.nii", label_path=tmp_path / "s.nii", liver_label=1,
                  trust_ct_spacing=True)
    img, liv = d["img"], d["liver"] > 0
    assert d["meta"]["label_ornt_fixed"] is True
    assert int(liv.sum()) == 10 * 20 * 20
    assert float(img[liv].min()) > 150, "maske goruntudeki organin USTUNDE degil (aynalanmis)"
    assert float(img[~liv].max()) < 50


def test_liver_only_vakada_kapali_delik_hacim_duzeyinde_dolduruluyor(tmp_path):
    """TS/AMOS organ etiketi ic lezyonu dislarsa o bolge 'karaciger disi' sayilip tumor-negatif
    ogretilirdi. full (LiTS) vakada etikete DOKUNULMAZ."""
    import json
    import numpy as np
    from src.jepa.store import load_case
    img = np.full((40, 40, 40), 120, np.uint8)
    liv = np.zeros((40, 40, 40), np.uint8); liv[8:32, 8:32, 8:32] = 1; liv[18:22, 18:22, 18:22] = 0
    meta = json.dumps({"spacing": [1, 1, 1]})
    np.savez(tmp_path / "a.npz", img=img, liver=liv, meta=meta)
    lab = np.zeros_like(liv); lab[10:12, 10:12, 10:12] = 1
    np.savez(tmp_path / "b.npz", img=img, liver=liv, lab=lab, meta=meta)
    rng = np.random.default_rng(0)
    a = load_case(tmp_path / "a.npz", [32, 32, 32], True, rng, fill_liver_holes=True)
    b = load_case(tmp_path / "b.npz", [32, 32, 32], True, rng, fill_liver_holes=True)
    assert load_case(tmp_path / "a.npz", [32, 32, 32], True, rng)["seg"][20, 20, 20] == 0   # varsayilan: KAPALI
    assert a["group"] == "liver_only" and a["seg"][20, 20, 20] == 1
    assert b["group"] == "full" and b["seg"][20, 20, 20] == 0


def test_pretrain_drop_path_yalniz_on_egitim_encoderini_etkiler_kaydedilen_cfg_degismez():
    """I-JEPA'da cevrimici encoder'da DropPath yok. pretrain.drop_path verilirse on egitim encoder'i
    onunla kurulur; pretrain.py'nin encoder.pt'ye yazdigi cfg (ince ayar profili) DEGISMEZ."""
    from src.jepa.encoder import load_cfg
    from src.jepa.objective import PretrainModel
    cfg = load_cfg("cok_kucuk"); cfg["drop_path"] = 0.1
    cfg["pretrain"] = dict(cfg["pretrain"], drop_path=0.0)
    m = PretrainModel(cfg, "jepa")
    assert float(m.encoder.cfg["drop_path"]) == 0.0 and float(cfg["drop_path"]) == 0.1
    assert float(PretrainModel(load_cfg("cok_kucuk"), "jepa").encoder.cfg["drop_path"]) == 0.0  # anahtar yok -> profil
