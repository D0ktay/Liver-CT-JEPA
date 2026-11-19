"""Kendi 3D CT encoder'imiz -- VERI OLMADAN calisir.

Her test, gercek egitimde ancak haftalar sonra fark edilecek bir hatayi hedefliyor:
token'in fiziksel boyutunun sessizce buyumesi (Rad-JEPA hatasi), B=1'de patlayan
BatchNorm istatistigi, ince ayarda kirilan agirlik yuklemesi, bolunemeyen girdi.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from src.jepa.blocks import count_params
from src.jepa.data import hu_window
from src.jepa.encoder import (
    LiverEncoder,
    STRIDE_KEYS,
    lesion_tokens,
    load_cfg,
    load_train_yaml,
    pick_config,
    token_mm,
    tokens_per_axis,
)

# Olculen medyanlar (CLAUDE.md / outputs/kaggle131/lits_gt_lesions.csv)
BILIC_MEDIAN_MM3 = 2490.0      # Bilic ve ark. LiTS medyan tumor hacmi
OUR_MEDIAN_DIAM_MM = 10.03     # bizim 908 bilesende olctugumuz esdeger cap


def _profil_butceleri():
    return {"cok_kucuk": 0.80e6, "kucuk": 2.75e6, "orta": 19.84e6}


# ------------------------------------------------------------------ konfig
def test_konfig_uc_profil_ve_param_butcesi():
    """Profil buyurse (ya da kucukse) GPU butcesi sessizce degisir -- yakala."""
    raw = load_train_yaml()
    assert {"cok_kucuk", "kucuk", "orta"} <= set(raw["profiles"])
    for name, beklenen in _profil_butceleri().items():
        n = count_params(load_cfg(name))
        assert abs(n - beklenen) / beklenen < 0.10, f"{name}: {n} vs {beklenen}"


def test_konfig_secim_kurali():
    assert pick_config(131, 24) == "kucuk"      # bugunku korpus
    assert pick_config(500, 24) == "orta"
    assert pick_config(1500, 24) == "orta"
    assert pick_config(1500, 4) == "kucuk"      # GPU yoksa profil buyutulmez
    for n in (10, 131, 299, 300, 999, 1000, 5000):
        for g in (0, 4, 8, 24, 80):
            assert pick_config(n, g) != "buyuk"


def test_hu_penceresi_paths_yamldan_okunuyor_kopyalanmiyor():
    """Iki ayri pencere tanimi = egitim/cikarim ayrismasi. Tek kaynak: paths.yaml."""
    cfg = load_cfg("kucuk")
    ref = yaml.safe_load(open("configs/paths.yaml"))["metrics"]["hu_window"]
    assert list(cfg["hu_window"]) == [float(x) for x in ref]
    metin = open("configs/train.yaml").read()
    assert "-200" not in metin and "250" not in metin, "HU penceresi train.yaml'a kopyalanmis"


def test_hu_window_out_range_encoder_icin_eksi_bir_bir():
    """Encoder [-1,1] ister; ikinci bir normalize fonksiyonu YAZILMAZ (tek kaynak)."""
    ct = np.array([-1000, -200, 25, 250, 3000], np.float32)
    z01 = hu_window(ct, (-200, 250))
    assert z01.min() == 0.0 and z01.max() == 1.0
    z11 = hu_window(ct, (-200, 250), out_range=(-1.0, 1.0))
    assert z11.dtype == np.float32
    assert z11.min() == -1.0 and z11.max() == 1.0
    np.testing.assert_allclose(z11, z01 * 2 - 1, atol=1e-6)


# ------------------------------------------------------- token fizigi (yildiz)
def test_token_fiziksel_boyutu_ve_medyan_lezyon():
    """Stride 4 -> 4 mm token. Biri stride'i 8'e cikarirsa burasi kirmizi olur.

    Rad-JEPA token'i 20x12.8x12.8 mm = 3277 mm^3; bizim medyanimiz orada 0.16 token
    ediyor -- olculmus basarisizligin sayisal aciklamasi budur.
    """
    cfg = load_cfg("kucuk")
    mm = token_mm(cfg, (1.0, 1.0, 1.0))
    assert mm["s4"] == (4.0, 4.0, 4.0)
    assert mm["s16"] == (16.0, 16.0, 16.0)
    # medyan lezyonumuz stride-4 izgarasinda eksen basina >= 2 token olmali
    assert tokens_per_axis(OUR_MEDIAN_DIAM_MM, 1.0, stride=4) >= 2.0
    # Bilic medyani (2490 mm^3) stride-4'te >= 30 token
    assert lesion_tokens(BILIC_MEDIAN_MM3, spacing=1.0, stride=4) >= 30.0
    # durust alt sinir: 5 mm lezyon ~1 token -> orada kazanc beklemiyoruz
    assert lesion_tokens(65.0, spacing=1.0, stride=4) < 2.0


# --------------------------------------------------------------- encoder
def test_piramit_sekilleri():
    cfg = load_cfg("cok_kucuk")
    enc = LiverEncoder(cfg).eval()
    with torch.no_grad():
        p = enc(torch.zeros(2, 1, 32, 32, 32))
    assert list(p) == list(STRIDE_KEYS)
    c1, c2, c3 = cfg["conv_width"]
    bekle = {"s1": (c1, 32), "s2": (c2, 16), "s4": (c3, 8),
             "s8": (cfg["s8"]["dim"], 4), "s16": (cfg["s16"]["dim"], 2)}
    for k, (c, n) in bekle.items():
        assert p[k].shape == (2, c, n, n, n), f"{k}: {tuple(p[k].shape)}"


def test_bolunemeyen_girdi_firlatir():
    """Sessizce kirpmak = maskenin GT ile hizasi kayar. Firlat."""
    enc = LiverEncoder(load_cfg("cok_kucuk")).eval()
    with pytest.raises(ValueError, match="16"):
        enc(torch.zeros(2, 1, 30, 32, 32))


def test_batchnorm_yok():
    """B=1/2 yamada BatchNorm istatistigi coper; cikarimda sessizce baska sonuc verir."""
    enc = LiverEncoder(load_cfg("kucuk"))
    kotu = [type(m).__name__ for m in enc.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert kotu == []


def test_farkli_girdi_boyutunda_ayni_agirlik_calisir():
    """On egitim 48^3, ince ayar 128^3 -- sabit konum gomulmesi olsa burasi kirilirdi."""
    enc = LiverEncoder(load_cfg("cok_kucuk")).eval()
    with torch.no_grad():
        a = enc(torch.zeros(1, 1, 32, 32, 32))["s16"]
        b = enc(torch.zeros(1, 1, 48, 32, 16))["s16"]
    assert a.shape[2:] == (2, 2, 2) and b.shape[2:] == (3, 2, 1)


def test_state_dict_gidis_donus(tmp_path):
    cfg = load_cfg("cok_kucuk")
    torch.manual_seed(0)
    enc = LiverEncoder(cfg).eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        y0 = enc(x)["s16"]
    f = tmp_path / "enc.pt"
    torch.save(enc.state_dict(), f)
    torch.manual_seed(1)
    enc2 = LiverEncoder(cfg).eval()
    eksik, fazla = enc2.load_state_dict(torch.load(f), strict=True)
    assert list(eksik) == [] and list(fazla) == []
    with torch.no_grad():
        y1 = enc2(x)["s16"]
    assert torch.equal(y0, y1)


def test_deterministik_ve_sonlu():
    torch.manual_seed(3)
    enc = LiverEncoder(load_cfg("cok_kucuk")).eval()
    x = torch.randn(2, 1, 32, 32, 32)
    with torch.no_grad():
        a, b = enc(x), enc(x)
    for k in STRIDE_KEYS:
        assert torch.isfinite(a[k]).all(), k
        assert torch.equal(a[k], b[k]), k


# ---- conv bloklarindan sonra yeniden maskeleme (on egitim secenegi; varsayilan kapali) ----
def _kucuk_enc():
    import torch
    from src.jepa.encoder import LiverEncoder, load_cfg
    torch.manual_seed(0)
    return LiverEncoder(load_cfg("cok_kucuk")).eval()


def test_keep_mask_None_ise_cikti_birebir_eski():
    import torch
    enc = _kucuk_enc(); x = torch.randn(1, 1, 32, 32, 32)
    a, b = enc(x), enc(x, keep_mask=None)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_yeniden_maskeleme_s4te_maskeli_konumlari_sifirlar_ve_icerik_sizmaz():
    import torch
    enc = _kucuk_enc()
    keep = torch.ones(8, 8, 8, dtype=torch.bool); keep[2:5, 2:5, 2:5] = False
    x = torch.randn(1, 1, 32, 32, 32)
    x2 = x.clone(); x2[:, :, 8:20, 8:20, 8:20] = 7.0           # yalniz MASKELI bolgenin icerigi degisti
    from src.jepa.objective import apply_input_mask
    with torch.no_grad():                                      # gercek hat: once girdi doldurulur (-1 = hava)
        p = enc(apply_input_mask(x, keep.numpy(), 4, fill=-1.0), keep_mask=keep)
        p2 = enc(apply_input_mask(x2, keep.numpy(), 4, fill=-1.0), keep_mask=keep)
    assert float(p["s4"][0, :, 3, 3, 3].abs().max()) == 0.0
    assert float(p["s2"][0, :, 5:9, 5:9, 5:9].abs().max()) == 0.0
    # maskeli icerik, yogun conv'a ragmen GORUNEN s4 token'larina hic ulasmiyor
    assert torch.equal(p["s4"][0][:, keep], p2["s4"][0][:, keep])


def test_keep_mask_yanlis_izgarada_firlatir():
    import pytest, torch
    enc = _kucuk_enc()
    with pytest.raises(ValueError):
        enc(torch.randn(1, 1, 32, 32, 32), keep_mask=torch.ones(4, 4, 4, dtype=torch.bool))


def test_load_encoder_target_anahtari_secilir_yoksa_firlatir(tmp_path):
    import pytest, torch
    from src.jepa.encoder import LiverEncoder, load_cfg, load_encoder
    cfg = load_cfg("cok_kucuk")
    a, b = LiverEncoder(cfg), LiverEncoder(cfg)
    torch.save({"cfg": cfg, "encoder": a.state_dict(), "target_encoder": b.state_dict()}, tmp_path / "e.pt")
    torch.save({"cfg": cfg, "encoder": a.state_dict()}, tmp_path / "eski.pt")
    w = lambda m: next(iter(m.state_dict().values()))
    assert torch.equal(w(load_encoder(tmp_path / "e.pt")), w(a))
    assert torch.equal(w(load_encoder(tmp_path / "e.pt", key="target_encoder")), w(b))
    with pytest.raises(KeyError):
        load_encoder(tmp_path / "eski.pt", key="target_encoder")


# ---- s8/s16 maskeli token dusurme (B1-b; varsayilan kapali) ----
def test_drop_tokens_gorunur_token_ciktisi_maskeli_icerikten_bagimsiz_ve_maskeli_yer_sifir():
    import torch
    from src.jepa.objective import apply_input_mask
    enc = _kucuk_enc()
    keep = torch.ones(8, 8, 8, dtype=torch.bool); keep[0:4, 0:4, 0:4] = False     # s16'da tam 1 token maskeli
    x = torch.randn(1, 1, 32, 32, 32); x2 = x.clone(); x2[:, :, 0:16, 0:16, 0:16] = 5.0
    with torch.no_grad():
        p = enc(apply_input_mask(x, keep.numpy(), 4, -1.0), keep_mask=keep, drop_tokens=True)
        p2 = enc(apply_input_mask(x2, keep.numpy(), 4, -1.0), keep_mask=keep, drop_tokens=True)
    k16 = p["keep_s16"]
    assert k16.shape == (2, 2, 2) and not bool(k16[0, 0, 0]) and int(k16.sum()) == 7
    assert float(p["s16"][0, :, 0, 0, 0].abs().max()) == 0.0                # dusurulen token cikista sifir
    assert torch.equal(p["s16"][0][:, k16], p2["s16"][0][:, k16])           # gorunurler maskeli icerigi gormedi
    assert not bool(p["s8"][0, :, 0, 0, 0].abs().max() > 0)


def test_drop_tokens_keep_mask_olmadan_firlatir_ve_kapaliyken_keep_s16_yok():
    import pytest, torch
    enc = _kucuk_enc(); x = torch.randn(1, 1, 32, 32, 32)
    with pytest.raises(ValueError):
        enc(x, drop_tokens=True)
    assert "keep_s16" not in enc(x, keep_mask=torch.ones(8, 8, 8, dtype=torch.bool))


def test_drop_tokens_pretrain_adimi_kosar_ve_predictor_yalniz_gorunur_baglam_alir():
    import numpy as np, torch
    from src.jepa.encoder import load_cfg
    from src.jepa.objective import PretrainModel, pretrain_step
    cfg = load_cfg("cok_kucuk"); cfg["pretrain"] = dict(cfg["pretrain"], drop_tokens=True, remask_conv=True)
    m = PretrainModel(cfg, "jepa")
    seen = {}
    orig = m.predictor.forward
    def spy(ctx_tok, ctx_coords, q_coords):
        seen["n"] = ctx_tok.shape[1]; return orig(ctx_tok, ctx_coords, q_coords)
    m.predictor.forward = spy
    out = pretrain_step(m, torch.randn(1, 1, 48, 48, 48), np.random.default_rng(0))
    assert torch.isfinite(out["loss"]) and seen["n"] < 27                     # 3^3 s16 token'dan azi gitti
    out["loss"].backward()
