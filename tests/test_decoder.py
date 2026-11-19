"""Segmentasyon cozucusu + kayiplar + kayan pencere -- VERI OLMADAN calisir.

Hedeflenen sessiz hatalar:
  * ince skip'lerin (stride 1/2) bagli gorunup aslinda kullanilmamasi -> kucuk
    lezyonlarda cozunurluk kaybi, Dice'ta gorunmez;
  * marj bandinin `binary_dilation(iterations=N)` ile uretilmesi -> anizotropik
    vokselde 5 mm degil 5 VOKSEL (CLAUDE.md kural 6);
  * kayan pencerede dikis izi / yanlis spacing ile tahmin;
  * fp16 altinda Dice'in tasmasi.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

import src.jepa.decoder as decoder_mod
from src.jepa.decoder import (
    SegDecoder,
    SegModel,
    deep_supervision_loss,
    deep_supervision_weights,
    dice_ce_loss,
    focal_loss,
    make_loss,
    margin_weight_map,
    margin_weighted_ce,
    predict_volume,
    tversky_loss,
)
from src.jepa.data import hu_window
from src.jepa.encoder import LiverEncoder, load_cfg


def _cfg():
    return load_cfg("cok_kucuk")


# ------------------------------------------------------------------ cozucu
def test_cikti_tam_cozunurluk_ve_derin_denetim():
    cfg = _cfg()
    m = SegModel(cfg).eval()
    x = torch.zeros(2, 1, 32, 32, 32)
    with torch.no_grad():
        logits = m(x)
    assert len(logits) == 3, "derin denetim basliklari: stride 1/2/4"
    assert logits[0].shape == x.shape                 # tam cozunurluk
    assert logits[1].shape[2:] == (16, 16, 16)
    assert logits[2].shape[2:] == (8, 8, 8)
    w = deep_supervision_weights(cfg)
    assert len(w) == 3 and sum(w) == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]


def test_ince_skipler_gercekten_kullaniliyor():
    """s1 skip'i sifirlaninca cikti DEGISMELI; degismiyorsa skip olu baglantidir."""
    cfg = _cfg()
    torch.manual_seed(0)
    enc, dec = LiverEncoder(cfg).eval(), SegDecoder(cfg).eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        pyr = enc(x)
        a = dec(pyr)[0]
        for k in ("s1", "s2"):
            bozuk = dict(pyr)
            bozuk[k] = torch.zeros_like(pyr[k])
            b = dec(bozuk)[0]
            assert not torch.allclose(a, b, atol=1e-6), f"{k} skip'i ciktiyi etkilemiyor"


# ------------------------------------------------------------------ kayiplar
def test_dice_ce_bos_maskede_sonlu_ve_mukemmelde_kucuk():
    bos = torch.zeros(1, 1, 4, 4, 4)
    l = dice_ce_loss(torch.full_like(bos, -10.0), bos)
    assert torch.isfinite(l) and float(l) < 0.05          # bos GT + bos tahmin: ceza yok
    t = torch.zeros(1, 1, 4, 4, 4)
    t[..., 1:3, 1:3, 1:3] = 1.0
    iyi = dice_ce_loss(torch.where(t > 0, 10.0, -10.0), t)
    kotu = dice_ce_loss(torch.where(t > 0, -10.0, 10.0), t)
    assert float(iyi) < 0.05 < float(kotu)
    assert torch.isfinite(dice_ce_loss(torch.zeros(1, 1, 4, 4, 4), bos))


def test_tversky_ve_focal_kapali_formla():
    """logit 0 -> p = 0.5; elle hesap: Tversky 0.5, focal 0.25*ln2."""
    t = torch.zeros(1, 1, 2, 2, 2)
    t[..., 0, :, :] = 1.0                                  # 4 pozitif / 4 negatif
    z = torch.zeros_like(t)
    assert float(tversky_loss(z, t, alpha=0.3, beta=0.7, eps=0.0)) == pytest.approx(0.5, abs=1e-6)
    beklenen = 0.25 * float(np.log(2.0))
    assert float(focal_loss(z, t, gamma=2.0)) == pytest.approx(beklenen, abs=1e-6)


def test_marj_agirligi_anizotropide_EDT():
    """CLAUDE.md kural 6: mm marji EDT ile. spacing (1,1,5)'te 5 mm = z'de 1 voksel."""
    gt = np.zeros((11, 11, 11), np.uint8)
    gt[5, 5, 5] = 1
    spacing = (1.0, 1.0, 5.0)
    w = margin_weight_map(gt, spacing, margin_mm=5.0, lam=2.0)
    assert w.shape == gt.shape
    band = w > 1.0
    assert band[0, 5, 5] == band[10, 5, 5] == True          # eksen 0: 5 mm = 5 voksel
    assert band[5, 5, 4] and band[5, 5, 6]                  # eksen 2 (5 mm): 1 voksel
    assert not band[5, 5, 3] and not band[5, 5, 7]          # eksen 2: 10 mm -> disarida
    dil = ndimage.binary_dilation(gt.astype(bool), iterations=5)
    assert not np.array_equal(band, dil), "marj bandi voksel dilatasyonuyla ayni -- kural 6 ihlali"
    assert float(w.max()) == pytest.approx(2.0)
    assert float(w.min()) == pytest.approx(1.0)


def test_margin_weighted_ce_marj_icindeki_hatayi_daha_cok_cezalandiriyor():
    gt = np.zeros((9, 9, 9), np.uint8)
    gt[4, 4, 4] = 1
    t = torch.from_numpy(gt[None, None].astype(np.float32))
    w = margin_weight_map(gt, (1.0, 1.0, 1.0), margin_mm=3.0, lam=2.0)
    wt = torch.from_numpy(w[None, None])
    yakin, uzak = torch.zeros_like(t), torch.zeros_like(t)
    yakin[0, 0, 4, 4, 6] = 6.0        # marj icinde yanlis pozitif (3 mm)
    uzak[0, 0, 0, 0, 0] = 6.0         # marjin disinda ayni buyuklukte hata
    assert float(margin_weighted_ce(yakin, t, wt)) > float(margin_weighted_ce(uzak, t, wt))


def test_kayip_fp32_hesaplaniyor():
    t = torch.zeros(1, 1, 4, 4, 4)
    t[..., :2, :2, :2] = 1.0
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        x = torch.randn(1, 1, 4, 4, 4).half()
        for l in (dice_ce_loss(x, t), tversky_loss(x, t), focal_loss(x, t)):
            assert l.dtype == torch.float32


# ------------------------------------------------------------- kayan pencere
class _Sabit(nn.Module):
    """Her yerde ayni logit ureten sahte model -- dikis izi varsa gorunur."""

    def __init__(self, v: float = 2.0):
        super().__init__()
        self.v = v
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x, spacing=None):
        return [torch.full_like(x, self.v)]


def test_kayan_pencere_dikissiz():
    cfg = _cfg()
    vol = np.zeros((70, 52, 48), np.float32)
    prob = predict_volume(_Sabit(2.0).eval(), vol, cfg["spacing_mm"], cfg)
    assert prob.shape == vol.shape
    bekle = 1.0 / (1.0 + np.exp(-2.0))
    assert np.allclose(prob, bekle, atol=1e-5), f"dikis izi: {prob.min()} .. {prob.max()}"


def test_kayan_pencere_yanlis_spacingde_firlatir():
    cfg = _cfg()
    vol = np.zeros((32, 32, 32), np.float32)
    with pytest.raises(ValueError, match="spacing"):
        predict_volume(_Sabit().eval(), vol, (1.0, 1.0, 2.5), cfg)


def test_kayan_pencere_hacim_yamadan_kucukse_calisir():
    cfg = _cfg()
    vol = np.zeros((20, 33, 48), np.float32)
    prob = predict_volume(_Sabit(0.0).eval(), vol, cfg["spacing_mm"], cfg)
    assert prob.shape == vol.shape and np.allclose(prob, 0.5, atol=1e-5)


def _dice_terimi(logits: torch.Tensor, target: torch.Tensor) -> float:
    """dice_ce_loss = CE + Dice. CE'yi cikarip YALNIZ Dice terimini olc."""
    ce = float(F.binary_cross_entropy_with_logits(logits.float(), target.float()))
    return float(dice_ce_loss(logits, target)) - ce


def test_kucuk_lezyonu_kacirma_cezasi_buyuk_lezyonunkiyle_karsilastirilabilir():
    """eps PAYDA'da buyukse (eps=1.0) 1 vokselli lezyonu tamamen kacirmanin Dice
    cezasi 0.50, 512 vokselliyi kacirmanin cezasi ~1.0 olur -> en kucuk lezyonda
    gradyan iskontolu. Tasiyici sayi: <=2 cm duyarlilik %39.6 (RadGPT); kayip tam
    o populasyonu sistematik olarak cezasizlastiramaz."""
    kacirma = torch.full((1, 1, 32, 32, 32), -20.0)     # hicbir sey tahmin etme
    cezalar = []
    for n in (1, 512):
        t = torch.zeros(1, 1, 32, 32, 32)
        t.view(-1)[:n] = 1.0
        cezalar.append(_dice_terimi(kacirma, t))
    kucuk, buyuk = cezalar
    assert kucuk > 0.9, f"1 vokselli lezyonu tamamen kacirmak neredeyse cezasiz: {kucuk}"
    assert buyuk / kucuk <= 1.2, f"kucuk lezyon iskontolu: buyuk={buyuk} kucuk={kucuk}"


def test_dice_terimi_bos_gt_bos_tahminde_sifir():
    """Bos GT + bos tahmin -> Dice terimi ~0 (kural 5: bos maskede Dice tanimsiz,
    ceza yanlis pozitifi zaten cezalandiran CE'den gelir)."""
    bos = torch.zeros(1, 1, 32, 32, 32)
    assert abs(_dice_terimi(torch.full_like(bos, -10.0), bos)) < 0.02


def test_dice_eps_varsayilani_train_yaml_ile_ayni():
    """Imza varsayilani ile configs/train.yaml iki kopya olmamali."""
    import yaml

    from src.jepa.decoder import DICE_EPS
    ref = yaml.safe_load(open("configs/train.yaml"))["loss"]["dice_eps"]
    assert float(ref) == DICE_EPS
    assert DICE_EPS < 1e-3, "eps duzenlestirici degil sayisal koruyucu olmali"


def test_predict_volume_resample_ciktisinin_ULASILAN_spacingini_kabul_eder():
    """`data.resample` ULASILAN spacing'i dondurur; sekil yuvarlandigi icin hedeften
    sapar. 512x512 @ 0.7 mm (LiTS'in EN YAYGIN in-plane geometrisi) 1 mm'ye
    orneklenince ulasilan (1.0, 1.001117, 1.001117) -- mutlak 1e-3 tolerans bunu
    REDDEDIYORDU, yani gercek LiTS hacimleri cikarima giremiyordu.
    Girdi spacing ELLE YAZILMAZ, resample'in dondurdugu deger kullanilir."""
    from src.jepa.data import resample
    cfg = _cfg()
    arr = np.zeros((4, 512, 512), np.float32)
    out, got = resample(arr, (1.0, 0.7, 0.7), cfg["spacing_mm"], order=1)
    sapma = max(abs(a - b) for a, b in zip(got, cfg["spacing_mm"]))
    assert sapma > 1.0e-3, f"test anlamsiz: izgara yuvarlamasi sapma uretmiyor ({got})"
    prob = predict_volume(_Sabit(0.0).eval(), out[:, :96, :96], got, cfg)
    assert prob.shape == (4, 96, 96)


class _SpacingKaydeden(nn.Module):
    """Aldigi spacing'i kaydeden sahte model -- encoder konum kodlamasi mm tabanli."""

    def __init__(self):
        super().__init__()
        self.gorulen: list = []

    def forward(self, x, spacing=None):
        self.gorulen.append(None if spacing is None else tuple(float(s) for s in spacing))
        return [torch.zeros_like(x)]


def test_predict_volume_spacingi_modele_iletiyor():
    """Spacing dogrulanip ATILIRSA encoder cfg['spacing_mm']'e duser; oysa konum
    kodlamasi mm tabanli -- tolerans icindeki gercek spacing modele gitmeli."""
    cfg = _cfg()
    m = _SpacingKaydeden().eval()
    sp = (1.0, 1.001117318435754, 1.001117318435754)
    predict_volume(m, np.zeros((48, 48, 48), np.float32), sp, cfg)
    assert m.gorulen, "model hic cagrilmadi"
    assert all(g == pytest.approx(sp) for g in m.gorulen), \
        f"modele giden spacing: {set(m.gorulen)}"


def test_predict_volume_train_modunda_firlatir():
    """@torch.no_grad() gradyani kapatir, train() modunu KAPATMAZ: drop_path acik
    kalirsa cikti stokastik olur (olculen maks olasilik farki 0.0958 @ drop_path=0.3)
    ve 0.5 esiginde voksel cevirip mm hedefleme metriklerini degistirir.
    Sessiz duzeltme (model.eval() cagirmak) YOK -- acik mesajla firlatilir."""
    cfg = _cfg()
    m = _Sabit(2.0).train()
    with pytest.raises(ValueError, match="eval"):
        predict_volume(m, np.zeros((32, 32, 32), np.float32), cfg["spacing_mm"], cfg)


def test_derin_denetimsiz_model_kayip_hesaplayabiliyor():
    """deep_supervision=False tek logit dondururdu ama agirliklar hep 3 taneydi ->
    `1 baslik ama 3 agirlik` ile patliyordu; derin denetimsiz ablasyon imkansizdi.
    Agirliklar baslik sayisina kirpilip YENIDEN NORMALIZE edilmeli (toplam 1.0),
    yoksa ablasyon kolunun kaybi ana koldan 0.57 kat kucuk olur."""
    cfg = _cfg()
    m = SegModel(cfg, deep_supervision=False).eval()
    x = torch.zeros(1, 1, 32, 32, 32)
    t = torch.zeros_like(x)
    t[..., 8:12, 8:12, 8:12] = 1.0
    with torch.no_grad():
        logits = m(x)
        assert len(logits) == 1
        kayip = deep_supervision_loss(logits, t, cfg)
        tek = make_loss(cfg["loss"]["name"], cfg)(logits[0], t)
    assert np.isfinite(float(kayip))
    assert float(kayip) == pytest.approx(float(tek), rel=1e-6), \
        "tek baslikta agirlik 1.0'a normalize edilmemis"


def test_baslik_agirliktan_COKSA_firlatir():
    """Kirpma yalniz asagi yonde: 4 baslik + 3 agirlik hala hatadir (sessiz dusurme yok)."""
    cfg = _cfg()
    t = torch.zeros(1, 1, 8, 8, 8)
    lo = [torch.zeros(1, 1, 8, 8, 8)] * 4
    with pytest.raises(ValueError, match="agirlik"):
        deep_supervision_loss(lo, t, cfg)


def test_tversky_fn_cezasi_fp_cezasindan_agir():
    """tp=fp=fn olan bir vakada sonuc alpha/beta'dan BAGIMSIZ (hep 0.5) -- yani
    alpha ile beta'yi ters yazan hata gorunmez. Ayni buyuklukte FN ile FP ayri
    ayri kurulur: Salehi konvansiyonunda beta FN katsayisidir (beta > alpha)."""
    t_fn = torch.zeros(1, 1, 4, 4, 4)
    t_fn.view(-1)[:16] = 1.0
    lo_fn = torch.full_like(t_fn, -20.0)
    lo_fn.view(-1)[:8] = 20.0                      # tp=8, fn=8, fp=0
    t_fp = torch.zeros(1, 1, 4, 4, 4)
    t_fp.view(-1)[:8] = 1.0
    lo_fp = torch.full_like(t_fp, -20.0)
    lo_fp.view(-1)[:16] = 20.0                     # tp=8, fp=8, fn=0
    a, b = 0.3, 0.7
    l_fn = float(tversky_loss(lo_fn, t_fn, alpha=a, beta=b))
    l_fp = float(tversky_loss(lo_fp, t_fp, alpha=a, beta=b))
    assert l_fn > l_fp, f"FN cezasi FP'den agir olmali: FN={l_fn} FP={l_fp}"
    assert l_fn == pytest.approx(1 - 1 / (1 + b), abs=1e-3)    # 0.4118
    assert l_fp == pytest.approx(1 - 1 / (1 + a), abs=1e-3)    # 0.2308


def test_focal_kolay_negatifi_zor_negatiften_daha_az_cezalandirir():
    """Butun vokseller p=0.5 iken dogru pt = p*t + (1-p)*(1-t) ile hatali pt = p
    AYNI sayiyi verir; negatif sinif kolu ancak p != 0.5 ile denetlenir."""
    t = torch.zeros(1, 1, 2, 2, 2)
    kolay = torch.full_like(t, float(np.log(0.01 / 0.99)))     # p=0.01, hedef 0
    zor = torch.full_like(t, float(np.log(0.99 / 0.01)))       # p=0.99, hedef 0
    lk = float(focal_loss(kolay, t, gamma=2.0))
    lz = float(focal_loss(zor, t, gamma=2.0))
    assert lk < lz, f"kolay negatif zor negatiften pahali: {lk} vs {lz}"
    assert lk == pytest.approx((0.01 ** 2) * -np.log(0.99), rel=1e-3)
    assert lz == pytest.approx((0.99 ** 2) * -np.log(0.01), rel=1e-3)
    assert lz / lk > 1e3                                        # modulasyon gamma ile


class _Konum(nn.Module):
    """Yama ICINDEKI konuma bagli logit: merkezde +4, kosede -4.

    Sabit logit ureten sahte modelde agirlikli ortalama HANGI agirlikla yapilirsa
    yapilsin ayni cikar -- Gauss penceresi hic kullanilmasa da test gecer.
    """

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x, spacing=None):
        ax = [1.0 - 2.0 * torch.abs(torch.arange(n, dtype=torch.float32)
                                    - (n - 1) / 2.0) / ((n - 1) / 2.0) for n in x.shape[2:]]
        r = (ax[0][:, None, None] + ax[1][None, :, None] + ax[2][None, None, :]) / 3.0
        return [(4.0 * r)[None, None].expand_as(x).contiguous()]


def test_kayan_pencere_gauss_agirligi_gercekten_uygulaniyor(monkeypatch):
    """Gauss penceresi uniforma cevrilirse sonuc DEGISMELI; degismiyorsa
    `_gauss_window` olu koddur ve dikis kenarlarindaki kaba tahminler merkezi
    kadar agirlik tasiyor demektir."""
    cfg = _cfg()
    vol = np.zeros((70, 52, 48), np.float32)
    m = _Konum().eval()
    gauss = predict_volume(m, vol, cfg["spacing_mm"], cfg)
    monkeypatch.setattr(decoder_mod, "_gauss_window",
                        lambda patch, frac: np.ones(tuple(int(p) for p in patch)))
    uniform = predict_volume(m, vol, cfg["spacing_mm"], cfg)
    fark = float(np.abs(gauss - uniform).max())
    assert fark > 0.1, f"Gauss agirligi sonuca girmiyor (maks fark {fark})"


def test_marj_mm_paths_yamldan_okunuyor_kopyalanmiyor():
    """Marj yaricapi TEK kaynak: paths.yaml#metrics.ablation_margin_mm. Kopya olursa
    degerlendirme marji 7 mm'ye cekildiginde egitim marji sessizce 5 mm kalir."""
    import inspect

    import yaml

    from src.jepa.decoder import resolve_margin_mm
    cfg = _cfg()
    ref = yaml.safe_load(open("configs/paths.yaml"))["metrics"]["ablation_margin_mm"]
    assert resolve_margin_mm(cfg) == float(ref)
    m = yaml.safe_load(open("configs/train.yaml"))["loss"]["margin"]
    assert "margin_mm" not in m, "marj yaricapi train.yaml'a KOPYALANMIS"
    assert "margin_mm_ref" in m
    var = inspect.signature(margin_weight_map).parameters["margin_mm"]
    assert var.default is inspect.Parameter.empty, "imza varsayilani ucuncu kopya"


def test_make_loss_margin_kolu_agirligi_gercekten_uyguluyor():
    """'margin' kolu olmadan marj-farkinda egitim UYGULANAMAZ durumdaydi
    (margin_weighted_ce yalniz testlerden cagriliyordu)."""
    cfg = _cfg()
    fn = make_loss("margin", cfg)
    t = torch.zeros(1, 1, 24, 24, 24)
    t[0, 0, 12, 12, 12] = 1.0
    yakin = torch.full_like(t, -20.0)
    yakin[0, 0, 12, 12, 15] = 6.0                 # 3 mm: 5 mm'lik marj bandinda
    uzak = torch.full_like(t, -20.0)
    uzak[0, 0, 0, 0, 0] = 6.0                     # ayni buyuklukte hata, marj disi
    assert float(fn(yakin, t)) > float(fn(uzak, t))
    dc = make_loss("dice_ce", cfg)
    assert float(dc(yakin, t)) == pytest.approx(float(dc(uzak, t)), rel=1e-6), \
        "agirliksiz kol iki hatayi zaten ayirt ediyorsa test marji olcmuyor"


def test_margin_kolu_derin_denetimde_spacingi_stride_ile_olcekliyor():
    """Stride 2 basliginda hedef max-pool ile kuculur -> voksel 2 mm olur. Marj
    haritasi 1 mm spacing ile kurulursa band mm cinsinden 2 kat kucuk cikar."""
    cfg = _cfg()
    m = SegModel(cfg).eval()
    x = torch.zeros(1, 1, 32, 32, 32)
    t = torch.zeros_like(x)
    t[..., 12:20, 12:20, 12:20] = 1.0
    with torch.no_grad():
        kayip = deep_supervision_loss(m(x), t, cfg, make_loss("margin", cfg))
    assert np.isfinite(float(kayip))
    w1 = margin_weight_map(t[0, 0].numpy(), (1.0, 1.0, 1.0), 5.0, lam=2.0)
    w2 = margin_weight_map(t[0, 0].numpy(), (2.0, 2.0, 2.0), 5.0, lam=2.0)
    assert (w2 > 1).sum() < (w1 > 1).sum(), "spacing olcegi banda yansimiyor"


def test_marj_bandi_HALKA_tumor_ici_ayirt_edilebiliyor():
    """`dist = edt(~g)` tumor icinde 0'dir; `dist <= margin` tumorun icini de lam
    yapardi. Ablasyon marji semantigi HALKA'dir (tumor cevresi)."""
    gt = np.zeros((9, 9, 9), np.uint8)
    gt[4, 4, 4] = 1
    w = margin_weight_map(gt, (1.0, 1.0, 1.0), 3.0, lam=2.0)
    assert float(w[4, 4, 4]) == pytest.approx(1.0), "tumor ici marj bandiyla ayni agirlikta"
    assert float(w[4, 4, 6]) == pytest.approx(2.0)
    w2 = margin_weight_map(gt, (1.0, 1.0, 1.0), 3.0, lam=2.0, inside_lam=3.0)
    assert float(w2[4, 4, 4]) == pytest.approx(3.0)
    assert float(w2[4, 4, 6]) == pytest.approx(2.0)


def test_n_classes_birden_buyukse_acikca_reddediliyor():
    """dice_ce_loss kanallar uzerinden de toplam alir -> cok sinifta tek BIRLESIK
    Dice cikar (sessizce yanlis). Acik red, sessiz yanlistan iyidir."""
    cfg = _cfg()
    with pytest.raises(NotImplementedError, match="n_classes"):
        SegDecoder(cfg, n_classes=3)
    with pytest.raises(NotImplementedError, match="n_classes"):
        SegModel(cfg, n_classes=3)


def test_predict_volume_kullanilmayan_batch_parametresi_yok():
    """`batch` parametresi hic kullanilmiyordu; API'de kalmasi yaniltici."""
    import inspect
    assert "batch" not in inspect.signature(predict_volume).parameters


class _GirdiKaydeden(nn.Module):
    """Modele giden yamayi kaydeden sahte model."""

    def __init__(self):
        super().__init__()
        self.yamalar: list = []

    def forward(self, x, spacing=None):
        self.yamalar.append(x.detach().clone())
        return [torch.zeros_like(x)]


def test_dolgu_degeri_hava_HUsuna_denk():
    """`hu_window`'dan SONRA sifirla doldurmak out_range=(-1,1)'de 25 HU (yumusak
    doku) demekti; hacim kenarinda doku benzeri dolgu lezyon-bazli yanlis pozitifi
    sisirir (precision iddiasi buna bagli). Dolgu pencerenin alt ucu olmali."""
    cfg = _cfg()
    m = _GirdiKaydeden().eval()
    vol = np.zeros((20, 33, 48), np.float32)          # z ve y yamadan kucuk -> dolgu
    predict_volume(m, vol, cfg["spacing_mm"], cfg)
    x = m.yamalar[0]
    alt = float(cfg["out_range"][0])
    assert float(x[0, 0, -1, -1, 0]) == pytest.approx(alt), "dolgu pencere altina esit degil"
    doku = float(hu_window(np.zeros(1, np.float32), cfg["hu_window"],
                           out_range=cfg["out_range"])[0])
    assert float(x[0, 0, -1, -1, 0]) < doku, "dolgu 0 HU (yumusak doku) gibi gorunuyor"
