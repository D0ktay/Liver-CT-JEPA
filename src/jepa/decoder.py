"""Segmentasyon cozucusu (U-Net tarzi), kayiplar ve kayan pencere cikarimi.

`src/jepa/seg.py` DEGISTIRILMEDI -- o dosya donmus Rad-JEPA token'lariyla yayimlanmis
sonuclara bagli. Burasi kendi encoder'imizin cozucusu.

Bu bir **nnU-Net degildir** ve raporda oyle adlandirilmaz: kendi kurdugumuz 4 seviyeli
trilineer yukari orneklemeli cozucu, derin denetimli.

Kayiplar fp32'de hesaplanir (autocast kapatilarak): fp16'da Dice paydasi buyuk
hacimlerde tasip sessizce yanlis gradyan verir.

Marj agirligi `distance_transform_edt(sampling=spacing)` ile -- CLAUDE.md kural 6:
anizotropik vokselde `binary_dilation(iterations=N)` mm degil VOKSEL genisletir.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from src.jepa.blocks import LayerNorm3d
from src.jepa.data import hu_window, pad_to_patch, patch_starts
from src.jepa.encoder import DEFAULT_TRAIN_CFG, LiverEncoder, _resolve_ref

SKIP_KEYS = ("s8", "s4", "s2", "s1")        # kaba -> ince
# Imza varsayilani; TEK KAYNAK configs/train.yaml -> loss.dice_eps (esitligi
# tests/test_decoder.py:test_dice_eps_varsayilani_train_yaml_ile_ayni denetler).
# Sayisal koruyucu, duzenlestirici DEGIL -- gerekce dice_ce_loss docstring.
DICE_EPS = 1.0e-5
# EDT mesafesi kayan noktadir: spacing 5.0'da 'tam 5 mm' 5.000000001 cikabilir.
EDGE_TOL_MM = 1e-6
HEAD_STRIDES = (1, 2, 4)                    # derin denetim basliklarinin stride'i


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False), LayerNorm3d(cout), nn.GELU(),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False), LayerNorm3d(cout), nn.GELU())


def deep_supervision_weights(cfg: dict) -> tuple[float, ...]:
    """Toplami 1.0'a normalize edilmis derin denetim agirliklari (stride 1/2/4)."""
    w = [float(x) for x in cfg["decoder"]["deep_supervision_weights"]]
    s = sum(w)
    if s <= 0:
        raise ValueError(f"derin denetim agirliklari pozitif olmali: {w}")
    return tuple(x / s for x in w)


class SegDecoder(nn.Module):
    """{s1,s2,s4,s8,s16} piramidi -> [tam cozunurluk, /2, /4] logitleri."""

    def __init__(self, cfg: dict, deep_supervision: bool = True, n_classes: int = 1):
        super().__init__()
        if int(n_classes) not in (1, 2):
            raise NotImplementedError(
                f"n_classes={n_classes}: kayiplar (dice_ce_loss/tversky) kanallar "
                "uzerinden de toplam aliyor, yani cok sinifta tek BIRLESIK Dice "
                "cikar -- sessizce yanlis. 2 kanal YALNIZ bolge-tabanli kol icindir "
                "(src/jepa/region.py: kanal basina Dice); dice_ce_loss'a 2 kanal VERME.")
        self.cfg = cfg
        self.deep_supervision = deep_supervision
        c1, c2, c3 = [int(c) for c in cfg["conv_width"]]
        d8, d16 = int(cfg["s8"]["dim"]), int(cfg["s16"]["dim"])
        w0, w1, w2, w3 = [int(w) for w in cfg["decoder"]["width"]]
        self.up8 = _block(d16 + d8, w0)          # s16 -> s8
        self.up4 = _block(w0 + c3, w1)           # -> s4
        self.up2 = _block(w1 + c2, w2)           # -> s2
        self.up1 = _block(w2 + c1, w3)           # -> s1 (tam cozunurluk)
        self.head1 = nn.Conv3d(w3, n_classes, 1)
        self.head2 = nn.Conv3d(w2, n_classes, 1)
        self.head4 = nn.Conv3d(w1, n_classes, 1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, pyr: dict) -> list[torch.Tensor]:
        eksik = [k for k in ("s1", "s2", "s4", "s8", "s16") if k not in pyr]
        if eksik:
            raise KeyError(f"piramitte eksik seviye: {eksik}")
        y4 = self.up8(torch.cat([self._up(pyr["s16"], pyr["s8"]), pyr["s8"]], 1))
        y4 = self.up4(torch.cat([self._up(y4, pyr["s4"]), pyr["s4"]], 1))
        y2 = self.up2(torch.cat([self._up(y4, pyr["s2"]), pyr["s2"]], 1))
        y1 = self.up1(torch.cat([self._up(y2, pyr["s1"]), pyr["s1"]], 1))
        out = [self.head1(y1)]
        if self.deep_supervision:
            out += [self.head2(y2), self.head4(y4)]
        return out


class SegModel(nn.Module):
    """Encoder + cozucu. `forward` HER ZAMAN logit LISTESI dondurur."""

    def __init__(self, cfg: dict, encoder: LiverEncoder | None = None,
                 deep_supervision: bool = True, n_classes: int = 1):
        super().__init__()
        self.cfg = cfg
        self.encoder = LiverEncoder(cfg) if encoder is None else encoder
        self.decoder = SegDecoder(cfg, deep_supervision, n_classes)

    def forward(self, x: torch.Tensor, spacing=None) -> list[torch.Tensor]:
        return self.decoder(self.encoder(x, spacing))


# -------------------------------------------------------------------- kayiplar
def _fp32(*ts):
    return [t.float() for t in ts]


def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = DICE_EPS,
                 weight: torch.Tensor | None = None) -> torch.Tensor:
    """ANA KOL. Yumusak Dice + (agirlikli) BCE; bos maskede sonlu.

    `eps` YALNIZ sayisal koruyucudur (0/0 yok), duzenlestirici degil. Buyuk eps
    (orn. 1.0) paydada kucuk lezyonu sistematik olarak cezasizlastirir: 1 vokselli
    bir lezyonu tamamen kacirmanin cezasi eps=1.0'da 0.50, 512 vokselliyi
    kacirmanin cezasi 0.998 -- en kucuk lezyonda gradyan 2x (derin denetimli tam
    kayipta 4x) iskontolu. Tasiyici sayi: <=2 cm duyarlilik %39.6 (RadGPT).

    BOS GT (target tamamen 0): Dice tanimsizdir (CLAUDE.md kural 5 -- bos maskede
    Dice 1.0 degil nan). Kucuk eps ile formul burada sigmoid sizintisi yuzunden
    ~1.0 ceza uretirdi; bunun yerine Dice terimi 0 birakilir, yanlis pozitifi
    CE cezalandirir (kural 5'in "spesifiklik / yanlis pozitif ile raporla" karsiligi).
    """
    if logits.ndim == 5 and logits.shape[1] != 1:
        raise ValueError(f"dice_ce_loss tek kanal bekler, {logits.shape[1]} geldi -- "
                         "cok kanalda BIRLESIK Dice sessizce yanlis; region.region_loss kullan")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        lg, tg = _fp32(logits, target)
        ce = F.binary_cross_entropy_with_logits(lg, tg, weight=weight)
        if float(tg.sum()) == 0.0:
            return ce
        p = torch.sigmoid(lg)
        inter = (p * tg).sum()
        dice = 1 - (2 * inter + eps) / (p.sum() + tg.sum() + eps)
        return ce + dice


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3,
                 beta: float = 0.7, eps: float = DICE_EPS) -> torch.Tensor:
    """beta > alpha -> yanlis negatif daha agir (kucuk lezyonu kacirmamak icin).

    `eps` icin `dice_ce_loss` ile ayni gerekce: sayisal koruyucu, duzenlestirici degil.
    Bos GT'de bolge terimi 0 (kural 5). DIKKAT: bu kol CE icermez, dolayisiyla
    bos GT'li yamada kayip 0 olur -- tumorsuz yamalarda yanlis pozitifi bastiran
    tek sey kalmaz. Tversky ANA KOL DEGILDIR (ablasyon icin); ana kol dice_ce.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        lg, tg = _fp32(logits, target)
        if float(tg.sum()) == 0.0:
            return torch.zeros((), dtype=lg.dtype, device=lg.device) + 0.0 * lg.sum()
        p = torch.sigmoid(lg)
        tp = (p * tg).sum()
        fp = (p * (1 - tg)).sum()
        fn = ((1 - p) * tg).sum()
        return 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
               alpha: float | None = None) -> torch.Tensor:
    with torch.autocast(device_type=logits.device.type, enabled=False):
        lg, tg = _fp32(logits, target)
        ce = F.binary_cross_entropy_with_logits(lg, tg, reduction="none")
        p = torch.sigmoid(lg)
        pt = p * tg + (1 - p) * (1 - tg)
        loss = (1 - pt).pow(gamma) * ce
        if alpha is not None:
            loss = loss * (alpha * tg + (1 - alpha) * (1 - tg))
        return loss.mean()


def resolve_margin_mm(cfg: dict) -> float:
    """Marj yaricapi TEK kaynaktan: configs/paths.yaml#metrics.ablation_margin_mm.

    train.yaml'da `margin_mm` KOPYASI bulunursa firlatilir: degerlendirme marji
    (src/metrics/targeting.py) 7 mm'ye cekilseydi egitim marji sessizce 5 mm
    kalir ve "marj-farkinda egitim" iddiasi olculen marjla ayni seyi olcmezdi.
    """
    m = cfg["loss"]["margin"]
    if "margin_mm" in m:
        raise ValueError("configs/train.yaml loss.margin.margin_mm KOPYADIR -- "
                         "margin_mm_ref ile paths.yaml#metrics.ablation_margin_mm kullan")
    return float(_resolve_ref(m["margin_mm_ref"], DEFAULT_TRAIN_CFG))


def margin_weight_map(gt: np.ndarray, spacing, margin_mm: float,
                      lam: float = 2.0, inside_lam: float = 1.0) -> np.ndarray:
    """GT'yi CEVRELEYEN `margin_mm`'lik HALKA `lam`, tumor ici `inside_lam`, disi 1.0.

    `spacing` DIZI EKSENI SIRASINDADIR: (z, y, x) -- yani gt.shape ile ayni sirada.
    (x, y, z) verilirse anizotropik hacimde marj yanlis eksende buyur.

    Band bir HALKA'dir: `dist = edt(~g)` tumor icinde 0 oldugu icin `dist <= margin`
    tumorun ICINI de lam yapardi; ablasyon marji semantigi ise tumor cevresindeki
    saglam dokudur. Tumor ici agirligi ayri parametre (`inside_lam`, varsayilan 1.0).
    Ust sinir kayan nokta icin `margin + EDGE_TOL_MM`: EDT mesafesi tam sinirdaki
    vokselde (orn. spacing 5.0, margin 5.0) 5.0 yerine 5.0000000001 cikabilir ve
    ciplak `<=` bandin son halkasini gorunmez sekilde dusururdu.

    KURAL 6: mesafe `distance_transform_edt(sampling=spacing)` ile mm cinsinden
    olculur. `binary_dilation(iterations=N)` spacing'i yok sayar ve anizotropik
    vokselde marji sessizce z ekseninde 5 kat buyutur.

    `margin_mm` icin VARSAYILAN YOK: tek kaynak paths.yaml (bkz. resolve_margin_mm).
    """
    g = np.asarray(gt) > 0
    sp = tuple(float(s) for s in spacing)
    if len(sp) != g.ndim:
        raise ValueError(f"spacing {len(sp)}D, maske {g.ndim}D")
    if margin_mm < 0 or lam <= 0 or inside_lam <= 0:
        raise ValueError(f"margin_mm >= 0, lam > 0, inside_lam > 0 olmali: "
                         f"{margin_mm}, {lam}, {inside_lam}")
    if not g.any():
        warnings.warn("margin_weight_map: GT bos -- marj agirligi uygulanmadi (hepsi 1.0)")
        return np.ones(g.shape, np.float32)
    dist = ndimage.distance_transform_edt(~g, sampling=sp)
    w = np.ones(g.shape, np.float32)
    w[(dist > 0) & (dist <= float(margin_mm) + EDGE_TOL_MM)] = float(lam)
    w[g] = float(inside_lam)
    return w


def margin_weight_batch(target: torch.Tensor, spacing, margin_mm: float,
                        lam: float) -> torch.Tensor:
    """(B, C, D, H, W) hedef -> ayni sekilde marj agirlik tensoru (ornek basina EDT)."""
    if target.ndim != 5:
        raise ValueError(f"hedef (B, C, D, H, W) olmali: {tuple(target.shape)}")
    arr = target.detach().float().cpu().numpy()
    flat = arr.reshape(-1, *arr.shape[-3:])
    w = np.stack([margin_weight_map(g, spacing, margin_mm, lam) for g in flat])
    return torch.from_numpy(w.reshape(arr.shape)).to(target.device, dtype=torch.float32)


def margin_weighted_ce(logits: torch.Tensor, target: torch.Tensor,
                       weight: torch.Tensor) -> torch.Tensor:
    """Marj bandinda hatayi `lam` kat cezalandiran BCE."""
    if weight.shape != target.shape:
        raise ValueError(f"agirlik {tuple(weight.shape)} hedef {tuple(target.shape)} ile uyusmuyor")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        lg, tg, wt = _fp32(logits, target, weight)
        return F.binary_cross_entropy_with_logits(lg, tg, weight=wt)


def make_loss(name: str, cfg: dict):
    """cfg['loss'] adindan kayip fonksiyonu (imza: (logits, target) -> skaler)."""
    lc = cfg["loss"]
    if name == "dice_ce":
        return lambda lo, t: dice_ce_loss(lo, t, eps=float(lc["dice_eps"]))
    if name == "tversky":
        return lambda lo, t: tversky_loss(lo, t, float(lc["tversky"]["alpha"]),
                                          float(lc["tversky"]["beta"]),
                                          float(lc["dice_eps"]))
    if name == "focal":
        return lambda lo, t: focal_loss(lo, t, float(lc["focal"]["gamma"]))
    if name == "margin":
        mm, lam = resolve_margin_mm(cfg), float(lc["margin"]["lam"])
        eps = float(lc["dice_eps"])
        sp = tuple(float(s) for s in cfg["spacing_mm"])

        def _margin(lo, t, stride: int = 1):
            # Derin denetimde hedef max-pool ile kuculur -> voksel `stride` kat
            # buyur. Spacing olceklenmezse band mm cinsinden `stride` kat kucuk
            # kurulur ve marj-farkinda egitim yanlis yaricapta yapilir.
            w = margin_weight_batch(t, tuple(s * stride for s in sp), mm, lam)
            return dice_ce_loss(lo, t, eps=eps, weight=w)

        _margin.needs_stride = True
        return _margin
    raise ValueError(f"bilinmeyen kayip: {name} (dice_ce | tversky | focal | margin)")


def deep_supervision_loss(logits: list[torch.Tensor], target: torch.Tensor,
                          cfg: dict, loss_fn=None) -> torch.Tensor:
    """Basliklarin agirlikli toplami. Hedef max-pool ile kuculur (kucuk lezyon kaybolmasin).

    Baslik sayisi agirlik sayisindan AZ olabilir (deep_supervision=False -> tek
    baslik): agirliklar kirpilip yeniden normalize edilir, boylece derin denetimsiz
    ablasyon kolu ana kolla ayni olcekte kayip uretir. Baslik sayisi FAZLA ise
    hatadir (sessizce baslik dusurmeyiz).
    """
    fn = make_loss(cfg["loss"]["name"], cfg) if loss_fn is None else loss_fn
    w = deep_supervision_weights(cfg)
    if not logits:
        raise ValueError("logit listesi bos")
    if len(logits) > len(w):
        raise ValueError(f"{len(logits)} baslik ama {len(w)} agirlik")
    if len(logits) < len(w):
        kirpik = w[:len(logits)]
        w = tuple(x / sum(kirpik) for x in kirpik)
    total = 0.0
    for lo, wi, st in zip(logits, w, HEAD_STRIDES):
        t = target if st == 1 else F.max_pool3d(target, st)
        if lo.shape[2:] != t.shape[2:]:
            raise ValueError(f"baslik {tuple(lo.shape[2:])} hedef {tuple(t.shape[2:])} ile uyusmuyor")
        # spacing'e bagli kayiplar (marj kolu) stride'i bilmek ZORUNDA: stride 2'de
        # voksel 2 mm olur, mm cinsinden band aksi halde yari yaricapta kurulur.
        total = total + wi * (fn(lo, t, st) if getattr(fn, "needs_stride", False)
                              else fn(lo, t))
    return total


# ------------------------------------------------------------- kayan pencere
def _gauss_window(patch, sigma_frac: float) -> np.ndarray:
    ax = []
    for p in patch:
        i = np.arange(p, dtype=np.float64) - (p - 1) / 2.0
        ax.append(np.exp(-(i ** 2) / (2.0 * (sigma_frac * p) ** 2)))
    w = ax[0][:, None, None] * ax[1][None, :, None] * ax[2][None, None, :]
    return np.maximum(w, 1e-6)                   # sifir agirlik = kapsanmayan voksel


@torch.no_grad()
def predict_volume(model, vol: np.ndarray, spacing, cfg: dict,
                   device: str = "cpu") -> np.ndarray:
    """Tum hacim icin olasilik haritasi: %50 ortusmeli, Gauss agirlikli kayan pencere.

    `vol` HU cinsinden 3D dizidir; normalizasyon TEK yerde (`data.hu_window`) yapilir.
    `spacing` DIZI EKSENI SIRASINDADIR: (z, y, x) -- yani vol.shape ile ayni sirada.
    (x, y, z) verilirse anizotropik hacimde mm metrikleri sessizce yanlis cikar.

    Hedef spacing'den `inference.spacing_rtol` BAGIL toleransindan fazla sapma ->
    FIRLATIR: yanlis aralikli bir hacmi sessizce tahmin etmek mm metriklerini
    gecersiz kilar. Tolerans neden bagil ve neden 1e-3 degil: `data.resample` cikis
    seklini yuvarlar (`round(n*s/t)`), dolayisiyla ULASILAN spacing hedeften ~0.5/m
    bagil sapar (m = eksendeki voksel sayisi). 512x512 @ 0.7 mm -> 1 mm'de ulasilan
    1.001117 (sapma 1.12e-3); mutlak 1e-3 tolerans LiTS'in en yaygin in-plane
    geometrisini REDDEDIYORDU.
    """
    if vol.ndim != 3:
        raise ValueError(f"hacim 3D olmali: {vol.shape}")
    if getattr(model, "training", False):
        raise ValueError(
            "model train() modunda -- once model.eval() cagir. @torch.no_grad() yalniz "
            "gradyani kapatir; drop_path/dropout acik kalirsa cikti stokastik olur "
            "(olculen maks olasilik farki 0.0958 @ drop_path=0.3) ve 0.5 esiginde "
            "voksel cevirip mm hedefleme metriklerini degistirir. Sessiz eval() YOK.")
    hedef = tuple(float(s) for s in cfg["spacing_mm"])
    sp = tuple(float(s) for s in spacing)
    rtol = float(cfg["inference"]["spacing_rtol"])
    if len(sp) != 3:
        raise ValueError(f"spacing 3 bilesenli olmali (z, y, x): {sp}")
    if any(abs(a - b) > rtol * b for a, b in zip(sp, hedef)):
        raise ValueError(f"spacing {sp} hedeften ({hedef}) bagil %{rtol * 100:g}'den fazla "
                         "sapiyor -- once src.jepa.data.resample ile yeniden ornekle")
    patch = tuple(int(p) for p in cfg["patch"])
    overlap = float(cfg["inference"]["overlap"])
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"ortusme [0,1) araliginda olmali: {overlap}")
    x = hu_window(vol, cfg["hu_window"], out_range=cfg["out_range"])
    # Dolgu pencere ALT ucuyla: normalize uzayda 0.0 = 25 HU (yumusak doku),
    # hacim kenarinda doku benzeri serit yanlis pozitifi sisirir.
    x, pad = pad_to_patch(x, patch, fill=float(cfg["out_range"][0]))
    gw = _gauss_window(patch, float(cfg["inference"]["gauss_sigma_frac"]))
    acc = np.zeros(x.shape, np.float64)
    wsum = np.zeros(x.shape, np.float64)
    starts = [patch_starts(n, p, max(1, int(round(p * (1 - overlap)))))
              for n, p in zip(x.shape, patch)]
    model = model.to(device) if hasattr(model, "to") else model
    for i in starts[0]:
        for j in starts[1]:
            for k in starts[2]:
                sl = (slice(i, i + patch[0]), slice(j, j + patch[1]), slice(k, k + patch[2]))
                t = torch.from_numpy(x[sl]).to(device)[None, None]
                out = model(t, sp)          # konum kodlamasi mm tabanli: spacing SART
                lo = out[0] if isinstance(out, (list, tuple)) else out
                if tuple(lo.shape[2:]) != patch:
                    raise ValueError(f"model ciktisi {tuple(lo.shape[2:])} yama {patch} degil")
                p = torch.sigmoid(lo.float())[0, 0].cpu().numpy().astype(np.float64)
                acc[sl] += p * gw
                wsum[sl] += gw
    if not (wsum > 0).all():
        raise ValueError("kayan pencere hacmin tamamini kapsamadi (agirlik toplami 0)")
    prob = (acc / wsum).astype(np.float32)
    sl = tuple(slice(0, n - b) if b else slice(None) for n, (_, b) in zip(prob.shape, pad))
    return prob[sl]
