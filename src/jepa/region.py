"""Bolge-tabanli (karaciger bolgesi + tumor) egitim kaybi, hizli cikarim, son isleme.

22 EYL 2026 TESHISI (olculdu, `outputs/kaggle/mve/fp_where.json`):
  MVE modelinin tahmin ettigi voksellerin cogu KARACIGERIN DISINDAYDI (4 IRCADb vakasinin
  3'unde karaciger-ici pay <= %30, cogunda < %6; tumorsuz 2 vakada 10-70 bin voksel
  yanlis pozitif). Guvenilir hizalanan tek vakada (lits-37) yalniz karaciger maskesi
  uygulamak Dice'i 0.30-0.61 -> 0.73-0.77 yapti (TS ayni vakada 0.67).
  Kok neden: egitim yalniz GT karaciger kutusundan, test TUM govdeden.

COZUM (LiTS'te standart uygulama): model karacigeri de ogrenir; tumor tahmini modelin
KENDI karaciger tahmininin en buyuk bagli bileseniyle sinirlanir. GT'ye HIC dokunulmaz
(L15 degerlendirme sizintisi geri gelmez).

Kanal 0 = karaciger BOLGESI (karaciger ∪ tumor) · kanal 1 = tumor. Sigmoid, kanal basina
BCE + yumusak Dice. 3 sinifli softmax yerine bolge: (a) MSD Task08 gibi karaciger etiketi
OLMAYAN kaynaklarda kanal 0 kaybi vaka bazinda kapatilabilir, (b) tumor ⊂ bolge iliskisi
son islemeyi dogal kilar.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from src.jepa.data import hu_window, pad_to_patch, patch_starts
from src.jepa.decoder import DICE_EPS, HEAD_STRIDES, _gauss_window, deep_supervision_weights

N_REGION = 2
TTA_FLIPS = ((), (2,), (3,), (4,))          # yok + her eksende tek ayna (5D tensor eksenleri)


def region_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                eps: float = DICE_EPS, channel_weights=(1.0, 1.0)) -> torch.Tensor:
    """Kanal basina BCE + parti-havuzlu yumusak Dice; `valid` (B,C) gecersiz kanali KAPATIR.

    Bos hedefli kanalda Dice terimi 0 birakilir (CLAUDE.md kural 6: bos maskede Dice
    tanimsiz), yanlis pozitifi BCE cezalandirir. Hicbir gecerli kanal yoksa FIRLATIR --
    sessizce 0 kayipla egitmek yerine.
    """
    if logits.shape != target.shape or logits.shape[1] != N_REGION:
        raise ValueError(f"logit {tuple(logits.shape)} hedef {tuple(target.shape)}; "
                         f"{N_REGION} kanal bekleniyor")
    if valid.shape != logits.shape[:2] and valid.shape != logits.shape:
        raise ValueError(f"valid {tuple(valid.shape)} != (B, C) {tuple(logits.shape[:2])} "
                         f"ya da voksel duzeyi {tuple(logits.shape)}")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        lg, tg, vd = logits.float(), target.float(), valid.float()
        total, wsum = lg.new_zeros(()), 0.0
        for c in range(N_REGION):
            v = vd[:, c]
            if float(v.sum()) == 0.0:
                continue
            # (B,C): ornek duzeyi anahtar. (B,C,D,H,W): VOKSEL duzeyi -- liver_only vakada
            # tumor kanali yalniz karaciger DISINDA denetlenir (bkz. voxel_valid).
            m = v[:, None, None, None] if v.dim() == 1 else v
            l, t = lg[:, c], tg[:, c]
            bce = (F.binary_cross_entropy_with_logits(l, t, reduction="none") * m).sum() \
                / (m.expand_as(l).sum())
            term = bce
            if float((t * m).sum()) > 0.0:
                pr = torch.sigmoid(l) * m
                inter = (pr * t).sum()
                term = term + 1 - (2 * inter + eps) / (pr.sum() + (t * m).sum() + eps)
            total = total + float(channel_weights[c]) * term
            wsum += float(channel_weights[c])
        if wsum == 0.0:
            raise ValueError("partide gecerli kanal yok (valid tamamen 0)")
        return total / wsum


def voxel_valid(target: torch.Tensor, valid: torch.Tensor, margin_vox: int) -> torch.Tensor:
    """(B,C) ornek gecerliligi -> (B,C,D,H,W) voksel gecerliligi.

    liver_only vakada (karaciger etiketli, tumor etiketsiz) tumor kanali tamamen kapaliydi;
    oysa karaciger tumoru karacigerin DISINDA olamaz -> o vakalarda tumor kanali, karaciger
    maskesinin `margin_vox` genisletilmisinin DISINDA negatif olarak denetlenir. Icerisi
    (etiketlenmemis lezyon olabilir) kapali kalir. Gerekce: 22 Eyl gece kosusu, adim 20k --
    tumor kanalinin uzak negatifleri yalniz 96 LiTS hacminden geliyordu; gorulmemis govdede
    karaciger disi yanlis pozitif payi 0.09 -> 0.42'ye cikti.
    Onbellek 1 mm izotropik oldugu icin voksel = mm (kural 7 burada max-pool ile esdeger DEGIL
    ama guvenli yonde: kup, kureden BUYUK -> daha fazla voksel kapali kalir).
    """
    if margin_vox < 0:
        raise ValueError(f"margin_vox >= 0 olmali: {margin_vox}")
    vm = valid.float()[:, :, None, None, None].expand(target.shape).clone()
    lo = (valid[:, 0] > 0) & (valid[:, 1] == 0)
    if bool(lo.any()):
        liv = target[lo, 0:1].float()
        if margin_vox > 0:
            # kup cekirdek AYRISABILIR: uc 1-B havuz = tek k^3 havuzla BIREBIR ayni, ~k^2/3 kat ucuz
            k, q = 2 * margin_vox + 1, margin_vox
            liv = F.max_pool3d(liv, (k, 1, 1), stride=1, padding=(q, 0, 0))
            liv = F.max_pool3d(liv, (1, k, 1), stride=1, padding=(0, q, 0))
            liv = F.max_pool3d(liv, (1, 1, k), stride=1, padding=(0, 0, q))
        vm[lo, 1] = 1.0 - liv[:, 0]
    return vm


def region_ds_loss(logits: list[torch.Tensor], target: torch.Tensor, valid: torch.Tensor,
                   cfg: dict) -> torch.Tensor:
    """Derin denetimli bolge kaybi. Hedef max-pool ile kuculur (kucuk lezyon kaybolmasin)."""
    w = deep_supervision_weights(cfg)
    if not logits or len(logits) > len(w):
        raise ValueError(f"{len(logits)} baslik, {len(w)} agirlik")
    w = tuple(x / sum(w[:len(logits)]) for x in w[:len(logits)])
    eps = float(cfg["loss"]["dice_eps"])
    cw = tuple(float(x) for x in cfg["loss"].get("region_weights", (1.0, 1.0)))
    total = 0.0
    for lo, wi, st in zip(logits, w, HEAD_STRIDES):
        t = target if st == 1 else F.max_pool3d(target, st)
        # voksel gecerliligi MIN-pool ile kuculur: kaba voksel, ancak tum alt vokselleri
        # gecerliyse gecerli (belirsiz bolge kaba baslikta da cezalandirilmasin)
        vd = valid if (st == 1 or valid.dim() == 2) else -F.max_pool3d(-valid.float(), st)
        total = total + wi * region_loss(lo, t, vd, eps, cw)
    return total


# ----------------------------------------------------------------- cikarim
@torch.no_grad()
def predict_volume_fast(model, vol_hu: np.ndarray, spacing, cfg: dict, device: str = "cpu",
                        batch: int = 4, tta: bool = True, amp_dtype=None) -> np.ndarray:
    """(C, D, H, W) olasilik. Toplu yama, GPU'da Gauss birikimi, istege bagli ayna TTA.

    `decoder.predict_volume` ile AYNI sozlesme (HU girdi, (z,y,x) spacing, bagil tolerans,
    eval modu sart, hava dolgusu) -- fark: cok kanal + hiz. MVE'de surenin %51'i tek-tek
    yama cikarimiydi (27 hacim 51 dk).
    """
    if vol_hu.ndim != 3:
        raise ValueError(f"hacim 3D olmali: {vol_hu.shape}")
    if getattr(model, "training", False):
        raise ValueError("model train() modunda -- once model.eval() (drop_path stokastik)")
    hedef = tuple(float(s) for s in cfg["spacing_mm"])
    sp = tuple(float(s) for s in spacing)
    rtol = float(cfg["inference"]["spacing_rtol"])
    if len(sp) != 3 or any(abs(a - b) > rtol * b for a, b in zip(sp, hedef)):
        raise ValueError(f"spacing {sp} hedeften ({hedef}) bagil %{rtol * 100:g}'den fazla sapiyor")
    patch = tuple(int(p) for p in cfg["patch"])
    overlap = float(cfg["inference"]["overlap"])
    x = hu_window(vol_hu, cfg["hu_window"], out_range=cfg["out_range"])
    x, pad = pad_to_patch(x, patch, fill=float(cfg["out_range"][0]))
    xt = torch.from_numpy(x).to(device)
    gw = torch.from_numpy(_gauss_window(patch, float(cfg["inference"]["gauss_sigma_frac"]))
                          .astype(np.float32)).to(device)
    acc = torch.zeros((N_REGION,) + tuple(x.shape), dtype=torch.float32, device=device)
    wsum = torch.zeros(tuple(x.shape), dtype=torch.float32, device=device)
    starts = [patch_starts(n, p, max(1, int(round(p * (1 - overlap)))))
              for n, p in zip(x.shape, patch)]
    origins = [(i, j, k) for i in starts[0] for j in starts[1] for k in starts[2]]
    flips = TTA_FLIPS if tta else ((),)
    use_amp = amp_dtype is not None and str(device).startswith("cuda")
    for b0 in range(0, len(origins), batch):
        chunk = origins[b0:b0 + batch]
        xb = torch.stack([xt[i:i + patch[0], j:j + patch[1], k:k + patch[2]]
                          for i, j, k in chunk])[:, None]
        pb = torch.zeros((len(chunk), N_REGION) + patch, dtype=torch.float32, device=device)
        for fl in flips:
            xi = xb.flip(fl) if fl else xb
            # 🔴 Egitim `model(x)` cagiriyor (konum kodlamasi cfg spacing'i = tek kafes).
            # Cikarimda ULASILAN spacing'i (orn. 1.0011) vermek modelin HIC GORMEDIGI bir
            # kafes demek (olculdu: s16 ozellik farki %16.9 @1.005). Geometri zaten
            # `reconstruct` ile meta'dan geri kurulur; tolerans korumasi yukarida duruyor.
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(xi)
            lo = out[0] if isinstance(out, (list, tuple)) else out
            if tuple(lo.shape[1:]) != (N_REGION,) + patch:
                raise ValueError(f"model ciktisi {tuple(lo.shape)} beklenen (B,{N_REGION},{patch})")
            pr = torch.sigmoid(lo.float())
            pb += pr.flip(fl) if fl else pr
        pb /= len(flips)
        for n, (i, j, k) in enumerate(chunk):
            acc[:, i:i + patch[0], j:j + patch[1], k:k + patch[2]] += pb[n] * gw
            wsum[i:i + patch[0], j:j + patch[1], k:k + patch[2]] += gw
    if not bool((wsum > 0).all()):
        raise ValueError("kayan pencere hacmin tamamini kapsamadi")
    prob = (acc / wsum).cpu().numpy()
    sl = (slice(None),) + tuple(slice(0, n - b) if b else slice(None)
                                for n, (_, b) in zip(prob.shape[1:], pad))
    return prob[sl]


def liver_mask_from_prob(p_region: np.ndarray, thr: float = 0.5, close_vox: int = 2,
                         dilate_mm: float = 0.0, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Modelin KENDI karaciger-bolgesi tahmini -> en buyuk bagli bilesen, delikler dolu.

    GT KULLANILMAZ. Bos donerse (model karaciger bulamadi) tumor tahmini de bos olur --
    bu durum cagirana BILDIRILIR (`postprocess` sayar), sessizce gecilmez.
    """
    full = np.asarray(p_region) > thr
    if not full.any():
        return full
    # 1) en buyuk bagli bilesen TUM hacimde (uzaktaki sahte adaciklar kutuyu sisirmesin)
    lab, n = ndimage.label(full)
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        full = lab == (int(np.argmax(sizes)) + 1)
    del lab
    # 2) kapama + delik doldurma + mm genisletme yalniz o bilesenin KUTUSUNDA (+pay):
    #    80M voksellik hacimde bunlar ~47 sn / 4 GB suruyordu (olculdu); kutu hacmin ~%10'u.
    sp = tuple(float(x) for x in spacing)
    pad = [int(np.ceil((float(dilate_mm) + 2.0) / s_)) + int(close_vox) + 1 for s_ in sp]
    idx = (np.flatnonzero(full.any(axis=(1, 2))), np.flatnonzero(full.any(axis=(0, 2))),
           np.flatnonzero(full.any(axis=(0, 1))))
    box = tuple(slice(max(0, int(i[0]) - p_), min(n_, int(i[-1]) + 1 + p_))
                for i, p_, n_ in zip(idx, pad, full.shape))
    m = full[box]
    if close_vox > 0:
        st = ndimage.generate_binary_structure(3, 1)
        m = ndimage.binary_closing(m, st, iterations=int(close_vox), border_value=0) | m
    m = ndimage.binary_fill_holes(m)
    if dilate_mm > 0:
        # Sinirdaki (egzofitik) tumor karaciger kanalinin kenar hatasiyla kesilmesin.
        # KURAL 7: mm genisletme EDT ile, `binary_dilation(iterations=N)` ile DEGIL.
        m = ndimage.distance_transform_edt(~m, sampling=sp) <= float(dilate_mm)
    out = np.zeros(full.shape, bool)
    out[box] = m
    return out


def postprocess(prob: np.ndarray, thr: float = 0.5, dilate_mm: float = 3.0,
                spacing=(1.0, 1.0, 1.0)) -> tuple[np.ndarray, dict]:
    """(2,D,H,W) olasilik -> karaciger-ici tumor OLASILIGI (disi 0) + tani bilgisi.

    Olasilik doner, maske degil: orijinal izgaraya `reconstruct` ile (trilineer) donup
    ESIK ORADA uygulanir (maskeyi yeniden orneklemek kenari bozar).
    """
    if prob.ndim != 4 or prob.shape[0] != N_REGION:
        raise ValueError(f"(2, D, H, W) bekleniyor: {prob.shape}")
    liver = liver_mask_from_prob(prob[0], thr, dilate_mm=dilate_mm, spacing=spacing)
    raw = prob[1] > thr
    kept = raw & liver
    info = {"liver_vox": int(liver.sum()), "tumor_vox_raw": int(raw.sum()),
            "tumor_vox_kept": int(kept.sum()),
            "outside_liver_frac": float(1 - kept.sum() / max(1, raw.sum()))}
    return np.where(liver, prob[1], 0.0).astype(np.float32), info


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Ikili Dice; GT bos ise nan (kural 6)."""
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    if not b.any():
        return float("nan")
    return float(2.0 * (a & b).sum() / (a.sum() + b.sum()))
