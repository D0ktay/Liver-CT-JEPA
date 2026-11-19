"""On egitim onbellegi -- VERI OLMADAN calisir (sentetik NIfTI yazilir).

Onbellek, gercek veride ancak mm cinsinden sonuclar yanlis ciktiktan sonra fark
edilecek hatalarin toplandigi yerdir: kaybolan spacing, yon karisikligi,
ROI disinda kalan lezyon, sessizce atlanan vaka.
"""
import json

import nibabel as nib
import numpy as np
import pytest

from src.jepa.cache_build import (
    CacheError,
    build_one,
    load_volume,
    reconstruct,
    roi_bbox,
    to_hu,
    to_model_input,
    write_cache,
)


def _nifti(tmp_path, name, arr, spacing=(1.0, 1.0, 1.0), affine=None):
    if affine is None:
        affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    p = tmp_path / name
    nib.save(nib.Nifti1Image(arr, affine), str(p))
    return p


def _blob(shape=(40, 50, 60), center=(20, 25, 30), r=6, val=120.0, bg=-900.0):
    a = np.full(shape, bg, np.float32)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    m = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2 <= r * r
    a[m] = val
    return a


# ------------------------------------------------------------------- yukleme
def test_load_volume_spacing_ve_yonu_birlikte_dondurur(tmp_path):
    p = _nifti(tmp_path, "ct.nii.gz", _blob(), spacing=(0.7, 0.8, 2.5))
    arr, sp, aff, ornt, src = load_volume(p)
    assert arr.shape == (40, 50, 60) and src == (40, 50, 60)
    assert sp == pytest.approx((0.7, 0.8, 2.5))
    assert aff.shape == (4, 4) and len(ornt) == 3


def test_load_volume_ORIJINAL_sekli_ayri_dondurur(tmp_path):
    """RAS'a cevrilince eksenler takas olursa 'orijinal sekil' degisir. Geri donuste
    RAS seklini orijinal sanmak, eksenleri takas edilmis bir maske uretir ve
    evaluate bunu 'sekil uyusuyor' diye kabul eder -- sessiz ve olumcul."""
    aff = np.array([[0.0, 0.8, 0, 0], [-0.7, 0, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1.0]])
    p = _nifti(tmp_path, "ct.nii.gz", _blob((41, 51, 61)), spacing=(0.7, 0.8, 2.5), affine=aff)
    arr, _, _, _, src = load_volume(p)
    assert src == (41, 51, 61)          # diskteki sekil
    assert arr.shape == (51, 41, 61)    # RAS sekli -- ayni sayilar, FARKLI eksenler
    assert src != arr.shape


def test_bozuk_dosya_sessizce_atlanmaz(tmp_path):
    bad = tmp_path / "bozuk.nii.gz"
    bad.write_bytes(b"bu bir nifti degil")
    with pytest.raises(CacheError, match="okunamadi"):
        load_volume(bad)


# ----------------------------------------------------------------------- ROI
def test_roi_kutusu_pay_birakir_ve_siniri_asmaz():
    m = np.zeros((40, 50, 60), np.uint8)
    m[18:22, 23:27, 28:32] = 1
    bb = roi_bbox(m, spacing=(1.0, 1.0, 1.0), margin_mm=5.0)
    assert bb[0].start == 13 and bb[0].stop == 27
    bb2 = roi_bbox(m, spacing=(1.0, 1.0, 1.0), margin_mm=1000.0)
    assert bb2[0].start == 0 and bb2[0].stop == 40          # kirpilir, tasmaz


def test_roi_payi_mm_cinsinden_anizotropide_farkli_voksel():
    """5 mm pay, 1 mm eksende 5 voksel; 5 mm eksende 1 voksel olmali."""
    m = np.zeros((40, 40, 40), np.uint8)
    m[20, 20, 20] = 1
    bb = roi_bbox(m, spacing=(1.0, 1.0, 5.0), margin_mm=5.0)
    assert (bb[0].stop - bb[0].start) == 11                 # 1 + 2*5
    assert (bb[2].stop - bb[2].start) == 3                  # 1 + 2*1


def test_bos_maskede_roi_firlatir():
    with pytest.raises(CacheError, match="bos"):
        roi_bbox(np.zeros((8, 8, 8), np.uint8), (1.0, 1.0, 1.0), 5.0)


# ------------------------------------------------------------------ build_one
def test_onbellek_hedef_spacinge_yakin_ve_ULASILANI_yazar(tmp_path):
    """41 x 0.7 mm = 28.7 mm; 1.0 mm istenince 29 voksel cikar ve GERCEK aralik
    28.7/29 = 0.9897 mm olur. Istenen degeri (1.0) yazmak mm hatasini sessizce buyutur.
    DIKKAT: sekil bilerek tam bolunmeyecek secildi -- (40, 50, 60) @ (0.7, 0.8, 2.5)
    tam bolunuyor ve bu hatayi GIZLIYOR."""
    shape, sp_in = (41, 51, 61), (0.7, 0.8, 2.5)
    p = _nifti(tmp_path, "ct.nii.gz", _blob(shape, center=(20, 25, 30)), spacing=sp_in)
    d = build_one(p, target=(1.0, 1.0, 1.0), hu_window=(-200, 250))
    assert all(abs(s - 1.0) < 0.1 for s in d["spacing"])
    assert d["spacing"] != (1.0, 1.0, 1.0)          # ULASILAN != ISTENEN
    # fiziksel uzunluk korunur
    for n_in, s_in, n_out, s_out in zip(shape, sp_in, d["img"].shape, d["spacing"]):
        assert n_in * s_in == pytest.approx(n_out * s_out)


def test_onbellek_uint8_ve_hu_penceresi_uygulanmis(tmp_path):
    p = _nifti(tmp_path, "ct.nii.gz", _blob(val=3000.0, bg=-2000.0))
    d = build_one(p, target=(1.0, 1.0, 1.0), hu_window=(-200, 250))
    assert d["img"].dtype == np.uint8
    assert d["img"].min() == 0 and d["img"].max() == 255


def test_etiket_en_yakin_komsu_ile_orneklenir(tmp_path):
    """Trilineer ornekleme + uint8'e cast, maskeyi SESSIZCE eritir: ara degerler
    (0.4 gibi) sifira kirpilir ve unique testi yine {0, 1} gorur. Bu yuzden
    degerlere degil ETIKET HACMINE (mm^3) bakilir -- fiziksel hacim korunmali."""
    sp_in = (1.0, 1.0, 2.0)
    ct = _nifti(tmp_path, "ct.nii.gz", _blob(), spacing=sp_in)
    lab = np.zeros((40, 50, 60), np.uint8)
    lab[18:22, 23:27, 28:32] = 2                            # 4x4x4 = 64 voksel
    lp = _nifti(tmp_path, "seg.nii.gz", lab, spacing=sp_in)
    d = build_one(ct, target=(1.0, 1.0, 1.0), hu_window=(-200, 250),
                  label_path=lp, tumor_label=2)
    assert set(np.unique(d["lab"])).issubset({0, 1})
    hacim_once = 64 * sp_in[0] * sp_in[1] * sp_in[2]        # 128 mm^3
    hacim_sonra = int(d["lab"].sum()) * d["spacing"][0] * d["spacing"][1] * d["spacing"][2]
    assert hacim_sonra == pytest.approx(hacim_once, rel=0.05), (   # olculdu: en yakin komsu 128.0, trilineer+uint8 96.0 mm^3
        f"etiket hacmi {hacim_once:.0f} -> {hacim_sonra:.0f} mm^3: maske eridi/sisti")


def test_bos_hacim_firlatir(tmp_path):
    p = _nifti(tmp_path, "ct.nii.gz", np.zeros((4, 4, 4), np.float32))
    with pytest.raises(CacheError):
        build_one(p, target=(1.0, 1.0, 1.0), hu_window=(-200, 250), min_voxels=10**6)


# ----------------------------------------------------------------- write/read
def test_yazilan_npz_geri_donus_bilgisini_tasir(tmp_path):
    p = _nifti(tmp_path, "ct.nii.gz", _blob(), spacing=(0.7, 0.8, 2.5))
    d = build_one(p, target=(1.0, 1.0, 1.0), hu_window=(-200, 250))
    out = write_cache(d, tmp_path / "cache" / "case1.npz")
    z = np.load(out, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for k in ("spacing", "target", "affine", "ornt", "src_shape", "hu_window"):
        assert k in meta, k
    assert np.array_equal(z["img"], d["img"])


def test_ayni_girdi_ayni_cikti(tmp_path):
    p = _nifti(tmp_path, "ct.nii.gz", _blob(), spacing=(0.7, 0.8, 2.5))
    a = build_one(p, target=(1.0, 1.0, 1.0), hu_window=(-200, 250))
    b = build_one(p, target=(1.0, 1.0, 1.0), hu_window=(-200, 250))
    assert np.array_equal(a["img"], b["img"]) and a["spacing"] == b["spacing"]


# ------------------------------------------------------------- geri donus
def test_reconstruct_orijinal_sekle_ve_yone_doner(tmp_path):
    """Tahmin onbellek uzayindan (1 mm, kirpilmis, RAS) ORIJINAL izgaraya donmeli:
    yoksa src.metrics.evaluate GT ile hizalayamaz ve mm metrikleri anlamsiz olur."""
    shape, sp = (41, 51, 61), (0.7, 0.8, 2.5)
    # RAS olmayan bir affine: eksenler takas + yon cevrimi
    aff = np.array([[0.0, 0.8, 0, 0], [-0.7, 0, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1.0]])
    ct = _nifti(tmp_path, "ct.nii.gz", _blob(shape, center=(20, 25, 30)), spacing=sp, affine=aff)
    roi = np.zeros(shape, np.uint8); roi[10:35, 12:40, 15:50] = 1
    rp = _nifti(tmp_path, "roi.nii.gz", roi, spacing=sp, affine=aff)
    d = build_one(ct, target=(1.0, 1.0, 1.0), hu_window=(-200, 250), roi_path=rp, margin_mm=5.0)
    prob = np.ones(d["img"].shape, np.float32)
    out = reconstruct(prob, d["meta"])
    assert out.shape == tuple(shape), f"{out.shape} != {shape}"
    assert out.max() == pytest.approx(1.0, abs=0.05)
    assert out.min() == 0.0                      # ROI disi sifir


def test_reconstruct_sekil_uyusmazliginda_firlatir(tmp_path):
    p = _nifti(tmp_path, "ct.nii.gz", _blob(), spacing=(1.0, 1.0, 1.0))
    d = build_one(p, target=(1.0, 1.0, 1.0), hu_window=(-200, 250))
    with pytest.raises(CacheError, match="onbellek sekli"):
        reconstruct(np.ones((3, 3, 3), np.float32), d["meta"])


def test_reconstruct_roi_disini_sifir_birakir(tmp_path):
    shape = (40, 40, 40)
    p = _nifti(tmp_path, "ct.nii.gz", _blob(shape), spacing=(1.0, 1.0, 1.0))
    roi = np.zeros(shape, np.uint8); roi[15:25, 15:25, 15:25] = 1
    rp = _nifti(tmp_path, "roi.nii.gz", roi, spacing=(1.0, 1.0, 1.0))
    d = build_one(p, target=(1.0, 1.0, 1.0), hu_window=(-200, 250), roi_path=rp, margin_mm=0.0)
    out = reconstruct(np.ones(d["img"].shape, np.float32), d["meta"])
    assert out[:15].sum() == 0 and out[25:].sum() == 0
    assert out[15:25, 15:25, 15:25].min() > 0.9


# --------------------------------------------- egitim/cikarim normalizasyonu
def test_onbellekten_model_girdisine_araligi_cfg_out_range_ile_ayni():
    """🔴 metrics-reviewer 1a: onbellek [0,1] ile yaziliyordu, predict_volume girdiyi
    cfg out_range (-1,1) ile olcekliyordu ve arada donusum yapan HICBIR kod yoktu.
    Egitim ve cikarim sessizce AYRISIR. Tek donusum yolu burasi."""
    cfg = {"hu_window": (-200.0, 250.0), "out_range": (-1.0, 1.0)}
    img = np.array([0, 128, 255], np.uint8)
    x = to_model_input(img, cfg)
    assert x.dtype == np.float32
    assert x[0] == pytest.approx(-1.0) and x[-1] == pytest.approx(1.0)


def test_onbellek_hu_gidis_donusu_predict_volume_ile_ayni_sonucu_verir():
    """to_hu -> hu_window(out_range) yolu, to_model_input ile BIREBIR ayni olmali;
    yoksa predict_volume'a beslenen hacim egitimdekinden farkli olceklenir."""
    from src.jepa.data import hu_window as _hu
    cfg = {"hu_window": (-200.0, 250.0), "out_range": (-1.0, 1.0)}
    img = np.arange(256, dtype=np.uint8)
    a = to_model_input(img, cfg)
    b = _hu(to_hu(img, cfg), cfg["hu_window"], out_range=cfg["out_range"])
    assert np.allclose(a, b, atol=1e-6)


def test_float16_onbellek_de_ayni_yoldan_gecer():
    cfg = {"hu_window": (-200.0, 250.0), "out_range": (-1.0, 1.0)}
    x = to_model_input(np.array([0.0, 0.5, 1.0], np.float16), cfg)
    assert x[0] == pytest.approx(-1.0) and x[-1] == pytest.approx(1.0)


def test_bilinmeyen_dtype_sessizce_gecmez():
    cfg = {"hu_window": (-200.0, 250.0), "out_range": (-1.0, 1.0)}
    with pytest.raises(CacheError, match="dtype"):
        to_model_input(np.array([1, 2, 3], np.int32), cfg)


# ------------------------------------------------- ROI etiketi (degerlendirme sizintisi)
def test_roi_yalniz_belirtilen_etiketi_kullanir(tmp_path):
    """ROI 'roi > 0' ile kurulursa GT TUMOR etiketi de kutuyu belirler -- yani
    'tumor etiketi on egitimde kullanilmiyor' iddiasi YANLIS olur. --roi-label 1
    yalniz karacigeri kullanir."""
    shape = (40, 40, 40)
    ct = _nifti(tmp_path, "ct.nii.gz", _blob(shape), spacing=(1.0, 1.0, 1.0))
    seg = np.zeros(shape, np.uint8)
    seg[18:22, 18:22, 18:22] = 1                 # karaciger
    seg[2:5, 2:5, 2:5] = 2                       # tumor, karacigerden UZAKTA
    sp = _nifti(tmp_path, "seg.nii.gz", seg, spacing=(1.0, 1.0, 1.0))
    her = build_one(ct, target=(1.0, 1.0, 1.0), hu_window=(-200, 250),
                    roi_path=sp, margin_mm=0.0)
    yalniz_kc = build_one(ct, target=(1.0, 1.0, 1.0), hu_window=(-200, 250),
                          roi_path=sp, roi_label=1, margin_mm=0.0)
    assert her["img"].shape != yalniz_kc["img"].shape
    assert yalniz_kc["meta"]["bbox"][0] == [18, 22]      # yalniz karaciger kutusu
    assert her["meta"]["bbox"][0] == [2, 22]            # tumor kutuyu buyuttu


def test_roi_verilmezse_tum_hacim(tmp_path):
    """Dondurulmus test tahmini ROI'ye hapsolmamali: model kutu disinda yanlis
    pozitif URETEBILMELI, yoksa precision/marj sayilari yukari yanli cikar."""
    shape = (40, 40, 40)
    ct = _nifti(tmp_path, "ct.nii.gz", _blob(shape), spacing=(1.0, 1.0, 1.0))
    d = build_one(ct, target=(1.0, 1.0, 1.0), hu_window=(-200, 250))
    assert d["meta"]["bbox"] == [[0, 40], [0, 40], [0, 40]]
    out = reconstruct(np.ones(d["img"].shape, np.float32), d["meta"])
    assert out.min() > 0.9, "ROI'siz onbellekte kutu disi bolge OLMAMALI"


# ----------------------------------------- LiTS 48-52: sahte etiket spacing'i
def test_etiket_spacing_farkliysa_varsayilan_FIRLATIR(tmp_path):
    """Gercek bulgu (Kaggle MVE v2): lits-48..52'nin segmentasyon basliklari
    (1,1,1) tasiyor, CT'ler gercek degeri (0.9766, 0.9766, 2.5). Sessizce devam
    etmek mm metriklerini gecersiz kilar."""
    shape = (20, 20, 20)
    ct = _nifti(tmp_path, "ct.nii.gz", _blob(shape), spacing=(0.9766, 0.9766, 2.5))
    lab = np.zeros(shape, np.uint8); lab[8:12, 8:12, 8:12] = 2
    lp = _nifti(tmp_path, "seg.nii.gz", lab, spacing=(1.0, 1.0, 1.0))
    with pytest.raises(CacheError, match="etiket spacing"):
        build_one(ct, target=(1.0, 1.0, 1.0), hu_window=(-200, 250), label_path=lp)


def test_trust_ct_spacing_acikca_istenirse_CT_basligini_kullanir(tmp_path):
    """Sekil ayniysa etiket izgarasi ZATEN CT izgarasidir; iki basliktan biri
    yanlistir. Bu bir VARSAYIMDIR, bu yuzden acik bayrakla ve uyariyla yapilir."""
    shape = (20, 20, 20)
    ct = _nifti(tmp_path, "ct.nii.gz", _blob(shape), spacing=(0.9766, 0.9766, 2.5))
    lab = np.zeros(shape, np.uint8); lab[8:12, 8:12, 8:12] = 2
    lp = _nifti(tmp_path, "seg.nii.gz", lab, spacing=(1.0, 1.0, 1.0))
    d = build_one(ct, target=(1.0, 1.0, 1.0), hu_window=(-200, 250),
                  label_path=lp, trust_ct_spacing=True)
    assert d["lab"].shape == d["img"].shape and d["lab"].any()


def test_trust_ct_spacing_SEKIL_farkliysa_yine_firlatir(tmp_path):
    """Sekil de farkliysa 'ayni izgara' varsayimi COKER -- o zaman guvenilmez."""
    ct = _nifti(tmp_path, "ct.nii.gz", _blob((20, 20, 20)), spacing=(1.0, 1.0, 2.5))
    lp = _nifti(tmp_path, "seg.nii.gz", np.zeros((20, 20, 10), np.uint8), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(CacheError, match="etiket"):
        build_one(ct, target=(1.0, 1.0, 1.0), hu_window=(-200, 250),
                  label_path=lp, trust_ct_spacing=True)
