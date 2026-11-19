"""Metrik testleri -- VERI OLMADAN calisir. Once bunlari gecir, sonra CT'ye dokun.

Her test, gercek veride yakalanmasi imkansiz olan bir hatayi hedefliyor.
"""
import numpy as np
import pytest

from src.metrics.lesionwise import (
    connected_components, dice, equivalent_diameter_mm,
    match_lesions, stratum_of, summarize,
)
from src.metrics.targeting import (
    _dilate_mm, summarize_targeting, targeting_errors,
)

SP = (1.0, 1.0, 1.0)


def _cube(shape, center, half):
    a = np.zeros(shape, dtype=np.uint8)
    z, y, x = center
    a[z-half:z+half+1, y-half:y+half+1, x-half:x+half+1] = 1
    return a


# ---------------------------------------------------------------- temel
def test_dice_bos_maskede_nan_dondurur():
    """Iki taraf da bossa Dice 1.0 DEGIL, tanimsizdir.
    Bircok makale bunu sessizce 1.0 sayip tumorsuz vakalarda skoru siisiriyor."""
    assert np.isnan(dice(np.zeros((4, 4, 4)), np.zeros((4, 4, 4))))


def test_dice_tam_ortusme():
    m = _cube((20, 20, 20), (10, 10, 10), 2)
    assert dice(m, m) == pytest.approx(1.0)


def test_connected_components_ayri_lezyonlari_ayirir():
    a = _cube((40, 40, 40), (10, 10, 10), 2) | _cube((40, 40, 40), (30, 30, 30), 2)
    _, n = connected_components(a)
    assert n == 2


def test_esdeger_cap_kure_formulu():
    # 1mm^3 voksel, 4188 voksel ~ r=10mm kure -> cap ~20mm
    d = equivalent_diameter_mm(4188, SP)
    assert d == pytest.approx(20.0, abs=0.1)


def test_stratum_sinirlari():
    assert stratum_of(10.0) == "<=10mm"
    assert stratum_of(10.1) == "10-20mm"
    assert stratum_of(20.0) == "10-20mm"
    assert stratum_of(20.1) == ">20mm"


# ---------------------------------------------------------------- eslestirme
def test_lezyon_eslestirme_tp_fp_fn():
    shape = (60, 60, 60)
    gt = _cube(shape, (10, 10, 10), 3) | _cube(shape, (40, 40, 40), 3)
    pred = _cube(shape, (10, 10, 10), 3) | _cube(shape, (10, 40, 40), 3)
    s = summarize(match_lesions(gt, pred, SP))
    assert (s["tp"], s["fn"], s["fp"]) == (1, 1, 1)
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(0.5)


def test_yuksek_dice_dusuk_precision_ayni_anda_olabilir():
    """Projenin tasiyici bulgusu: LiTS birincisi Dice 0.702 / precision 0.156.
    Metrigin bunu gosterebildigini kanitla."""
    shape = (80, 80, 80)
    gt = _cube(shape, (40, 40, 40), 8)               # tek buyuk lezyon
    pred = gt.copy()
    for k in range(10):                               # 10 kucuk yanlis pozitif
        pred |= _cube(shape, (5 + k * 7, 5, 70), 1)
    d = dice(gt, pred)
    s = summarize(match_lesions(gt, pred, SP))
    assert d > 0.9                                    # voksel metrigi 'mukemmel'
    assert s["precision"] < 0.2                       # tespit metrigi felaket
    assert s["recall"] == pytest.approx(1.0)


def test_boyut_katmanli_recall():
    shape = (80, 80, 80)
    small = _cube(shape, (10, 10, 10), 1)             # ~2.5mm cap
    big = _cube(shape, (50, 50, 50), 8)               # ~20mm cap
    gt = small | big
    pred = big.copy()                                 # kucugu kaciriyor
    s = summarize(match_lesions(gt, pred, SP))
    assert s["by_size"]["<=10mm"]["recall"] == pytest.approx(0.0)
    assert s["by_size"][">20mm"]["recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------- hedefleme
def test_dilate_mm_anizotropik_vokselde_dogru():
    """5mm marj, z-spacing 5mm ise z ekseninde 1 voksel olmali; 1mm x/y'de 5.
    binary_dilation(iterations=5) kullanilsaydi z'de 25mm olurdu -- sessiz bug."""
    shape = (21, 21, 21)
    m = np.zeros(shape, bool)
    m[10, 10, 10] = True
    grown = _dilate_mm(m, (5.0, 1.0, 1.0), 5.0)
    assert grown[9, 10, 10] and grown[11, 10, 10]     # z: +-1 voksel = +-5mm
    assert not grown[8, 10, 10]                        # z: +-2 voksel = 10mm, disarida
    assert grown[10, 15, 10] and not grown[10, 16, 10] # y: +-5 voksel = +-5mm


def test_hedefleme_hatasi_bilinen_kayma():
    shape = (60, 60, 60)
    gt = _cube(shape, (30, 30, 30), 4)
    pred = _cube(shape, (30, 30, 33), 4)              # x ekseninde 3 voksel kayma
    r = targeting_errors(gt, pred, SP, margin_mm=5.0)
    assert len(r) == 1 and r[0].detected
    assert r[0].centroid_error_mm == pytest.approx(3.0, abs=0.01)


def test_marj_ihlali_mantigi():
    shape = (60, 60, 60)
    gt = _cube(shape, (30, 30, 30), 5)
    # tahmin cok kucuk: 5mm marj bile gercek tumoru kapsamiyor
    pred_small = _cube(shape, (30, 30, 30), 1)
    r = targeting_errors(gt, pred_small, SP, margin_mm=1.0)
    assert r[0].margin_violation is True
    assert r[0].margin_coverage < 1.0
    # tahmin tam: 5mm marj rahat kapsiyor
    r2 = targeting_errors(gt, gt, SP, margin_mm=5.0)
    assert r2[0].margin_violation is False
    assert r2[0].margin_coverage == pytest.approx(1.0)


def test_kacirilan_lezyon_ortalamayi_kirletmez():
    """Tespit edilemeyen lezyonun centroid hatasi TANIMSIZ. 0 sayilirsa
    metrik yalanci olur -- bu testin varlik sebebi bu."""
    shape = (80, 80, 80)
    gt = _cube(shape, (20, 20, 20), 4) | _cube(shape, (60, 60, 60), 4)
    pred = _cube(shape, (20, 20, 20), 4)
    res = targeting_errors(gt, pred, SP)
    s = summarize_targeting(res)
    assert s["overall"]["n_lesions"] == 2
    assert s["overall"]["n_detected"] == 1
    assert s["overall"]["overlap_hit_rate"] == pytest.approx(0.5)
    assert s["overall"]["centroid_error_mm_median"] == pytest.approx(0.0, abs=1e-6)
    assert s["overall"]["margin_violation_rate_all"] == pytest.approx(0.5)


def test_bos_tahmin_cokmez():
    shape = (40, 40, 40)
    gt = _cube(shape, (20, 20, 20), 3)
    res = targeting_errors(gt, np.zeros(shape, np.uint8), SP)
    assert len(res) == 1 and res[0].detected is False
    assert res[0].margin_violation is True


# ---------------------------------------------------------------- hedefleme: ortusme tabanli planlama maskesi
def test_buyuk_tahmin_gt_yi_kapsiyorsa_ihlal_degil():
    """Tahmin tumoru her yonden 2mm tasiyor: IoU<0.5 (tespitte FN) ama klinikte
    EN GUVENLI durum. IoU esikli hedefleme bunu 'kacirildi + ihlal' sayiyordu."""
    shape = (40, 40, 40)
    gt = _cube(shape, (20, 20, 20), 2)
    pred = _cube(shape, (20, 20, 20), 4)
    r = targeting_errors(gt, pred, SP, margin_mm=5.0)
    assert r[0].detected and r[0].margin_violation is False
    assert r[0].margin_coverage == pytest.approx(1.0)
    assert r[0].centroid_error_mm == pytest.approx(0.0, abs=1e-6)
    # tespit metrigi ayni lezyonu FN sayar -- iki tanim bilerek ayrisir
    assert summarize(match_lesions(gt, pred, SP))["fn"] == 1


def test_iki_bitisik_gt_tek_tahmin():
    """Tahmin iki yakin lezyonu tek blob olarak ciziyor: ikisi de kapsaniyor,
    ama planlama merkezi ikisinin arasina kayiyor -- hata gorunur olmali."""
    shape = (40, 40, 60)
    gt = _cube(shape, (20, 20, 20), 3) | _cube(shape, (20, 20, 36), 3)
    pred = np.zeros(shape, np.uint8)
    pred[17:24, 17:24, 17:40] = 1
    r = targeting_errors(gt, pred, SP, margin_mm=5.0)
    assert len(r) == 2 and all(x.detected for x in r)
    assert all(x.margin_violation is False for x in r)
    assert all(x.centroid_error_mm == pytest.approx(8.0, abs=0.01) for x in r)


def test_parcali_tahmin_birlesimi_planlama_maskesidir():
    shape = (40, 40, 40)
    gt = _cube(shape, (20, 20, 20), 4)
    pred = gt.copy()
    pred[:, :, 20] = 0                               # tahmini ikiye bol
    r = targeting_errors(gt, pred, SP, margin_mm=2.0)
    assert r[0].detected and r[0].n_plan_components == 2
    assert r[0].centroid_error_mm == pytest.approx(0.0, abs=1e-6)
    assert r[0].margin_violation is False


def test_hedefleme_anizotropik_spacing():
    """z-spacing 5mm: z'de 1 voksel kayma = 5mm. Izotropik varsayan kod 1mm derdi."""
    shape = (20, 40, 40)
    sp = (5.0, 0.7, 0.7)
    gt = _cube(shape, (10, 20, 20), 3)
    pred = _cube(shape, (11, 20, 20), 3)
    r = targeting_errors(gt, pred, sp, margin_mm=5.0)
    assert r[0].centroid_error_mm == pytest.approx(5.0, abs=1e-6)
    assert r[0].margin_violation is False            # 5mm marj z'de 1 voksel kapatir
    r2 = targeting_errors(gt, pred, sp, margin_mm=3.0)
    assert r2[0].margin_violation is True            # 3mm z'de 1 voksele yetmez


def test_hacim_kenarindaki_ve_tek_voksel_lezyon():
    shape = (20, 20, 20)
    gt = np.zeros(shape, np.uint8)
    gt[0, 0, 0] = 1                                  # kose + tek voksel
    r = targeting_errors(gt, gt, SP, margin_mm=5.0)
    assert r[0].detected and r[0].margin_coverage == pytest.approx(1.0)
    assert r[0].centroid_error_mm == pytest.approx(0.0)


def test_hedefleme_sekil_uyusmazligi_firlatir():
    with pytest.raises(ValueError):
        targeting_errors(np.zeros((10, 10, 10)), np.zeros((10, 10, 11)), SP)


def test_kirpilmis_edt_tam_hacimle_ayni_sonucu_verir():
    """Performans icin EDT lezyon kutusuna kirpiliyor; kirpma sonucu DEGISTIRMEMELI."""
    rng = np.random.default_rng(1)
    shape = (30, 60, 60)
    sp = (2.5, 0.8, 0.8)
    gt = _cube(shape, (15, 30, 30), 5)
    pred = np.logical_and(_cube(shape, (15, 31, 29), 5), rng.random(shape) > 0.3)
    r = targeting_errors(gt, pred, sp, margin_mm=1.5)[0]
    # kaba kuvvet: tam hacim EDT
    from scipy import ndimage
    lab, _ = connected_components(pred)
    plan = np.isin(lab, np.unique(lab[(gt > 0) & (lab > 0)]))
    grown = ndimage.distance_transform_edt(~plan, sampling=sp) <= 1.5
    expect = (grown & (gt > 0)).sum() / (gt > 0).sum()
    assert r.margin_coverage == pytest.approx(expect)


def test_dev_tahmin_ihlalsiz_ama_fazla_hacim_raporlanir():
    """Tum karacigeri 'tumor' cizen model marji hep tutturur. Metrik bunu
    oduullendirmemeli: fazla (saglikli doku) hacmi gorunur olmali."""
    shape = (40, 40, 40)
    gt = _cube(shape, (20, 20, 20), 2)               # 5^3 = 125 mm^3
    pred = _cube(shape, (20, 20, 20), 10)            # 21^3 = 9261 mm^3
    r = targeting_errors(gt, pred, SP, margin_mm=5.0)[0]
    assert r.margin_violation is False
    assert r.gt_volume_ml == pytest.approx(0.125)
    assert r.plan_volume_ml == pytest.approx(9.261)
    assert r.excess_volume_ml == pytest.approx(9.261 - 0.125)
    s = summarize_targeting([r])
    assert s["overall"]["excess_volume_ml_median_detected"] == pytest.approx(9.136)
    assert s["overall"]["plan_to_gt_volume_ratio_median_detected"] == pytest.approx(9261 / 125)


def test_paylasilan_blob_iki_lezyonda_gorunur_ve_hacmi_cift_sayilmaz():
    shape = (40, 40, 60)
    gt = _cube(shape, (20, 20, 20), 3) | _cube(shape, (20, 20, 36), 3)
    pred = np.zeros(shape, np.uint8)
    pred[17:24, 17:24, 17:40] = 1
    r = targeting_errors(gt, pred, SP, margin_mm=5.0)
    assert all(x.plan_shared_with_n_gt == 2 for x in r)
    # fazla hacim = plan - TUM GT'ler (kardes lezyon saglikli doku sayilmaz)
    expect = (pred.sum() - (pred & gt).sum()) / 1000.0
    assert all(x.excess_volume_ml == pytest.approx(expect) for x in r)
    # tespit tarafinda ayni durum: iki FN + bir FP
    sm = summarize(match_lesions(gt, pred, SP))
    assert (sm["tp"], sm["fn"], sm["fp"]) == (0, 2, 1)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_gecersiz_marj_firlatir(bad):
    """Negatif marj sessizce 0mm'ye donuyordu."""
    gt = _cube((20, 20, 20), (10, 10, 10), 2)
    with pytest.raises(ValueError):
        targeting_errors(gt, gt, SP, margin_mm=bad)


def test_tek_voksel_temas_tespit_sayilir():
    shape = (30, 30, 30)
    gt = _cube(shape, (10, 10, 10), 2)
    pred = _cube(shape, (10, 10, 16), 3)             # x: gt 8-12, pred 13-19 -> temas yok
    pred[10, 10, 12] = 1                             # tek voksel kopru
    r = targeting_errors(gt, pred, SP, margin_mm=1.0)[0]
    assert r.detected and r.n_plan_components == 1
    assert r.margin_violation is True


def test_kirpma_cok_lezyon_rastgele_spacing_kaba_kuvvetle_ayni():
    from scipy import ndimage
    rng = np.random.default_rng(7)
    for _ in range(15):
        shape = (24, 40, 40)
        sp = tuple(rng.uniform(0.5, 5.0, 3))
        margin = float(rng.uniform(0.5, 8.0))
        gt = ndimage.binary_dilation(rng.random(shape) > 0.995, iterations=2)
        pred = ndimage.binary_dilation(rng.random(shape) > 0.995, iterations=2) | (gt & (rng.random(shape) > 0.4))
        res = targeting_errors(gt, pred, sp, margin_mm=margin)
        gl, _ = connected_components(gt)
        pl, _ = connected_components(pred)
        for r in res:
            g = gl == r.gt_label
            hit = np.unique(pl[g & (pl > 0)])
            if hit.size == 0:
                assert not r.detected
                continue
            plan = np.isin(pl, hit)
            grown = ndimage.distance_transform_edt(~plan, sampling=sp) <= margin
            assert r.margin_coverage == pytest.approx((grown & g).sum() / g.sum())
            c = lambda m: np.argwhere(m).mean(0) * np.asarray(sp)
            assert r.centroid_error_mm == pytest.approx(float(np.linalg.norm(c(plan) - c(g))))
