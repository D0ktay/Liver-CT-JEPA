"""L6 'kismi' adaylarini ozgullukle ayirma testleri (veri gerektirmez).

match_crop 'ncc_partial' derken: tek yama >= 0.97 ama iki yama ayni kirpma kokeninde
bulusmadi. Bu adaylarin ELLE bakilmasi gerekiyordu; burada hangisine bakilacagini
ayiran olcut var: MUTLAK skor degil, OZGULLUK -- en iyi ile ikinci en iyi (baska hasta)
arasindaki fark.

Hedeflenen hatalar:
  - Mutlak NCC ile karar verilir (0.99 hem gercek kopyada hem sade karaciger benzerliginde cikar).
  - Sabit esik tek basina kullanilir: gurultu tabani veri kumesine gore degisiyor
    (akran kontrolunde 0.906, bu CSV'de ~0.98) -> aykiri deger kurali da gerekir.
  - Ikinci en iyi AYNI hastadan geldiginde ozguluk kaniti sayilir; '<id>@ras' ayni hacmin
    baska GORUNUMU oldugu halde farkli hasta sanilir (match_crop iki gorunum de arar).
  - Ikinci skoru olmayan satir sessizce 'temiz' sayilir (aslinda karsilastirilamaz).
  - Ozet, denetlenen sorgu sayisini sisirir ya da toplam kapsami hic vermez.
"""
import pandas as pd
import pytest

from src.provenance.l6_verdict import GAP_FLAG, ROBUST_Z, classify, summarize


def _row(qid, verdict="ncc_partial", best=None, second=None, bref="lits-1", sref="lits-2",
         candidates=2, checked=2, complete=True, structured=3, integer=False):
    return {"query_id": qid, "verdict": verdict, "ncc_best": best, "ncc_second": second,
            "ncc_best_ref": bref, "ncc_second_ref": sref, "ncc_candidates": candidates,
            "ncc_checked": checked, "ncc_complete": complete, "n_structured": structured,
            "integer_valued": integer}


# ---------------------------------------------------------------- ozguluk

def test_gercek_kopya_imzasi_isaretlenir():
    df = pd.DataFrame([_row("kopya", best=0.991, second=0.905)])
    out = classify(df)
    assert out.loc[0, "gap"] == pytest.approx(0.086)
    assert out.loc[0, "karar"] == "elle bak"


def test_yuksek_skor_tek_basina_yetmez():
    """0.998 gibi yuksek bir skor, ikinci de 0.997 ise kopya degil -- yama ozgul degil."""
    df = pd.DataFrame([_row("sade", best=0.998, second=0.997)])
    out = classify(df)
    assert out.loc[0, "gap"] == pytest.approx(0.001)
    assert out.loc[0, "karar"] == "ozgul degil"


def test_sabit_esik_sinirda_isaretler():
    """gap == GAP_FLAG tam sinirda: isaretlenmeli (>= degil > yazilirsa bu test duser)."""
    df = pd.DataFrame([_row("sinir", best=0.9 + GAP_FLAG, second=0.9)])
    assert classify(df).loc[0, "kural"] == "sabit esik"


def test_esik_bu_veri_kumesinin_gurultu_tabanina_gore_degil():
    """Sabit esik akran kontrolunden (0.906) turedi; bu CSV'de taban ~0.98.
    Bu yuzden sabit esik TEK BASINA kullanilamaz -- ampirik kural da olmali."""
    assert GAP_FLAG == pytest.approx(0.027)
    assert ROBUST_Z >= 3.0


# ---------------------------------------------------------------- ampirik aykiri deger

def test_ampirik_kural_sabit_esigin_altindaki_aykiriyi_yakalar():
    """Gurultu tabani dar oldugunda 0.012'lik fark bile 7 robust-sigma uzaktir."""
    sade = [_row(f"s{i}", best=0.98 + i * 1e-4, second=0.9783 + i * 1e-4) for i in range(30)]
    df = pd.DataFrame(sade + [_row("aykiri", best=0.9702, second=0.9584)])   # gap 0.0118
    out = classify(df)
    a = out[out.query_id == "aykiri"].iloc[0]
    assert a["gap"] < GAP_FLAG                      # sabit esik kacirirdi
    assert a["robust_z"] > ROBUST_Z
    assert a["karar"] == "elle bak"
    assert a["kural"] == "aykiri deger"
    assert (out[out.query_id != "aykiri"]["karar"] == "ozgul degil").all()


def test_tum_farklar_ayniysa_aykiri_deger_uretilmez():
    """MAD = 0 -> sifira bolme yok, kimse isaretlenmez."""
    df = pd.DataFrame([_row(f"s{i}", best=0.98, second=0.978) for i in range(10)])
    out = classify(df)
    assert out["robust_z"].isna().all()
    assert (out["karar"] == "ozgul degil").all()


# ---------------------------------------------------------------- ayni hacim

def test_ikinci_ayni_hastadan_ise_ozguluk_kaniti_degil():
    df = pd.DataFrame([_row("ayni", best=0.99, second=0.90, bref="lits-7", sref="lits-7")])
    out = classify(df)
    assert pd.isna(out.loc[0, "gap"])
    assert out.loc[0, "karar"] == "karsilastirilamaz"


def test_ras_gorunumu_ayni_hacim_sayilir():
    """match_crop referansi '<id>' ve '<id>@ras' olarak IKI gorunumde arar; ikisi AYNI hacim."""
    df = pd.DataFrame([_row("ras", best=0.99, second=0.90, bref="lits-7", sref="lits-7@ras")])
    out = classify(df)
    assert pd.isna(out.loc[0, "gap"])
    assert out.loc[0, "karar"] == "karsilastirilamaz"


def test_ikinci_skor_yoksa_temiz_sayilmaz():
    df = pd.DataFrame([_row("tek", best=0.99, second=None, candidates=1, checked=1)])
    out = classify(df)
    assert out.loc[0, "karar"] == "karsilastirilamaz"


def test_kismi_olmayan_satirlar_siniflandirilmaz():
    df = pd.DataFrame([_row("yok", verdict="no_match", best=None, second=None,
                            candidates=0, checked=0)])
    assert classify(df).loc[0, "karar"] == "-"


# ---------------------------------------------------------------- kapsam

def test_kapsam_match_crop_ile_ayni_tanimi_kullanir():
    """Korluk: aday yok VEYA butce kesildi (ncc_complete False) VEYA yapili yama < 2."""
    df = pd.DataFrame([
        _row("a", verdict="no_match", candidates=3),                      # denetlendi (NCC)
        _row("b", verdict="no_match", candidates=0),                      # aday yok
        _row("c", verdict="no_match", candidates=3, complete=False),      # butce kesildi
        _row("d", verdict="no_match", candidates=3, structured=1),        # zayif yama
    ])
    s = summarize(df)
    assert s["n_no_match_denetlenen"] == 1


def test_birebir_kanali_calistiysa_denetlenmis_sayilir():
    """integer_valued=True -> exact arama yapildi; spacing filtresi YOK, tum referanslari tarar.
    NCC adayi cikmamasi bu satiri denetlenmemis yapmaz (match_crop da boyle sayar)."""
    df = pd.DataFrame([
        _row("exact", verdict="no_match", candidates=0, integer=True),
        _row("kor", verdict="no_match", candidates=0, integer=False),
    ])
    s = summarize(df)
    assert s["n_no_match_denetlenen"] == 1


def test_ozet_toplam_kapsami_verir():
    df = pd.DataFrame([
        _row("a", best=0.991, second=0.905),                              # kismi, olculebilir
        _row("b", best=0.998, second=0.997),                              # kismi, olculebilir
        _row("c", best=0.99, second=None, candidates=1),                  # kismi, olculemez
        _row("d", verdict="no_match", candidates=3),                      # denetlendi
        _row("e", verdict="no_match", candidates=0),                      # kor
    ])
    s = summarize(df)
    assert s["n_sorgu"] == 5
    assert s["n_kismi"] == 3
    assert s["n_karsilastirilamaz"] == 1
    assert s["n_denetlenen"] == 3            # 1 no_match + 2 olculebilir kismi
    assert s["oran_denetlenen"] == pytest.approx(0.6)
    assert s["elle_bakilacak_var"] is True         # 'a' satiri


def test_hicbiri_ozgul_degilse_isaret_yok():
    df = pd.DataFrame([_row("b", best=0.998, second=0.997),
                       _row("c", best=0.9979, second=0.9970, bref="lits-3", sref="lits-9")])
    s = summarize(df)
    assert s["elle_bakilacak_var"] is False
    assert s["n_ozgul_degil"] == 2
