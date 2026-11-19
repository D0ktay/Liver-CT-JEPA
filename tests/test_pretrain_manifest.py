"""On egitim korpusu manifesti + 4 kesisim denetimi -- VERI OLMADAN calisir.

Kural 10 (dokunulmaz): test / dis test hastasinin HICBIR taramasi, etiketsiz bile
olsa, on egitim korpusuna giremez. Buradaki her test, gercek veride ancak sonuc
yayimlandiktan sonra fark edilecek bir sizintiyi hedefliyor.
"""
import json

import pandas as pd
import pytest

from src.data.pretrain_manifest import (
    CorpusError,
    assert_clean,
    audit,
    build_manifest,
    canonical_for,
    drop_by_split,
    drop_excluded,
    expand_contains,
    source_conflicts,
)

EQUIV = {"LITS": "LITS|MSD_TASK03", "MSD_TASK03": "LITS|MSD_TASK03"}


def _rec(vid, source, path=None):
    return {"volume_id": vid, "source": source, "path": path or f"data/{source}/{vid}.nii.gz",
            "n_bytes": 1000}


def _splits(rows):
    return pd.DataFrame(rows, columns=["canonical_id", "split"])


# ------------------------------------------------------------ icerme kapanisi
def test_icerme_gecisli_kapanis():
    """AbdomenAtlas -> FLARE23 -> HCC_TACE_SEG zinciri tek adimda gorunmez."""
    c = expand_contains({"ABDOMENATLAS": ["FLARE23"], "FLARE23": ["LITS", "HCC_TACE_SEG"]})
    assert "HCC_TACE_SEG" in c["ABDOMENATLAS"]
    assert "LITS" in c["ABDOMENATLAS"]
    assert c["LITS"] == {"LITS"}          # kendini icerir


def test_icerme_dongusu_sonsuz_donguye_girmez():
    c = expand_contains({"A": ["B"], "B": ["A"]})
    assert c["A"] == {"A", "B"} and c["B"] == {"A", "B"}


def test_flare23_dis_test_korumasini_ihlal_eder():
    """L7: FLARE23 HCC-TACE-Seg iceriyor -> korpusa girerse dis testimiz olur."""
    c = expand_contains({"FLARE23": ["LITS", "HCC_TACE_SEG"]})
    assert source_conflicts("FLARE23", c, {"HCC_TACE_SEG"}) == {"HCC_TACE_SEG"}
    assert source_conflicts("KITS23", c, {"HCC_TACE_SEG"}) == set()


# ------------------------------------------------------------- kanonik kimlik
def test_lits_ve_msd_ayni_kanonik_kimlige_duser():
    """LiTS = MSD Task03. Iki isim, tek tarama -- korpus ve split AYNI kimligi gormeli."""
    assert canonical_for("lits-38", "LITS", EQUIV) == canonical_for("liver_38", "MSD_TASK03", EQUIV)


def test_kanonik_kimlik_make_splits_bicimiyle_ayni():
    assert canonical_for("lits-38", "LITS", EQUIV) == "LITS|MSD_TASK03::38"


def test_kimlik_turetilemeyen_vaka_firlatir():
    """Sessizce 'UNKNOWN::x' uydurmak = denetlenemeyen korpus."""
    with pytest.raises(CorpusError):
        canonical_for("", "KITS23", EQUIV)


def test_sayisiz_kimlik_oldugu_gibi_kullanilir():
    """TCIA hasta kodlari sayi tasimayabilir -- kimlik kaybedilmez."""
    assert canonical_for("HCC_003", "HCC_TACE_SEG", EQUIV) == "HCC_TACE_SEG::003"
    assert canonical_for("abc", "WORD", EQUIV) == "WORD::abc"


def test_sessiz_birlesme_firlatir():
    """Son-sayi kurali 'case_1_scan_2'/'case_1_scan_3' tuzagini cozuyor (2 != 3),
    ama ortak SON sayi tasiyan adlarda risk suruyor: 'case_2_scan_1' ve
    'case_3_scan_1' ikisi de '1'e iner -> iki hacim tek kimlige coker, korpus
    kendini oldugundan kucuk sanar ve kesisim denetimi bir hacmi hic gormez."""
    with pytest.raises(CorpusError, match="sessizce"):
        build_manifest([_rec("case_2_scan_1", "WORD"), _rec("case_3_scan_1", "WORD")],
                       equiv_map=EQUIV)


def test_son_sayi_kurali_farkli_taramalari_ayirir():
    m = build_manifest([_rec("case_1_scan_2", "WORD"), _rec("case_1_scan_3", "WORD")],
                       equiv_map=EQUIV)
    assert m["canonical_id"].nunique() == 2


# ----------------------------------------------------------------- manifest
def test_manifest_checksum_pending_birakir():
    """Kural 18: bu gece >5 GB indirme yok -> icerik hash'i Lambda'da dolacak."""
    m = build_manifest([_rec("case_1", "KITS23")], equiv_map=EQUIV)
    assert list(m["checksum"]) == ["pending"]


def test_manifest_yasakli_kaynagi_reddeder():
    with pytest.raises(CorpusError, match="HCC_TACE_SEG"):
        build_manifest([_rec("f1", "FLARE23")], equiv_map=EQUIV,
                       contains={"FLARE23": ["LITS", "HCC_TACE_SEG"]},
                       protected_sources={"HCC_TACE_SEG"})


def test_manifest_ayni_kanonik_kimligi_tekilleştirir():
    """Ayni tarama iki isimle gelirse korpus onu iki kez gormemelidir."""
    m = build_manifest([_rec("lits-38", "LITS"), _rec("liver_38", "MSD_TASK03")], equiv_map=EQUIV)
    assert len(m) == 2 and m["canonical_id"].nunique() == 1


# -------------------------------------------------------------- 4 kesisim
def test_korpus_test_hastasini_iceriyorsa_kirmizi():
    m = build_manifest([_rec("lits-38", "LITS")], equiv_map=EQUIV)
    s = _splits([("LITS|MSD_TASK03::38", "test")])
    r = audit(m, s)
    assert r["checks"]["pretrain∩test"]["n"] == 1
    assert r["clean"] is False
    with pytest.raises(CorpusError, match="pretrain∩test"):
        assert_clean(r)


def test_korpus_dis_test_hastasini_iceriyorsa_kirmizi():
    m = build_manifest([_rec("HCC_003", "HCC_TACE_SEG")], equiv_map=EQUIV)
    s = _splits([("HCC_TACE_SEG::003", "external_test")])
    r = audit(m, s)
    assert r["checks"]["pretrain∩dis_test"]["n"] == 1 and r["clean"] is False


def test_korpus_egitim_hastasini_icerebilir():
    """On egitim, etiketli egitim goruntuleriyle ortusebilir -- bu sizinti DEGIL."""
    m = build_manifest([_rec("lits-5", "LITS")], equiv_map=EQUIV)
    r = audit(m, _splits([("LITS|MSD_TASK03::5", "train")]))
    assert r["clean"] is True


def test_egitim_dogrulama_kesisimi_de_denetlenir():
    s = _splits([("X::1", "train"), ("X::1", "val")])
    r = audit(build_manifest([_rec("y1", "KITS23")], equiv_map=EQUIV), s)
    assert r["checks"]["egitim∩dogrulama"]["n"] == 1 and r["clean"] is False


def test_bos_test_kumesi_temiz_degil_denetlenmedi_sayilir():
    """En sinsi hata: test seti bos -> kesisim bos -> 'temiz' goruntusu.
    Bos kumede gecen denetim DENETLENMEMISTIR."""
    r = audit(build_manifest([_rec("k1", "KITS23")], equiv_map=EQUIV),
              _splits([("LITS|MSD_TASK03::1", "train")]))
    assert r["checks"]["pretrain∩test"]["vacuous"] is True
    assert r["clean"] is True          # ihlal yok
    assert r["audited"] is False       # ama denetlenmis de degil
    # canonical_members kapisi (B7) once patlar; onu gecince DENETLENMEDI gelir
    s2 = _splits([("LITS|MSD_TASK03::1", "train")]).assign(canonical_members="LITS|MSD_TASK03::1")
    r2 = audit(build_manifest([_rec("k1", "KITS23")], equiv_map=EQUIV), s2)
    with pytest.raises(CorpusError, match="DENETLENMEDI"):
        assert_clean(r2, require_audited=True)


def test_audit_json_serilestirilebilir():
    r = audit(build_manifest([_rec("k1", "KITS23")], equiv_map=EQUIV),
              _splits([("LITS|MSD_TASK03::1", "test")]))
    json.dumps(r)          # set/np tipi sizmasin


def test_audit_ornekleri_kirpilmis_ama_sayi_tam():
    m = build_manifest([_rec(f"lits-{i}", "LITS") for i in range(30)], equiv_map=EQUIV)
    s = _splits([(f"LITS|MSD_TASK03::{i}", "test") for i in range(30)])
    r = audit(m, s)
    assert r["checks"]["pretrain∩test"]["n"] == 30
    assert len(r["checks"]["pretrain∩test"]["examples"]) <= 10


# ------------------------------------------------- cozulemeyen ortusme (kor nokta)
def test_yasak_listesi_kaynagi_reddeder():
    """CT-ORG icerme grafigiyle yakalanmaz: LiTS hacimleri anonim, kanonik kimlige
    inmiyor -> 'pretrain∩test' onlari HIC GORMEZ. Acik yasak listesi tek savunma."""
    with pytest.raises(CorpusError, match="CT_ORG"):
        build_manifest([_rec("c1", "CT_ORG")], equiv_map=EQUIV,
                       blocked_sources={"CT_ORG": "L8: TCIA sayfasi LiTS'ten odunc diyor"})


def test_yasak_gerekcesi_hata_mesajinda_gorunur():
    with pytest.raises(CorpusError, match="odunc"):
        build_manifest([_rec("c1", "CT_ORG")], equiv_map=EQUIV,
                       blocked_sources={"CT_ORG": "L8: TCIA sayfasi LiTS'ten odunc diyor"})


def test_yasak_olmayan_kaynak_gecer():
    m = build_manifest([_rec("k1", "KITS23")], equiv_map=EQUIV,
                       blocked_sources={"CT_ORG": "x"})
    assert len(m) == 1


# =====================================================================
# leakage-auditor bulgulari (21 Eyl 2026) -- her testin basinda bulgu no
# =====================================================================

# ---- B14: _NUM_RE ilk sayi grubunu aliyordu -> 3Dircadb1.17 => "3" ----
def test_kanonik_kimlik_surum_numarasina_takilmaz():
    """'3Dircadb1.17' ILK sayi grubuyla '3'e iniyordu -> 20 IRCADb vakasinin
    HEPSI ayni kimlige cokuyordu. Vaka numarasi SONDA olur."""
    assert canonical_for("3Dircadb1.17", "IRCADB", EQUIV) == "IRCADB::17"
    assert canonical_for("3Dircadb1.3", "IRCADB", EQUIV) == "IRCADB::3"
    assert canonical_for("case_00123", "KITS23", EQUIV) == "KITS23::00123"
    assert canonical_for("amos_0001", "AMOS", EQUIV) == "AMOS::0001"
    assert canonical_for("liver_38", "MSD_TASK03", EQUIV) == "LITS|MSD_TASK03::38"


# ---- B5: volume_id de coktugunde sessiz birlesme gorunmuyordu ----
def test_ayni_kaynakta_tekrar_eden_volume_id_firlatir():
    """kits23/case_00123/imaging_resampled.nii.gz ve .../case_00456/... ikisi de
    'imaging_resampled' kimligi aliyordu; groupby(volume_id).nunique()==1 oldugu
    icin sessiz birlesme assert'i de kordu."""
    with pytest.raises(CorpusError, match="tekrar"):
        build_manifest([_rec("imaging_resampled", "KITS23", "a/case_00123/x.nii.gz"),
                        _rec("imaging_resampled", "KITS23", "b/case_00456/x.nii.gz")],
                       equiv_map=EQUIV)


# ---- B1: kanonik kok alfabetik secildigi icin split tarafi farkli ad tasiyor ----
def test_kesisim_canonical_members_uzerinden_de_bakar():
    """resolve_canonical union-find kokunu SOZLUK SIRASINDA KUCUK olani secer:
    LiTS <-> AbdomenCT-1K onayli birlesmesinde split'in kimligi 'ABDOMENCT_1K::123'
    olur, korpusunki 'LITS|MSD_TASK03::38' kalir -> kesisim SESSIZCE bos cikardi."""
    m = build_manifest([_rec("lits-38", "LITS")], equiv_map=EQUIV)
    s = pd.DataFrame([{"canonical_id": "ABDOMENCT_1K::123", "split": "test",
                       "canonical_members": "ABDOMENCT_1K::123;LITS|MSD_TASK03::38"}])
    r = audit(m, s)
    assert r["checks"]["pretrain∩test"]["n"] == 1, "kanonik kok kaymasi yakalanmadi"
    assert r["clean"] is False


# ---- B4: excluded_ircadb_overlap dort kontrolun hicbirine girmiyordu ----
def test_egitim_disindaki_HER_split_denetlenir():
    """K3 LiTS 28-47'yi egitimden cikariyor ama on egitim onlari serbestce yutuyordu.
    Beyaz liste: izin verilen yalniz train ve val."""
    m = build_manifest([_rec("lits-38", "LITS")], equiv_map=EQUIV)
    r = audit(m, _splits([("LITS|MSD_TASK03::38", "excluded_ircadb_overlap")]))
    assert r["checks"]["pretrain∩egitim_disi"]["n"] == 1 and r["clean"] is False


def test_yarin_eklenen_yeni_split_etiketi_de_otomatik_kapsanir():
    m = build_manifest([_rec("lits-7", "LITS")], equiv_map=EQUIV)
    r = audit(m, _splits([("LITS|MSD_TASK03::7", "bambaska_bir_etiket")]))
    assert r["checks"]["pretrain∩egitim_disi"]["n"] == 1


def test_egitim_ve_dogrulama_korpusta_olabilir():
    m = build_manifest([_rec("lits-5", "LITS"), _rec("lits-6", "LITS")], equiv_map=EQUIV)
    r = audit(m, _splits([("LITS|MSD_TASK03::5", "train"), ("LITS|MSD_TASK03::6", "val")]))
    assert r["clean"] is True


# ---- B12: 'audited' yalniz isim duzeyinde denetimi kapsiyordu ----
def test_audited_seviyesi_isim_duzeyinde_oldugunu_soyler():
    """Tum satirlar checksum=pending iken 'denetlendi' demek, 🟡 kaniti 🟢 gibi
    gostermektir. audited_level bunu acikca yazar."""
    m = build_manifest([_rec("k1", "KITS23")], equiv_map=EQUIV)
    r = audit(m, _splits([("KITS23::9", "test"), ("X::1", "train"), ("X::2", "val")]))
    assert r["audited_level"] == "name_only"
    with pytest.raises(CorpusError, match="isim duzeyinde"):
        assert_clean(r, require_content=True)


def test_checksum_dolunca_seviye_yukselir():
    m = build_manifest([{**_rec("k1", "KITS23"), "checksum": "ab" * 32,
                         "canonical_status": "hash_confirmed"}], equiv_map=EQUIV)
    r = audit(m, _splits([("KITS23::9", "test"), ("X::1", "train"), ("X::2", "val")]))
    assert r["audited_level"] == "hash_confirmed"


# ---- B3: gercek degerlendirme kumesi (CV fold) hicbir split'te degil ----
def test_fold_degerlendirme_vakalari_korpusta_ise_kirmizi():
    """make_splits 'test' split'i URETMIYOR; gercek degerlendirme seg.make_folds'tan
    geliyor ve splits.csv'yi hic okumuyor. Korpus 131 LiTS'in tamamiysa fold
    degerlendirme vakalari ON EGITIMDE GORULMUS olur -- baskalarini sucladigimiz sey."""
    m = build_manifest([_rec("lits-3", "LITS"), _rec("lits-9", "LITS")], equiv_map=EQUIV)
    r = audit(m, _splits([("LITS|MSD_TASK03::3", "train"), ("LITS|MSD_TASK03::9", "train")]),
              eval_ids={"LITS|MSD_TASK03::9"})
    assert r["checks"]["pretrain∩ic_test"]["n"] == 1 and r["clean"] is False


def test_ic_test_verilmezse_bu_kontrol_denetlenmemis_sayilir():
    m = build_manifest([_rec("lits-3", "LITS")], equiv_map=EQUIV)
    r = audit(m, _splits([("LITS|MSD_TASK03::3", "train")]))
    assert r["checks"]["pretrain∩ic_test"]["vacuous"] is True
    assert r["audited"] is False


# ---- B2/B7/B8: korpus kaynagi split tablosunda hic gorunmuyorsa denetim KOR ----
def test_splitte_hic_gorunmeyen_kaynak_kor_nokta_olarak_raporlanir():
    """KITS23 korpusa girdi ama split tablosunda 'KITS' ailesi var (build_provenance
    upper().replace('-','_') uretiyor) -> kimlikler HIC eslesmez, kesisim yapisal
    olarak bos cikar. Sessizce 'temiz' demek yerine kor nokta diye raporla."""
    m = build_manifest([_rec("case_00123", "KITS23")], equiv_map=EQUIV)
    s = _splits([("LITS|MSD_TASK03::3", "train"), ("LITS|MSD_TASK03::9", "test")])
    r = audit(m, s)
    assert "KITS23" in r["corpus"]["sources_not_in_splits"]
    with pytest.raises(CorpusError, match="KITS23"):
        assert_clean(r, require_sources_covered=True)


def test_splitte_gorunen_kaynak_kor_nokta_degil():
    m = build_manifest([_rec("lits-3", "LITS")], equiv_map=EQUIV)
    s = _splits([("LITS|MSD_TASK03::3", "train"), ("LITS|MSD_TASK03::9", "test")])
    r = audit(m, s)
    assert r["corpus"]["sources_not_in_splits"] == []
    assert_clean(r, require_sources_covered=True)      # firlatmamali


# ---- B14b: blocked_until anahtarini hicbir kod okumuyordu ----
def test_blocked_until_de_yasak_sayilir():
    with pytest.raises(CorpusError, match="dedup"):
        build_manifest([_rec("m1", "MSD_MSK")], equiv_map=EQUIV,
                       blocked_sources={"MSD_MSK": "L10: Task07<->08<->09<->10 dedup kosulmadi"})


# ---- B3 yapisal duzeltme: dondurulmus ic test korpustan DUSURULUR ----
def test_exclude_ids_korpustan_duser_ve_sayiyi_raporlar():
    m = build_manifest([_rec("lits-3", "LITS"), _rec("lits-9", "LITS")], equiv_map=EQUIV)
    kept, dropped = drop_excluded(m, {"LITS|MSD_TASK03::9"})
    assert list(kept["volume_id"]) == ["lits-3"]
    assert dropped == ["LITS|MSD_TASK03::9"]


def test_exclude_ids_hicbirini_tutmuyorsa_uyarir():
    """Kimlik bicimi tutmuyorsa dusurme SESSIZCE hicbir sey yapmaz ve korpus
    degerlendirme vakalarini tasimaya devam eder -- en sinsi hali."""
    m = build_manifest([_rec("lits-3", "LITS")], equiv_map=EQUIV)
    with pytest.raises(CorpusError, match="BULUNAMADI"):
        drop_excluded(m, {"KITS23::9"}, strict=True)


# =====================================================================
# FINAL leakage-auditor bulgulari (21 Eyl, teslimden once)
# =====================================================================

# ---- B3: drop_excluded yalniz HICBIRI eslesmezse firliyordu ----
def test_kismi_eslesme_de_firlatir():
    """26 dondurulmus kimligin 25'i duser, 1'i korpusta kalirsa HATA YOKTU;
    CLI '25 dusuruldu' yazar, operator 26 bekledigini fark etmez."""
    m = build_manifest([_rec("lits-3", "LITS")], equiv_map=EQUIV)
    with pytest.raises(CorpusError, match="eslesmeyen"):
        drop_excluded(m, {"LITS|MSD_TASK03::3", "LITS|MSD_TASK03::9"}, strict=True)


def test_tam_eslesme_gecer():
    m = build_manifest([_rec("lits-3", "LITS"), _rec("lits-9", "LITS")], equiv_map=EQUIV)
    kept, dropped = drop_excluded(m, {"LITS|MSD_TASK03::3", "LITS|MSD_TASK03::9"}, strict=True)
    assert len(kept) == 0 and len(dropped) == 2


# ---- B2: __eval__ kolu canonical_members ile genisletilmiyordu ----
def test_ic_test_kontrolu_de_canonical_members_gorur():
    """frozen_test 'LITS|MSD_TASK03::38' uretir; korpusta ayni fiziksel tarama
    dedup-onayli 'ABDOMENCT_1K::123' olarak duruyorsa kesisim SESSIZCE bos cikardi."""
    m = build_manifest([_rec("case_123", "ABDOMENCT_1K")], equiv_map=EQUIV)
    s = pd.DataFrame([{"canonical_id": "ABDOMENCT_1K::123", "split": "train",
                       "canonical_members": "ABDOMENCT_1K::123;LITS|MSD_TASK03::38"}])
    r = audit(m, s, eval_ids={"LITS|MSD_TASK03::38"})
    assert r["checks"]["pretrain∩ic_test"]["n"] == 1, "ic test uye genisletmesi yok"


# ---- B7: canonical_members sutunu yoksa SESSIZCE eski davranisa donuyordu ----
def test_canonical_members_sutunu_yoksa_uyarir():
    m = build_manifest([_rec("lits-38", "LITS")], equiv_map=EQUIV)
    r = audit(m, _splits([("LITS|MSD_TASK03::9", "test")]))
    assert r["corpus"]["canonical_members_var"] is False
    with pytest.raises(CorpusError, match="canonical_members"):
        assert_clean(r, require_audited=True)


# ---- B16: uye genisletmesi kor kaynagi MASKELEYEBILIYORDU ----
def test_uye_genisletmesi_kor_kaynagi_maskelemez():
    """resolve'un 'vol::<id>' dugumleri '::' tasimayan uyeler uretebiliyor;
    bunlar onek sanilirsa CT_ORG gibi bir kaynak 'kapsanmis' gorunur."""
    m = build_manifest([_rec("c1", "CT_ORG")], equiv_map=EQUIV)
    s = pd.DataFrame([{"canonical_id": "LITS|MSD_TASK03::5", "split": "train",
                       "canonical_members": "LITS|MSD_TASK03::5;CT_ORG"}])
    r = audit(m, s)
    assert "CT_ORG" in r["corpus"]["sources_not_in_splits"]


# ---- B4: kimlik kurali ikiligi (ILK vs SON sayi) ----
def test_ilk_ve_son_sayi_farkliysa_uyarir():
    """build_provenance ILK sayi grubunu, biz SONDAKINI aliyoruz. '3Dircadb1.17'
    provenance'ta IRCADB::3, bizde IRCADB::17 -> kesisim 19/20 vakayi gormez."""
    with pytest.raises(CorpusError, match="SAYI KURALI AYRISIYOR"):
        build_manifest([_rec("3Dircadb1.17", "IRCADB"), _rec("3Dircadb1.18", "IRCADB")],
                       equiv_map=EQUIV, strict_id_rule=True)


def test_tek_sayili_kimlikte_iki_kural_ayni_sorun_yok():
    m = build_manifest([_rec("lits-38", "LITS"), _rec("lits-39", "LITS")],
                       equiv_map=EQUIV, strict_id_rule=True)
    assert len(m) == 2


# ---- B9: gercek make_splits ciktisinda pretrain∩egitim_disi KIRMIZI yaniyor ----
def test_exclude_splits_egitim_disi_vakalari_dusurur():
    """make_splits LiTS 28-47'yi `excluded_ircadb_overlap` yapiyor ama korpus yaml'i
    LiTS'in 131'ini aliyor -> beyaz liste kontrolu HER KOSUDA kirmizi yanardi ve tek
    'cozum' gibi gorunen sey assert'i kismakti. Dogrusu: veriyi dusurmek."""
    m = build_manifest([_rec("lits-5", "LITS"), _rec("lits-38", "LITS")], equiv_map=EQUIV)
    s = _splits([("LITS|MSD_TASK03::5", "train"),
                 ("LITS|MSD_TASK03::38", "excluded_ircadb_overlap")])
    kept, dropped = drop_by_split(m, s, ("excluded_ircadb_overlap",))
    assert list(kept["volume_id"]) == ["lits-5"]
    assert dropped == ["LITS|MSD_TASK03::38"]


def test_exclude_splits_canonical_members_de_gorur():
    m = build_manifest([_rec("lits-38", "LITS")], equiv_map=EQUIV)
    s = pd.DataFrame([{"canonical_id": "ABDOMENCT_1K::123", "split": "excluded_ircadb_overlap",
                       "canonical_members": "ABDOMENCT_1K::123;LITS|MSD_TASK03::38"}])
    kept, dropped = drop_by_split(m, s, ("excluded_ircadb_overlap",))
    assert len(kept) == 0 and dropped


def test_exclude_splits_bilinmeyen_etiket_firlatir():
    m = build_manifest([_rec("lits-5", "LITS")], equiv_map=EQUIV)
    s = _splits([("LITS|MSD_TASK03::5", "train")])
    with pytest.raises(CorpusError, match="split tablosunda yok"):
        drop_by_split(m, s, ("boyle_bir_split_yok",))


def test_LiTSi_iceren_derleme_kendi_kimligiyle_gelse_bile_REDDEDILIR():
    """ULS23 ⊃ LITS: kimlikler eslesmez (`ULS23::7` vs `LITS|MSD_TASK03::7`), kesisim yesil gecer."""
    import pytest
    from src.data.pretrain_manifest import CorpusError, build_manifest
    recs = [{"volume_id": "uls_7", "source": "ULS23", "path": "/x/uls_7.nii.gz", "n_bytes": 1}]
    with pytest.raises(CorpusError, match="DEGERLENDIRME ailesini"):
        build_manifest(recs, {}, {"ULS23": ["LITS"]}, {"HCC_TACE_SEG"}, {})
    # LiTS'in kendisi serbest (test vakalari --exclude-ids ile duser)
    ok = build_manifest([{"volume_id": "lits-7", "source": "LITS", "path": "/x/volume-7.nii",
                          "n_bytes": 1}], {}, {"ULS23": ["LITS"], "MSD_TASK03": ["LITS"]},
                        {"HCC_TACE_SEG"}, {})
    assert len(ok) == 1
