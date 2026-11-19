"""Dondurulmus ic test kumesi -- VERI OLMADAN calisir.

B3 (leakage-auditor): make_splits 'test' split'i uretmiyor, gercek degerlendirme
seg.make_folds'tan geliyor -> on egitim korpusu degerlendirme vakalarini gorebiliyordu.
"""
import pytest

from src.data.frozen_test import FrozenTestError, choose, to_payload

EQUIV = {"LITS": "LITS|MSD_TASK03", "MSD_TASK03": "LITS|MSD_TASK03"}
IDS = [f"lits-{i}" for i in range(131)]
HAS = {c: (int(c.split("-")[1]) % 10 != 0) for c in IDS}      # 118 tumorlu / 13 tumorsuz


def test_belirleyici_ayni_tohum_ayni_kume():
    assert choose(IDS, HAS, seed=0) == choose(IDS, HAS, seed=0)


def test_farkli_tohum_farkli_kume():
    assert choose(IDS, HAS, seed=0) != choose(IDS, HAS, seed=1)


def test_tabakali_tumorsuz_oran_korunur():
    """Tumorsuzler tek kata yigilirsa o kat Dice'ta bos kalir."""
    sec = choose(IDS, HAS, k=5, fold=0, seed=0)
    tumorsuz = sum(1 for c in sec if not HAS[c])
    assert 1 <= tumorsuz <= 4, f"13 tumorsuz 5 kata dagilmali, katta {tumorsuz} var"


def test_katlar_ortusmez_ve_hepsini_kapsar():
    kovalar = [set(choose(IDS, HAS, k=5, fold=f, seed=0)) for f in range(5)]
    birlesim = set().union(*kovalar)
    assert birlesim == set(IDS)
    for i in range(5):
        for j in range(i + 1, 5):
            assert not (kovalar[i] & kovalar[j])


def test_tumor_durumu_bilinmeyen_vaka_firlatir():
    with pytest.raises(FrozenTestError, match="tumor durumu"):
        choose(IDS, {c: True for c in IDS[:5]})


def test_bos_liste_firlatir():
    with pytest.raises(FrozenTestError, match="bos"):
        choose([], {})


def test_gecersiz_fold_firlatir():
    with pytest.raises(FrozenTestError):
        choose(IDS, HAS, k=5, fold=5)


def test_payload_kanonik_kimlik_uretir():
    sec = choose(IDS, HAS, seed=0)
    p = to_payload(sec, "LITS", EQUIV, 5, 0, 0, len(IDS))
    assert p["n_frozen"] == len(sec)
    assert all(x.startswith("LITS|MSD_TASK03::") for x in p["canonical_ids"])


def test_kimlik_cakismasi_firlatir():
    """Iki farkli vaka tek kanonik kimlige inerse dondurulmus kume guvenilmez:
    korpus disinda tutulmasi gereken bir vaka gorunmez olur (B5/B14).
    'case_2_scan_1' ve 'case_3_scan_1' son-sayi kuralinda ikisi de '1'e iner."""
    with pytest.raises(FrozenTestError, match="cakisiyor"):
        to_payload(["case_2_scan_1", "case_3_scan_1"], "X", {}, 5, 0, 0, 2)


def test_ircadb_surum_numarasi_artik_cakismiyor():
    """B14 duzeltmesinden sonra 3Dircadb1.1 ve 3Dircadb1.2 ayri kimlik aliyor."""
    p = to_payload(["3Dircadb1.1", "3Dircadb1.2"], "IRCADB", {}, 5, 0, 0, 2)
    assert p["canonical_ids"] == ["IRCADB::1", "IRCADB::2"]
