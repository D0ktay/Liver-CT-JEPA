"""configs/pretrain_corpus.yaml'in KENDISINI denetler -- veri gerektirmez.

Bu dosya bir regresyondan sonra yazildi: IRCADB / ABDOMENCT_1K / FLARE22 / M3D_SEG
yasak kayitlari, commit'lenmeden once calistirilan bir `git checkout --` ile SESSIZCE
silinmisti ve commit mesaji onlarin eklendigini soyluyordu. Yapilandirma dosyalari da
test edilmeli: bir yasagin kaybolmasi, kodun kaybolmasindan daha sessizdir.
"""
import pathlib

import pytest
import yaml

from src.data.pretrain_manifest import CorpusError, build_manifest, expand_contains, source_conflicts

CFG = yaml.safe_load(pathlib.Path("configs/pretrain_corpus.yaml").read_text())
SOURCES = {s["name"]: s for s in CFG["sources"]}


def _blocked():
    out = {}
    for s in CFG["sources"]:
        why = s.get("reason_blocked") or (
            f"kosul: {s['blocked_until']}" if s.get("blocked_until") else None)
        if why:
            out[s["name"]] = str(why).strip()
    return out


# LiTS iceren ama dosyalari anonim oldugu icin kanonik kimligi ESLESMEYEN kaynaklar:
# kesisim denetimi onlarin icinde bir LiTS test hastasini GOREMEZ -> acik yasak sart.
KOR_NOKTALI = ["CT_ORG", "IRCADB", "ABDOMENCT_1K", "FLARE22", "M3D_SEG", "WORD_RAOS"]
# Korunan kaynaga (dis test) icerme zinciriyle ulasanlar
ZINCIRLI = ["FLARE23", "ABDOMENATLAS"]


@pytest.mark.parametrize("ad", KOR_NOKTALI + ZINCIRLI + ["HCC_TACE_SEG", "MSD_MSK"])
def test_yasakli_kaynak_yapilandirmada_isaretli(ad):
    assert ad in _blocked(), (
        f"{ad} yasak listesinde YOK. Icerme grafigi tek basina yetmez: dosyalari anonim "
        "oldugu icin kanonik kimlik eslesmez ve kesisim denetimi kor kalir.")


@pytest.mark.parametrize("ad", KOR_NOKTALI + ZINCIRLI)
def test_yasakli_kaynak_gercekten_reddediliyor(ad):
    with pytest.raises(CorpusError):
        build_manifest([{"volume_id": "case_1", "source": ad, "path": "a", "n_bytes": 1}],
                       blocked_sources=_blocked())


def test_hicbir_kaynak_hem_acik_hem_yasakli_degil():
    celiski = [n for n, s in SOURCES.items()
               if s.get("enabled") and (s.get("reason_blocked") or s.get("blocked_until"))]
    assert not celiski, f"hem enabled hem yasakli: {celiski}"


def test_dis_test_korunuyor():
    assert CFG["protected_sources"] == ["HCC_TACE_SEG"]
    cl = expand_contains(CFG["contains"])
    assert source_conflicts("FLARE23", cl, {"HCC_TACE_SEG"}) == {"HCC_TACE_SEG"}


def test_acik_kaynaklar_BILINCLI_liste_ve_her_biri_koken_beyanli():
    """Tuzak teli: yeni kaynak acmak BILINCLI bir karar olmali.

    22 Eyl 2026 gece kosusu icin uc kaynak acildi (dataset-verifier ayni gun dogruladi:
    URL, dosya deseni, kurum). Icerik dedup'i YAPILMADI -> her biri `disjoint_evidence`
    (kurum duzeyi beyan, 🟡) tasimak ZORUNDA ve hicbiri icerme grafiginde korunan bir
    kaynagi (HCC-TACE-Seg) ya da LiTS'i iceremez.
    """
    acik = [n for n, s in SOURCES.items() if s.get("enabled")]
    assert acik == ["LITS", "KITS23", "AMOS", "TOTALSEGMENTATOR_V2", "MSD_TASK08", "MSD_TASK07", "MSD_TASK10"], (
        f"acik kaynaklar {acik}. Yeni bir kaynak acmadan once: dataset-verifier + "
        "icerme grafigi + koken beyani (docs/EGITIM-PLANI.md §1).")
    cl = expand_contains(CFG["contains"])
    for n in acik:
        if n == "LITS":
            continue
        assert str(SOURCES[n].get("disjoint_evidence", "")).strip(), f"{n}: koken beyani yok"
        assert not source_conflicts(n, cl, {"HCC_TACE_SEG", "LITS", "MSD_TASK03"}), (
            f"{n} icerme grafiginde LiTS/HCC-TACE-Seg iceriyor")
