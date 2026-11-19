"""On egitim korpusu manifesti + sizinti denetimi (6 kesisim).

NEDEN BU DOSYA VAR
------------------
Projenin ikinci iddiasi: "mevcut 3D CT temel modellerinin benchmark'lari veri
sizintisiyla kirli" (kamu abdomen CT setlerinde vaka tekrari %59.1,
arXiv:2508.13626). Hazir bir encoder indirip kullanirsak, o encoder'in on egitim
verisi denetlenemedigi icin kendi iddiamizi kendimiz curutmus oluruz.
Kendi modelimizde tersi gecerli: on egitim korpusunu vaka duzeyinde BIZ biliriz.
Bu dosya o bilgiyi diske yazan ve dogrulugunu ASSERT eden yerdir.

KURAL 10 (dokunulmaz)
---------------------
Test ya da dis test hastasinin HICBIR taramasi, ETIKETSIZ BILE OLSA, on egitim
korpusuna giremez. ALTI kesisim ayri ayri assert edilir (bkz. _CHECKS).
Kirmiziysa egitim baslamaz.

DIKKAT: bu modul `resolve_canonical`'i CAGIRMAZ; kanonik kimligi ayni BICIMDE uretir.
Tek basina yetmiyordu -- `resolve_canonical` union-find kokunu sozluk sirasinda kucuk
olani sectigi icin kesisimin sag tarafi `canonical_members` ile genisletilir.

IKI AYRI KAVRAM, KARISTIRMA
---------------------------
  clean=True    ihlal BULUNMADI
  audited=True  denetim BOS KUME UZERINDE GECMEDI
Test seti bossa kesisim de bos cikar ve rapor "temiz" gorunur. Bu, sizintinin
olmadigini degil, BAKILMADIGINI gosterir. `require_audited=True` bunu reddeder.

ICERME (containment) GRAFIGI
----------------------------
Sizinti yalnizca vaka duzeyinde degil, VERI SETI duzeyinde de tasiniyor:
FLARE23 ⊃ HCC-TACE-Seg (L7), AbdomenAtlas ⊃ FLARE23 ⊃ ... Dis test adayimiz
HCC-TACE-Seg oldugu icin, FLARE23'u korpusa almak dis testimizi oldurur.
`contains:` kenarlari `configs/pretrain_corpus.yaml`'da; burada gecisli kapanisi
alinir ve korunan kaynaklardan birine ulasan her kaynak REDDEDILIR.

Icerik hash'i (`checksum`) bu asamada `pending`. Gerekce: kural 18, bu gece
>5 GB indirme yok. `scripts/lambda_bootstrap.sh` indirmeden sonra doldurur ve
denetimi YENIDEN kosar.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from src.data.case_id import case_id_from_path, glob_paths

MAX_EXAMPLES = 10
PENDING = "pending"

# Vaka numarasi: kimligin SONUNDAKI sayi grubu.
# B14 (leakage-auditor, 21 Eyl 2026): ILK grup aliniyordu ve "3Dircadb1.17" -> "3"
# oluyordu; diskteki 20 IRCADb vakasinin HEPSI "IRCADB::3"e cokuyordu. Ayni tuzak
# "Task03_Liver", "AbdomenCT-1K", "amos22" gibi surum/versiyon tasiyan her adda var.
# NOT: build_provenance ILK grubu aliyor. Tek sayi tasiyan kimliklerde (liver_38,
# BDMAP_00000091) ikisi ayni; farkli olan durumlarda dogru olan SONDAKI numaradir.
_NUM_RE = re.compile(r"(\d+)(?!.*\d)")
# build_provenance ILK sayi grubunu aliyor (`str.extract(r"(\d+)")`). Tek sayili
# kimliklerde ikisi ayni; iki sayili adlarda AYRISIRLAR ve kesisim kacirir.
_FIRST_NUM_RE = re.compile(r"(\d+)")


class CorpusError(RuntimeError):
    """Korpus sizintili ya da denetlenemez. Egitim baslamaz."""


# --------------------------------------------------------------- icerme grafigi
def expand_contains(contains: dict[str, list[str]] | None) -> dict[str, set[str]]:
    """Gecisli kapanis; her kaynak kendini de icerir. Dongude sonsuza gitmez.

    {"ABDOMENATLAS": ["FLARE23"], "FLARE23": ["HCC_TACE_SEG"]}
      -> ABDOMENATLAS icinde HCC_TACE_SEG gorunur.
    """
    contains = contains or {}
    nodes = set(contains) | {c for v in contains.values() for c in v}
    out: dict[str, set[str]] = {}
    for n in nodes:
        seen, stack = {n}, list(contains.get(n, []))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(contains.get(x, []))
        out[n] = seen
    return out


def source_conflicts(source: str, closure: dict[str, set[str]], protected: set[str]) -> set[str]:
    """`source` korpusa alinirsa hangi korunan kaynaklar kirlenir."""
    return (closure.get(source, {source}) | {source}) & set(protected)


# ---------------------------------------------------------------- kanonik kimlik
def canonical_for(case_id: str, source: str, equiv_map: dict[str, str] | None = None) -> str:
    """Vaka kimligi + kaynak -> kanonik kimlik; make_splits/resolve_canonical ile AYNI bicim.

    'lits-38' (LITS) ve 'liver_38' (MSD_TASK03) ayni fiziksel tarama -> ayni kimlik.
    Sayi tasimayan kimliklerde vaka kimliginin kendisi kullanilir (orn. TCIA hasta kodu).
    """
    equiv_map = equiv_map or {}
    case_id = str(case_id).strip()
    source = str(source).strip()
    if not case_id or not source:
        raise CorpusError(
            f"kanonik kimlik turetilemedi (vaka={case_id!r}, kaynak={source!r}). "
            "Kimligi olmayan hacim korpusa giremez -- denetlenemez.")
    family = equiv_map.get(source, source)
    m = _NUM_RE.search(case_id)
    return f"{family}::{m.group(1) if m else case_id}"


def equiv_map_from_config(cfg: dict) -> dict[str, str]:
    """configs/paths.yaml'daki equivalent_sources -> {kaynak: aile}. build_provenance ile ayni."""
    out: dict[str, str] = {}
    for group in cfg.get("leakage", {}).get("equivalent_sources", []) or []:
        head = "|".join(sorted(group))
        for s in group:
            out[s] = head
    return out


# -------------------------------------------------------------------- manifest
MANIFEST_COLUMNS = ["volume_id", "canonical_id", "source", "source_family", "path",
                    "n_bytes", "checksum", "checksum_kind", "canonical_status",
                    "license", "url"]


def assert_id_rule_agrees(records: list[dict]) -> None:
    """Vaka kimliginden sayi cikarma kurali `build_provenance` ile AYNI sonucu vermeli.

    B4 (leakage-auditor, final): biz SONDAKI sayi grubunu aliyoruz (`3Dircadb1.17` -> 17),
    `build_provenance` ILK grubu aliyor (-> 3). Iki sayili her adda ayrisirlar:
      3Dircadb1.17            provenance IRCADB::3      korpus IRCADB::17
      04-12-2000-NA-CT-12345  provenance ::04           korpus ::12345
      Task03_Liver_liver_43   provenance ::03           korpus ::43
      amos22_0507             provenance ::22           korpus ::0507
    Sonuc: provenance 20 IRCADb vakasini tek kimlige cokertirken korpus 20 ayri kimlik
    uretir -> kesisim denetimi 19 vakayi HIC GORMEZ. Iki kuralin birlestirilmesi
    `build_provenance`'a dokunmayi gerektiriyor (baska oturumun alani) -> simdilik
    AYRISMAYI SAPTAYIP REDDEDIYORUZ. Sessizce ayrismaktan iyidir.
    """
    bad = []
    for r in records:
        cid = str(r["volume_id"]).strip()
        a, b = _FIRST_NUM_RE.search(cid), _NUM_RE.search(cid)
        if a and b and a.group(1) != b.group(1):
            bad.append((cid, a.group(1), b.group(1)))
    if bad:
        raise CorpusError(
            f"{len(bad)} vaka kimliginde SAYI KURALI AYRISIYOR (build_provenance ILK sayiyi, "
            f"bu modul SONDAKINI aliyor): {bad[:5]}. Ayni tarama iki tarafta iki farkli "
            "kanonik kimlik alir ve kesisim denetimi onu GOREMEZ. Kuralin birlestirilmesi "
            "gerekiyor (docs/leakage-map.md L14).")


def build_manifest(records: list[dict], equiv_map: dict[str, str] | None = None,
                   contains: dict[str, list[str]] | None = None,
                   protected_sources: set[str] | None = None,
                   blocked_sources: dict[str, str] | None = None,
                   strict_id_rule: bool = False) -> pd.DataFrame:
    """Hacim kayitlari -> manifest tablosu. Yasakli kaynak varsa FIRLATIR.

    records: {"volume_id", "source", "path", "n_bytes", ("license", "url",
              "checksum", "canonical_status")}

    IKI AYRI SAVUNMA, IKISI DE GEREKLI:
      protected_sources + contains  kaynak ADIYLA yakalanan zincir (FLARE23 ⊃ HCC-TACE-Seg)
      blocked_sources               kaynak adiyla YAKALANAMAYAN ortusme

    Ikincisi neden sart: CT-ORG'un hacimlerinin cogu LiTS'ten "odunc alinmis" (TCIA'nin
    kendi sayfasi). Ama CT-ORG dosyalari anonim ve LiTS'e eslestiren bir tablo yok ->
    kanonik kimlikleri `CT_ORG::N` olur, `LITS|MSD_TASK03::N` ile ESLESMEZ. Yani
    `pretrain∩test` denetimi bir LiTS test hastasini CT-ORG icinde GOREMEZ.
    Denetimin kor oldugu yerde tek dogru davranis kaynagi hic almamaktir.
    """
    if strict_id_rule:
        assert_id_rule_agrees(records)
    protected = set(protected_sources or ())
    blocked = dict(blocked_sources or {})
    if blocked:
        hit = sorted({r["source"] for r in records} & set(blocked))
        if hit:
            raise CorpusError(
                "Yasakli kaynak korpusa alinamaz:\n"
                + "\n".join(f"  {s}: {blocked[s]}" for s in hit))
    closure = expand_contains(contains)
    if protected:
        bad = {r["source"]: sorted(c) for r in records
               if (c := source_conflicts(r["source"], closure, protected))}
        if bad:
            raise CorpusError(
                "Korpus korunan kaynaklari icerecek (dis test olurdu): "
                + "; ".join(f"{s} ⊃ {', '.join(v)}" for s, v in sorted(bad.items())))

    # DEGERLENDIRME AILESI (LiTS) de kapanisa karsi sinanir (bagimsiz denetim, 22 Eyl 2026):
    # LiTS'i ICEREN bir derleme (ULS23 ⊃ LITS gibi) kendi kimlik onekiyle gelir
    # (`ULS23::N` != `LITS|MSD_TASK03::N`) -> 6 kesisim de YESIL gecer ama test hastasi korpustadir.
    # LiTS'in kendisi korpusa girebilir (test vakalari --exclude-ids ile duser); onu iceren
    # BASKA bir kaynak giremez -- dosyalari anonim, hangi satirin test hastasi oldugu bilinemez.
    eval_family = {"LITS", "MSD_TASK03"}
    bad_eval = {r["source"]: sorted(c) for r in records
                if r["source"] not in eval_family
                and (c := source_conflicts(r["source"], closure, eval_family))}
    if bad_eval:
        raise CorpusError(
            "Korpus kaynagi DEGERLENDIRME ailesini (LiTS) iceriyor; kimlikleri eslesmedigi icin "
            "kesisim denetimi bunu GOREMEZ: "
            + "; ".join(f"{s} ⊃ {', '.join(v)}" for s, v in sorted(bad_eval.items())))

    rows = []
    for r in records:
        src = str(r["source"])
        rows.append({
            "volume_id": str(r["volume_id"]),
            "canonical_id": canonical_for(r["volume_id"], src, equiv_map),
            "source": src,
            "source_family": (equiv_map or {}).get(src, src),
            "path": str(r.get("path", "")),
            "n_bytes": int(r.get("n_bytes") or 0),
            "checksum": str(r.get("checksum", PENDING)),
            "checksum_kind": str(r.get("checksum_kind", "sha256")),
            # name_only: kimlik yalniz isim/yoldan turetildi, icerik dedup'i gormedi
            "canonical_status": str(r.get("canonical_status", "name_only")),
            "license": str(r.get("license", "")),
            "url": str(r.get("url", "")),
        })
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    dup = df[df.duplicated("path", keep=False)]
    if len(dup):
        raise CorpusError(f"{len(dup)} kayit ayni yolu paylasiyor (ornek: {list(dup['path'][:3])})")
    _assert_no_silent_merge(df)
    return df


def _assert_no_silent_merge(df: pd.DataFrame) -> None:
    """AYNI kaynaktan iki FARKLI vaka kimligi tek kanonik kimlige inmemeli, VE
    ayni kaynakta ayni vaka kimligi iki kez gorunmemeli.

    B5 (leakage-auditor, 21 Eyl 2026): ilk kontrol tek basina KORDU. Yol
    `kits23/case_00123/imaging_resampled.nii.gz` ve `.../case_00456/...` ikisi de
    'imaging_resampled' kimligi aliyor (case_id.py'nin jenerik ad listesinde
    'imaging_resampled' YOK, 'imaging' var). Iki farkli tarama TEK satir kimligi
    tasidigi icin `nunique() > 1` hic tetiklenmiyor: 489 KiTS hacmi tek kimlige
    coker ve kesisim denetimi 488'ini HIC GORMEZ.
    """
    dup = df[df.duplicated(["source", "volume_id"], keep=False)]
    if len(dup):
        ex = dup.groupby(["source", "volume_id"])["path"].apply(lambda v: list(v)[:2]).head(3)
        raise CorpusError(
            f"{dup['volume_id'].nunique()} vaka kimligi AYNI kaynakta tekrar ediyor "
            f"(farkli taramalar tek kimlige cokuyor): {ex.to_dict()}. "
            "src/data/case_id.py GENERIC_STEMS listesini genislet ya da kimlik kuralini duzelt.")
    g = df.groupby(["source", "canonical_id"])["volume_id"].nunique()
    bad = g[g > 1]
    if len(bad):
        ex = [f"{s}/{c}" for s, c in list(bad.index)[:5]]
        raise CorpusError(
            f"{len(bad)} kanonik kimlik AYNI kaynakta birden fazla vakayi sessizce birlestiriyor: "
            f"{ex}. Vaka kimligi kuralini duzelt (src/data/case_id.py) -- korpus kucuk gorunur.")


def drop_excluded(manifest: pd.DataFrame, exclude: set[str],
                  strict: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Dondurulmus ic test kimliklerini korpustan DUSUR; dusenleri de dondur.

    B3'un yapisal cozumu: denetimin "bu vaka korpusta" demesi yetmez, korpus onu
    hic tasimamalidir. `strict=True` iken hicbir kimlik tutmuyorsa FIRLATIR --
    kimlik bicimi uyusmazliginda dusurme sessizce hicbir sey yapar ve korpus
    degerlendirme vakalarini tasimaya devam eder (en sinsi hali).
    """
    exclude = {str(x) for x in (exclude or set())}
    if not exclude:
        return manifest, []
    mask = manifest["canonical_id"].astype(str).isin(exclude)
    hit = sorted(set(manifest.loc[mask, "canonical_id"].astype(str)))
    if strict:
        # B3 (leakage-auditor, final): eskiden yalniz HICBIRI eslesmezse firliyordu.
        # 26 kimligin 25'i duser 1'i kalirsa hata yoktu ve CLI "25 dusuruldu" yazardi;
        # operator 26 bekledigini fark etmezdi. Artik KISMI eslesme de kirmizi.
        kayip = sorted(exclude - set(hit))
        if kayip:
            raise CorpusError(
                f"Dusurulecek {len(exclude)} kimligin {len(kayip)} tanesi korpusta "
                f"BULUNAMADI (eslesmeyen: {kayip[:5]}). Kimlik bicimleri uyusmuyor olabilir "
                "-- dusurme sessizce eksik kaliyor. Korpus {len(manifest)} satir, "
                f"orn. {list(manifest['canonical_id'][:3])}.")
    return manifest.loc[~mask].reset_index(drop=True), hit


def drop_by_split(manifest: pd.DataFrame, splits: pd.DataFrame,
                  labels: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    """Belirtilen split etiketlerindeki kanonik kimlikleri korpustan DUSUR.

    B9 (leakage-auditor, final): `make_splits` LiTS 28-47'yi `excluded_ircadb_overlap`
    yapiyor, ama `configs/pretrain_corpus.yaml` LiTS globu 131 hacmin HEPSINI aliyor.
    Beyaz liste kontrolu (`pretrain∩egitim_disi`) bu yuzden HER KOSUDA kirmizi yaniyordu
    ve tek "cozum" gibi gorunen sey assert'i kismakti. Dogru cozum veriyi dusurmek.

    Bilinmeyen bir etiket verilirse FIRLATIR: yazim hatasi yuzunden hicbir sey
    dusurmemek, sessizce sizintili korpusla egitmek demektir.
    """
    mevcut = set(splits["split"].astype(str))
    bilinmeyen = [x for x in labels if x not in mevcut]
    if bilinmeyen:
        raise CorpusError(
            f"{bilinmeyen} split tablosunda yok (mevcut: {sorted(mevcut)}). "
            "Yazim hatasi sessizce hicbir sey dusurmez -- ve korpus sizintili kalir.")
    hedef = _expand_members(splits, splits["split"].astype(str).isin(labels))
    mask = manifest["canonical_id"].astype(str).isin(hedef)
    hit = sorted(set(manifest.loc[mask, "canonical_id"].astype(str)))
    return manifest.loc[~mask].reset_index(drop=True), hit


def scan_source(source: dict, equiv_map: dict[str, str] | None = None) -> list[dict]:
    """Diskteki bir kaynagi tarar: {'name', 'glob': [...], 'license', 'url'} -> kayitlar.

    Dosya yoksa BOS liste doner ve sayiyi cagiran raporlar; sessizce 'tamam' demez.
    """
    recs = []
    for pat in source.get("glob", []) or []:
        for p in glob_paths(pat):
            if not p.is_file():
                continue
            recs.append({"volume_id": case_id_from_path(p), "source": source["name"],
                         "path": str(p), "n_bytes": p.stat().st_size,
                         "license": source.get("license", ""), "url": source.get("url", "")})
    return recs


# --------------------------------------------------------------------- denetim
# On egitimin GORMESINE IZIN VERILEN split'ler. Beyaz liste, kara liste DEGIL:
# yarin make_splits yeni bir etiket uretirse otomatik olarak denetime girer.
# B4 (leakage-auditor): 'excluded_ircadb_overlap' (= LiTS 28-47 = IRCADb) eski
# kara liste mantiginda HICBIR kontrole girmiyordu; K3 onlari egitimden cikariyor
# ama on egitim serbestce yutuyordu.
CORPUS_ALLOWED_SPLITS = ("train", "val")

#  (ad, sol kume, sag kume)
_CHECKS = (
    ("egitim∩dogrulama", ("train",), ("val",)),
    ("egitim∩test", ("train", "val"), ("test",)),
    ("pretrain∩test", ("__corpus__",), ("test",)),
    ("pretrain∩dis_test", ("__corpus__",), ("external_test",)),
    ("pretrain∩egitim_disi", ("__corpus__",), ("__not_train__",)),
    ("pretrain∩ic_test", ("__corpus__",), ("__eval__",)),
)


def _expand_members(splits: pd.DataFrame, mask) -> set[str]:
    """canonical_id + canonical_members.

    B1 (leakage-auditor): `resolve_canonical` union-find kokunu SOZLUK SIRASINDA
    KUCUK olani secer. LiTS <-> AbdomenCT-1K onayli birlesmesinde split'in kimligi
    'ABDOMENCT_1K::123' olur, korpusunki 'LITS|MSD_TASK03::38' kalir ve kesisim
    SESSIZCE bos cikar -- bir LiTS test hastasi korpusta dururken rapor "temiz" der.
    `make_splits.apply_ircadb_rule` K3'te zaten uyelere bakiyor; ayni seyi burada da
    yapmak zorundayiz. Kok keyfi, UYELER degil.
    """
    out = set(splits.loc[mask, "canonical_id"].astype(str))
    if "canonical_members" in splits.columns:
        for mem in splits.loc[mask, "canonical_members"].fillna(""):
            out |= {x for x in str(mem).split(";") if x}
    return out


def expand_with_members(ids: set[str], splits: pd.DataFrame) -> set[str]:
    """Verilen kanonik kimlikleri, split tablosundaki AYNI BILESENIN tum uyeleriyle genislet.

    B2 (leakage-auditor, final): `_expand_members` yalniz split'ten turetilen kumelere
    uygulaniyordu. `frozen_test` `LITS|MSD_TASK03::38` uretirken korpusta ayni fiziksel
    tarama dedup-onayli `ABDOMENCT_1K::123` olarak duruyorsa, `pretrain∩ic_test`
    SESSIZCE bos cikiyordu -- dondurulmus testin tek korumasi bir tam dize eslesmesiydi.
    """
    if not ids or "canonical_members" not in splits.columns:
        return set(ids)
    out = set(ids)
    for cid, mem in zip(splits["canonical_id"].astype(str),
                        splits["canonical_members"].fillna("")):
        uyeler = {x for x in str(mem).split(";") if x} | {cid}
        if uyeler & out:
            out |= uyeler
    return out


def _ids(manifest: pd.DataFrame, splits: pd.DataFrame, keys: tuple[str, ...],
         eval_ids: set[str] | None = None) -> set[str]:
    if keys == ("__corpus__",):
        return set(manifest["canonical_id"].astype(str))
    if keys == ("__eval__",):
        return expand_with_members(set(eval_ids or ()), splits)
    if keys == ("__not_train__",):
        return _expand_members(splits, ~splits["split"].isin(CORPUS_ALLOWED_SPLITS))
    return _expand_members(splits, splits["split"].isin(keys))


def _sources_not_in_splits(manifest: pd.DataFrame, splits: pd.DataFrame) -> list[str]:
    """Korpus kaynaklarindan HICBIR kanonik kimligi split tablosunda gorunmeyenler.

    B2/B7/B8 (leakage-auditor, 21 Eyl 2026): korpus adlari (`KITS23`,
    `TCIA_PANCREAS`, `TOTALSEGMENTATOR_V2`) ile provenance'in urettigi aile adlari
    (`KITS`, `PANCREAS_CT`, `TOTALSEGMENTATOR`) ayri sozluklerden geliyor. Onek
    farkli oldugu icin o kaynaklarin test/dis-test vakalari kesisimde HIC GORUNMEZ:
    denetim "temiz" der ama aslinda BAKMAMISTIR.

    Ayni koku paylasan ikinci bir tuzak: bir derleme (ABDOMENCT_1K, FLARE22, M3D_SEG)
    LiTS iceriyor ama dosyalari anonim -> kimlikleri `ABDOMENCT_1K::N` olur ve
    `LITS|MSD_TASK03::N` ile eslesmez. CT-ORG (L8) ile birebir ayni korluk.

    Bos split tablosunda anlamsiz -> bos liste doner (o durumu `vacuous` yakalar).
    """
    if not len(splits) or not len(manifest):
        return []
    known = set(splits["canonical_id"].astype(str))
    if "canonical_members" in splits.columns:
        for mem in splits["canonical_members"].fillna(""):
            known |= {x for x in str(mem).split(";") if x}
    # B16 (leakage-auditor, final): `resolve_canonical`'in `vol::<id>` dugumleri "::"
    # tasimayan uyeler uretebiliyor; bunlar onek sayilirsa CT_ORG gibi bir kaynak
    # "kapsanmis" gorunur ve --require-sources-covered sessizce gecer.
    prefixes = {k.split("::", 1)[0] for k in known if "::" in k}
    out = sorted({str(src) for src, cid in zip(manifest["source"], manifest["canonical_id"])
                  if str(cid).split("::", 1)[0] not in prefixes})
    return out


def audit(manifest: pd.DataFrame, splits: pd.DataFrame,
          eval_ids: set[str] | None = None) -> dict:
    """Kesisimleri ayri ayri olcer. FIRLATMAZ -- rapor doner (kirmizi da diske yazilsin).

    eval_ids: gercek degerlendirme kumesinin kanonik kimlikleri (dondurulmus ic test
      ya da CV fold degerlendirme kovasi). B3 (leakage-auditor): `make_splits` hicbir
      zaman 'test' split'i URETMIYOR (train/val/external_test/excluded_ircadb_overlap),
      ve gercek degerlendirme `src/jepa/seg.py:make_folds` ile ozellik dizininden
      kuruluyor, `splits.csv` hic okunmuyor. Yani `pretrain∩test` YAPISAL OLARAK
      daima bos cikiyordu ve fold degerlendirme vakalari on egitimde gorulebiliyordu
      -- baskalarini sucladigimiz seyin ta kendisi. eval_ids verilmezse bu kontrol
      `vacuous` isaretlenir ve `audited=False` olur.

    Dondurulen sozluk JSON'a birebir yazilabilir. `assert_clean` karari verir.
    """
    for col in ("canonical_id", "split"):
        if col not in splits.columns:
            raise CorpusError(f"split tablosunda '{col}' sutunu yok -- denetim yapilamaz")
    if "canonical_id" not in manifest.columns:
        raise CorpusError("manifest'te 'canonical_id' yok -- denetim yapilamaz")

    checks: dict[str, dict] = {}
    for name, left, right in _CHECKS:
        a = _ids(manifest, splits, left, eval_ids)
        b = _ids(manifest, splits, right, eval_ids)
        inter = sorted(a & b)
        checks[name] = {
            "n": len(inter),
            "examples": inter[:MAX_EXAMPLES],
            "n_left": len(a), "n_right": len(b),
            # bos kumede gecen denetim = denetlenmemis denetim
            "vacuous": not a or not b,
        }
    n_vol = len(manifest)
    status = manifest["canonical_status"].value_counts().to_dict() if n_vol else {}
    pend = int((manifest["checksum"].astype(str) == PENDING).sum()) if n_vol else 0
    # B12 (leakage-auditor): 'audited' yalniz kume boslugu na bakiyordu. Tum satirlar
    # checksum=pending ve canonical_status=name_only iken "denetlendi" demek, 🟡 kaniti
    # 🟢 gibi gostermektir. Seviye ayri raporlanir.
    level = "hash_confirmed" if (n_vol and pend == 0 and
                                 set(status) <= {"hash_confirmed", "content_verified"}) else "name_only"
    return {
        "clean": all(c["n"] == 0 for c in checks.values()),
        "audited": all(not c["vacuous"] for c in checks.values()),
        "audited_level": level,
        "checks": checks,
        "corpus": {
            # B7: sutun yoksa uye genisletmesi SESSIZCE devre disi kaliyordu
            "canonical_members_var": bool("canonical_members" in splits.columns),
            "sources_not_in_splits": _sources_not_in_splits(manifest, splits),
            "n_volumes": n_vol,
            "n_canonical": int(manifest["canonical_id"].nunique()) if n_vol else 0,
            "n_bytes": int(manifest["n_bytes"].sum()) if n_vol else 0,
            "by_source": manifest["source"].value_counts().to_dict() if n_vol else {},
            "canonical_status": {str(k): int(v) for k, v in status.items()},
            "checksum_pending": pend,
        },
        "splits": splits["split"].value_counts().to_dict() if len(splits) else {},
    }


def assert_clean(result: dict, require_audited: bool = False,
                 require_content: bool = False,
                 require_sources_covered: bool = False) -> None:
    """Kirmizi denetimde FIRLATIR. Assert'i gevsetme -- veriyi duzelt."""
    bad = {k: v for k, v in result["checks"].items() if v["n"]}
    if bad:
        raise CorpusError(
            "KURAL 10 IHLALI -- on egitim korpusu split'e tabidir:\n"
            + "\n".join(f"  {k}: {v['n']} kanonik tarama ortak, orn. {v['examples']}"
                        for k, v in sorted(bad.items())))
    if require_audited and not result.get("corpus", {}).get("canonical_members_var", True):
        raise CorpusError(
            "split tablosunda 'canonical_members' sutunu YOK -> kanonik kok kaymasi "
            "denetlenemedi (L11). `resolve_canonical` kokleri sozluk sirasinda sectigi icin "
            "kesisim sessizce bos cikabilir. Split'i resolve_canonical ciktisindan uret.")
    if require_audited and not result["audited"]:
        vac = [k for k, v in result["checks"].items() if v["vacuous"]]
        raise CorpusError(
            f"Denetim BOS kume uzerinde gecti -> DENETLENMEDI: {vac}. "
            "'Ihlal bulunmadi' ile 'bakildi' ayni sey degil. "
            "'pretrain∩ic_test' bossa --eval-ids ver (dondurulmus ic test kumesi).")
    if require_sources_covered:
        blind = result.get("corpus", {}).get("sources_not_in_splits") or []
        # 22 Eyl 2026: isimle denetlenemeyen (kor) kaynak, yaml'da yazili KURUM DUZEYI
        # koken kaniti (`disjoint_evidence`) varsa kabul edilir -- ama bu 🟡 kanittir
        # (isim/icerik degil, beyan) ve denetim raporunda AYRI listelenir.
        ev = result.get("corpus", {}).get("blind_source_evidence") or {}
        blind = [b for b in blind if not str(ev.get(b, "")).strip()]
        if blind:
            raise CorpusError(
                f"Su korpus kaynaklarinin HICBIR kimligi split tablosunda yok: {blind}. "
                "Kesisim denetimi bu kaynaklar icin YAPISAL OLARAK kor -- 'temiz' raporu "
                "onlar hakkinda hicbir sey soylemez. Ya kaynak adini provenance ailesiyle "
                "esle (configs/paths.yaml source_aliases), ya kaynagi korpustan cikar.")
    if require_content and result.get("audited_level") != "hash_confirmed":
        raise CorpusError(
            "Denetim yalniz isim duzeyinde yapildi (checksum=pending / canonical_status=name_only). "
            "Icerik hash'i olmadan 'ayriklik kanitlandi' DENEMEZ -- once "
            "scripts/lambda_bootstrap.sh manifest adimi, sonra dedup_hash.")


# ------------------------------------------------------------------------- CLI
def _read_ids(path: Path) -> set[str]:
    """JSON liste ya da src.data.frozen_test ciktisi ({"canonical_ids": [...]})."""
    d = json.loads(Path(path).read_text())
    ids = d.get("canonical_ids") if isinstance(d, dict) else d
    if not isinstance(ids, list) or not ids:
        raise CorpusError(f"{path}: kanonik kimlik listesi okunamadi ya da bos")
    return {str(x) for x in ids}

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("configs/pretrain_corpus.yaml"))
    ap.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    ap.add_argument("--splits", type=Path, default=Path("outputs/splits.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/pretrain_manifest.csv"))
    ap.add_argument("--audit-out", type=Path, default=Path("outputs/split_audit.json"))
    ap.add_argument("--require-audited", action="store_true",
                    help="bos kumede gecen denetimi de reddet (Lambda'da ZORUNLU)")
    ap.add_argument("--exclude-splits", default="",
                    help="korpustan dusurulecek split etiketleri, virgulle "
                         "(orn. 'excluded_ircadb_overlap,external_test'). make_splits K3'u "
                         "uyguladigi icin bu olmadan pretrain∩egitim_disi HER KOSUDA kirmizi yanar.")
    ap.add_argument("--exclude-ids", type=Path, default=None,
                    help="korpustan DUSURULECEK kanonik kimlikler (src.data.frozen_test ciktisi). "
                         "--eval-ids ile ayni dosya verilir: korpus onlari tasimaz VE denetim dogrular.")
    ap.add_argument("--eval-ids", type=Path, default=None,
                    help="gercek degerlendirme kumesinin kanonik kimlikleri (JSON liste). "
                         "make_splits 'test' split'i URETMIYOR; degerlendirme katlari "
                         "seg.make_folds'tan geliyor -> bu dosya olmadan pretrain∩ic_test kor.")
    ap.add_argument("--require-content", action="store_true",
                    help="isim duzeyinde denetimi reddet; icerik hash'i sart (Lambda)")
    ap.add_argument("--require-sources-covered", action="store_true",
                    help="split'te hic gorunmeyen korpus kaynagini reddet")
    a = ap.parse_args(argv)

    corpus = yaml.safe_load(a.corpus.read_text())
    equiv = equiv_map_from_config(yaml.safe_load(a.paths.read_text()))
    protected = set(corpus.get("protected_sources", []) or [])

    records, missing = [], []
    for s in corpus.get("sources", []) or []:
        if not s.get("enabled", True):
            continue
        r = scan_source(s, equiv)
        exp = s.get("n_expected")
        print(f"[korpus] {s['name']}: {len(r)} hacim bulundu"
              + (f" (beklenen {exp})" if exp else ""))
        if exp and len(r) != exp:
            missing.append(f"{s['name']}: {len(r)} != beklenen {exp}")
        records += r
    if missing:
        print("[UYARI] beklenen hacim sayisi tutmuyor: " + "; ".join(missing))

    # `blocked_until` de yasaktir: B14 (leakage-auditor) -- bu anahtari HICBIR kod
    # okumuyordu, yani MSD_MSK `enabled: true` yapilirsa L10 korumasi yalnizca yorumdu.
    blocked = {}
    for src in corpus.get("sources", []) or []:
        why = src.get("reason_blocked") or (
            f"kosul saglanmadi: {src['blocked_until']}" if src.get("blocked_until") else None)
        if why:
            blocked[src["name"]] = str(why).strip()
    man = build_manifest(records, equiv, corpus.get("contains"), protected, blocked)
    if a.exclude_ids:
        ex = _read_ids(a.exclude_ids)
        man, dropped = drop_excluded(man, ex, strict=True)
        print(f"[korpus] dondurulmus ic test: {len(dropped)} kanonik tarama korpustan DUSURULDU "
              f"({a.exclude_ids})")
    if a.exclude_splits.strip():
        etiketler = tuple(x.strip() for x in a.exclude_splits.split(",") if x.strip())
        man, dusen = drop_by_split(man, pd.read_csv(a.splits), etiketler)
        print(f"[korpus] {etiketler}: {len(dusen)} kanonik tarama korpustan DUSURULDU")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(a.out, index=False)
    print(f"[korpus] {len(man)} hacim / {man['canonical_id'].nunique()} kanonik -> {a.out}")

    if a.exclude_splits.strip() and not a.splits.exists():
        raise CorpusError(f"--exclude-splits verildi ama {a.splits} yok")
    if not a.splits.exists():
        raise CorpusError(f"{a.splits} yok. On egitim korpusu split OLMADAN denetlenemez "
                          "-- once src.data.make_splits.")
    eval_ids = _read_ids(a.eval_ids) if a.eval_ids else None
    if eval_ids:
        print(f"[korpus] degerlendirme kumesi: {len(eval_ids)} kanonik kimlik ({a.eval_ids})")
    else:
        print("[UYARI] --eval-ids verilmedi -> 'pretrain∩ic_test' DENETLENMEYECEK. "
              "make_splits 'test' split'i uretmiyor; gercek degerlendirme katlari "
              "src/jepa/seg.py:make_folds'tan geliyor.")
    res = audit(man, pd.read_csv(a.splits), eval_ids=eval_ids)
    res["corpus"]["blind_source_evidence"] = {
        str(x["name"]): str(x.get("disjoint_evidence", "")).strip()
        for x in corpus.get("sources", []) or []
        if x.get("enabled", True) and x["name"] in res["corpus"]["sources_not_in_splits"]}
    a.audit_out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[korpus] denetim -> {a.audit_out}")
    blind = res["corpus"]["sources_not_in_splits"]
    for k, v in res["checks"].items():
        if v["n"]:
            flag = "🔴"
        elif v["vacuous"]:
            flag = "⚠️ bos kume"
        elif blind:
            # Kor bir kaynak varken "✅" yazmak yaniltir: o kaynagin kimlikleri
            # karsi tarafla HIC eslesmedigi icin kesisim yapisal olarak bos cikar.
            flag = "⚠️ kor kaynak var"
        else:
            flag = "✅"
        print(f"  {flag} {k}: {v['n']} ortak ({v['n_left']} vs {v['n_right']})")
    for b, e in res["corpus"]["blind_source_evidence"].items():
        print(f"  🟡 {b}: isimle denetlenemez; koken beyani: {e or 'YOK'}")
    if blind:
        print(f"  ⚠️  split'te hic gorunmeyen kaynak (denetim KOR): {blind}")
        print("      -> Yukaridaki ✅'lar bu kaynaklar hakkinda HICBIR SEY SOYLEMEZ.")
    print(f"  denetim seviyesi: {res['audited_level']}")
    assert_clean(res, require_audited=a.require_audited,
                 require_content=a.require_content,
                 require_sources_covered=a.require_sources_covered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
