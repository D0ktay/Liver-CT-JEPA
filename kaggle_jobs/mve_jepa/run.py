"""Kaggle MVE: LiverJEPA'nin KENDI on egitimi -- jepa vs mae vs rastgele, T4, tek kosu.

Amac bir "sonuc" uretmek DEGIL: (1) hattin gercek GPU'da uctan uca calistigini
kanitlamak, (2) cokus esiklerini kalibre etmek icin taban olcmek, (3) abstract
taslagi bos kalmasin diye ilk sayiyi almak.

Hat: LiTS (Kaggle girdisi) -> dondurulmus ic test -> onbellek (1 mm izo, karaciger ROI)
  -> on egitim jepa + mae -> ince ayar jepa|mae|none -> evaluate + bootstrap.

🔴 KURAL 10: dondurulmus ic test vakalari on egitim korpusuna GIRMEZ. `src.jepa.pretrain`
   --eval-ids ile bunu ayrica assert eder; ihlalde is CAKAR (sessizce kosmaz).
🔴 ROI, IKI KUME ICIN FARKLI (leakage-auditor final, B5):
   - ON EGITIM + INCE AYAR EGITIMI: karaciger kutusu GT etiketi **1**'den (`--roi-label 1`),
     yani TUMOR etiketi (2) kutuyu BELIRLEMEZ. Bu yine de bir varsayimdir ve raporda yazilir;
     Lambda'da TS `liver` maskesiyle degistirilecek.
   - DONDURULMUS TEST: **ROI YOK, tum hacim.** Ilk surumde test vakalari da GT'den turetilmis
     kutuya kirpiliyordu; `reconstruct` kutu disini sifirladigi icin model GT anatomisi
     disinda YANLIS POZITIF URETEMIYORDU -> precision / ortusme / marj ihlali sistematik
     olarak yukari yanli ve TS tabaniyla (tum hacim) KARSILASTIRILAMAZ olurdu.

Otonom: bash scripts/kaggle_push.sh mve_jepa
KURAL: bu surec numpy/torch YUKLEMEZ (Kaggle'in numpy'siyle carpisir).
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

T0 = time.time()
INPUT = os.environ.get("JOB_INPUT", "/kaggle/input")
TMP = Path(os.environ.get("JOB_TMP", "/kaggle/tmp"))
WORK = Path(os.environ.get("JOB_WORK", "/kaggle/working"))
DRY = os.environ.get("JOB_DRY") == "1"
PROFILE = os.environ.get("JOB_PROFILE", "kucuk")
PRE_STEPS = os.environ.get("JOB_PRE_STEPS", "8000")
FT_STEPS = os.environ.get("JOB_FT_STEPS", "4000")
SEED = os.environ.get("JOB_SEED", "0")
ARMS = os.environ.get("JOB_ARMS", "jepa mae none").split()
CODE = TMP / "code"
OUT = WORK / "outputs" / "mve"
CACHE = TMP / "cache_iso1mm"


def log(*a):
    print(f"[{time.time() - T0:6.0f} sn]", *a, flush=True)


def bul(kok, *desenler):
    """Sembolik bagli klasorlere de giren dosya arama. Kaggle veri setlerini
    /kaggle/input/datasets/<sahip>/<ad> altina BAG olarak koyabiliyor; glob('**')
    baglara girmedigi icin oradaki dosyalari bulamaz (liver-l6-round v2 boyle dustu)."""
    import fnmatch
    bulunan = []
    for kk, _, dosyalar in os.walk(kok, followlinks=True):
        for d in dosyalar:
            yol = os.path.join(kk, d)
            if any(fnmatch.fnmatch(yol, f"*{x}") for x in desenler):
                bulunan.append(yol)
    return sorted(bulunan)


def sh(*args, check=True):
    log("$", " ".join(map(str, args)))
    r = subprocess.run([str(a) for a in args])
    if check and r.returncode != 0:
        raise RuntimeError(f"basarisiz ({r.returncode}): {' '.join(map(str, args))}")
    return r.returncode


# ------------------------------------------------------------------ 0) kod ----
if CODE.exists():
    shutil.rmtree(CODE)
tars = bul(INPUT, "code.tar.gz")
roots = sorted({str(Path(p).parent.parent.parent) for p in bul(INPUT, "src/jepa/pretrain.py")})
if tars:
    CODE.mkdir(parents=True)
    with tarfile.open(tars[0]) as t:
        t.extractall(CODE)
else:
    assert len(roots) == 1, f"kod bulunamadi / birden fazla: {roots}"
    shutil.copytree(roots[0], CODE)
os.chdir(CODE)
log("kod:", (CODE / "COMMIT").read_text().strip() if (CODE / "COMMIT").exists() else "?")
OUT.mkdir(parents=True, exist_ok=True)

chk = subprocess.run([sys.executable, "-c", "import torch, nibabel, scipy; "
                      "print('CUDA', torch.cuda.is_available(), 'NGPU', torch.cuda.device_count())"],
                     capture_output=True, text=True, check=True).stdout
log(chk.strip())
assert "CUDA True" in chk or DRY, "GPU yok -- MVE'nin anlami kalmaz"

# ------------------------------------------------------- 1) LiTS girdileri ----
# Iki LiTS veri seti (part-1 ve part-2) AYNI segmentasyonlari tasiyabiliyor: ilk kosuda
# 131 hacme karsilik 211 segmentasyon bulundu. Sayiya degil DOSYA ADINA gore tekillestir.
def tekil(yollar):
    d = {}
    for y in yollar:
        d.setdefault(Path(y).name, []).append(y)
    cift = {k: v for k, v in d.items() if len(v) > 1}
    return {k: v[0] for k, v in d.items()}, cift


vols_all = bul(INPUT, "volume-*.nii")
segs_all = bul(INPUT, "segmentation-*.nii")
vols_d, vols_dup = tekil(vols_all)
segs_d, segs_dup = tekil(segs_all)
log(f"LiTS: {len(vols_all)} hacim yolu -> {len(vols_d)} tekil; "
    f"{len(segs_all)} segmentasyon yolu -> {len(segs_d)} tekil")
if vols_dup or segs_dup:
    log(f"  (ayni ad birden fazla yolda: {len(vols_dup)} hacim, {len(segs_dup)} segmentasyon "
        f"-- ilk yol kullaniliyor, orn. {list(segs_dup)[:2]})")
assert len(vols_d) == 131 or DRY, f"131 tekil hacim beklenirdi: {len(vols_d)}"
eksik = [n for n in vols_d if n.replace("volume-", "segmentation-") not in segs_d]
assert not eksik or DRY, f"{len(eksik)} hacmin segmentasyonu yok: {eksik[:5]}"
vols, segs = sorted(vols_d.values()), sorted(segs_d.values())

raw = Path("data/raw/LiTS")
raw.mkdir(parents=True, exist_ok=True)
for f in vols + segs:
    dst = raw / Path(f).name
    if not dst.exists():
        dst.symlink_to(f)
n_raw_v = len(list(raw.glob("volume-*.nii")))
n_raw_s = len(list(raw.glob("segmentation-*.nii")))
log(f"baglandi: {n_raw_v} hacim, {n_raw_s} segmentasyon")
assert n_raw_v == n_raw_s or DRY, f"bag sayilari tutmuyor: {n_raw_v}/{n_raw_s}"

# ------------------------------------------- 2) dondurulmus ic test kumesi ----
FROZEN = OUT / "internal_test.json"
if not FROZEN.exists():
    sh(sys.executable, "-m", "src.data.frozen_test",
       "--ids-glob", "data/raw/LiTS/segmentation-*.nii", "--out", FROZEN)
frozen = json.loads(FROZEN.read_text())
test_ids = set(frozen["case_ids"])
log(f"dondurulmus ic test: {len(test_ids)} vaka")

# --------------------------- 3) manifest: split + kural 10 denetimi ----------
# Kaggle'da provenance tablosu yok. Minik ama GERCEK bir splits.csv uretiyoruz:
# dondurulmus test -> "test", geri kalan %85/%15 train/val. Boylece pretrain∩test
# kontrolu BOS KUME uzerinde gecmez (aksi halde denetim anlamsiz olurdu).
#
# ⚠️ BILINCLI FARK: burada K3 (LiTS 28-47 = IRCADb ortusmesi) UYGULANMIYOR, cunku bu
# deneyde IRCADb HICBIR SEKILDE degerlendirme kumesi degil -- degerlendirme dondurulmus
# LiTS alt kumesinde. Dolayisiyla MVE korpusu 131 - 27 = 104 hacim.
# Lambda kosusunda `make_splits` K3'u uygular ve o 16 vakayi `excluded_ircadb_overlap`
# yapar -> oradaki korpus 88 hacim olur (docs/EGITIM-PLANI.md §1). Iki sayi FARKLIDIR
# ve bu fark BILEREKTIR; sessiz bir tutarsizlik degildir.
import csv  # noqa: E402  (torch/numpy DEGIL -- cekirdek modul, guvenli)
import random  # noqa: E402

ids = sorted({Path(v).name[len("volume-"):-len(".nii")] for v in vols}, key=int)
rng = random.Random(1337)
kalan = [i for i in ids if f"lits-{i}" not in test_ids]
rng.shuffle(kalan)
n_val = max(1, int(0.15 * len(kalan)))
split_of = {i: ("val" if k < n_val else "train") for k, i in enumerate(kalan)}
for i in ids:
    if f"lits-{i}" in test_ids:
        split_of[i] = "test"

splits_p = Path("outputs/splits.csv")
splits_p.parent.mkdir(parents=True, exist_ok=True)
with splits_p.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["volume_id", "canonical_id", "source_family", "split"])
    for i in ids:
        w.writerow([f"lits-{i}", f"LITS|MSD_TASK03::{i}", "LITS|MSD_TASK03", split_of[i]])
log(f"splits.csv: {sum(1 for v in split_of.values() if v == 'train')} train / "
    f"{sum(1 for v in split_of.values() if v == 'val')} val / {len(test_ids)} test")
log(f"on egitim korpusu (MVE): {len(ids) - len(test_ids)} hacim "
    "-- K3 uygulanmadi (IRCADb degerlendirme kumesi degil); Lambda'da 88 olacak")

MAN_ALL = OUT / "manifest_all.csv"          # ince ayar onbellegi: TUM vakalar
MAN_PRE = OUT / "pretrain_manifest.csv"     # on egitim korpusu: test HARIC
with MAN_ALL.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["volume_id", "canonical_id", "source", "source_family", "path", "n_bytes",
                "checksum", "checksum_kind", "canonical_status", "license", "url"])
    for i in ids:
        w.writerow([f"lits-{i}", f"LITS|MSD_TASK03::{i}", "LITS", "LITS|MSD_TASK03",
                    str(raw / f"volume-{i}.nii"), 0, "pending", "sha256",
                    "name_only", "CC BY-NC-ND 4.0", ""])
sh(sys.executable, "-c",
   "import pandas as pd, json, sys;"
   f"m = pd.read_csv({str(MAN_ALL)!r});"
   f"t = set(json.load(open({str(FROZEN)!r}))['canonical_ids']);"
   "m = m[~m['canonical_id'].isin(t)];"
   f"m.to_csv({str(MAN_PRE)!r}, index=False);"
   "print('[korpus] on egitim hacmi:', len(m))")

# Denetim. GERCEK BULGU (v2): `configs/pretrain_corpus.yaml` LITS icin IKI glob tasiyor
# (yerel + /kaggle/input). run.py yerel yolu Kaggle yoluna SEMBOLIK BAG olarak kurdugu icin
# ayni hacim IKI YOLDAN bulunuyor -> _assert_no_silent_merge hakli olarak CorpusError
# firlatiyordu, ama `check=False` bunu YUTUYORDU (leakage-auditor final, B8).
# Cozum: Kaggle'a ozel korpus yaml'i -- LITS globu yalniz yerel bag dizini.
CORPUS_K = OUT / "pretrain_corpus_kaggle.yaml"
_satirlar = Path("configs/pretrain_corpus.yaml").read_text().splitlines(keepends=True)
CORPUS_K.write_text("".join(x for x in _satirlar if "/kaggle/input/**/volume-" not in x))
assert "/kaggle/input/**/volume" not in CORPUS_K.read_text(), "korpus yaml override tutmadi"

# Bu adim KIRMIZIYSA is CAKAR -- artik check=True.
sh(sys.executable, "-m", "src.data.pretrain_manifest",
   "--corpus", CORPUS_K, "--paths", "configs/paths.yaml",
   "--splits", splits_p, "--exclude-ids", FROZEN, "--eval-ids", FROZEN,
   "--out", OUT / "audit_manifest.csv", "--audit-out", OUT / "split_audit.json")

# ------------------------------------------------------------- 4) onbellek ----
# Egitim vakalari: karaciger ROI'si ile (etiket 1; tumor etiketi kutuyu BELIRLEMEZ).
# Dondurulmus test: ROI YOK, TUM HACIM -- yoksa degerlendirme yukari yanli olur (B5).
MAN_TEST = OUT / "manifest_test.csv"
with MAN_TEST.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["volume_id", "canonical_id", "source", "source_family", "path", "n_bytes",
                "checksum", "checksum_kind", "canonical_status", "license", "url"])
    for i in ids:
        if f"lits-{i}" in test_ids:
            w.writerow([f"lits-{i}", f"LITS|MSD_TASK03::{i}", "LITS", "LITS|MSD_TASK03",
                        str(raw / f"volume-{i}.nii"), 0, "pending", "sha256",
                        "name_only", "CC BY-NC-ND 4.0", ""])

if len(list(CACHE.glob("*.npz"))) < len(ids):
    # (a) egitim kumesi -- karaciger ROI'si (etiket 1)
    sh(sys.executable, "-m", "src.jepa.cache_build",
       "--manifest", MAN_PRE, "--out", CACHE,
       "--labels-glob", "data/raw/LiTS/segmentation-*.nii",
       "--roi-glob", "data/raw/LiTS/segmentation-*.nii", "--roi-label", "1",
       "--trust-ct-spacing")
    # (b) dondurulmus test -- ROI YOK (tum hacim), TS tabaniyla ayni sartlar
    sh(sys.executable, "-m", "src.jepa.cache_build",
       "--manifest", MAN_TEST, "--out", CACHE,
       "--labels-glob", "data/raw/LiTS/segmentation-*.nii", "--overwrite",
       "--trust-ct-spacing")
log("onbellek:", len(list(CACHE.glob("*.npz"))), "npz",
    f"{sum(f.stat().st_size for f in CACHE.glob('*.npz')) / 1e9:.2f} GB")
log(f"  egitim {len(ids) - len(test_ids)} vaka (karaciger ROI, etiket 1) + "
    f"dondurulmus test {len(test_ids)} vaka (ROI YOK, tum hacim)")

# ---------------------------------------------------------- 5) on egitim -----
for obj in [a for a in ARMS if a != "none"]:
    d = OUT / f"pre_{obj}"
    if (d / "encoder.pt").exists():
        log(f"{obj}: encoder zaten var, atlaniyor")
        continue
    sh(sys.executable, "-m", "src.jepa.pretrain", "--objective", obj,
       "--manifest", MAN_PRE, "--cache", CACHE, "--out", d, "--profile", PROFILE,
       "--eval-ids", FROZEN, "--steps", PRE_STEPS, "--seed", SEED,
       "--log-every", "200", "--ckpt-every", "1000", "--resume")

# ---------------------------------------------- 6) ince ayar + degerlendirme --
ozet = {}
for arm in ARMS:
    d = OUT / f"ft_{arm}"
    init = "random" if arm == "none" else str(OUT / f"pre_{arm}" / "encoder.pt")
    if not (d / "done.json").exists():
        sh(sys.executable, "-m", "src.jepa.finetune_e2e", "--init", init,
           "--cache", CACHE, "--test-ids", FROZEN, "--out", d, "--profile", PROFILE,
           "--steps", FT_STEPS, "--seed", SEED, "--log-every", "200",
           "--gt-dir", "data/raw/LiTS", "--resume")
    ev = d / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    sh(sys.executable, "-m", "src.metrics.make_pairs",
       "--gt-glob", "data/raw/LiTS/segmentation-*.nii",
       "--pred-glob", str(d / "pred" / "*.nii.gz"), "--out", ev / "pairs.csv", check=False)
    # LiTS GT tumor etiketi 2, tahminimiz ikili (1) -- etiketler AYRI
    sh(sys.executable, "-m", "src.metrics.evaluate", "--pairs", ev / "pairs.csv",
       "--out", ev, "--gt-label", "2", "--pred-label", "1", check=False)
    sh(sys.executable, "-m", "src.metrics.bootstrap", "--eval", ev, "--n-boot", "2000",
       check=False)
    sh(sys.executable, "-m", "src.metrics.margin_sweep", "--pairs", ev / "pairs.csv",
       "--out", ev, "--gt-label", "2", "--pred-label", "1", check=False)
    sp = ev / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        to = s.get("targeting", {}).get("overall", {})
        ozet[arm] = {"n_cases": s.get("n_cases"), "n_failed": s.get("n_failed"),
                     "dice_median": s.get("voxel", {}).get("dice_median"),
                     "lesion_recall": s.get("lesion", {}).get("recall"),
                     "lesion_precision": s.get("lesion", {}).get("precision"),
                     "overlap_hit_rate": to.get("overlap_hit_rate"),
                     "margin_violation_rate_all": to.get("margin_violation_rate_all"),
                     "centroid_error_mm_median": to.get("centroid_error_mm_median")}

# -------------------------------------------------------------- 7) rapor -----
cokus = {a: (OUT / f"pre_{a}" / "collapse.json").exists() for a in ARMS if a != "none"}
(OUT / "RESULT.json").write_text(json.dumps({
    "profile": PROFILE, "pre_steps": PRE_STEPS, "ft_steps": FT_STEPS, "seed": SEED,
    "n_frozen_test": len(test_ids), "arms": ARMS, "collapse": cokus,
    "summary": ozet,
    "n_corpus": len(ids) - len(test_ids),
    "k3_uygulanmadi": True,
    "uyari": [
        "Dondurulmus ic test 27 vaka -> guven araliklari GENIS.",
        "Egitim vakalari GT karaciger etiketinden (1) turetilmis ROI kutusunda; "
        "dondurulmus TEST vakalari ROI'siz (tum hacim) -- TS tabaniyla ayni sartlar.",
        "K3 (LiTS 28-47) UYGULANMADI: bu deneyde IRCADb degerlendirme kumesi degil. "
        "SONUC: bu encoder 3D-IRCADb'de HICBIR ZAMAN dis test edilemez.",
        "MVE profili 'kucuk' ve adim sayisi dusuk -> TS tabanini gecmesi BEKLENMIYOR. "
        "Amac hattin gercek GPU'da calistigini kanitlamak ve cokus tabanini olcmek.",
    ],
}, indent=1))
print(json.dumps(ozet, indent=1))
log("BITTI")
assert ozet, "hicbir kol degerlendirilemedi"
