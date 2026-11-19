"""Kaggle DOGRULAMA kosusu: duzeltilmis ince ayar hatti GERCEK veride ogreniyor mu?

22 Eyl 2026. MVE (liver-mve-jepa v3) Dice 0.05 verdi; teshis: egitim karaciger kutusunda,
test tum govdede -> tahminlerin cogu karaciger DISINDA (olculdu). Bu is, Lambda H100
gecesinden ONCE duzeltmeyi gercek LiTS verisinde sinar. ~2 saat, 2xT4.

🔴 DONDURULMUS IC TEST'E DOKUNMAZ. "Test" olarak verilen kume, EGITIM tarafindan ayrilmis
bir GELISTIRME kumesidir (N_DEV vaka). Dondurulmus 27 vaka onbellege bile ALINMAZ.

GPU0: `orta` profil (gece kosusunun profili) · GPU1: `kucuk` profil. Ikisi de rastgele
baslatma -- amac on egitimi degil SEGMENTASYON HATTINI dogrulamak.
Basari olcutu (onceden yazildi): dogrulama tumor Dice'i > 0.30 VE karaciger Dice'i > 0.85
VE karaciger-disi tahmin payi son islemeyle 0'a iniyor. MVE'de tumor Dice 0.05'ti.

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
SEED = os.environ.get("JOB_SEED", "0")
N_TRAIN = int(os.environ.get("JOB_N_TRAIN", "48"))
N_DEV = int(os.environ.get("JOB_N_DEV", "6"))
MAX_MIN = os.environ.get("JOB_MAX_MIN", "95")
CODE = TMP / "code"
OUT = WORK / "outputs" / "val_seg"
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
assert "CUDA True" in chk or DRY, "GPU yok"

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
_ref = json.loads(Path("configs/frozen_internal_test.json").read_text())["case_ids"]
assert sorted(_ref) == sorted(test_ids) or DRY, "dondurulmus test repodaki ile AYNI DEGIL"


# ------------------------------------ 3) alt kume: egitim + gelistirme (TEST HARIC) ----
import csv  # noqa: E402
import random  # noqa: E402

ids = sorted({Path(v).name[len("volume-"):-len(".nii")] for v in vols}, key=int)
havuz = [i for i in ids if f"lits-{i}" not in test_ids]
random.Random(2026).shuffle(havuz)
dev = havuz[:N_DEV]
egitim = havuz[N_DEV:N_DEV + N_TRAIN]
assert not ({f"lits-{i}" for i in dev + egitim} & test_ids), "dondurulmus test alt kumeye sizdi"
DEV = OUT / "dev_ids.json"
DEV.write_text(json.dumps({"case_ids": [f"lits-{i}" for i in dev],
                           "note": "GELISTIRME kumesi -- dondurulmus ic test DEGIL"}))
MAN = OUT / "manifest_sub.csv"
with MAN.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["volume_id", "canonical_id", "source", "source_family", "path", "n_bytes",
                "checksum", "checksum_kind", "canonical_status", "license", "url"])
    for i in dev + egitim:
        w.writerow([f"lits-{i}", f"LITS|MSD_TASK03::{i}", "LITS", "LITS|MSD_TASK03",
                    str(raw / f"volume-{i}.nii"), 0, "pending", "sha256", "name_only", "", ""])
log(f"alt kume: {len(egitim)} egitim + {len(dev)} gelistirme; dondurulmus {len(test_ids)} vaka DISARIDA")

# ------------------------------------------------ 4) onbellek: TAM HACIM, karaciger+tumor ----
sh(sys.executable, "-m", "src.jepa.cache_build", "--manifest", MAN, "--out", CACHE,
   "--labels-glob", "data/raw/LiTS/segmentation-*.nii", "--liver-label", "1",
   "--trust-ct-spacing", "--workers", "4")
log("onbellek:", len(list(CACHE.glob("*.npz"))), "npz",
    f"{sum(f.stat().st_size for f in CACHE.glob('*.npz')) / 1e9:.2f} GB")

# ------------------------------------------------ 5) iki profil PARALEL (GPU basina bir) ----
KOSU = [("orta", "0", "6000", "2"), ("kucuk", "1", "12000", "4")]
procs = []
for prof, gpu, steps, bs in KOSU:
    d = OUT / f"ft_{prof}"
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    cmd = [sys.executable, "-m", "src.jepa.finetune_e2e", "--init", "random",
           "--cache", str(CACHE), "--test-ids", str(DEV), "--out", str(d), "--profile", prof,
           "--steps", steps, "--batch", bs, "--seed", SEED, "--log-every", "250",
           "--val-n", "4", "--val-every", "1500", "--max-minutes", MAX_MIN,
           "--ram-gb", "9", "--no-tta", "--gt-dir", "data/raw/LiTS"]
    if DRY:
        cmd += ["--device", "cpu"]
    log("$", " ".join(cmd))
    procs.append((prof, subprocess.Popen(cmd, env=env, stdout=open(d / "train.log", "w"),
                                         stderr=subprocess.STDOUT)))
son = {p: 0 for p, _ in procs}
while any(pr.poll() is None for _, pr in procs):
    time.sleep(60)
    for prof, _ in procs:
        lines = (OUT / f"ft_{prof}" / "train.log").read_text().splitlines()
        for ln in lines[son[prof]:]:
            if "dogrulama" in ln or "takvim" in ln or "step=" in ln or "UYARI" in ln or "depo" in ln:
                log(f"[{prof}]", ln.strip())
        son[prof] = len(lines)
rcs = {prof: pr.returncode for prof, pr in procs}
log("bitis kodlari:", rcs)

# ------------------------------------------------ 6) lezyon duzeyi olcum (gelistirme kumesi) ----
ozet = {"commit": (CODE / "COMMIT").read_text().strip() if (CODE / "COMMIT").exists() else "?",
        "n_train": len(egitim), "n_dev": len(dev), "returncodes": rcs, "arms": {}}
for prof, _ in procs:
    d = OUT / f"ft_{prof}"
    if rcs[prof] != 0:
        log(f"[{prof}] BASARISIZ -- son satirlar:")
        print("\n".join((d / "train.log").read_text().splitlines()[-30:]), flush=True)
        continue
    ev = d / "eval"
    sh(sys.executable, "-m", "src.metrics.make_pairs", "--gt-glob", "data/raw/LiTS/segmentation-*.nii",
       "--pred-glob", str(d / "pred" / "*.nii.gz"), "--out", ev / "pairs.csv")
    sh(sys.executable, "-m", "src.metrics.evaluate", "--pairs", ev / "pairs.csv", "--out", ev,
       "--gt-label", "2", "--pred-label", "1")
    ozet["arms"][prof] = {"done": json.loads((d / "done.json").read_text()),
                          "val": [json.loads(x) for x in (d / "val.jsonl").read_text().splitlines()]
                          if (d / "val.jsonl").exists() else [],
                          "summary": json.loads((ev / "summary.json").read_text())}
    for f in (d / "state.pt", d / "best.pt"):
        if f.exists():
            f.unlink()                                     # cikti kotasi
(OUT / "RESULT.json").write_text(json.dumps(ozet, indent=1, default=str))
log("bitti ->", OUT / "RESULT.json")
if any(rc != 0 for rc in rcs.values()):
    raise SystemExit("en az bir kol basarisiz")
