"""Kaggle isi: L6 birebir (exact) kanali ACIK olarak yeniden arama, LiTS 131.

Neden: match_crop sorgunun tamsayi olup olmadigini `p == rint(p)` ile KATI sinar. TS egitim
goruntuleri float32 gosterim hatasi (~5e-4) tasidigi icin ilk kosuda 842/842 sorgu "exact'e
kor" sayildi -- en guclu kopya dedektoru hic calismadi. Sorgularin 416'si toleransla (1e-3)
tamsayi; yuvarlanip integer_valued=True yapildi (YOUR_KAGGLE_USERNAME/l6-probes-round veri seti).
match_crop.py DEGISTIRILMEDI; yalniz girdi duzeltildi.

Yerelde IRCADb'ye karsi denendi: kapsam %5 -> %97, isaretlenen aday 0.
Bu is ayni seyi LiTS 131 referansiyla yapar. CPU; GPU kotasi harcamaz.
"""
import glob
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
BUDGET = os.environ.get("JOB_BUDGET_MIN", "240")
CODE = TMP / "code"
OUT = WORK / "outputs" / "l6_round"


def log(*a):
    print(f"[{time.time() - T0:6.0f} sn]", *a, flush=True)


def bul(kok, *desenler):
    """Sembolik bagli klasorlere de giren dosya arama. Kaggle veri setlerini
    /kaggle/input/datasets/<sahip>/<ad> altina BAG olarak koyabiliyor; glob('**')
    baglara girmedigi icin oradaki dosyalari bulamaz."""
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


# ---- 0) kod (Kaggle .tar.gz'yi otomatik acmis olabilir) ----
if CODE.exists():
    shutil.rmtree(CODE)
log("girdi kokleri:", sorted(os.listdir(INPUT))[:8])
tars = bul(INPUT, "code.tar.gz")
roots = sorted({str(Path(p).parent.parent.parent) for p in bul(INPUT, "src/provenance/match_crop.py")})
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

# ---- 1) yuvarlanmis yamalar ----
# Kaggle yuklenen zip'i aciyor ve dosyalari VERI SETI KOKUNE koyuyor: klasor adi
# 'probes_round' degil 'l6-probes-round' oluyor (v3 bu yuzden dustu). Bu yuzden
# klasor adina degil, "probes" iceren bir yol altindaki .npz'lere bakilir.
pr = [p for p in bul(INPUT, "*.npz") if "probes" in p.lower()]
if not pr:
    import zipfile
    zips = [z for z in bul(INPUT, "*.zip")
            if "probes" in Path(z).name or "l6" in Path(z).name]
    assert zips, (f"yuvarlanmis yama yok ve zip de yok. {INPUT} altindaki ilk 20 dosya: "
                  f"{bul(INPUT, '*')[:20]}")
    dest = TMP / "probes_zip"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(dest)
    log("zip acildi:", zips[0])
    pr = bul(str(dest), "*.npz")
assert pr, "yuvarlanmis yama bulunamadi (YOUR_KAGGLE_USERNAME/l6-probes-round ekli mi?)"
PROBES = Path(pr[0]).parent
assert len({Path(x).parent for x in pr}) == 1, f"yamalar tek klasorde degil: {sorted({str(Path(x).parent) for x in pr})[:4]}"
assert len(pr) == 416, f"416 yuvarlanmis sorgu beklenirdi: {len(pr)}"
log("yuvarlanmis sorgu:", len(pr), "|", PROBES)

# integer_valued gercekten True mi? (yanlis veri setiyle kosmayi engelle)
chk = subprocess.run(
    [sys.executable, "-c",
     "import glob,sys,numpy as np;"
     "fs=sorted(glob.glob(sys.argv[1]+'/*.npz'));"
     "bad=[f for f in fs if not bool(np.load(f)['integer_valued'])];"
     "print('INTEGER_FALSE', len(bad))", str(PROBES)],
    capture_output=True, text=True, check=True).stdout.strip()
log(chk)
assert int(chk.split()[1]) == 0, "bazi sorgular integer_valued=False -- yanlis veri seti"

# ---- 2) LiTS hacimlerini bagla ----
raw = Path("data/raw/LiTS"); raw.mkdir(parents=True, exist_ok=True)
vols = bul(INPUT, "volume-*.nii")
assert len(vols) == 131 or DRY, f"131 hacim beklenirdi: {len(vols)}"
for f in vols:
    dst = raw / Path(f).name
    if not dst.exists():
        dst.symlink_to(f)
log("LiTS hacmi:", len(list(raw.glob("volume-*.nii"))))

# ---- 3) arama (exact kanali artik acik) ----
csv = OUT / "round_vs_lits.csv"
sh(sys.executable, "-m", "src.provenance.match_crop", "search",
   "--probes", PROBES, "--ref-glob", "data/raw/LiTS/volume-*.nii",
   "--out", csv, "--ncc-budget-min", BUDGET, "--expected", len(pr))

# ---- 4) karar ----
sh(sys.executable, "-m", "src.provenance.l6_verdict", csv, "--out", OUT / "round_vs_lits_karar.csv",
   "--top", "5")
log("BITTI")
