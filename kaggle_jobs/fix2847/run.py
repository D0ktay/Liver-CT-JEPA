"""Kaggle isi: LiTS 28-47 sahte spacing duzeltmesi + TS yeniden kosu + tam 131 degerlendirme.

Otonom: `bash scripts/kaggle_push.sh fix2847` ile gonderilir, cikti `kaggle kernels output` ile iner.
KURAL: bu surec numpy/pandas/torch YUKLEMEZ; agir isler ayri surecte (pip numpy'i degistirebilir).

Uc karsilastirma (yalniz 28-47, 20 vaka):
  A  eski TS tahmini + SAHTE baslik      -> kosu 1'i tekrar uretir (saglama)
  B  eski TS tahmini + GERCEK spacing    -> yalniz mm OLCUM duzeltmesinin etkisi
  C  YENI TS tahmini (dogru spacing ile kosuldu) + GERCEK spacing -> modele etkisi
Tam 131: 28-47 icin C, digerleri kosu 1 tahmini -> eval_all + ci + margin_sweep.
48-52: GT / hacim / tahmin baslik bilgisi (kosu 1'de affine uyusmazligi) -> headers_4852.json
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
# Yerel kuru deneme icin ortam degiskenleri (Kaggle'da hicbiri tanimli degil -> varsayilanlar)
INPUT = os.environ.get("JOB_INPUT", "/kaggle/input")
TMP = Path(os.environ.get("JOB_TMP", "/kaggle/tmp"))
WORK = Path(os.environ.get("JOB_WORK", "/kaggle/working"))
DRY = os.environ.get("JOB_DRY") == "1"        # pip yok, TS yerine eski tahmin kopyalanir
CODE = TMP / "code"
FIX = TMP / "fixed"
OUT = WORK / "outputs" / "fix2847"
TABLE = os.environ.get("JOB_SPACING_TABLE", "configs/lits_2847_true_spacing.csv")


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


def py(*args, check=True):
    return sh(sys.executable, "-m", *args, check=check)


# ---- 0) kod: ozel veri seti YOUR_KAGGLE_USERNAME/liver-ct-code icindeki code.tar.gz ----
# Kaggle veri setindeki .tar.gz'yi OTOMATIK acar (19 Eyl 2026, v1 bu yuzden dustu): iki hali de kabul et
if CODE.exists():
    shutil.rmtree(CODE)
tars = bul(INPUT, "code.tar.gz")
roots = sorted({str(Path(p).parent.parent.parent) for p in
                bul(INPUT, "src/baseline/respacing.py")})
if tars:
    assert len(tars) == 1, f"birden fazla code.tar.gz: {tars}"
    CODE.mkdir(parents=True)
    with tarfile.open(tars[0]) as t:
        t.extractall(CODE)
else:
    assert len(roots) == 1, f"kod bulunamadi / birden fazla: {roots} (girdi: {os.listdir(INPUT)})"
    shutil.copytree(roots[0], CODE, symlinks=False)
os.chdir(CODE)
log("kod:", (CODE / "COMMIT").read_text().strip() if (CODE / "COMMIT").exists() else "?")
OUT.mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# ---- 1) bagimliliklar: mevcut numpy/torch sabit ----
cons = subprocess.run([sys.executable, "-c",
                       "import numpy, torch; print(f'numpy=={numpy.__version__}'); print(f'torch=={torch.__version__}')"],
                      capture_output=True, text=True, check=True).stdout
(TMP / "constraints.txt").write_text(cons)
if not DRY:
    sh("pip", "install", "-q", "-r", "requirements.txt", "-c", TMP / "constraints.txt")
chk = subprocess.run([sys.executable, "-c",
                      "import torch; print('CUDA', torch.cuda.is_available(), 'NGPU', torch.cuda.device_count())"],
                     capture_output=True, text=True, check=True).stdout
log(chk.strip())
NGPU = int(chk.split("NGPU")[1].split()[0])
assert NGPU >= 1 or DRY, "GPU yok -- kernel-metadata machine_shape/enable_gpu kontrol et"

# ---- 2) LiTS + kosu 1 tahminleri ----
raw = Path("data/raw/LiTS"); raw.mkdir(parents=True, exist_ok=True)
vols = bul(INPUT, "volume-*.nii")
segs = bul(INPUT, "segmentation-*.nii")
assert len(vols) == 131 and len(segs) == 131, (len(vols), len(segs))
for f in vols + segs:
    dst = raw / Path(f).name
    if not dst.exists():
        dst.symlink_to(f)
old = sorted(bul(INPUT, "baseline/totalsegmentator/lits-*.nii.gz"))
assert len(old) == 131, f"kosu 1 tahminleri 131 degil: {len(old)}"
oldd = Path("outputs/old_preds"); oldd.mkdir(parents=True, exist_ok=True)
for f in old:
    dst = oldd / Path(f).name
    if not dst.exists():
        dst.symlink_to(f)
log("LiTS 131+131, kosu 1 tahmini 131")

# ---- 3) baslik duzeltmesi (yalniz 28-47) ----
py("src.baseline.respacing", "--src-glob", "data/raw/LiTS/volume-*.nii", "--out", FIX / "vol", "--table", TABLE)
py("src.baseline.respacing", "--src-glob", "data/raw/LiTS/segmentation-*.nii", "--out", FIX / "seg", "--table", TABLE)
py("src.baseline.respacing", "--src-glob", "outputs/old_preds/lits-*.nii.gz", "--out", FIX / "oldpred", "--table", TABLE)

# ---- 4) A / B (TS'siz) -> 5) TS yeniden kosu -> C ----
IDS = list(range(28, 48))


def evaluate(name, gt_of, pred_of):
    """Ciftler ACIKCA yazilir (glob yok): 20 vakanin hepsi var olmali."""
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    rows = ["case_id,gt_path,pred_path"]
    for i in IDS:
        g, p = Path(gt_of(i)), Path(pred_of(i))
        assert g.exists() and p.exists(), f"{name}: eksik {g} / {p}"
        rows.append(f"lits-{i},{g},{p}")
    (d / "pairs.csv").write_text("\n".join(rows) + "\n")
    rc = py("src.metrics.evaluate", "--pairs", d / "pairs.csv", "--out", d,
            "--gt-label", "2", "--pred-label", "1", check=False)
    log(name, "evaluate cikis kodu", rc)
    return d


# once ucuz olanlar: burada hata varsa GPU harcanmadan durur
A = evaluate("A_eski_tahmin_sahte_baslik", lambda i: f"data/raw/LiTS/segmentation-{i}.nii",
             lambda i: f"outputs/old_preds/lits-{i}.nii.gz")
B = evaluate("B_eski_tahmin_gercek_spacing", lambda i: FIX / "seg" / f"segmentation-{i}.nii.gz",
             lambda i: FIX / "oldpred" / f"lits-{i}.nii.gz")
assert (A / "summary.json").exists() and (B / "summary.json").exists(), "A/B degerlendirmesi basarisiz"

# ---- 4) TS yeniden kosu, dogru spacing'li 20 hacim (eski, dogrulanmis cli motoru; GPU basina parca) ----
ts_new = OUT / "ts_new"
if DRY:   # kuru deneme: TS yerine duzeltilmis eski tahmin
    ts_new.mkdir(parents=True, exist_ok=True)
    for f in (FIX / "oldpred").glob("lits-*.nii.gz"):
        shutil.copy(f, ts_new / f.name)
else:
    py("src.baseline.run_parallel", "--gpus", NGPU, "--log-dir", OUT / "logs", "--poll", 60, "--",
       "--in-glob", str(FIX / "vol" / "volume-*.nii.gz"), "--out", ts_new)
n_new = len(list(ts_new.glob("lits-*.nii.gz")))
log("yeni TS tahmini:", n_new)
assert n_new == 20, f"20 yeni tahmin beklenirdi, {n_new}"


C = evaluate("C_yeni_tahmin_gercek_spacing", lambda i: FIX / "seg" / f"segmentation-{i}.nii.gz",
             lambda i: ts_new / f"lits-{i}.nii.gz")

# ---- 6) tam 131: 28-47 icin C, digerleri kosu 1 ----
fixed_ids = {f"lits-{i}" for i in range(28, 48)}
rows = ["case_id,gt_path,pred_path"]
for i in range(131):
    cid = f"lits-{i}"
    if cid in fixed_ids:
        rows.append(f"{cid},{FIX / 'seg' / f'segmentation-{i}.nii.gz'},{ts_new / f'{cid}.nii.gz'}")
    else:
        rows.append(f"{cid},data/raw/LiTS/segmentation-{i}.nii,outputs/old_preds/{cid}.nii.gz")
allp = OUT / "all131"; allp.mkdir(parents=True, exist_ok=True)
(allp / "pairs.csv").write_text("\n".join(rows) + "\n")
py("src.metrics.evaluate", "--pairs", allp / "pairs.csv", "--out", allp,
   "--gt-label", "2", "--pred-label", "1", check=False)
py("src.metrics.bootstrap", "--eval", allp, "--n-boot", "2000", "--seed", "0", check=False)
py("src.metrics.margin_sweep", "--pairs", allp / "pairs.csv", "--out", allp,
   "--gt-label", "2", "--pred-label", "1", check=False)

# ---- 7) 48-52 baslik bilgisi ----
code_4852 = r'''
import json, nibabel as nib, numpy as np
out = {}
for i in range(48, 53):
    row = {}
    for k, p in (("volume", f"data/raw/LiTS/volume-{i}.nii"), ("segmentation", f"data/raw/LiTS/segmentation-{i}.nii"),
                 ("pred", f"outputs/old_preds/lits-{i}.nii.gz")):
        im = nib.load(p)
        row[k] = {"shape": list(im.shape), "zooms": [float(z) for z in im.header.get_zooms()[:3]],
                  "axcodes": list(nib.aff2axcodes(im.affine)), "affine": np.round(im.affine, 4).tolist(),
                  "sform": int(im.header["sform_code"]), "qform": int(im.header["qform_code"])}
    out[f"lits-{i}"] = row
import sys
json.dump(out, open(sys.argv[1], "w"), indent=1)
for c, r in out.items():
    print(c, {k: (v["axcodes"], v["zooms"], v["sform"], v["qform"]) for k, v in r.items()})
'''
sh(sys.executable, "-c", code_4852, OUT / "headers_4852.json", check=False)

# ---- 8) ozet ----
summ = {}
for name, d in (("A", A), ("B", B), ("C", C), ("all131", allp)):
    f = d / "summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        to = s["targeting"]["overall"]
        summ[name] = {"n_cases": s["n_cases"], "n_failed": s["n_failed"],
                      "dice_median": s["voxel"]["dice_median"],
                      "lesion_recall": s["lesion"]["recall"], "lesion_precision": s["lesion"]["precision"],
                      "overlap_hit_rate": to.get("overlap_hit_rate"),
                      "margin_violation_rate_all": to.get("margin_violation_rate_all"),
                      "centroid_error_mm_median": to.get("centroid_error_mm_median")}
(OUT / "RESULT.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
log("BITTI")
