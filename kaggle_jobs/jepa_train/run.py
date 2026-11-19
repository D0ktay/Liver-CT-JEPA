"""Kaggle isi: JEPA'li ve JEPA'siz tumor basligi, 5 kat, iki GPU paralel -> 131 kat-disi tahmin -> degerlendirme.

Girdi: liver-jepa-feats ciktisi (131 npz) + LiTS segmentasyonlari (affine) + kod.
GPU 0: variant=jepa, GPU 1: variant=nojepa (tek GPU varsa sirayla). Her kol: kat 0..4 -> pred/*.nii.gz.
Degerlendirme: LiTS 28-47'de GT + tahmin basligi gercek spacing'e (src.baseline.respacing), sonra
src.metrics.evaluate + bootstrap + margin_sweep. Otonom: bash scripts/kaggle_push.sh jepa_train.
KURAL: bu surec numpy/torch YUKLEMEZ.
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
ITERS = os.environ.get("JOB_ITERS", "4000")
K = int(os.environ.get("JOB_K", "5"))
TABLE = os.environ.get("JOB_SPACING_TABLE", "configs/lits_2847_true_spacing.csv")
VARIANTS = ["jepa", "nojepa"]
CODE = TMP / "code"
OUT = WORK / "outputs" / "jepa_seg"


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


# ---- 0) kod ----
if CODE.exists():
    shutil.rmtree(CODE)
tars = bul(INPUT, "code.tar.gz")
roots = sorted({str(Path(p).parent.parent.parent) for p in
                bul(INPUT, "src/jepa/train.py")})
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
NGPU = int(chk.split("NGPU")[1].split()[0])
assert NGPU >= 1 or DRY, "GPU yok"

# ---- 1) girdiler ----
npz = sorted(bul(INPUT, "jepa_feats/lits-*.npz"))
assert len(npz) == 131 or DRY, f"131 npz beklenirdi: {len(npz)}"
FEATS = Path(npz[0]).parent
raw = Path("data/raw/LiTS"); raw.mkdir(parents=True, exist_ok=True)
for f in bul(INPUT, "segmentation-*.nii"):
    dst = raw / Path(f).name
    if not dst.exists():
        dst.symlink_to(f)
log("npz:", len(npz), "| ozellik klasoru:", FEATS)

# ---- 2) egitim: GPU basina bir kol, katlar sirayla ----
def worker_cmd(variant):
    lines = [f"{sys.executable} -m src.jepa.train --feats {FEATS} --out {OUT} --variant {variant} "
             f"--fold {k} --k {K} --iters {ITERS}" + (" --device cpu" if DRY else "") for k in range(K)]
    return " && ".join(lines)

# katlari once tek surecte yaz (iki kol ayni folds.json'u kullansin, yaris kosulu olmasin)
sh(sys.executable, "-c", f"from pathlib import Path; from src.jepa.train import folds_for; "
   f"folds_for(Path('{FEATS}'), Path('{OUT}'), {K})")
procs = []
for j, v in enumerate(VARIANTS):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(j % max(NGPU, 1))}
    logf = open(OUT / f"train_{v}.log", "w")
    if NGPU >= 2 or j == 0:
        procs.append((v, subprocess.Popen(["bash", "-c", worker_cmd(v)], env=env, stdout=logf,
                                          stderr=subprocess.STDOUT), logf))
    else:   # tek GPU: ikinci kol birinci bitince
        procs[0][1].wait()
        procs.append((v, subprocess.Popen(["bash", "-c", worker_cmd(v)], env=env, stdout=logf,
                                          stderr=subprocess.STDOUT), logf))
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(60)
    for v, _, _ in procs:
        lines = (OUT / f"train_{v}.log").read_text(errors="replace").splitlines()
        last = [l for l in lines if l.startswith(("[train]", "  it"))]
        log(v, "|", last[-1] if last else "-")
bad = [v for v, p, f in procs if (f.close() or p.returncode)]
for v, p, _ in procs:
    log(v, "cikis kodu", p.returncode)
    if p.returncode:
        print("\n".join((OUT / f"train_{v}.log").read_text(errors="replace").splitlines()[-30:]))

# ---- 3) degerlendirme (her kol) ----
summ = {}
for v in VARIANTS:
    pred = OUT / v / "pred"
    n = len(list(pred.glob("lits-*.nii.gz")))
    log(v, "tahmin:", n)
    if n == 0:
        continue
    fix = TMP / "fixed" / v
    # 28-47 duzeltmesi ZORUNLU: basarisizsa sahte spacing'le SESSIZCE olculmez
    sh(sys.executable, "-m", "src.baseline.respacing", "--src-glob", str(pred / "lits-*.nii.gz"),
       "--out", fix / "pred", "--table", TABLE)
    sh(sys.executable, "-m", "src.baseline.respacing", "--src-glob", "data/raw/LiTS/segmentation-*.nii",
       "--out", fix / "seg", "--table", TABLE)
    rows = ["case_id,gt_path,pred_path"]
    for p in sorted(pred.glob("lits-*.nii.gz"), key=lambda q: int(q.name.split("-")[1].split(".")[0])):
        cid = p.name[:-7]; i = int(cid.split("-")[1])
        if 28 <= i <= 47:
            assert (fix / "pred" / p.name).exists(), f"{cid}: duzeltilmis baslik yok"
            rows.append(f"{cid},{fix / 'seg' / f'segmentation-{i}.nii.gz'},{fix / 'pred' / p.name}")
        else:
            rows.append(f"{cid},data/raw/LiTS/segmentation-{i}.nii,{p}")
    ev = OUT / v / "eval"; ev.mkdir(parents=True, exist_ok=True)
    (ev / "pairs.csv").write_text("\n".join(rows) + "\n")
    sh(sys.executable, "-m", "src.metrics.evaluate", "--pairs", ev / "pairs.csv", "--out", ev,
       "--gt-label", "2", "--pred-label", "1", check=False)
    sh(sys.executable, "-m", "src.metrics.bootstrap", "--eval", ev, "--n-boot", "2000", check=False)
    sh(sys.executable, "-m", "src.metrics.margin_sweep", "--pairs", ev / "pairs.csv", "--out", ev,
       "--gt-label", "2", "--pred-label", "1", check=False)
    if (ev / "summary.json").exists():
        s = json.loads((ev / "summary.json").read_text()); to = s["targeting"]["overall"]
        summ[v] = {"n_cases": s["n_cases"], "n_failed": s["n_failed"], "dice_median": s["voxel"]["dice_median"],
                   "lesion_recall": s["lesion"]["recall"], "lesion_precision": s["lesion"]["precision"],
                   "overlap_hit_rate": to.get("overlap_hit_rate"),
                   "margin_violation_rate_all": to.get("margin_violation_rate_all"),
                   "centroid_error_mm_median": to.get("centroid_error_mm_median")}
(OUT / "RESULT.json").write_text(json.dumps({"train_exit": {v: p.returncode for v, p, _ in procs},
                                             "summary": summ, "iters": ITERS, "k": K}, indent=1))
print(json.dumps(summ, indent=1))
log("BITTI")
assert not bad, f"egitim basarisiz: {bad}"
