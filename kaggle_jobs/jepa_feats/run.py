"""Kaggle isi: Rad-JEPA 3D (donmus) ozellikleri, LiTS 131 -> outputs/jepa_feats/*.npz

Adimlar: kod (ozel veri seti) -> Rad-JEPA reposu (sabit commit) + HF agirligi -> [istege bagli]
mamba_ssm kurulabilirse saf PyTorch yedegiyle BIREBIR karsilastirma -> src.jepa.extract (131 vaka).
Otonom: bash scripts/kaggle_push.sh jepa_feats. KURAL: bu surec numpy/torch YUKLEMEZ.
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
DRY = os.environ.get("JOB_DRY") == "1"          # yerel: mamba_ssm denemesi yok, cihaz cpu, 1 vaka
CODE = TMP / "code"
OUT = WORK / "outputs" / "jepa_feats"
RADJEPA_REPO = "https://github.com/huyquoctrinh/radjepa3d"
RADJEPA_COMMIT = "6a73d1b"
ENC_URL = "https://huggingface.co/huyquoctrinh/Rad-Jepa-3D/resolve/main/encoder/encoder.pt"
ENC_BYTES = 155603150


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


def sh(*args, check=True, timeout=None):
    log("$", " ".join(map(str, args)))
    try:
        r = subprocess.run([str(a) for a in args], timeout=timeout)
    except subprocess.TimeoutExpired:
        log("ZAMAN ASIMI")
        if check:
            raise
        return -1
    if check and r.returncode != 0:
        raise RuntimeError(f"basarisiz ({r.returncode}): {' '.join(map(str, args))}")
    return r.returncode


# ---- 0) kod (Kaggle .tar.gz'yi otomatik acabilir) ----
if CODE.exists():
    shutil.rmtree(CODE)
tars = bul(INPUT, "code.tar.gz")
roots = sorted({str(Path(p).parent.parent.parent) for p in
                bul(INPUT, "src/jepa/extract.py")})
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

# ---- 1) bagimlilik: yalniz einops (torch/scipy/nibabel Kaggle'da var) ----
if not DRY:
    sh("pip", "install", "-q", "einops")
chk = subprocess.run([sys.executable, "-c", "import torch, nibabel, scipy, einops; "
                      "print('CUDA', torch.cuda.is_available(), 'NGPU', torch.cuda.device_count())"],
                     capture_output=True, text=True)
log(chk.stdout.strip(), chk.stderr[-500:])
DEVICE = "cuda" if "CUDA True" in chk.stdout else "cpu"
assert DEVICE == "cuda" or DRY, "GPU yok"

# ---- 2) Rad-JEPA kodu + agirlik ----
repo = TMP / "radjepa3d"
if not repo.exists():
    sh("git", "clone", "-q", RADJEPA_REPO, repo)
    sh("git", "-C", repo, "checkout", "-q", RADJEPA_COMMIT)
enc = TMP / "encoder.pt"
if os.environ.get("JOB_ENCODER"):
    enc = Path(os.environ["JOB_ENCODER"])
elif not enc.exists():
    sh("curl", "-fsSL", "--retry", "5", "-o", enc, ENC_URL)
assert enc.stat().st_size == ENC_BYTES, f"encoder.pt boyutu {enc.stat().st_size} != {ENC_BYTES}"

# ---- 3) mamba_ssm (gercek CUDA cekirdegi) kurulabilirse: yedek ile birebir karsilastirma ----
cmp_json = OUT / "mamba_ref_vs_mamba_ssm.json"
if not DRY:
    rc = sh("pip", "install", "-q", "--no-build-isolation", "causal-conv1d", "mamba-ssm",
            check=False, timeout=900)
    code = f'''
import json, sys, torch
sys.path[:0] = ["{repo}", "{repo}/utils"]
from src.jepa.radjepa import load_radjepa_encoder
import importlib
x = torch.rand(2, 1, 32, 256, 256, device="cuda")
real = load_radjepa_encoder("{enc}", "{repo}", device="cuda", force_ref=False)
impl = real.mamba_impl
with torch.no_grad():
    a = real(x, indices=None).float()
for k in [k for k in list(sys.modules) if k in ("mamba_ssm", "vjepa_model", "vjepa_hybrid", "load_checkpoint")]:
    del sys.modules[k]
ref = load_radjepa_encoder("{enc}", "{repo}", device="cuda", force_ref=True)
with torch.no_grad():
    b = ref(x, indices=None).float()
d = (a - b).abs()
cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1)).min().item()
res = dict(real_impl=impl, ref_impl=ref.mamba_impl, max_abs=d.max().item(), mean_abs=d.mean().item(),
           out_abs_mean=a.abs().mean().item(), min_cosine=cos)
print(res); json.dump(res, open("{cmp_json}", "w"), indent=1)
'''
    sh(sys.executable, "-c", code, check=False, timeout=900)
else:
    log("DRY: mamba_ssm karsilastirmasi atlandi")

# ---- 4) LiTS + ozellik cikarma (saf PyTorch yedegi: mamba_ssm olsa da AYNI uygulama kullanilsin) ----
raw = Path("data/raw/LiTS"); raw.mkdir(parents=True, exist_ok=True)
vols = bul(INPUT, "volume-*.nii")
segs = bul(INPUT, "segmentation-*.nii")
assert (len(vols), len(segs)) == (131, 131) or DRY, (len(vols), len(segs))
for f in vols + segs:
    dst = raw / Path(f).name
    if not dst.exists():
        dst.symlink_to(f)
args = ["src.jepa.extract", "--ckpt", enc, "--repo", repo, "--out", OUT, "--device", DEVICE,
        "--batch", "4", "--force-ref"]
if DRY:
    args += ["--limit", "1", "--batch", "1"]
sh(sys.executable, "-m", *args)

n = len(list(OUT.glob("lits-*.npz")))
summary = {"n_cases": n, "seconds": round(time.time() - T0),
           "mamba_compare": json.loads(cmp_json.read_text()) if cmp_json.exists() else None}
(OUT / "RESULT.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
assert n == 131 or DRY, f"131 npz beklenirdi, {n}"
log("BITTI")
