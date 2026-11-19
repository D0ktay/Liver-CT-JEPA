#!/usr/bin/env python3
"""LiverJEPA -- Lambda H100 GECE KOSUSU. Tek komut, kaldigi yerden devam eder.

    tmux new -s gece
    python3 scripts/lambda_night.py --budget-hours 30 2>&1 | tee -a outputs/night/night.log

    python3 scripts/lambda_night.py --dry-run           # ne yapacagini yazar
    python3 scripts/lambda_night.py --only cache        # tek adim
    python3 scripts/lambda_night.py --from train        # bir adimdan itibaren
    python3 scripts/lambda_night.py --skip-corpus       # AMOS/TS indirme; korpus = LiTS + Task08

NEDEN `lambda_bootstrap.sh` DEGIL: o betik provenance -> make_splits zincirine bagli ve
gercek veriyle HIC kosmadi (`outputs/lambda/steps.tsv` iki kez `manifest`te kirmizi).
Bu betik, Kaggle'da gercek veriyle UC KEZ uctan uca kosmus akisi (kaggle_jobs/mve_jepa/run.py)
izler: minik ama GERCEK splits.csv -> manifest -> denetim -> onbellek -> egitim -> olcum.

SIRA (en onemli sonuc en once gelsin diye):
  env -> deps -> data -> frozen -> manifest(+denetim) -> cache -> bench ->
  ft_none -> pre_jepa -> ft_jepa -> pre_mae -> ft_mae -> report
Her ince ayar biter bitmez OLCULUR; gece yarida kesilse de eldeki kollar raporludur.

🔴 Denetim kirmiziysa (korpus ∩ dondurulmus test != bos) EGITIM BASLAMAZ.
🔴 Dondurulmus ic test `configs/frozen_internal_test.json`'dan gelir (MVE ve TS tabani ile
   AYNI 27 vaka). Lambda'daki veriyle yeniden uretilip KARSILASTIRILIR; farkliysa DURUR.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = Path("outputs/night")
STATE = OUT / "state"
RAW = Path("data/raw")
DL = RAW / "_dl"
C_LITS, C_T08, C_UNLAB = Path("data/cache/lits"), Path("data/cache/task08"), Path("data/cache/unlab")
C_LIVER = Path("data/cache/liveronly")          # TS-v2 + AMOS Tr/Va: YALNIZ karaciger etiketi
FROZEN = Path("outputs/internal_test.json")
GT_FIXED = Path("data/gt_fixed")
SPLITS = Path("outputs/splits.csv")
MAN_FULL = Path("outputs/night/manifest_full.csv")   # manifest adimi yazar (denetlenen)
MAN_PRE = Path("outputs/pretrain_manifest.csv")      # cache adimi yazar = FULL eksi onbelleksizler
STEPS = ["env", "deps", "data", "frozen", "manifest", "cache", "qc", "bench",
         "ft_none", "pre_jepa", "ft_jepa", "pre_mae", "ft_mae", "report"]
T0 = time.time()
ARGS = None
FAILED: list = []
LANE = threading.local()          # .gpu -> bu serit hangi GPU'da kosuyor
MAN_COLS = ["volume_id", "canonical_id", "source", "source_family", "path", "n_bytes",
            "checksum", "checksum_kind", "canonical_status", "license", "url"]

URLS = {
    "task08": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
    "task07": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
    "task10": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
    "amos": "https://zenodo.org/api/records/7155725/files/amos22.zip/content",
    "ts": "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip",
}
MD5 = {"amos": "67717b2a483ac0744c89c3016b7aaef7", "ts": "fe250e5718e0a3b5df4c4ea9d58a62fe"}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.0f}dk]", *a, flush=True)


def py() -> str:
    v = ROOT / ".venv/bin/python"
    return str(v) if v.exists() else sys.executable


def sh(*args, check=True, env=None, quiet=False) -> int:
    cmd = [str(a) for a in args]
    if not quiet:
        log("$", " ".join(cmd))
    if ARGS.dry_run:
        return 0
    gpu = getattr(LANE, "gpu", None)
    if gpu is not None and env is None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    try:
        r = subprocess.run(cmd, env=env)
    except FileNotFoundError:
        if check:
            raise SystemExit(f"[DUR] komut bulunamadi: {cmd[0]}")
        log(f"  (komut yok, atlandi: {cmd[0]})")
        return 127
    if check and r.returncode != 0:
        raise SystemExit(f"[DUR] basarisiz ({r.returncode}): {' '.join(cmd)}")
    return r.returncode


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


# ============================================================================ adimlar
def step_env():
    sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv", check=False)
    sh("df", "-h", ".", check=False)
    sh("free", "-g", check=False)
    sh("nproc", check=False)
    sh("git", "log", "--oneline", "-1", check=False)
    if not ARGS.dry_run:
        free = shutil.disk_usage(".").free / 2 ** 30
        need = 120 if ARGS.skip_corpus else 260
        if free < need:
            raise SystemExit(f"[DUR] bos disk {free:.0f} GB < gereken ~{need} GB")


def step_deps():
    if not (ROOT / ".venv").exists():
        sh(sys.executable, "-m", "venv", "--system-site-packages", ".venv")
    sh(py(), "-m", "pip", "install", "-q", "--upgrade", "pip")
    sh(py(), "-m", "pip", "install", "-q", "-r", "requirements-train.txt")
    rc = sh(py(), "-c", "import torch,sys; print('torch', torch.__version__, 'cuda', "
            "torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)",
            check=False)
    if rc != 0:
        log("CUDA'li torch yok -> kuruluyor")
        sh(py(), "-m", "pip", "install", "-q", "torch", "--index-url",
           "https://download.pytorch.org/whl/cu124")
        sh(py(), "-c", "import torch; assert torch.cuda.is_available(), 'CUDA yok'; "
           "print(torch.__version__, torch.cuda.get_device_name(0))")
    if not have("aria2c"):
        sh("sudo", "apt-get", "install", "-y", "-q", "aria2", "unzip", check=False)
    sh(py(), "-m", "pytest", "-q", "-x", "-m", "not slow", "-p", "no:cacheprovider")


def _fetch(name: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not (dst.parent / (dst.name + ".aria2")).exists():
        log(f"  {name}: zaten indirilmis ({dst.stat().st_size / 1e9:.1f} GB)")
        return
    if have("aria2c"):
        cmd = ["aria2c", "-x4", "-s4", "-c", "--always-resume=false", "--console-log-level=warn", "--summary-interval=60",
               "-d", str(dst.parent), "-o", dst.name, URLS[name]]
        if name in MD5:
            cmd.insert(1, f"--checksum=md5={MD5[name]}")
    else:
        cmd = ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(dst), URLS[name]]
    sh(*cmd)


def _lits():
    dst = RAW / "LiTS"
    if len(list(dst.glob("volume-*.nii"))) == 131 and len(list(dst.glob("segmentation-*.nii"))) == 131:
        log("  LiTS: zaten hazir (131 + 131)")
        return
    home = Path.home() / ".kaggle"
    if not ((home / "kaggle.json").exists() or (home / "access_token").exists()
            or os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")):
        raise SystemExit(
            "[DUR] Kaggle kimligi yok. LiTS yalniz Kaggle'dan guvenle iniyor (MSD Task03 aynasinda "
            "liver_N = volume-N esligi TEYITSIZ -> dondurulmus test kayabilir).\n"
            "  Yerel makineden:  scp ~/.kaggle/access_token ubuntu@<lambda>:~/.kaggle/access_token\n"
            "  (ya da kaggle.json). Sonra bu betigi yeniden calistir.")
    kdir = RAW / "LiTS_kaggle"
    kdir.mkdir(parents=True, exist_ok=True)
    for ds in ("andrewmvd/liver-tumor-segmentation", "andrewmvd/liver-tumor-segmentation-part-2"):
        sub = kdir / ds.split("/")[1]
        if not sub.exists() or not any(sub.rglob("*.nii")):
            kb = next((k for k in (ROOT / ".venv/bin/kaggle", Path.home() / ".local/bin/kaggle")
                       if k.exists()), "kaggle")            # CLI >=2.0 py3.11 ister -> `uv tool install kaggle`
            sh(kb, "datasets", "download", "-d", ds, "-p", sub, "--unzip")
    if ARGS.dry_run:
        return
    dst.mkdir(parents=True, exist_ok=True)
    seen = {}
    for f in sorted(kdir.rglob("*.nii")):
        if f.name.startswith(("volume-", "segmentation-")):
            seen.setdefault(f.name, f)               # iki veri seti AYNI segmentasyonlari tasiyor
    for name, f in seen.items():
        t = dst / name
        if not t.exists():
            t.symlink_to(f.resolve())
    nv, ns = len(list(dst.glob("volume-*.nii"))), len(list(dst.glob("segmentation-*.nii")))
    if nv != 131 or ns != 131:
        raise SystemExit(f"[DUR] LiTS eksik: {nv} hacim / {ns} segmentasyon (131/131 beklenir)")


def _task08():
    d = RAW / "Task08_HepaticVessel"
    if len(list((d / "imagesTr").glob("hepaticvessel_*.nii.gz"))) == 303:
        log("  Task08: zaten hazir")
        return
    _fetch("task08", DL / "Task08_HepaticVessel.tar")
    sh("tar", "-xf", DL / "Task08_HepaticVessel.tar", "-C", RAW)
    sh("find", d, "-name", "._*", "-delete")           # AppleDouble copu nibabel'i patlatir


def _msd(task: str, stem: str, n_tr: int):
    d = RAW / task
    if len(list((d / "imagesTr").glob(f"{stem}_*.nii.gz"))) == n_tr:
        log(f"  {task}: zaten hazir")
        return
    key = "task07" if "07" in task else "task10"
    _fetch(key, DL / f"{task}.tar")
    sh("tar", "-xf", DL / f"{task}.tar", "-C", RAW)
    sh("find", d, "-name", "._*", "-delete")


def _kits():
    d = RAW / "kits23"
    if len(list((d / "images").glob("case_*.nii.gz"))) >= 489:
        log("  KiTS23: zaten hazir")
        return
    sh(py(), "-m", "pip", "install", "-q", "huggingface_hub")
    sh(py(), "-c", "from huggingface_hub import snapshot_download; snapshot_download("
       "'neheller/KiTS-Challenge-Imaging', repo_type='dataset', local_dir='data/raw/kits23', "
       "allow_patterns=['images/*'], max_workers=16)")


def _amos():
    d = RAW / "amos22"
    if len(list(d.glob("images*/amos_0[0-4]*.nii.gz"))) >= 499 and any(d.glob("labelsTr/amos_0*.nii.gz")):
        log("  AMOS: zaten hazir")
        return
    _fetch("amos", DL / "amos22.zip")
    # labels*: karaciger (etiket 6) ince ayarda karaciger kanali icin kullanilir
    sh("unzip", "-q", "-o", DL / "amos22.zip", "amos22/images*/*", "amos22/labels*/*",
       "-x", "__MACOSX/*", "-d", RAW)


def _ts():
    d = RAW / "Totalsegmentator_dataset_v201"
    if not (d / "meta.csv").exists():
        _fetch("ts", DL / "Totalsegmentator_dataset_v201.zip")
        d.mkdir(parents=True, exist_ok=True)
        sh("unzip", "-q", "-o", DL / "Totalsegmentator_dataset_v201.zip", "meta.csv",
           "*/ct.nii.gz", "*/segmentations/liver.nii.gz", "-d", d)
    if ARGS.dry_run:
        return
    # karaciger iceren vakalari sec -> data/raw/ts_liver/<id> baglari
    sel = RAW / "ts_liver"
    sel.mkdir(parents=True, exist_ok=True)
    code = ("import sys, nibabel as nib, numpy as np\n"
            "for p in sys.argv[1:]:\n"
            "    try:\n"
            "        a = np.asanyarray(nib.load(p).dataobj)\n"
            "        print(p, int((a > 0).sum()))\n"
            "    except Exception as e:\n"
            "        print(p, -1)\n")
    livers = sorted(d.glob("s*/segmentations/liver.nii.gz"))
    chunks = [livers[i::16] for i in range(16)]

    def run(ch):
        if not ch:
            return ""
        return subprocess.run([py(), "-c", code, *map(str, ch)], capture_output=True,
                              text=True).stdout
    n_ok = 0
    with ThreadPoolExecutor(16) as ex:
        for out in ex.map(run, chunks):
            for ln in out.splitlines():
                p, n = ln.rsplit(" ", 1)
                # >= 50 cm^3 (1.5 mm izotropik = 3.375 mm^3/voksel): ucundan gorunen karaciger alinmaz
                if int(n) * 3.375 >= 50_000:
                    case = Path(p).parents[1]
                    t = sel / case.name
                    if not t.exists():
                        t.symlink_to(case.resolve())
                    n_ok += 1
    log(f"  TS v2: {n_ok}/{len(livers)} vakada karaciger var -> {sel}")


def step_data():
    RAW.mkdir(parents=True, exist_ok=True)
    # Task08 ZORUNLU: dusmesi etiketli veriyi sessizce ~4 kat kucultur (--lits-only ile bilincli kapatilir)
    jobs = [("LiTS", _lits, True), ("Task08", _task08, not ARGS.lits_only)]
    if not ARGS.skip_corpus:
        jobs += [("AMOS", _amos, False), ("TS-v2", _ts, False), ("KiTS23", _kits, False),
                 ("Task07", lambda: _msd("Task07_Pancreas", "pancreas", 281), False),
                 ("Task10", lambda: _msd("Task10_Colon", "colon", 126), False)]
    errs = {}

    def run(j):
        name, fn, must = j
        try:
            fn()
        except BaseException as e:                     # noqa: BLE001
            errs[name] = (must, str(e))
    with ThreadPoolExecutor(len(jobs)) as ex:
        list(ex.map(run, jobs))
    for name, (must, e) in errs.items():
        log(f"  [{'DUR' if must else 'UYARI'}] {name}: {e}")
    if any(m for m, _ in errs.values()):
        raise SystemExit("[DUR] zorunlu veri (LiTS / Task08) indirilemedi -- yukaridaki hataya bak")
    if errs:
        log("  ⚠️ istege bagli kaynak(lar) dustu -> korpus/etiketli veri KUCULDU, kosu DEVAM ediyor")
    (OUT / "data_errors.json").write_text(json.dumps({k: v[1] for k, v in errs.items()}, indent=1))


def step_frozen():
    ref = json.loads(Path("configs/frozen_internal_test.json").read_text())
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    if ARGS.dry_run:
        return
    chk = OUT / "frozen_regen.json"
    if chk.exists():
        chk.unlink()
    sh(py(), "-m", "src.data.frozen_test", "--ids-glob", "data/raw/LiTS/segmentation-*.nii",
       "--out", chk)
    got = json.loads(chk.read_text())
    if sorted(got["case_ids"]) != sorted(ref["case_ids"]):
        raise SystemExit(
            "[DUR] Lambda'da yeniden uretilen dondurulmus test, repodaki ile AYNI DEGIL.\n"
            f"  repoda: {sorted(ref['case_ids'])[:6]}...\n  burada: {sorted(got['case_ids'])[:6]}...\n"
            "  TS tabani ve MVE bu 27 vakada olculdu; farkli kume = karsilastirma gecersiz.")
    FROZEN.write_text(json.dumps(ref, indent=1))
    log(f"  dondurulmus ic test: {len(ref['case_ids'])} vaka (repo ile BIREBIR ayni)")


def _lits_rows(ids):
    return [[f"lits-{i}", f"LITS|MSD_TASK03::{i}", "LITS", "LITS|MSD_TASK03",
             str(RAW / "LiTS" / f"volume-{i}.nii"), 0, "pending", "sha256", "name_only",
             "CC BY-NC-ND 4.0", ""] for i in ids]


def _write_man(path: Path, rows) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(MAN_COLS)
        w.writerows(rows)


def step_manifest():
    if ARGS.dry_run:
        log("  (dry-run) splits.csv + manifest + denetim")
        return
    test = set(json.loads(FROZEN.read_text())["case_ids"])
    ids = sorted((int(p.name[len("volume-"):-4]) for p in (RAW / "LiTS").glob("volume-*.nii")))
    with SPLITS.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["volume_id", "canonical_id", "source_family", "split"])
        for i in ids:
            w.writerow([f"lits-{i}", f"LITS|MSD_TASK03::{i}", "LITS|MSD_TASK03",
                        "test" if f"lits-{i}" in test else "train"])
    _write_man(OUT / "manifest_lits_all.csv", _lits_rows(ids))
    # 🔴 Denetim: kirmiziysa burada DURUR (sh check=True).
    sh(py(), "-m", "src.data.pretrain_manifest", "--corpus", "configs/pretrain_corpus.yaml",
       "--paths", "configs/paths.yaml", "--splits", SPLITS, "--exclude-ids", FROZEN,
       "--eval-ids", FROZEN, "--require-sources-covered",
       "--out", MAN_FULL, "--audit-out", "outputs/split_audit.json")
    # KAPSAM: --require-audited KULLANILMIYOR cunku bu kosuda dis test yok. Bu bir kapsam
    # daraltmasidir ve ACIK olmali: bos kumede gecen kontroller TAM OLARAK su ikisi olmali.
    # Baska bir kontrol bos kumede gectiyse (orn. test kumesi okunamadi) denetim KORDUR -> DUR.
    au = json.loads(Path("outputs/split_audit.json").read_text())
    bos = sorted(k for k, v in au["checks"].items() if v["vacuous"])
    beklenen = sorted(["egitim∩dogrulama", "pretrain∩dis_test"])
    if bos != beklenen:
        raise SystemExit(f"[DUR] bos kumede gecen kontroller {bos}; beklenen yalniz {beklenen}. "
                         "Denetim kor -- splits.csv / internal_test.json okunamamis olabilir.")
    if not au["clean"]:
        raise SystemExit("[DUR] korpus denetimi KIRMIZI")
    import pandas as pd
    # Bu adim bir KAPIDIR ve --from/--only ile bile yeniden kosar. Bu yuzden cache adiminin
    # (bozuk hacimleri dusurerek) yazdigi MAN_PRE'yi EZMEZ; ama onu da yeniden denetler.
    frozen_can = set(json.loads(FROZEN.read_text())["canonical_ids"])
    for mp in (MAN_FULL, MAN_PRE):
        if not mp.exists():
            continue
        m = pd.read_csv(mp)
        bad = set(m["canonical_id"].astype(str)) & frozen_can
        if bad:
            raise SystemExit(f"[DUR] KURAL 3: dondurulmus test korpusta ({mp}): {sorted(bad)[:5]}")
        extra = set(m["canonical_id"].astype(str)) - set(pd.read_csv(MAN_FULL)["canonical_id"].astype(str))
        if extra:
            raise SystemExit(f"[DUR] {mp} denetlenen manifest'te OLMAYAN {len(extra)} satir tasiyor")
        log(f"  {mp.name}:", m["source"].value_counts().to_dict())


def step_cache():
    import pandas as pd
    w = str(max(2, min(ARGS.workers, (os.cpu_count() or 4) - 2)))
    # (1) LiTS -- 131 vakanin TAMAMI (egitim + dogrulama + dondurulmus test), TAM HACIM,
    #     karaciger (1) + tumor (2). ROI YOK: test GT'den turetilmis kutuya hapsedilmez (L15).
    sh(py(), "-m", "src.jepa.cache_build", "--manifest", OUT / "manifest_lits_all.csv",
       "--out", C_LITS, "--labels-glob", "data/raw/LiTS/segmentation-*.nii",
       "--liver-label", "1", "--tumor-label", "2", "--trust-ct-spacing", "--workers", w,
       "--spacing-table", "configs/lits_2847_true_spacing.csv")
    # Degerlendirme icin BASLIGI DUZELTILMIS GT (lits-28..47 ve 48..52; bkz. src/data/gt_fixed.py)
    sh(py(), "-m", "src.data.gt_fixed", "--ids", FROZEN, "--raw", RAW / "LiTS", "--out", GT_FIXED)
    if ARGS.dry_run:
        return
    m = pd.read_csv(MAN_FULL)
    # (2) Task08 ETIKETLI (imagesTr, 303): tumor=2, karaciger etiketi YOK (1 = DAMAR)
    t08 = m[(m["source"] == "MSD_TASK08") & m["path"].str.contains("/imagesTr/")]
    if len(t08):
        t08.to_csv(OUT / "manifest_task08_tr.csv", index=False)
        sh(py(), "-m", "src.jepa.cache_build", "--manifest", OUT / "manifest_task08_tr.csv",
           "--out", C_T08, "--tumor-label", "2", "--liver-label", "-1", "--workers", w,
           "--labels-glob", "data/raw/Task08_HepaticVessel/labelsTr/hepaticvessel_*.nii.gz")
    done = set(t08["volume_id"].astype(str))
    # (3) YALNIZ KARACIGER etiketli (tumor kanali KAPALI): TS-v2 (liver.nii.gz = 1) + AMOS Tr/Va (6)
    ts = m[m["source"] == "TOTALSEGMENTATOR_V2"]
    if len(ts):
        ts.to_csv(OUT / "manifest_liver_ts.csv", index=False)
        sh(py(), "-m", "src.jepa.cache_build", "--manifest", OUT / "manifest_liver_ts.csv",
           "--out", C_LIVER, "--workers", w, "--hu-offset-auto", "--allow-errors", "0.03",
           "--labels-glob", "data/raw/ts_liver/*/segmentations/liver.nii.gz",
           "--liver-label", "1", "--tumor-label", "-1",
           # v1 (22 Eyl): 60 mm ROI kirpigi -> ~920 hacimde pelvis/gluteal YOK, uzak negatif azaldi.
           # Ikinci kosu: --ts-roi-margin-mm 0 = kirpma (data/cache/liveronly_tsfull ile olculdu: 917, QC 842).
           *(["--roi-glob", "data/raw/ts_liver/*/segmentations/liver.nii.gz",
              "--roi-margin-mm", str(ARGS.ts_roi_margin_mm)] if ARGS.ts_roi_margin_mm > 0 else []))
        done |= set(ts["volume_id"].astype(str))
    lab_ids = {p.name[:-7] for p in (RAW / "amos22").glob("labels*/amos_0*.nii.gz")}
    am = m[(m["source"] == "AMOS") & m["volume_id"].astype(str).isin(lab_ids)]
    if len(am):
        am.to_csv(OUT / "manifest_liver_amos.csv", index=False)
        sh(py(), "-m", "src.jepa.cache_build", "--manifest", OUT / "manifest_liver_amos.csv",
           "--out", C_LIVER, "--workers", w, "--hu-offset-auto", "--allow-errors", "0.03",
           "--labels-glob", "data/raw/amos22/labels*/amos_0*.nii.gz",
           "--liver-label", "6", "--tumor-label", "-1")
        done |= set(am["volume_id"].astype(str))
    # (4) ETIKETSIZ korpus: geri kalan her sey
    rest = m[(m["source"] != "LITS") & ~m["volume_id"].astype(str).isin(done)]
    for src in sorted(rest["source"].unique()):
        sub = rest[rest["source"] == src]
        mp = OUT / f"manifest_unlab_{src}.csv"
        sub.to_csv(mp, index=False)
        sh(py(), "-m", "src.jepa.cache_build", "--manifest", mp, "--out", C_UNLAB,
           "--workers", w, "--allow-errors", "0.03", "--hu-offset-auto")
    n_lits = len(list(C_LITS.glob("*.npz")))
    if n_lits != 131:
        raise SystemExit(f"[DUR] LiTS onbellegi {n_lits}/131")
    _finalize_manifest()
    sh("du", "-sh", "data/cache", check=False)


def _finalize_manifest():
    """MAN_PRE = denetlenen manifest EKSI onbellegi olmayan (bozuk / QC'de reddedilen) hacimler."""
    import pandas as pd
    m = pd.read_csv(MAN_FULL)
    dirs = [C_LITS, C_T08, C_LIVER, C_UNLAB]
    ok = m["volume_id"].astype(str).map(lambda v: any((d / f"{v}.npz").exists() for d in dirs))
    if (~ok).any():
        log(f"  ⚠️ {int((~ok).sum())} hacim onbelleksiz/reddedilmis -> manifest'ten dusuruldu:",
            m.loc[~ok, "source"].value_counts().to_dict())
        m.loc[~ok].to_csv(OUT / "manifest_dropped.csv", index=False)
    m = m[ok]
    m.to_csv(MAN_PRE, index=False)
    log("  ON EGITIM KORPUSU (kesin):", len(m), m["source"].value_counts().to_dict())


def step_qc():
    """Egitime girecek HER hacim olculur; bozuklar _rejected/'a tasinir; ornekler PNG'ye cizilir."""
    sh(py(), "-m", "src.data.qc_cache", "--cache", C_LITS, C_T08, C_LIVER, C_UNLAB,
       "--out", OUT / "qc", "--fatal-dirs", "lits", "--workers", "24")
    if ARGS.dry_run:
        return
    test = set(json.loads(FROZEN.read_text())["case_ids"])
    have = {p.stem for p in C_LITS.glob("*.npz")}
    if not test <= have:
        raise SystemExit(f"[DUR] QC dondurulmus test vakasi reddetti: {sorted(test - have)}")
    _finalize_manifest()


def _caches():
    return [str(d) for d in (C_LITS, C_T08, C_LIVER, C_UNLAB) if d.exists() and any(d.glob("*.npz"))]


def _extra():
    ds = [str(d) for d in (C_T08, C_LIVER) if d.exists() and any(d.glob("*.npz"))]
    return ["--extra-cache", *ds] if ds and not ARGS.lits_only else []


def step_bench():
    ram = str(ARGS.ram_gb)
    LANE.gpu = 0 if ARGS.gpus >= 2 else None       # olcum TEK GPU'da
    sh(py(), "-m", "src.jepa.finetune_e2e", "--init", "random", "--cache", C_LITS, *_extra(),
       "--test-ids", FROZEN, "--out", OUT / "bench_ft", "--profile", ARGS.profile,
       "--bench", "80", "--ram-gb", "12")
    sh(py(), "-m", "src.jepa.pretrain", "--objective", "jepa", "--manifest", MAN_PRE,
       "--cache", *_caches(), "--eval-ids", FROZEN, "--out", OUT / "bench_pre",
       "--profile", ARGS.profile, "--bench", "60", "--ram-gb", "12")
    if ARGS.dry_run:
        return
    import yaml
    cfg = yaml.safe_load(Path("configs/train.yaml").read_text())
    ft = json.loads((OUT / "bench_ft/bench.json").read_text())["it_s"]
    pre = json.loads((OUT / "bench_pre/bench.json").read_text())["it_s"]
    B = ARGS.budget_hours * 3600
    if ARGS.gpus >= 2:
        # IKI SERIT paralel: GPU0 = ft_none -> pre_mae -> ft_mae (en uzun: 2 ft + 1 pre),
        # GPU1 = pre_jepa -> ft_jepa. Butce = DUVAR saati; uzun serit: 2x%33 + %20 + %14 pay.
        f_ft, f_pre = 0.33, 0.20
    else:
        # tek GPU, sirali: 3 ince ayar x %20 + 2 on egitim x %13 + %14 cikarim/olcum/yukleme
        f_ft, f_pre = 0.20, 0.13
    s_ft = int(min(cfg["finetune"]["steps"], ft * B * f_ft))
    s_pre = int(min(cfg["pretrain"]["steps"], pre * B * f_pre))
    plan = {"it_s_ft": ft, "it_s_pre": pre, "budget_hours": ARGS.budget_hours,
            "steps_ft": s_ft, "steps_pre": s_pre,
            "hours_ft_each": s_ft / ft / 3600, "hours_pre_each": s_pre / pre / 3600}
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    log("  PLAN:", json.dumps(plan))
    if s_ft < 30000:
        log(f"  ⚠️ ince ayar {s_ft} adim < 30000: butce dar. MVE'nin 4000 adimi yetmemisti; "
            "--budget-hours'u artir ya da --arms ile kol sayisini azalt.")
    _ = ram
    LANE.gpu = None


def _plan():
    p = OUT / "plan.json"
    if p.exists():
        return json.loads(p.read_text())
    if ARGS.dry_run:
        return {"steps_ft": "<plan.json>", "steps_pre": "<plan.json>"}
    raise SystemExit("[DUR] outputs/night/plan.json yok -- once `--only bench`. Plansiz kosu "
                     "butceden bagimsiz 80k/60k adim egitir.")


def _evaluate(d: Path):
    ev = d / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    sh(py(), "-m", "src.metrics.make_pairs", "--gt-glob", str(GT_FIXED / "segmentation-*.nii"),
       "--pred-glob", str(d / "pred" / "*.nii.gz"), "--out", ev / "pairs.csv")
    sh(py(), "-m", "src.metrics.evaluate", "--pairs", ev / "pairs.csv", "--out", ev,
       "--gt-label", "2", "--pred-label", "1")
    sh(py(), "-m", "src.metrics.bootstrap", "--eval", ev, "--n-boot", "2000", check=False)
    sh(py(), "-m", "src.metrics.margin_sweep", "--pairs", ev / "pairs.csv", "--out", ev,
       "--gt-label", "2", "--pred-label", "1", check=False)


def _finetune(arm: str):
    if arm not in ARGS.arms:
        log(f"  {arm}: --arms disinda, atlandi")
        return
    init = "random" if arm == "none" else str(OUT / f"pre_{arm}" / "encoder.pt")
    if arm != "none" and not Path(init).exists() and not ARGS.dry_run:
        raise SystemExit(f"[DUR] {init} yok -- once pre_{arm}")
    d = OUT / f"ft_{arm}"
    sh(py(), "-m", "src.jepa.finetune_e2e", "--init", init, "--cache", C_LITS, *_extra(),
       "--test-ids", FROZEN, "--out", d, "--profile", ARGS.profile, "--seed", ARGS.seed,
       "--steps", _plan()["steps_ft"], "--ram-gb", ARGS.ram_gb, "--gt-dir", GT_FIXED,
       "--resume", "--init-key", ARGS.init_key if arm != "none" else "encoder")
    _evaluate(d)
    step_report()


def _pretrain(obj: str):
    if obj not in ARGS.arms:
        log(f"  {obj}: --arms disinda, atlandi")
        return
    d = OUT / f"pre_{obj}"
    if (d / "encoder.pt").exists():
        log(f"  {obj}: encoder.pt zaten var")
        return
    sh(py(), "-m", "src.jepa.pretrain", "--objective", obj, "--manifest", MAN_PRE,
       "--cache", *_caches(), "--eval-ids", FROZEN, "--out", d, "--profile", ARGS.profile,
       "--seed", ARGS.seed, "--steps", _plan()["steps_pre"], "--ram-gb", ARGS.ram_gb, "--resume")


def step_report():
    if ARGS.dry_run:
        return
    rows = []
    for arm in ("none", "jepa", "mae"):
        sp = OUT / f"ft_{arm}" / "eval" / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        to = s.get("targeting", {}).get("overall", {})
        dn = json.loads((OUT / f"ft_{arm}" / "done.json").read_text())
        rows.append({"arm": arm, "n": s.get("n_cases"), "failed": s.get("n_failed"),
                     "dice_median": s.get("voxel", {}).get("dice_median"),
                     "recall": s.get("lesion", {}).get("recall"),
                     "precision": s.get("lesion", {}).get("precision"),
                     "overlap_hit": to.get("overlap_hit_rate"),
                     "margin_viol": to.get("margin_violation_rate_all"),
                     "centroid_mm": to.get("centroid_error_mm_median"),
                     "steps": dn.get("steps"), "best_val": dn.get("best_val_tumor_dice")})
    lines = ["# LiverJEPA gece kosusu", "",
             f"git: {subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}"
             f" | {time.strftime('%F %T')}", "",
             "| kol | n | Dice medyan | lezyon recall | precision | ortusme | marj ihlali | merkez mm | adim | en iyi val |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    for r in rows:
        lines.append("| " + " | ".join(f(r[k]) for k in ("arm", "n", "dice_median", "recall",
                     "precision", "overlap_hit", "margin_viol", "centroid_mm", "steps",
                     "best_val")) + " |")
    # TS tabani AYNI zeminde: ayni 27 vaka, ayni data/gt_fixed GT (liver-ct-13, outputs/night/ts_same_gt).
    # Dosya yoksa eski 25 vakalik satir ACIKCA etiketli kalir (48/49 eski kosuda spacing yuzunden dusmustu).
    tsp = OUT / "ts_same_gt" / "strict" / "summary.json"
    if tsp.exists():
        t = json.loads(tsp.read_text()); tt = t.get("targeting", {}).get("overall", {})
        ts_row = ("| TS `liver_lesions` (ayni vakalar, ayni gt_fixed GT; son isleme YOK) | " + " | ".join(
            f(v) for v in (t.get("n_cases"), t.get("voxel", {}).get("dice_median"),
                           t.get("lesion", {}).get("recall"), t.get("lesion", {}).get("precision"),
                           tt.get("overlap_hit_rate"), tt.get("margin_violation_rate_all"),
                           tt.get("centroid_error_mm_median"))) + " | - | - |")
    else:
        ts_row = ("| TS `liver_lesions` (25 vaka: lits-48/49 HARIC, outputs/frozen/common_ts) | 25 | 0.752 | 0.489 "
                  "| 0.450 | 0.774 | 0.505 | 1.18 | - | - |")
    lines += [ts_row,
              "", "⚠️ 27 vaka -> guven araliklari GENIS (eval/ci.csv). Kollar arasi kucuk fark IDDIA EDILMEZ (kural 8).",
              "⚠️ TS esit veriyle egitilmedi (842 vaka, Basel) ve LiTS'i gorup gormedigi kapanmadi (L6).",
              "⚠️ Test tahmini SON kontrol noktasindan; `best_val` yalniz tani.",
              "⚠️ TS'e son isleme uygulanmadi; kollarda LCC + 3 mm karaciger maskesi var. TS satiri 25 vakaliksa "
              "kollarla ayni kumeye indirgenmeden fark IDDIA EDILMEZ (scripts/frozen_baseline.py --intersect).",
              "⚠️ K1/K2/K3 (make_splits) bu kosuda devrede DEGIL; yerine egitim∩test ve pretrain∩ic_test calisti. "
              "Denetim ISIM duzeyinde; icerik ve hasta duzeyi dogrulanmadi."]
    ap = Path("outputs/split_audit.json")
    if ap.exists():
        a = json.loads(ap.read_text())
        lines += ["", "## Korpus denetimi", f"clean={a['clean']} seviye={a['audited_level']} "
                  f"hacim={a['corpus']['n_volumes']} kaynak={a['corpus']['by_source']}"]
        for k, v in a["checks"].items():
            lines.append(f"- {k}: n={v['n']} bos_kume={v['vacuous']}")
        for k, v in (a["corpus"].get("blind_source_evidence") or {}).items():
            lines.append(f"- 🟡 {k}: isimle denetlenemez; beyan: {v}")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "RESULT.json").write_text(json.dumps(rows, indent=1))
    log("  ->", OUT / "REPORT.md")
    print("\n".join(lines[4:4 + 3 + len(rows)]), flush=True)


FUNCS = {"env": step_env, "deps": step_deps, "data": step_data, "frozen": step_frozen,
         "manifest": step_manifest, "cache": step_cache, "qc": step_qc, "bench": step_bench,
         "ft_none": lambda: _finetune("none"), "pre_jepa": lambda: _pretrain("jepa"),
         "ft_jepa": lambda: _finetune("jepa"), "pre_mae": lambda: _pretrain("mae"),
         "ft_mae": lambda: _finetune("mae"), "report": step_report}
ALWAYS = {"env", "report"}            # ucuz, idempotent
GATE = {"frozen", "manifest"}         # 🔴 --from/--only ile bile ATLANMAZ (saniyeler surer)


def _run_step(s: str) -> None:
    stamp = STATE / f"{s}.done"
    if stamp.exists() and s not in ALWAYS | GATE and not ARGS.dry_run:
        log(f"== {s}: zaten bitmis (tekrar icin: rm {stamp})")
        return
    log(f"== {s}" + (f" [GPU {LANE.gpu}]" if getattr(LANE, "gpu", None) is not None else ""))
    t = time.time()
    arm = s.split("_", 1)[1] if s.startswith(("ft_", "pre_")) else None
    if arm is not None and arm not in ARGS.arms:
        log(f"  {arm}: --arms disinda -> atlandi (damga YAZILMADI)")
        return
    try:
        FUNCS[s]()
    except SystemExit as e:
        if arm is None:
            raise
        # Bir kolun dusmesi SONRAKI kollari oldurmesin (ft_X, encoder.pt yoksa zaten temiz cikar).
        log(f"!! {s} BASARISIZ: {e} -> sonraki adima geciliyor")
        with (OUT / "steps.tsv").open("a") as fh:
            fh.write(f"{s}\tKIRMIZI\t{time.time() - t:.0f}\n")
        FAILED.append(s)
        return
    if not ARGS.dry_run:
        stamp.write_text(f"{time.time() - t:.0f}\n")
        with (OUT / "steps.tsv").open("a") as fh:
            fh.write(f"{s}\tOK\t{time.time() - t:.0f}\n")
        if getattr(LANE, "gpu", None) is None:
            _reexec_in_venv()


def _run_lanes(lanes: dict) -> None:
    """Iki GPU'da iki serit PARALEL. Serit ici sira korunur (pre_X her zaman ft_X'ten once)."""
    def work(gpu, steps):
        LANE.gpu = gpu
        for st in steps:
            _run_step(st)
    ts = [threading.Thread(target=work, args=(g, st)) for g, st in lanes.items() if st]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def _reexec_in_venv() -> None:
    """pandas/yaml venv'de; betik sistem python'uyla baslatilmis olabilir -> venv'e gec."""
    v = ROOT / ".venv"
    if (v / "bin/python").exists() and Path(sys.prefix).resolve() != v.resolve():
        log("  venv python'una geciliyor")
        os.execv(str(v / "bin/python"), [str(v / "bin/python"), *sys.argv])


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-hours", type=float, default=30.0,
                    help="TOPLAM GPU egitim butcesi; adim sayilari olculen it/s'ten turetilir")
    ap.add_argument("--profile", default="orta")
    ap.add_argument("--arms", nargs="+", default=["none", "jepa", "mae"])
    ap.add_argument("--gpus", type=int, default=0,
                    help="0 -> otomatik (nvidia-smi). >=2: GPU0 = ft_none->pre_mae->ft_mae, "
                         "GPU1 = pre_jepa->ft_jepa PARALEL")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--ram-gb", default="110")
    ap.add_argument("--ts-roi-margin-mm", type=float, default=60.0,
                    help="TS-v2 liver_only onbellegi karaciger cevresine kirpilir; 0 = kirpma (tum govde)")
    ap.add_argument("--init-key", default="encoder", choices=("encoder", "target_encoder"),
                    help="ft_jepa/ft_mae icin encoder.pt anahtari (EMA = target_encoder; yalniz jepa'da var)")
    ap.add_argument("--workers", type=int, default=12,
                    help="onbellek surecleri; tam hacim basina birkac GB RAM")
    ap.add_argument("--skip-corpus", action="store_true", help="AMOS + TS-v2 indirme")
    ap.add_argument("--lits-only", action="store_true", help="ince ayarda Task08 etiketlerini KULLANMA")
    ap.add_argument("--until", help="bu adimi da kosup DUR (orn. --until bench: egitime baslama)")
    ap.add_argument("--only")
    ap.add_argument("--from", dest="from_")
    ap.add_argument("--dry-run", action="store_true")
    ARGS = ap.parse_args()
    if ARGS.gpus == 0:
        try:
            ARGS.gpus = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                           text=True).stdout.strip().splitlines()) or 1
        except FileNotFoundError:
            ARGS.gpus = 1
    for d in (OUT, STATE):
        d.mkdir(parents=True, exist_ok=True)
    if not ARGS.dry_run:
        _reexec_in_venv()
    for x in (ARGS.only, ARGS.from_, ARGS.until):
        if x and x not in STEPS:
            raise SystemExit(f"bilinmeyen adim: {x} (secenekler: {STEPS})")
    started = ARGS.from_ is None
    todo = []
    for s in STEPS:
        if s == ARGS.from_:
            started = True
        selected = (ARGS.only == s) if ARGS.only else started
        data_ready = (RAW / "LiTS").exists()
        if selected or (s in GATE and data_ready and not ARGS.dry_run):
            todo.append(s)
    if ARGS.until:
        todo = [x for x in todo if STEPS.index(x) <= STEPS.index(ARGS.until)]
    lanes = {0: ["ft_none", "pre_mae", "ft_mae"], 1: ["pre_jepa", "ft_jepa"]}
    parallel = ARGS.gpus >= 2 and not ARGS.dry_run
    for s in todo:
        if parallel and s in lanes[0] + lanes[1]:
            continue                                  # asagida seritlerde kosacak
        if parallel and s == "report":
            _run_lanes({g: [x for x in ls if x in todo] for g, ls in lanes.items()})
        _run_step(s)
    if FAILED:
        log(f"⚠️ BASARISIZ ADIMLAR: {FAILED} -- loglara bak; ayni komut onlari yeniden dener")
    log("BITTI ->", OUT / "REPORT.md")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
