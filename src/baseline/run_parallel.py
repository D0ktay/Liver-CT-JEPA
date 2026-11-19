"""TotalSegmentator'i N GPU'da paralel kostur: GPU basina bir surec, her biri bir parca.

    python -m src.baseline.run_parallel --gpus 2 -- --in-glob 'data/raw/LiTS/volume-*.nii' \\
        --out outputs/baseline/totalsegmentator --engine inproc --limit 0

'--' sonrasi run_totalsegmentator'a aynen gecer; --shard ve --device BURADA atanir
(CUDA_VISIBLE_DEVICES=k -> surec icinde 'gpu' = k. fiziksel GPU).
Loglar: <log-dir>/ts_gpu{k}.log. Herhangi bir surec basarisizsa cikis kodu != 0.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def build_jobs(ts_args: list[str], n_gpus: int, log_dir: Path, device: str = "gpu") -> list[dict]:
    if n_gpus < 1:
        raise ValueError(f"n_gpus >= 1 olmali: {n_gpus}")
    for flag in ("--shard", "--device"):
        if flag in ts_args:
            raise ValueError(f"{flag} run_parallel tarafindan atanir, disaridan verilmez")
    return [{"cmd": [sys.executable, "-m", "src.baseline.run_totalsegmentator", *ts_args,
                     "--device", device, "--shard", f"{k}/{n_gpus}"],
             "env": {"CUDA_VISIBLE_DEVICES": str(k)},
             "log": Path(log_dir) / f"ts_gpu{k}.log"} for k in range(n_gpus)]


def _last_progress(log: Path) -> str:
    try:
        lines = [l for l in log.read_text(errors="replace").splitlines() if l.startswith("[")]
    except FileNotFoundError:
        return "-"
    return lines[-1][:100] if lines else "-"


def run_jobs(jobs: list[dict], poll_s: float = 30.0) -> int:
    procs = []
    for j in jobs:
        Path(j["log"]).parent.mkdir(parents=True, exist_ok=True)
        fh = open(j["log"], "w")
        procs.append((subprocess.Popen(j["cmd"], env={**os.environ, **j["env"]},
                                       stdout=fh, stderr=subprocess.STDOUT), fh, j))
    t0 = time.time()
    while any(p.poll() is None for p, _, _ in procs):
        time.sleep(poll_s)
        if poll_s >= 1:
            print(f"[{time.time() - t0:5.0f} sn] " + " | ".join(
                f"gpu{j['env'].get('CUDA_VISIBLE_DEVICES', '?')}: {_last_progress(j['log'])}"
                for _, _, j in procs), flush=True)
    worst = 0
    for p, fh, j in procs:
        fh.close()
        if p.returncode != 0:
            print(f"[HATA] {j['log']} cikis kodu {p.returncode}; son satirlar:\n"
                  + "\n".join(Path(j["log"]).read_text(errors="replace").splitlines()[-15:]))
            worst = worst or p.returncode
    return worst


def main() -> int:
    argv = sys.argv[1:]
    ts_args = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--log-dir", type=Path, default=Path("outputs/logs"))
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args(own)
    jobs = build_jobs(ts_args, a.gpus, a.log_dir, a.device)
    for j in jobs:
        print(f"[paralel] CUDA_VISIBLE_DEVICES={j['env']['CUDA_VISIBLE_DEVICES']} -> {j['log']}")
    return run_jobs(jobs, a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
