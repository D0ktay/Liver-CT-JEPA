"""Gun-0 baseline: TotalSegmentator 'liver_lesions'.

Egitim GEREKTIRMEZ. GPU yoksa CPU'da calisir (vaka basina ~5-15 dk).
Yayinlanmis performansi: CT duyarlilik %75.8, Dice 0.658
(J Imaging Inform Med 2025, doi:10.1007/s10278-025-01716-y)

!! ONCE SUNU DOGRULA !!
TotalSegmentator'in liver_lesions egitim verisi (Zenodo 10.5281/zenodo.20272572,
842 vaka, CC BY 4.0) LiTS iceriyor mu? Iceriyorsa LiTS uzerindeki skoru DIS
dogrulama degildir. Zenodo'daki zip'in dosya adlarina bak (indirmeden, preview ile).
Sonucu docs/leakage-map.md'ye yaz.

HIZLANDIRMA (Kaggle T4 x2, bkz. docs/kaggle.md):
  --shard k/N       vakalarin k. parcasi (N GPU -> N surec; src.baseline.run_parallel baslatir)
  --engine inproc   TS'in KENDI CLI main()'i ayni surecte: vaka basina python/torch acilisi yok.
                    Argumanlar CLI ile birebir ayni (ts_argv) -> ayni maske; Kaggle'da
                    src.baseline.compare_masks ile voksel duzeyinde dogrulanir.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.data.case_id import assert_unique_ids, glob_paths


def ts_argv(ct: Path, out_file: Path, task: str = "liver_lesions",
            fast: bool = False, device: str = "cpu") -> list[str]:
    """IKI motorun da TS'e verdigi TEK arguman listesi."""
    argv = ["-i", str(ct), "-o", str(out_file), "-ta", task, "--device", device, "--ml"]
    return argv + (["--fast"] if fast else [])


def _ts_main():
    from totalsegmentator.bin.TotalSegmentator import main
    return main


def run_case(ct: Path, out_file: Path, task: str = "liver_lesions",
             fast: bool = False, device: str = "cpu", engine: str = "cli") -> Path:
    """--ml ile TotalSegmentator -o'yu DOSYA yolu olarak bekler (klasor degil).
    Cikti: <out>/<case_id>.nii.gz, liver_lesions'ta lezyon etiketi = 1."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    argv = ts_argv(ct, out_file, task, fast, device)
    print(f"[TS:{engine}] TotalSegmentator", " ".join(argv), flush=True)
    if engine == "cli":
        subprocess.run(["TotalSegmentator", *argv], check=True)
    elif engine == "inproc":
        saved = sys.argv
        sys.argv = ["TotalSegmentator", *argv]
        try:
            _ts_main()()
        except SystemExit as e:   # TS argparse hatasi: tum donguyu degil yalniz bu vakayi dusur
            if e.code not in (None, 0):
                raise RuntimeError(f"TotalSegmentator cikti kodu {e.code}: {ct}") from None
        finally:
            sys.argv = saved
    else:
        raise ValueError(f"bilinmeyen motor: {engine}")
    if not out_file.exists():
        raise FileNotFoundError(f"TotalSegmentator cikti yazmadi: {out_file}")
    return out_file


def parse_shard(s: str) -> tuple[int, int]:
    """'k/N' -> (k, N), 0 <= k < N."""
    parts = str(s).split("/")
    if len(parts) != 2:
        raise ValueError(f"parca 'k/N' bicimde olmali: {s!r}")
    try:
        k, n = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"parca 'k/N' bicimde olmali: {s!r}") from None
    if n < 1 or not 0 <= k < n:
        raise ValueError(f"parca 0 <= k < N olmali: {s!r}")
    return k, n


def select_cases(cases: list, limit: int = 0, shard: tuple[int, int] = (0, 1)) -> list:
    """ONCE limit (ilk N vaka, sirali), SONRA parca: parcalarin birlesimi = ayni ilk N vaka.
    Sira-atlamali bolme (k, k+N, ...) -> parcalar ayrik ve yuk dengeli."""
    cases = sorted(cases)
    if limit:
        cases = cases[:limit]
    k, n = shard
    return cases[k::n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-glob", required=True, help="orn: 'data/raw/3Dircadb1/**/ct.nii.gz'")
    ap.add_argument("--out", type=Path, default=Path("outputs/baseline/totalsegmentator"))
    ap.add_argument("--task", default="liver_lesions", choices=["liver_lesions", "total", "liver_segments"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="ilk N vaka (deneme icin)")
    ap.add_argument("--overwrite", action="store_true", help="var olan ciktiyi yeniden hesapla")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), help="k/N: vakalarin k. parcasi")
    ap.add_argument("--engine", default="cli", choices=["cli", "inproc"],
                    help="inproc: TS ayni surecte (vaka basina acilis maliyeti yok)")
    a = ap.parse_args()

    paths = glob_paths(a.in_glob)
    if not paths:
        print(f"Hicbir dosya eslesmedi: {a.in_glob}", file=sys.stderr)
        return 1
    # IRCADb'de her dosya 'ct.nii.gz': kimlik klasorden gelir, cakisma FIRLATIR
    cases = select_cases(list(assert_unique_ids(paths).items()), a.limit, a.shard)

    print(f"[TS] {len(cases)} vaka, gorev={a.task}, cihaz={a.device}, "
          f"parca={a.shard[0]}/{a.shard[1]}, motor={a.engine}", flush=True)
    failed = []
    for i, (cid, ct) in enumerate(cases, 1):
        out_file = a.out / f"{cid}.nii.gz"
        if out_file.exists() and not a.overwrite:
            print(f"[{i}/{len(cases)}] {cid}: mevcut, atlandi")
            continue
        print(f"\n--- [{i}/{len(cases)}] {cid} <- {ct}", flush=True)
        t0 = time.time()
        try:
            run_case(ct, out_file, a.task, a.fast, a.device, a.engine)
            print(f"[{i}/{len(cases)}] {cid}: {time.time() - t0:.0f} sn", flush=True)
        except Exception as e:   # inproc'ta TS istisnasi herhangi bir tur olabilir
            print(f"[HATA] {ct}: {e}", file=sys.stderr)
            failed.append(str(ct))
    if failed:
        print(f"\n[!] {len(failed)} vaka basarisiz:")
        for f in failed:
            print("   ", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
