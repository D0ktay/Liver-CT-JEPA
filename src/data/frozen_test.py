"""Dondurulmus IC TEST kumesi -- on egitimin asla gormeyecegi vakalar.

    python -m src.data.frozen_test --ids-glob 'data/raw/LiTS/segmentation-*.nii' \
        --out outputs/internal_test.json

NEDEN VAR (B3, leakage-auditor 21 Eyl 2026)
-------------------------------------------
`src/data/make_splits.py` HICBIR ZAMAN `split == "test"` uretmiyor; ciktilari
train / val / external_test / excluded_ircadb_overlap. Gercek degerlendirme ise
`src/jepa/seg.py:make_folds` ile ozellik dizininden kuruluyor ve `splits.csv`'yi
hic okumuyor. Iki sonucu vardi:

  1. `pretrain∩test` kontrolu YAPISAL OLARAK daima bos kume uzerinde geciyordu
     ("temiz" gorunuyordu ama hicbir sey denetlenmiyordu);
  2. korpus 131 LiTS hacminin tamamiysa, her CV katinin degerlendirme vakalari
     on egitimde GORULMUS oluyordu -- projenin baskalarini sucladigi seyin aynisi.

5 katli CV'de sorunun temeli su: butun katlarin degerlendirme kovalarinin BIRLESIMI
tum vaka kumesidir. Yani "korpus = tum vakalar" ile hicbir kat temiz kalamaz.
Iki cozum vardi:

  (A) kat basina ayri on egitim  -> 5x maliyet, butceye sigmiyor (EGITIM-PLANI §5)
  (B) TEK dondurulmus ic test kumesi, on egitim + CV geri kalanda  <-- SECILEN

Bu modul (B)'yi uygular. Kume belirleyicidir (tohum sabit), hasta duzeyindedir ve
tumorlu/tumorsuz oraninda TABAKALIDIR -- `src/jepa/seg.py:make_folds` ile ayni
fonksiyon kullanilir ki iki yerde iki farkli bolme olmasin.

Cikti `outputs/internal_test.json`; `pretrain_manifest` onu hem `--eval-ids`
(denetim) hem `--exclude-ids` (korpustan dusurme) olarak okur. Ikisi birlikte:
korpus o vakalari ICEREMEZ ve denetim bunu DOGRULAR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.case_id import case_id_from_path, glob_paths
from src.data.pretrain_manifest import canonical_for, equiv_map_from_config
from src.jepa.seg import make_folds


class FrozenTestError(RuntimeError):
    pass


def choose(case_ids: list[str], has_tumor: dict[str, bool], k: int = 5,
           fold: int = 0, seed: int = 0) -> list[str]:
    """Dondurulmus ic test = `make_folds`'un `fold` numarali kovasi.

    `make_folds` ile AYNI fonksiyon bilerek kullaniliyor: ince ayar katlari ve
    dondurulmus test iki ayri koddan gelirse er ya da gec ayrisirlar.
    """
    if not case_ids:
        raise FrozenTestError("vaka listesi bos")
    missing = [c for c in case_ids if c not in has_tumor]
    if missing:
        raise FrozenTestError(f"{len(missing)} vakanin tumor durumu bilinmiyor: {missing[:5]}")
    if not (0 <= fold < k):
        raise FrozenTestError(f"fold {fold} / k {k} gecersiz")
    return list(make_folds(sorted(case_ids), has_tumor, k=k, seed=seed)[fold]["test"])


def to_payload(case_ids: list[str], source: str, equiv_map: dict[str, str],
               k: int, fold: int, seed: int, all_n: int) -> dict:
    canon = sorted({canonical_for(c, source, equiv_map) for c in case_ids})
    if len(canon) != len(set(case_ids)):
        raise FrozenTestError(
            f"{len(case_ids)} vaka -> {len(canon)} kanonik kimlik: kimlikler cakisiyor, "
            "dondurulmus test kumesi guvenilmez (bkz. B5/B14).")
    return {
        "case_ids": sorted(case_ids),
        "canonical_ids": canon,
        "source": source, "k": k, "fold": fold, "seed": seed,
        "n_total": all_n, "n_frozen": len(canon),
        "note": "On egitim korpusu bu kimlikleri ICEREMEZ (kural 10 + B3). "
                "pretrain_manifest --eval-ids ve --exclude-ids ile okunur.",
    }


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids-glob", required=True,
                    help="GT maskeleri, orn. 'data/raw/LiTS/segmentation-*.nii'")
    ap.add_argument("--source", default="LITS")
    ap.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    ap.add_argument("--out", type=Path, default=Path("outputs/internal_test.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tumor-label", type=int, default=2)
    a = ap.parse_args(argv)

    if a.out.exists():
        # Dondurulmus kume DONDURULMUSTUR. Yeniden uretmek, sonuc gorduikten sonra
        # test kumesini degistirmenin kapisini acar.
        raise FrozenTestError(
            f"{a.out} zaten var -- dondurulmus test kumesi YENIDEN URETILMEZ. "
            "Gercekten degistirmen gerekiyorsa once elle sil ve nedenini gunluge yaz.")

    import nibabel as nib
    import numpy as np
    paths = glob_paths(a.ids_glob)
    if not paths:
        raise FrozenTestError(f"--ids-glob hicbir dosya bulmadi: {a.ids_glob}")
    has_tumor, ids = {}, []
    for p in paths:
        cid = case_id_from_path(p)
        ids.append(cid)
        has_tumor[cid] = bool((np.asanyarray(nib.load(str(p)).dataobj) == a.tumor_label).any())
    n_t = sum(has_tumor.values())
    print(f"[frozen] {len(ids)} vaka ({n_t} tumorlu, {len(ids) - n_t} tumorsuz)")

    chosen = choose(ids, has_tumor, k=a.k, fold=a.fold, seed=a.seed)
    equiv = equiv_map_from_config(yaml.safe_load(a.paths.read_text()))
    payload = to_payload(chosen, a.source, equiv, a.k, a.fold, a.seed, len(ids))
    n_t_frozen = sum(has_tumor[c] for c in chosen)
    print(f"[frozen] dondurulmus ic test: {len(chosen)} vaka "
          f"({n_t_frozen} tumorlu, {len(chosen) - n_t_frozen} tumorsuz)")
    print(f"[frozen] on egitim ve ince ayar geri kalan {len(ids) - len(chosen)} vakada kalir")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"[frozen] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
