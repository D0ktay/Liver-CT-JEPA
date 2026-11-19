"""Sizintisiz split uretimi.

Uc kural, hicbiri pazarliga acik degil:
  K1  Split KANONIK HASTA duzeyinde. Ayni fiziksel taramanin tum kopyalari
      ayni bolumde.
  K2  Dis test seti KAYNAK VERI SETI duzeyinde ayrik. Egitimdeki hicbir kaynak
      dis testte gorunmez.
  K3  LiTS'te egitiyorsan 3D-IRCADb dis test DEGILDIR (IRCADb, LiTS 28-47).
      Ya IRCADb vakalarini egitimden cikar, ya IRCADb'yi test olarak kullanma.

Script bunlari sadece uygulamakla kalmaz, ASSERT eder. Ihlal varsa hata firlatir.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class LeakageError(RuntimeError):
    pass


def assert_no_leakage(df: pd.DataFrame,
                      split_col: str = "split",
                      canon_col: str = "canonical_id",
                      source_col: str = "source_family") -> None:
    # K1: kanonik ID birden fazla split'te olamaz
    bad = df.groupby(canon_col)[split_col].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise LeakageError(
            f"K1 IHLALI: {len(bad)} kanonik ID birden fazla split'te.\n"
            f"Ornekler: {list(bad.index[:10])}")

    # K2: external_test kaynaklari egitimde olmamali
    train_sources = set(df.loc[df[split_col].isin(["train", "val"]), source_col])
    ext_sources = set(df.loc[df[split_col] == "external_test", source_col])
    overlap = train_sources & ext_sources
    if overlap:
        raise LeakageError(
            f"K2 IHLALI: dis test kaynaklari egitimde de var: {sorted(overlap)}")
    print("[splits] K1 + K2 dogrulandi: sizinti yok.")


def apply_ircadb_rule(df: pd.DataFrame, cfg: dict,
                      lits_source: str = "LITS") -> pd.DataFrame:
    """K3: IRCADb'ye karsilik gelen LiTS vakalarini egitimden cikar.

    KANONIK ID duzeyinde: LiTS 28-47'nin baska isimle gelen kopyalari (orn.
    dedup'in onayladigi bir BDMAP_ID) da ayni kanonik ID'yi tasir ve cikarilir.
    """
    idx = set(cfg["leakage"]["lits_ircadb_indices"])
    hit = (df["source_family"].str.contains(lits_source, na=False)) & (
        df["source_case_num"].isin(idx))
    tainted = set(df.loc[hit, "canonical_id"])
    # resolve_canonical'in yazdigi bilesen uyeleri: ham lits-N ile onaylanan kopya da yakalanir
    if "canonical_members" in df.columns:
        pat = re.compile(rf"{lits_source}[^;]*::(\d+)")
        for cid, mem in zip(df["canonical_id"], df["canonical_members"].fillna("")):
            if any(int(n) in idx for n in pat.findall(mem)):
                tainted.add(cid)
    mask = df["canonical_id"].isin(tainted)
    df.loc[mask, "split"] = "excluded_ircadb_overlap"
    print(f"[splits] K3: {len(tainted)} kanonik tarama / {int(mask.sum())} hacim "
          f"(LiTS {min(idx)}-{max(idx)} ve kopyalari) egitimden cikarildi.")
    return df


def make(provenance_csv: Path, config: Path, out: Path,
         val_frac: float = 0.15, seed: int = 1337,
         external_sources: tuple[str, ...] = ("HCC_TACE_SEG",)) -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config).read_text())
    df = pd.read_csv(provenance_csv)

    if "canonical_id" not in df.columns:
        # aday ile devam etmek = dedup onayini atlamak. Durdur.
        raise LeakageError(
            "'canonical_id' yok: once dedup ciktisini elle onayla ve "
            "src.provenance.resolve_canonical ile provenance_resolved.csv uret.")
    df["source_case_num"] = pd.to_numeric(
        df["source_case"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")

    df["split"] = "train"
    df.loc[df["source_family"].isin(external_sources), "split"] = "external_test"
    df = apply_ircadb_rule(df, cfg)

    pool = df[df["split"] == "train"]["canonical_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    n_val = max(1, int(len(pool) * val_frac))
    val_ids = set(pool[:n_val])
    df.loc[df["canonical_id"].isin(val_ids) & (df["split"] == "train"), "split"] = "val"

    assert_no_leakage(df[df["split"] != "excluded_ircadb_overlap"])
    if not (df["split"] == "external_test").any():
        print(f"[UYARI] external_test BOS: {external_sources} provenance'ta yok. "
              "K2 bos kumede gecti -- dis test sizintisi DENETLENMEDI.")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[splits] -> {out}")
    print(df["split"].value_counts().to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provenance", type=Path, default=Path("outputs/provenance.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/paths.yaml"))
    ap.add_argument("--out", type=Path, default=Path("outputs/splits.csv"))
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    make(a.provenance, a.config, a.out, seed=a.seed)
