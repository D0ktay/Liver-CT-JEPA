"""Provenance (koken) tablosu: her hacim -> kaynak veri seti -> kanonik ID.

Neden: AbdomenAtlas 3.0, 17 kamu veri setinden derlenmis. Ayni fiziksel tarama
icinde birden fazla kez, farkli BDMAP_ID ile bulunabiliyor. ID eslestirme
dosyasindaki 'original id' alani kaynak veri setinin adini tasiyor:

    LiTS_liver_38,BDMAP_00000091
    Decathlon_liver_43,BDMAP_00001114
    MSD-Liver_liver_148,BDMAP_00000064

Decathlon == MSD-Liver == MSD Task03_Liver, ve MSD Task03_Liver == LiTS.
Yani yukaridaki uc satir potansiyel olarak AYNI havuzdan geliyor.

Bu modul birinci savunma hatti. Ikincisi dedup_hash.py (icerik tabanli).
Ikisi birlikte kullanilmali -- tek basina hicbiri yeterli degil.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

# 'SourceName_caseid' formatindaki original id'yi ayristir
_ID_RE = re.compile(r"^(?P<source>[A-Za-z0-9\-\.]+?)_(?P<case>.+)$")


def parse_original_id(original_id: str) -> tuple[str, str]:
    m = _ID_RE.match(str(original_id).strip())
    if not m:
        return "UNKNOWN", str(original_id)
    return m.group("source"), m.group("case")


def canonical_source(source: str, aliases: dict[str, str]) -> str:
    return aliases.get(source, source.upper().replace("-", "_"))


def build(mapping_csv: Path, config: Path, out_csv: Path) -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config).read_text())
    aliases = cfg["leakage"]["source_aliases"]
    equiv_groups = cfg["leakage"].get("equivalent_sources", [])

    # esdeger kaynaklari tek bir etikete indir (LITS ve MSD_TASK03 -> LITS_FAMILY)
    equiv_map: dict[str, str] = {}
    for group in equiv_groups:
        head = "|".join(sorted(group))
        for s in group:
            equiv_map[s] = head

    df = pd.read_csv(mapping_csv)
    cols = {c.lower().strip(): c for c in df.columns}
    oid = cols.get("original id") or cols.get("original_id")
    aid = cols.get("abdomenatlas_id") or cols.get("bdmap_id")
    if oid is None or aid is None:
        raise SystemExit(f"Beklenen sutunlar bulunamadi. Mevcut: {list(df.columns)}")

    parsed = df[oid].map(parse_original_id)
    df["source_raw"] = [p[0] for p in parsed]
    df["source_case"] = [p[1] for p in parsed]
    df["source"] = df["source_raw"].map(lambda s: canonical_source(s, aliases))
    df["source_family"] = df["source"].map(lambda s: equiv_map.get(s, s))
    df["volume_id"] = df[aid]

    # Kanonik ID: ayni aileden gelen ayni case numarasi ayni fiziksel tarama
    # OLABILIR -- kesin degil, o yuzden 'candidate'. dedup_hash.py teyit eder.
    df["canonical_candidate"] = df["source_family"] + "::" + df["source_case"].str.extract(r"(\d+)", expand=False).fillna(df["source_case"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[provenance] {len(df)} hacim -> {out_csv}")
    print("\n[provenance] kaynak dagilimi:")
    print(df["source_family"].value_counts().to_string())

    dup = df[df.duplicated("canonical_candidate", keep=False)].sort_values("canonical_candidate")
    if len(dup):
        print(f"\n[UYARI] {len(dup)} hacim, {dup['canonical_candidate'].nunique()} kanonik adayi PAYLASIYOR:")
        print(dup[["volume_id", "source_raw", "source_case", "canonical_candidate"]].head(40).to_string(index=False))
        print("\n-> Bunlari dedup_hash.py ile TEYIT ET. Onaylanmadan split kurma.")
    else:
        print("\n[provenance] onek duzeyinde kopya bulunamadi (icerik hash'i yine de sart).")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping-csv", type=Path, required=True,
                    help="original id,AbdomenAtlas_id sutunlu CSV")
    ap.add_argument("--config", type=Path, default=Path("configs/paths.yaml"))
    ap.add_argument("--out", type=Path, default=Path("outputs/provenance.csv"))
    a = ap.parse_args()
    build(a.mapping_csv, a.config, a.out)
