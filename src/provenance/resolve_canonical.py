"""Isim tabanli aday + ELLE onaylanmis icerik kopyalari -> canonical_id.

Provenance (onek) ve dedup (icerik) iki ayri savunma hatti; bu modul ikisini
birlestirir. Ayni fiziksel tarama oldugu dusunulen her sey tek bir kanonik
ID'ye duser (union-find):

  - ayni canonical_candidate'i paylasan hacimler            (provenance)
  - candidate_duplicates.csv'de confirmed=1 isaretli ciftler (dedup, elle)

Onaysiz dedup ciktisi KULLANILMAZ: 'confirmed' sutunu yoksa hata firlatir.
Cikti: provenance tablosu + 'canonical_id' sutunu. make_splits.py bunu okur.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.data.case_id import case_id_from_path

_TRUE = {"1", "true", "yes", "y", "evet", "e", "x"}
_LITS_ID_RE = re.compile(r"^lits-(\d+)$")
LITS_FAMILY = "LITS|MSD_TASK03"   # build_provenance'in equivalent_sources ciktisi


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministik: sozluk sirasinda kucuk olan kok olur
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def _ids(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """dedup'in yazdigi a_id/b_id varsa onlar; yoksa yoldan turet."""
    if {"a_id", "b_id"} <= set(df.columns):
        return df["a_id"].astype(str), df["b_id"].astype(str)
    return df["a"].map(case_id_from_path), df["b"].map(case_id_from_path)


def confirmed_pairs(dups: pd.DataFrame, current: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    """current: dedup'in EN SON candidate_duplicates.csv'si. Incelenmis dosyada
    olmayan bir aday varsa (yeni veri eklendi) eski onay kullanilmaz."""
    if "confirmed" not in dups.columns:
        raise SystemExit(
            "candidate_duplicates.csv'de 'confirmed' sutunu yok. Her satiri ELLE "
            "incele ve confirmed=1 (ayni tarama) / 0 (degil) yaz. Onaysiz dedup kullanilmaz.")
    if dups["confirmed"].isna().any():
        raise SystemExit(f"{int(dups['confirmed'].isna().sum())} satirda 'confirmed' bos -- hepsini incele.")
    a, b = _ids(dups)
    if current is not None and len(current):
        ca, cb = _ids(current)
        seen = {frozenset(x) for x in zip(a, b)}
        new = [(x, y) for x, y in zip(ca, cb) if frozenset((x, y)) not in seen]
        if new:
            raise SystemExit(f"{len(new)} incelenmemis yeni kopya adayi var (orn. {new[:3]}). "
                             "reviewed.csv'yi guncelle.")
    ok = dups["confirmed"].astype(str).str.strip().str.lower().isin(_TRUE).to_numpy()
    return list(zip(a[ok], b[ok]))


def _raw_node(vid: str) -> str:
    """Provenance'ta olmayan hacim. Ham LiTS dosyasi LiTS ailesinin adayina baglanir
    ki K3 onu gorebilsin; digerleri kendi dugumu."""
    m = _LITS_ID_RE.match(vid)
    return f"cand::{LITS_FAMILY}::{int(m.group(1))}" if m else f"vol::{vid}"


def resolve(prov: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Her volume_id'ye bir canonical_id atar."""
    prov = prov.copy()
    uf = _UnionFind()
    vol_to_node = {}
    for vid, cand in zip(prov["volume_id"].astype(str), prov["canonical_candidate"].astype(str)):
        node = f"cand::{cand}"
        vol_to_node[vid] = node
        uf.find(node)

    unknown = []
    for a, b in pairs:
        # provenance'ta olmayan hacim: ham lits-N LiTS adayina, digerleri kendi dugumune
        na = vol_to_node.get(a) or _raw_node(a)
        nb = vol_to_node.get(b) or _raw_node(b)
        unknown += [x for x in (a, b) if x not in vol_to_node]
        uf.union(na, nb)
    if unknown:
        print(f"[resolve] {len(set(unknown))} onayli kopya provenance tablosunda yok "
              f"(orn. {sorted(set(unknown))[:5]}). Bunlar split disi; yalniz birlesme icin kullanildi.")

    # kok "cand::X" ise X; "cand::" < "vol::" oldugu icin aday varken kok hep aday olur
    prov["canonical_id"] = [uf.find(vol_to_node[str(v)]).split("::", 1)[1]
                            for v in prov["volume_id"]]
    # bilesendeki TUM adaylar: K3 kok hangisi secilirse secilsin LiTS 28-47'yi gorur
    members: dict[str, set[str]] = {}
    for node in list(uf.parent):
        members.setdefault(uf.find(node), set()).add(node.split("::", 1)[1])
    prov["canonical_members"] = [";".join(sorted(members[uf.find(vol_to_node[str(v)])]))
                                 for v in prov["volume_id"]]
    n_merged = prov["canonical_candidate"].nunique() - prov["canonical_id"].nunique()
    print(f"[resolve] {len(pairs)} onayli cift -> {n_merged} aday birlestirildi. "
          f"{prov['canonical_id'].nunique()} kanonik tarama.")
    return prov


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provenance", type=Path, default=Path("outputs/provenance.csv"))
    ap.add_argument("--reviewed", type=Path, required=True,
                    help="ELLE onaylanmis candidate_duplicates (confirmed sutunlu)")
    ap.add_argument("--out", type=Path, default=Path("outputs/provenance_resolved.csv"))
    ap.add_argument("--current", type=Path, default=Path("outputs/dedup/candidate_duplicates.csv"),
                    help="dedup'in en son ciktisi; reviewed bunu kapsamali")
    a = ap.parse_args()
    cur = pd.read_csv(a.current) if a.current.exists() else None
    df = resolve(pd.read_csv(a.provenance), confirmed_pairs(pd.read_csv(a.reviewed), cur))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[resolve] -> {a.out}")
