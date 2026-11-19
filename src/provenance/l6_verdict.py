"""L6 'ncc_partial' adaylarini ozgullukle ele: hangisine gercekten elle bakilacak?

    python -m src.provenance.l6_verdict outputs/kaggle/l6/outputs/l6/ts_vs_lits.csv

`match_crop.search` bir sorguyu 'ncc_partial' isaretlerken: tek yama NCC_SINGLE'i (0.97)
gecti ama iki yama ayni kirpma kokeninde bulusmadi. Bunlar "elle bak" kutusuna atiliyordu.
Burasi o kutuyu ayikliyor.

Olcut MUTLAK skor DEGIL. Gercek bir kopyada "kendi hastasi" ile "baska hasta" arasinda
belirgin fark olmali; sade karaciger benzerliginde her hasta birbirine yakin skor verir.
Ayirici olcut bu FARK (ozguluk). Iki kural birlikte:

  1. Sabit esik GAP_FLAG. Akran oturumun kontrolunden turedi (set disi yama 0.906, gercek
     kopya ~0.99). ⚠️ Bu taban VERI KUMESINE GORE DEGISIR: ts_vs_lits.csv'de ikinci en iyi
     skorlarin medyani 0.984, yani gurultu tabani 0.906 degil ~0.98. Orada gercek bir
     kopyanin farki ~0.01 olurdu ve sabit esik onu KACIRIR. Sabit esik tek basina yeterli
     degildir; asagidaki kuralla birlikte kullanilir.
  2. Ampirik aykiri deger: farkin kendi dagilimina gore robust z (medyan + ROBUST_Z x MAD).
     Gurultu tabani neyse ona gore kalibre olur.

Ikinci en iyi AYNI hacimden geliyorsa ozguluk olculemez. match_crop referansi iki gorunumde
arar ('<id>' ve '<id>@ras'), ikisi ayni hacimdir -> karsilastirma temel kimlik uzerinden.
Ikinci skoru olmayan satir 'temiz' degil 'karsilastirilamaz' sayilir.

Kapsam: bu modul "kac sorgu GERCEKTEN denetlendi" sayisini verir. match_crop ile ayni korluk
tanimi: NCC adayi yok (xy yeniden orneklenmis kopya), butce kesildi (ncc_complete False),
yapili yama < 2. Denetlenmeyen sorgu "kopya yok" kaniti DEGILDIR.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GAP_FLAG = 0.027          # sabit esik; gerekcesi ve sinirlari icin modul aciklamasi
ROBUST_Z = 5.0            # ampirik aykiri deger: medyandan kac robust sigma
MAD_K = 1.4826            # MAD -> standart sapma olcegi (normal dagilim)
PARTIAL = "ncc_partial"


def _base(s: pd.Series) -> pd.Series:
    """'lits-7@ras' -> 'lits-7' (ayni hacmin baska gorunumu)."""
    return s.astype("string").str.split("@").str[0]


def _denetlendi(df: pd.DataFrame) -> pd.Series:
    """match_crop korluk tanimi. Sorgu iki kanaldan BIRI calistiysa denetlenmis sayilir:
      - birebir (exact): integer_valued ise calisir, spacing filtresi YOK, tum referanslari tarar
      - NCC: aday var, butce kesilmedi, en az 2 yapili yama
    Not: TS goruntulerinin degerleri float32 gosterim hatasi (~5e-4) tasidigi icin
    match_crop'un kati testi (== rint) 842/842 sorguyu exact'e kor sayiyor; oysa 416'si
    toleransla tamsayi. Yuvarlanmis yamalarla kosuldugunda bu sutun True olur."""
    integer = df["integer_valued"].fillna(False).astype(bool) if "integer_valued" in df \
        else pd.Series(False, index=df.index)
    cand = pd.to_numeric(df.get("ncc_candidates"), errors="coerce").fillna(0) > 0
    comp = df["ncc_complete"].fillna(True).astype(bool) if "ncc_complete" in df else True
    stru = pd.to_numeric(df.get("n_structured"), errors="coerce").fillna(0) >= 2 \
        if "n_structured" in df else True
    return integer | (cand & comp & stru)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """ncc_partial satirlarina gap / robust_z / karar / kural sutunlari ekler (kopya dondurur)."""
    out = df.copy()
    is_p = out["verdict"] == PARTIAL
    farkli = _base(out["ncc_best_ref"]).notna() & _base(out["ncc_second_ref"]).notna() & \
        (_base(out["ncc_best_ref"]) != _base(out["ncc_second_ref"]))
    olcu = is_p & farkli & out["ncc_best"].notna() & out["ncc_second"].notna()
    out["gap"] = (out["ncc_best"] - out["ncc_second"]).where(olcu)

    g = out.loc[olcu, "gap"]
    med = float(g.median()) if len(g) else np.nan
    # MAD; yarisindan fazlasi ayni degerdeyse MAD 0 cikar -> ortalama mutlak sapmaya dus.
    # Ikisi de 0 ise dagilimda hic yayilma yok, aykiri deger tanimlanamaz.
    mad = float((g - med).abs().median()) if len(g) else np.nan
    if len(g) and not (mad > 0):
        mad = float((g - med).abs().mean())
    out["robust_z"] = ((out["gap"] - med) / (MAD_K * mad)) if mad and mad > 0 else np.nan

    sabit = olcu & (out["gap"] >= GAP_FLAG)
    ampirik = olcu & (out["robust_z"] >= ROBUST_Z) & ~sabit
    out["karar"] = "-"
    out["kural"] = ""
    out.loc[is_p & ~olcu, "karar"] = "karsilastirilamaz"
    out.loc[olcu, "karar"] = "ozgul degil"
    out.loc[sabit | ampirik, "karar"] = "elle bak"
    out.loc[sabit, "kural"] = "sabit esik"
    out.loc[ampirik, "kural"] = "aykiri deger"
    return out


def summarize(df: pd.DataFrame) -> dict:
    c = classify(df)
    nm = c["verdict"] == "no_match"
    nm_ok = int((nm & _denetlendi(c)).sum())
    olculebilir = int(((c["karar"] == "ozgul degil") | (c["karar"] == "elle bak")).sum())
    gaps = c["gap"].dropna()
    n = int(len(c))
    return {
        "n_sorgu": n,
        "n_kismi": int((c["verdict"] == PARTIAL).sum()),
        "n_elle_bak": int((c["karar"] == "elle bak").sum()),
        "n_elle_bak_sabit": int((c["kural"] == "sabit esik").sum()),
        "n_elle_bak_ampirik": int((c["kural"] == "aykiri deger").sum()),
        "n_ozgul_degil": int((c["karar"] == "ozgul degil").sum()),
        "n_karsilastirilamaz": int((c["karar"] == "karsilastirilamaz").sum()),
        "n_no_match": int(nm.sum()),
        "n_no_match_denetlenen": nm_ok,
        "n_denetlenen": nm_ok + olculebilir,
        "oran_denetlenen": (nm_ok + olculebilir) / n if n else 0.0,
        "gap_max": float(gaps.max()) if len(gaps) else None,
        "gap_medyan": float(gaps.median()) if len(gaps) else None,
        "elle_bakilacak_var": bool((c["karar"] == "elle bak").any()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="match_crop search ciktisi (ts_vs_lits.csv)")
    ap.add_argument("--out", type=Path, default=None, help="siniflandirilmis CSV yaz")
    ap.add_argument("--top", type=int, default=3, help="en buyuk kac farki listele")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.csv)
    c, s = classify(df), summarize(df)

    print(f"sorgu {s['n_sorgu']} | kismi {s['n_kismi']} | no_match {s['n_no_match']}")
    print(f"KAPSAM: gercekten denetlenen {s['n_denetlenen']}/{s['n_sorgu']} "
          f"(%{100 * s['oran_denetlenen']:.0f}) -- geri kalani 'kopya yok' KANITI DEGIL")
    print(f"  kismi ayrimi: elle bak {s['n_elle_bak']} "
          f"(sabit esik {s['n_elle_bak_sabit']}, aykiri deger {s['n_elle_bak_ampirik']}) | "
          f"ozgul degil {s['n_ozgul_degil']} | karsilastirilamaz {s['n_karsilastirilamaz']}")
    if s["gap_max"] is not None:
        print(f"  ozguluk farki: medyan {s['gap_medyan']:.4f} | en buyuk {s['gap_max']:.4f} "
              f"(sabit esik {GAP_FLAG})")
    top = c.dropna(subset=["gap"]).nlargest(a.top, "gap")
    if len(top):
        print(f"  en buyuk {len(top)} fark (esik gecmese de goz atilmali):")
        for r in top.itertuples():
            z = f"{r.robust_z:.1f}" if pd.notna(r.robust_z) else "-"
            print(f"    {r.query_id}  {r.ncc_best:.4f} {r.ncc_best_ref} vs "
                  f"{r.ncc_second:.4f} {r.ncc_second_ref}  fark {r.gap:.4f}  z {z}")
    if s["elle_bakilacak_var"]:
        print(f"SONUC: {s['n_elle_bak']} satir ELLE INCELENMELI. Bu bir kopya kaniti DEGIL; "
              "bakilmadan 'kopya yok' da denemez.")
    else:
        print("SONUC: isaretlenen satir yok. Bu, denetlenen kisimda kopya kaniti "
              "bulunmadigi anlamina gelir; 'LiTS yok' anlamina GELMEZ.")
    print("Kor noktalar ACIK: xy yeniden orneklenmis kopya gorulmez; kismi satirlarda "
          "tamsayi olmayan veri exact aramaya kor; karsilastirilamaz satirlar hic sinanmadi; "
          "referans havuzu yalniz LiTS egitim (131 hacim).")
    if a.out:
        c.to_csv(a.out, index=False)
        print("yazildi:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
