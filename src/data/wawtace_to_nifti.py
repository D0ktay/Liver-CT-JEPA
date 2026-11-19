"""WAW-TACE (Zenodo 10.5281/zenodo.12741586, CC BY 4.0) -> pipeline'in bekledigi NIfTI duzeni.

Varsova, 233 tedavi-oncesi HCC hastasi, cok fazli CT, 377 ELLE cizilmis tumor maskesi.
LiTS ve HCC-TACE-Seg ile kurum ortusmesi yok. ⚠️ FLARE PancancerCTSeg'in icinde (leakage-map L19).

Cikti:
  <out>/images/wawtace-<hasta>-<faz>.nii.gz   CT (HU, kaynaktaki tip)
  <out>/labels/wawtace-<hasta>-<faz>.nii.gz   uint8; 0 = arka plan, 2 = tumor (LiTS kurali, CLAUDE.md #10)
  <out>/manifest.csv                          vaka basina: maske sayisi, klinik lezyon sayisi, complete
  <out>/report.json                           ozet + yok sayilan dosyalar + etiketsiz fazlar

Gercek veride OLCULEN (21 Eyl 2026, maske zip'i + ct_hcc_metadata_v2.csv + klinik tablo):
  - maske adi `<hasta>_<faz>_<tumor>_tumor_seg.nrrd`, int16, tumor basina AYRI dosya
  - 377 maske <-> meta `tumor_count` birebir; her hastada maske TEK fazda (faz1 126, faz2 74,
    faz3 30, faz0 3) -> 233 etiketli hacim; 854 CT'nin kalani ETIKETSIZ, buradan yazilmaz
  - klinik `lesions_number` ile kiyas: 224 hastada maske >= lezyon sayisi, 9 hastada EKSIK
    -> `complete=False`: o vakada maske disi bolge "tumorsuz" SAYILAMAZ (kismi cizim)
  - zip'te kurala uymayan 1 dosya var (63_1_0.nrrd) -> report.json `ignored`
  - maske degerleri: 368 maske {0,1}, 7 maske {0,2}, 2 maske {0,1,2}; hepsi tek bilesen ve
    65_2_0'in 2-degerli bolgesi = 65_2_1'in tamami -> deger tumor SIRA numarasi, sifir-disi = tumor
  - 377 maskenin 357'si kendi CT'sinin izgarasinda; 20 maske / 17 hasta BASKA seride cizilmis
    (farkli kesit sayisi/kalinligi; 15'i ayni hastanin baska fazinin izgarasiyla eslesiyor ama hasta
    31'de iki faz birden eslesiyor) -> tahminle faz atamasi YAPILMAZ, `--exclude-mismatched` ile
    hasta tumuyle dislanir -> 216 vaka
KARACIGER ETIKETI YOK (1 yazilmaz) -> ince ayarda MSD Task08 gibi "yalniz tumor kanali gecerli".
CT zip'lerinin ic duzeni (Lambda'da dogrulandi, 21 Eyl 2026): `<hasta>/<hasta>_<faz>_scan.nii.gz`;
meta `ct_file_name` = `<hasta>_<faz>` -> `_scan` sonekiyle (yoksa soneksiz) aranir, TEK eslesme sart.

Sessiz hata yok: ad kurali, meta<->maske sayisi, sekil, spacing/origin/yon, bos maske -> FIRLATIR.
Maske CT izgarasinda degilse yeniden ORNEKLENMEZ.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TUMOR_LABEL = 2
MASK_VALUES = {0, 1, 2}      # 2 = Slicer segment indeksi (9/377 maske); sifir-disi = tumor
SPACING_TOL_REL = 0.01      # < %1
ORIGIN_TOL_VOX = 0.5        # < yarim voksel
_MASK_RE = re.compile(r"^(\d+)_(\d+)_(\d+)_tumor_seg\.nrrd$")
_CT_EXT = (".nrrd", ".nii.gz", ".nii", ".mha")
MANIFEST_COLS = ["case_id", "patient_id", "phase", "n_lesion_masks", "lesions_number_clinical",
                 "complete", "has_liver_label", "tumor_voxels", "image", "label"]


class WawTaceError(ValueError):
    pass


@dataclass
class Case:
    patient_id: int
    phase: int
    ct_file_name: str
    masks: list[Path]
    lesions_clinical: int | None = None

    @property
    def case_id(self) -> str:
        return f"wawtace-{self.patient_id}-{self.phase}"

    @property
    def complete(self) -> bool | None:
        """Maske sayisi klinik lezyon sayisini karsiliyor mu? Klinik tablo yoksa None (bilinmiyor)."""
        return None if self.lesions_clinical is None else len(self.masks) >= self.lesions_clinical


@dataclass
class Plan:
    cases: list[Case]
    unlabeled_ct: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


def parse_mask_name(name: str) -> tuple[int, int, int]:
    """`10_2_0_tumor_seg.nrrd` -> (hasta 10, faz 2, tumor 0). Kurala uymayan ad -> FIRLAT."""
    m = _MASK_RE.match(name)
    if not m:
        raise WawTaceError(f"maske adi kurala uymuyor: {name} (beklenen <hasta>_<faz>_<tumor>_tumor_seg.nrrd)")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def merge_masks(arrays: list[np.ndarray]) -> np.ndarray:
    """Tumor basina ikili maskelerin BIRLESIMI -> uint8, tumor = TUMOR_LABEL."""
    if not arrays:
        raise WawTaceError("birlestirilecek maske yok")
    out = np.zeros(arrays[0].shape, bool)
    for a in arrays:
        if a.shape != out.shape:
            raise WawTaceError(f"maske sekil uyusmazligi: {a.shape} != {out.shape}")
        vals = set(np.unique(a).tolist())
        if not vals <= MASK_VALUES:
            raise WawTaceError(f"maskede beklenmeyen deger: {sorted(vals)[:6]} (izinli {sorted(MASK_VALUES)})")
        if vals == {0}:
            raise WawTaceError("bos tumor maskesi -- bozuk dosya, 'tumorsuz' sayilmaz")
        out |= a != 0
    return out.astype(np.uint8) * np.uint8(TUMOR_LABEL)


def _read_clinical(path: Path) -> dict[int, int]:
    """CSV (PATPRI, lesions_number). xlsx ise once CSV'ye cevir -- openpyxl bagimliligi eklenmedi."""
    if path.suffix.lower() != ".csv":
        raise WawTaceError(f"klinik tablo CSV olmali (PATPRI, lesions_number): {path}")
    out = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("lesions_number") or "").strip():
                out[int(float(r["PATPRI"]))] = int(float(r["lesions_number"]))
    return out


def build_plan(meta_csv: Path, masks_dir: Path, clinical: Path | None) -> Plan:
    with Path(meta_csv).open(newline="") as f:
        meta = {(int(r["patient_id"]), int(r["ct_phase"])): r for r in csv.DictReader(f)}
    clin = _read_clinical(Path(clinical)) if clinical else {}

    found: dict[tuple[int, int], dict[int, Path]] = defaultdict(dict)
    ignored = []
    for p in sorted(Path(masks_dir).rglob("*.nrrd")):
        try:
            pid, ph, idx = parse_mask_name(p.name)
        except WawTaceError:
            ignored.append(p.name)
            continue
        if (pid, ph) not in meta:
            raise WawTaceError(f"{p.name}: (hasta {pid}, faz {ph}) meta CSV'de yok")
        if idx in found[(pid, ph)]:
            raise WawTaceError(f"ayni maske iki kez: {p.name}")
        found[(pid, ph)][idx] = p

    cases, unlabeled = [], []
    for key, r in sorted(meta.items()):
        n_meta, got = int(r["tumor_count"]), found.get(key, {})
        if n_meta != len(got):
            raise WawTaceError(f"hasta {key[0]} faz {key[1]}: meta tumor_count={n_meta}, maske dosyasi={len(got)}")
        if n_meta == 0:
            unlabeled.append(r["ct_file_name"])
            continue
        cases.append(Case(key[0], key[1], r["ct_file_name"], [got[i] for i in sorted(got)], clin.get(key[0])))
    return Plan(cases, unlabeled, ignored)


def _find_ct(ct_dir: Path, stem: str) -> Path:
    hits = [p for suf in ("_scan", "") for ext in _CT_EXT for p in Path(ct_dir).rglob(stem + suf + ext)]
    if len(hits) != 1:
        raise WawTaceError(f"CT bulunamadi ya da birden fazla: {stem} -> {[str(h) for h in hits]}")
    return hits[0]


def _check_geometry(ct, mask, name: str) -> None:
    """Ayni voksel izgarasi mi? Boyut esit + spacing farki < %1 + origin farki < YARIM voksel + ayni yon.
    (Gercek veride 4 maskede 0.03-0.28 mm origin / 0.8 vs 0.801 spacing yuvarlamasi var -> ayni izgara.)"""
    if ct.GetSize() != mask.GetSize():
        raise WawTaceError(f"{name}: sekil CT {ct.GetSize()} != maske {mask.GetSize()}")
    csp, msp = np.array(ct.GetSpacing()), np.array(mask.GetSpacing())
    if np.any(np.abs(csp - msp) / csp >= SPACING_TOL_REL):
        raise WawTaceError(f"{name}: spacing CT {tuple(csp)} != maske {tuple(msp)} -- yeniden ornekleme YAPILMAZ")
    if not np.allclose(ct.GetDirection(), mask.GetDirection(), atol=1e-4):
        raise WawTaceError(f"{name}: yon CT {ct.GetDirection()} != maske {mask.GetDirection()}")
    d = np.abs(np.array(ct.GetOrigin()) - np.array(mask.GetOrigin()))
    if np.any(d >= ORIGIN_TOL_VOX * csp):
        raise WawTaceError(f"{name}: origin CT {ct.GetOrigin()} != maske {mask.GetOrigin()} -- yeniden ornekleme YAPILMAZ")


def convert(ct_dir: Path, masks_dir: Path, meta_csv: Path, out_dir: Path,
            clinical: Path | None = None, on_mismatch: str = "raise") -> list[dict]:
    """on_mismatch="exclude": maskelerinden BIRI bile CT izgarasinda olmayan hasta TUMUYLE dislanir
    (tek maskeyi atmak yanlis negatif bolge uretir) ve report.json `excluded`'a gerekcesiyle yazilir."""
    import SimpleITK as sitk

    if on_mismatch not in ("raise", "exclude"):
        raise WawTaceError(f"on_mismatch={on_mismatch!r}")
    excluded = []

    plan = build_plan(meta_csv, masks_dir, clinical)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    rows = []
    for c in plan.cases:
        ct = sitk.ReadImage(str(_find_ct(ct_dir, c.ct_file_name)))
        arrays = []
        try:
            for mp in c.masks:
                m = sitk.ReadImage(str(mp))
                _check_geometry(ct, m, mp.name)
                arrays.append(sitk.GetArrayFromImage(m))
        except WawTaceError as e:
            if on_mismatch == "raise":
                raise
            excluded.append({"case_id": c.case_id, "reason": str(e)})
            print(f"[DISLANDI] {c.case_id}: {e}", flush=True)
            continue
        lab_arr = merge_masks(arrays)
        lab = sitk.GetImageFromArray(lab_arr)
        lab.CopyInformation(ct)
        img_p = out_dir / "images" / f"{c.case_id}.nii.gz"
        lab_p = out_dir / "labels" / f"{c.case_id}.nii.gz"
        sitk.WriteImage(ct, str(img_p))
        sitk.WriteImage(lab, str(lab_p))
        if c.complete is False:
            print(f"[UYARI] {c.case_id}: {len(c.masks)} maske < klinik {c.lesions_clinical} lezyon -> "
                  "kismi cizim, maske disi 'tumorsuz' sayilmaz", flush=True)
        rows.append({"case_id": c.case_id, "patient_id": c.patient_id, "phase": c.phase,
                     "n_lesion_masks": len(c.masks), "lesions_number_clinical": c.lesions_clinical,
                     "complete": c.complete, "has_liver_label": False,
                     "tumor_voxels": int((lab_arr > 0).sum()), "image": str(img_p), "label": str(lab_p)})
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "report.json").write_text(json.dumps({
        "n_cases": len(rows), "n_patients": len({r["patient_id"] for r in rows}),
        "n_incomplete": sum(r["complete"] is False for r in rows),
        "n_excluded": len(excluded), "excluded": excluded,
        "n_unlabeled_ct": len(plan.unlabeled_ct), "unlabeled_ct": plan.unlabeled_ct,
        "ignored": plan.ignored, "tumor_label": TUMOR_LABEL}, indent=2))
    if plan.ignored:
        print(f"[UYARI] kurala uymayan {len(plan.ignored)} dosya kullanilmadi: {plan.ignored}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ct-dir", required=True, type=Path, help="acilmis ct_scans_*_wawtace zip'leri")
    ap.add_argument("--masks-dir", required=True, type=Path, help="acilmis tumor_masks_wawtace zip'i")
    ap.add_argument("--meta", required=True, type=Path, help="ct_hcc_metadata_v2.csv")
    ap.add_argument("--clinical", type=Path, help="CSV: PATPRI, lesions_number (tamlik bayragi icin)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude-mismatched", action="store_true",
                    help="maskesi CT izgarasinda olmayan hastayi FIRLATMAK yerine tumuyle disla + raporla")
    ap.add_argument("--plan-only", action="store_true", help="CT'ye dokunmadan plani ve sayilari yaz")
    a = ap.parse_args()
    if a.plan_only:
        p = build_plan(a.meta, a.masks_dir, a.clinical)
        print(json.dumps({"n_cases": len(p.cases), "n_masks": sum(len(c.masks) for c in p.cases),
                          "n_incomplete": sum(c.complete is False for c in p.cases),
                          "n_unlabeled_ct": len(p.unlabeled_ct), "ignored": p.ignored}, indent=2))
        return
    rows = convert(a.ct_dir, a.masks_dir, a.meta, a.out, a.clinical,
                   on_mismatch="exclude" if a.exclude_mismatched else "raise")
    print(f"{len(rows)} vaka yazildi -> {a.out}")


if __name__ == "__main__":
    main()
