"""3D-IRCADb-01 (DICOM) -> pipeline'in bekledigi NIfTI duzeni.

Cikti (vaka basina):
  <out>/3Dircadb1.N/ct.nii.gz      int16 HU
  <out>/3Dircadb1.N/tumor.nii.gz   uint8 ikili; karaciger tumoru maskelerinin birlesimi (beyaz liste)
  <out>/3Dircadb1.N/masks.json     hangi maskeler birlestirildi, hangileri disarida kaldi

Girdi: resmi zip (https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download) ya da
acilmis klasor. Hasta klasorlerinde PATIENT_DICOM / MASKS_DICOM ic zip'li ya da acik olabilir.

Sessiz hata yok:
  - CT ve maske ayni dosya indeksleriyle (image_0..N) siralanir; boyut uyusmazsa FIRLATIR.
  - Maskeye CT'nin geometrisi yalnizca boyutlar ayniysa kopyalanir.
  - Tumorsuz vaka bos tumor.nii.gz alir ve UYARI basar (Dice'a girmez, bkz. evaluate).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

_IDX_RE = re.compile(r"(\d+)$")


def numeric_sorted(files: list[Path]) -> list[Path]:
    """image_0, image_1, ..., image_10 (sozluk sirasi degil). Numarasiz dosya -> FIRLAT."""
    keyed = []
    for f in files:
        m = _IDX_RE.search(f.stem if f.suffix.lower() == ".dcm" else f.name)
        if not m:
            raise ValueError(f"DICOM dosya adinda indeks yok: {f}")
        keyed.append((int(m.group(1)), f))
    idx = [k for k, _ in keyed]
    if len(set(idx)) != len(idx):
        raise ValueError(f"Tekrarlanan dilim indeksi: {files[0].parent}")
    return [f for _, f in sorted(keyed)]


_LIVER_TUMOR_RE = re.compile(r"^liver_?tumou?rs?_?\d*$", re.IGNORECASE)
_AMBIGUOUS_TUMOR_RE = re.compile(r"tumou?r", re.IGNORECASE)


def is_tumor_mask(name: str) -> bool:
    """Beyaz liste: livertumor, livertumor01, livertumors, liverTumor_2.
    'tumor' ALT DIZGISI yetmez (19 Eyl 2026, gercek veride):
      1.5  leftsurretumor/rightsurretumor = bobrekustu (surrenale) tumoru
      1.7  'tumor' = karaciger DISINDA (%0.03 karaciger ici), sag bobrek-karaciger arasi kitle
    Sonuc: 15/20 vakada karaciger tumoru -- literaturdeki %75 ile tutarli."""
    return bool(_LIVER_TUMOR_RE.match(name))


def fraction_inside(mask: np.ndarray, region: np.ndarray) -> float:
    """mask'in region icinde kalan orani. Bos maske -> nan (tanimsiz, 1.0 degil)."""
    n = int(mask.sum())
    return float("nan") if n == 0 else float((mask.astype(bool) & region.astype(bool)).sum() / n)


def _dicom_files(d: Path) -> list[Path]:
    return [f for f in d.iterdir() if f.is_file() and not f.name.startswith(".")]


def _materialize(patient_dir: Path, name: str, tmp: Path) -> Path:
    """PATIENT_DICOM / MASKS_DICOM: acik klasor varsa onu, yoksa ic zip'i ac."""
    d = patient_dir / name
    if d.is_dir():
        return d
    z = patient_dir / f"{name}.zip"
    if not z.exists():
        raise FileNotFoundError(f"{name} ne klasor ne zip olarak var: {patient_dir}")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(tmp)
    found = [p for p in tmp.rglob(name) if p.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f"{z} icinde tek bir {name} klasoru beklenirdi: {found}")
    return found[0]


def read_series(d: Path):
    import SimpleITK as sitk
    files = numeric_sorted(_dicom_files(d))
    r = sitk.ImageSeriesReader()
    r.SetFileNames([str(f) for f in files])
    return r.Execute()


def convert_patient(patient_dir: Path, out_dir: Path) -> dict:
    import SimpleITK as sitk
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        ct = read_series(_materialize(patient_dir, "PATIENT_DICOM", t / "p"))
        masks_root = _materialize(patient_dir, "MASKS_DICOM", t / "m")
        names = sorted(p.name for p in masks_root.iterdir() if p.is_dir())
        tumor_names = [n for n in names if is_tumor_mask(n)]

        size = ct.GetSize()
        tumor = np.zeros(sitk.GetArrayFromImage(ct).shape, dtype=np.uint8)
        for n in tumor_names:
            m = read_series(masks_root / n)
            if m.GetSize() != size:
                raise ValueError(f"{patient_dir.name}/{n}: maske boyutu {m.GetSize()} != CT {size}")
            tumor |= (sitk.GetArrayFromImage(m) > 0).astype(np.uint8)

        # Hizalama + dogru organ kontrolu: karaciger tumoru karaciger maskesinin icinde olmali
        in_liver = float("nan")
        if "liver" in names:
            liver = read_series(masks_root / "liver")
            if liver.GetSize() != size:
                raise ValueError(f"{patient_dir.name}/liver: boyut {liver.GetSize()} != CT {size}")
            in_liver = fraction_inside(tumor, sitk.GetArrayFromImage(liver) > 0)

    ct16 = sitk.Cast(ct, sitk.sitkInt16)
    sitk.WriteImage(ct16, str(out_dir / "ct.nii.gz"))
    t_img = sitk.GetImageFromArray(tumor)
    t_img.CopyInformation(ct)
    sitk.WriteImage(t_img, str(out_dir / "tumor.nii.gz"))

    info = {"case": out_dir.name, "size": list(size), "spacing": list(ct.GetSpacing()),
            "tumor_masks": tumor_names, "other_masks": [n for n in names if n not in tumor_names],
            "tumor_voxels": int(tumor.sum()),
            "tumor_in_liver_frac": None if np.isnan(in_liver) else round(in_liver, 4)}
    (out_dir / "masks.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    if not np.isnan(in_liver) and in_liver < 0.9:
        print(f"[UYARI] {out_dir.name}: tumorun yalnizca %{100*in_liver:.0f}'i karaciger icinde "
              f"-> yanlis organ ya da hizalama hatasi olabilir. Tumor maskeleri: {tumor_names}")
    if skipped := [n for n in names if _AMBIGUOUS_TUMOR_RE.search(n) and n not in tumor_names]:
        print(f"[UYARI] {out_dir.name}: karaciger tumoru SAYILMAYAN tumor maskeleri: {skipped}")
    if not tumor_names:
        print(f"[UYARI] {out_dir.name}: tumor maskesi yok -> bos tumor.nii.gz. "
              f"Maskeler: {names}")
    return info


def find_patients(root: Path) -> list[Path]:
    """PATIENT_DICOM(.zip) iceren klasorler = hasta klasorleri."""
    hits = {p.parent for p in root.rglob("PATIENT_DICOM*")
            if p.name in ("PATIENT_DICOM", "PATIENT_DICOM.zip")}
    return sorted(hits, key=lambda p: [int(x) if x.isdigit() else x
                                       for x in re.split(r"(\d+)", p.name)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="resmi zip ya da acilmis klasor")
    ap.add_argument("--out", type=Path, default=Path("data/raw/3Dircadb1"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    tmp = None
    root = a.src
    if a.src.is_file():
        tmp = Path(tempfile.mkdtemp(prefix="ircadb_"))
        with zipfile.ZipFile(a.src) as zf:
            zf.extractall(tmp)
        root = tmp
    try:
        patients = find_patients(root)
        if len(patients) != 20:
            print(f"[UYARI] 20 hasta beklenirdi, {len(patients)} bulundu.")
        if not patients:
            raise SystemExit(f"Hasta klasoru bulunamadi: {a.src}")
        infos = []
        for p in patients:
            case = p.name.replace("3D-IRCADb-01-", "3Dircadb1.").rstrip("/")
            out = a.out / case
            if (out / "tumor.nii.gz").exists() and not a.overwrite:
                print(f"[ircadb] {case}: mevcut, atlandi")
                continue
            info = convert_patient(p, out)
            print(f"[ircadb] {case}: {info['size']} sp={tuple(round(s, 3) for s in info['spacing'])} "
                  f"tumor={info['tumor_masks']} ({info['tumor_voxels']} voksel, "
                  f"karaciger ici={info['tumor_in_liver_frac']})")
            infos.append(info)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
