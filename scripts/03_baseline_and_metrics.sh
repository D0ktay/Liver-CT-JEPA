#!/usr/bin/env bash
# Gun-0 baseline + metrikler. GPU gerekmez.
# !! L6 dogrulanmadan (TS egitim verisi LiTS iceriyor mu) IRCADb/LiTS skoru DIS dogrulama DEGIL.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

CT_GLOB=${CT_GLOB:-'data/raw/3Dircadb1/**/ct.nii.gz'}
# IRCADb GT'si LiTS formatinda degil: tumor maskeleri NIfTI'ye cevrilip vaka klasorune
# tek dosya (orn. tumor.nii.gz, ikili) olarak konmali. LiTS icin:
#   GT_GLOB='data/raw/LiTS/**/segmentation-*.nii*' GT_LABEL=2
GT_GLOB=${GT_GLOB:-'data/raw/3Dircadb1/**/tumor.nii.gz'}
GT_LABEL=${GT_LABEL:-any}
PRED_DIR=outputs/baseline/totalsegmentator
LIMIT=${LIMIT:-5}          # once 5 vakayla dogrula, sonra LIMIT=0

echo "== TotalSegmentator (CPU, vaka basina ~5-15 dk) =="
python -m src.baseline.run_totalsegmentator \
  --in-glob "$CT_GLOB" --out "$PRED_DIR" --device cpu --limit "$LIMIT"

echo
echo "== GT <-> tahmin eslestirme =="
python -m src.metrics.make_pairs \
  --gt-glob "$GT_GLOB" --pred-glob "$PRED_DIR/*.nii.gz" --out outputs/pairs.csv

echo
echo "== Degerlendirme =="
# TotalSegmentator liver_lesions: lezyon etiketi = 1 (class_map ile dogrulandi)
python -m src.metrics.evaluate \
  --pairs outputs/pairs.csv --out outputs/eval \
  --gt-label "$GT_LABEL" --pred-label 1 --margin-mm 5.0 --iou 0.5
