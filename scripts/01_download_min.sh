#!/usr/bin/env bash
# Minimum veri: ilk gun icin yeterli, ~30 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

echo "[1/3] 3D-IRCADb (0.82 GB) -- EN HIZLI. Ama DIS TEST DEGIL (LiTS 28-47)."
# https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/

echo "[2/3] LiTS (16.66 GB) -- varsayilan egitim seti"
# huggingface-cli download Novel-BioMedAI/Liver_Segmentation_Datasets \
#   --repo-type dataset --local-dir data/raw/LiTS

echo "[3/3] AbdomenAtlas mask_only + meta (~13 GB) -- 586 GB'in tamamini INDIRME"
# huggingface-cli download AbdomenAtlas/AbdomenAtlas3.0Mini --repo-type dataset \
#   --include "mask_only/*" "*.csv" "TrainTestIDS/*" --local-dir data/raw/AbdomenAtlas3.0Mini

echo
echo "!! Komutlar bilerek yorumda. Once disk yerini kontrol et: df -h"
echo "!! Ticari niyet YOK (akademik) -> tum setler kullanilabilir."
