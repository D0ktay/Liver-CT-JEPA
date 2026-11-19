#!/usr/bin/env bash
# Otonom Kaggle gonderimi: kodu (git HEAD) ozel veri seti olarak yukle, isi baslat.
#   KAGGLE_USER=<kullanici> bash scripts/kaggle_push.sh <is_adi>   # kaggle_jobs/<is_adi>/{run.py,kernel-metadata.json}
# kernel-metadata.json icindeki YOUR_KAGGLE_USERNAME yer tutucusunu da kendi kullanici adinla degistir.
# Kimlik: ~/.kaggle/access_token (repo disinda). CLI: ~/.local/kaggle-cli (proje venv'inden ayri).
set -euo pipefail
JOB="${1:?is adi ver: kaggle_jobs/<ad>}"
K="${KAGGLE_BIN:-$HOME/.local/kaggle-cli/bin/kaggle}"
KU="${KAGGLE_USER:?KAGGLE_USER ayarla (Kaggle kullanici adin)}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
[ -f "kaggle_jobs/$JOB/kernel-metadata.json" ] || { echo "yok: kaggle_jobs/$JOB"; exit 1; }
if [ -n "$(git status --porcelain -- src configs requirements.txt kaggle_jobs/$JOB)" ]; then
  echo "commit edilmemis degisiklik var (src/configs/requirements/is) -- once commit et"; exit 1
fi
DS="$(mktemp -d)"
git archive --format=tar HEAD | gzip > "$DS/code.tar.gz"
git rev-parse --short HEAD > "$DS/COMMIT"
# COMMIT dosyasini arsivin icine de koy
( cd "$DS" && mkdir -p x && tar -xzf code.tar.gz -C x && cp COMMIT x/ && tar -C x -czf code.tar.gz . && rm -rf x )
cat > "$DS/dataset-metadata.json" <<JSON
{"title": "liver-ct-code", "id": "$KU/liver-ct-code", "licenses": [{"name": "other"}]}
JSON
if "$K" datasets status $KU/liver-ct-code >/dev/null 2>&1; then
  "$K" datasets version -p "$DS" -m "kod $(cat "$DS/COMMIT")" -q
else
  "$K" datasets create -p "$DS" -q
fi
echo "[push] kod veri seti: $(cat "$DS/COMMIT"); hazir olmasi bekleniyor"
for _ in $(seq 1 60); do
  st="$("$K" datasets status $KU/liver-ct-code 2>/dev/null || true)"
  case "$st" in *ready*) break;; esac
  sleep 10
done
echo "[push] veri seti durumu: $st"
"$K" kernels push -p "kaggle_jobs/$JOB"
rm -rf "$DS"
