#!/usr/bin/env bash
# Ortam kurulumu. Bir kere calistir.
set -euo pipefail
cd "$(dirname "$0")/.."

# Python >=3.10 sec. Sistem python3 (macOS'ta 3.9) python-gdcm'i derleyemiyor.
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1 &&
     "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "HATA: Python >=3.10 bulunamadi (python3 = $(python3 --version 2>&1))." >&2
  exit 1
fi
echo "Python: $PY ($("$PY" --version))"

"$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "--- torch/numpy uyumu ---"
# Intel Mac'te torch 2.2.2'de kalir; numpy 2 ile sessizce bozulur.
python - <<'PYEOF'
import numpy, torch
torch.from_numpy(numpy.zeros(3))
print(f"torch {torch.__version__} + numpy {numpy.__version__}: ok")
PYEOF

echo "--- test ---"
pytest -q
echo
echo "Tum testler gecerse ortam hazir. Simdi: scripts/01_download_min.sh"
