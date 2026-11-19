#!/usr/bin/env bash
# JEPA egitim isinin ciktisini indir -> ADIL kiyasla degerlendir -> panel belgeleri uret.
#   KAGGLE_USER=<kullanici> bash scripts/collect_jepa.sh liver-jepa-train-ras ras
# Adil kiyas: TS 126 vakada olculdu (lits-48..52 TS'te dustu), JEPA 131'de. Ayni cetvel
# icin --common-with ile ORTAK vakalara indirgenir; hazir ci.csv kullanilmaz.
set -euo pipefail
SLUG="${1:?kernel slug: liver-jepa-train-ras}"
TAG="${2:?panel anahtar eki: ras}"
K="${KAGGLE_BIN:-$HOME/.local/kaggle-cli/bin/kaggle}"
KU="${KAGGLE_USER:?KAGGLE_USER ayarla (Kaggle kullanici adin)}"
PY="${PY:-.venv/bin/python}"
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
TS_EVAL="outputs/kaggle/fix2847/outputs/fix2847/all131"
D="outputs/kaggle/$SLUG"

[ -f "$TS_EVAL/per_case.csv" ] || { echo "TS kiyas kumesi yok: $TS_EVAL"; exit 1; }
mkdir -p "$D"
if [ "${SKIP_DL:-0}" = "1" ]; then
  echo "[collect] SKIP_DL=1 -> indirme atlandi, $D icindeki cikti kullanilacak"
else
  echo "[collect] $SLUG indiriliyor -> $D"
  "$K" kernels output "$KU/$SLUG" -p "$D" >/dev/null
fi

SEG="$(find "$D" -type d -name jepa_seg | head -1)"
[ -n "$SEG" ] || { echo "jepa_seg klasoru bulunamadi (is basarisiz olabilir); log:"; \
                   tail -5 "$D/$SLUG.log" 2>/dev/null; exit 1; }
echo "[collect] cikti: $SEG"
[ -f "$SEG/RESULT.json" ] && "$PY" -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['summary'],indent=1))" "$SEG/RESULT.json"

n=0
for V in jepa nojepa; do
  EV="$SEG/$V/eval"
  if [ ! -f "$EV/summary.json" ]; then echo "[collect] $V: degerlendirme yok, atlandi"; continue; fi
  case "$V" in
    jepa)   MODEL=jepa;   LABEL="Rad-JEPA + baslik";  SHORT="JEPA";       ORDER=3 ;;
    nojepa) MODEL=nojepa; LABEL="JEPA'siz kontrol";   SHORT="JEPA'siz";   ORDER=4 ;;
  esac
  "$PY" -m src.viz.panel_data run --eval "$EV" --common-with "$TS_EVAL" \
    --key "${MODEL}_${TAG}" --model "$MODEL" --label "$LABEL" --short "$SHORT" --order "$ORDER" \
    --note "5 kat capraz dogrulama, TS ile ortak vaka kumesi"
  [ -f "$EV/margin_sweep.json" ] && "$PY" -m src.viz.panel_data margin \
    --sweep "$EV/margin_sweep.json" --key "${MODEL}_${TAG}" --run "${MODEL}_${TAG}" --short "$SHORT"
  n=$((n+1))
done
echo "[collect] $n kol islendi. Panel belgeleri: outputs/panel/docs/{runs,margin}/"
