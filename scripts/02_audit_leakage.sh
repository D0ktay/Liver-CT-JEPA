#!/usr/bin/env bash
# SIZINTI DENETIMI -- baska hicbir sey yapmadan once bu.
# Iki gecisli: 1. gecis provenance + dedup uretir ve DURUR (elle inceleme).
# candidate_duplicates.reviewed.csv (confirmed sutunlu) varsa 2. gecis split kurar.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p outputs
REVIEWED=outputs/dedup/candidate_duplicates.reviewed.csv

echo "== 1. Provenance tablosu =="
python -m src.provenance.build_provenance \
  --mapping-csv data/raw/AbdomenAtlas3.0Mini/AbdomenAtlas3.0MiniWithMeta.csv \
  --out outputs/provenance.csv

echo
echo "== 2. Icerik tabanli kopya tespiti =="
# Her calismada yeniden: yeni veri eklendiyse eski onay dosyasi yetersiz kalir
# (resolve_canonical incelenmemis yeni adayi gorurse durur). Atlamak icin SKIP_DEDUP=1.
# CT modunda maskeler is_mask_path ile dislanir. Sadece mask_only varsa: --kind mask
if [ "${SKIP_DEDUP:-0}" != "1" ]; then
  python -m src.provenance.dedup_hash \
    --glob 'data/raw/**/*.nii*' --kind "${DEDUP_KIND:-ct}" \
    --out outputs/dedup --threshold 0.995
fi

if [ ! -f "$REVIEWED" ]; then
  echo
  echo "DUR: outputs/dedup/candidate_duplicates.csv'yi ELLE incele,"
  echo "     'confirmed' sutunu ekle (1=ayni tarama, 0=degil), $REVIEWED olarak kaydet,"
  echo "     sonra bu script'i tekrar calistir. Onaysiz split KURULMAZ."
  exit 0
fi

echo
echo "== 2b. Onayli kopyalar -> canonical_id =="
python -m src.provenance.resolve_canonical \
  --provenance outputs/provenance.csv --reviewed "$REVIEWED" \
  --current outputs/dedup/candidate_duplicates.csv \
  --out outputs/provenance_resolved.csv

echo
echo "== 3. Sizintisiz split (K1/K2/K3 assert) =="
python -m src.data.make_splits \
  --provenance outputs/provenance_resolved.csv --out outputs/splits.csv

echo
echo "Ciktilar: outputs/provenance_resolved.csv, outputs/splits.csv"
