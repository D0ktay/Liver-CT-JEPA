#!/usr/bin/env bash
# LiverJEPA — sifir makineden tam deneye, TEK KOMUT.
#
#   bash scripts/lambda_bootstrap.sh              # bastan sona (resume edilebilir)
#   bash scripts/lambda_bootstrap.sh --only cache # tek adim
#   bash scripts/lambda_bootstrap.sh --from pretrain
#   bash scripts/lambda_bootstrap.sh --dry-run    # ne yapacagini yazar, yapmaz
#
# RESUME: her adim bitince `$STATE/<adim>.done` yazilir; yeniden calistirilinca atlanir.
# Bir adimi zorla tekrarlamak icin: rm outputs/lambda/state/<adim>.done
#
# 🔴 KURAL 10: `audit` adimi kirmizi ise ON EGITIM BASLAMAZ. Bu betik orada durur.
#    Assert'i gevsetme -- veriyi duzelt (`outputs/split_audit.json` -> checks.*.examples).
#
# Adim suresi ve cikti boyutu `outputs/lambda/steps.tsv` dosyasina yazilir.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

OUT="${OUT:-outputs/lambda}"
STATE="$OUT/state"
LOGS="$OUT/logs"
STEPS="$OUT/steps.tsv"
CACHE="${CACHE:-data/cache/iso1mm}"
CORPUS="${CORPUS:-configs/pretrain_corpus.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train.yaml}"
PROFILE="${PROFILE:-auto}"          # auto -> pick_config(n_vol, gpu_h)
GPU_HOURS="${GPU_HOURS:-24}"
FOLDS="${FOLDS:-0 1}"               # asgari senaryo: 2 kat. Tam: "0 1 2 3 4"
ARMS="${ARMS:-jepa mae none}"       # kontrol kollari KESILMEZ (EGITIM-PLANI §5)
SEED="${SEED:-0}"
DRY=0
ONLY=""
FROM=""

ALL_STEPS="env deps data frozen manifest audit cache probe pretrain finetune evaluate report"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="${2:?adim adi}"; shift ;;
    --from) FROM="${2:?adim adi}"; shift ;;
    --profile) PROFILE="${2:?}"; shift ;;
    --folds) FOLDS="${2:?}"; shift ;;
    --arms) ARMS="${2:?}"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "bilinmeyen secenek: $1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$STATE" "$LOGS"
[ -f "$STEPS" ] || printf 'adim\tdurum\tsaniye\tnot\n' > "$STEPS"

log()  { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[33m[UYARI] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m[DUR] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- adim cercevesi
skip_until_reached=0
[ -n "$FROM" ] && skip_until_reached=1

should_run() {  # $1 = adim adi
  local s="$1"
  # 🔴 `audit` HICBIR KOSULDA atlanmaz -- bu kontrol en BASTA olmali.
  # B1b (leakage-auditor, final): bu satir eskiden ONLY/FROM kontrollerinden SONRAYDI,
  # yani `--from pretrain` ve `--only cache` denetimi ATLIYORDU. Olculdu:
  # `--dry-run --from pretrain` ciktisinda step_audit HIC gorunmuyordu.
  # Senaryo: audit kirmizi -> operator `--from cache` ile devam eder -> denetim
  # atlanir -> on egitim denetlenmemis korpusla kosar. Saniyeler suruyor, idempotent.
  if [ "$s" = "audit" ]; then return 0; fi
  if [ -n "$ONLY" ]; then [ "$s" = "$ONLY" ] && return 0 || return 1; fi
  if [ "$skip_until_reached" = 1 ]; then
    [ "$s" = "$FROM" ] && skip_until_reached=0 || return 1
  fi
  # B11: damga yalniz dosya varligina bakiyordu -> girdi parmak izi yazildi.
  if [ -f "$STATE/$s.done" ]; then
    local want got
    want="$(_fingerprint)"
    got="$(cat "$STATE/$s.done" 2>/dev/null || true)"
    if [ "$want" = "$got" ]; then
      echo "  ($s zaten bitmis -- atlaniyor; tekrar icin: rm $STATE/$s.done)"
      return 1
    fi
    echo "  ($s daha once bitmisti ama GIRDI DEGISTI -> yeniden kosuluyor)"
  fi
  return 0
}

_fingerprint() {  # korpus yaml + split + egitim konfigi + git HEAD
  {
    cat "$CORPUS" 2>/dev/null
    cat "$TRAIN_CFG" 2>/dev/null
    cat outputs/splits.csv 2>/dev/null
    git rev-parse HEAD 2>/dev/null
  } | shasum -a 256 2>/dev/null | cut -d" " -f1 || echo "nofingerprint"
}

run_step() {  # $1 = ad, geri kalani komut
  local name="$1"; shift
  should_run "$name" || return 0
  log "$name"
  if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; return 0; fi
  local t0 rc
  t0=$(date +%s)
  set +e
  # B10 (leakage-auditor): alt kabuk `errexit`'i KAPALI devraliyordu -> adimin ilk
  # komutu cokse bile ikincisi kosuyor ve rc=0 damgalaniyordu. Somut sonucu:
  # step_manifest'in denetim komutu CorpusError ile cokse bile checksum doldurucu
  # BAYAT manifest uzerinde calisip adimi "OK" yapabiliyordu. `set -e` geri verildi.
  ( set -e; "$@" ) 2>&1 | tee "$LOGS/$name.log"
  rc=${PIPESTATUS[0]}
  set -e
  local dt=$(( $(date +%s) - t0 ))
  if [ "$rc" -ne 0 ]; then
    printf '%s\tKIRMIZI\t%s\t%s\n' "$name" "$dt" "bkz $LOGS/$name.log" >> "$STEPS"
    die "$name basarisiz (${dt}sn). Log: $LOGS/$name.log"
  fi
  printf '%s\tOK\t%s\t\n' "$name" "$dt" >> "$STEPS"
  _fingerprint > "$STATE/$name.done"
  echo "  -> $name bitti (${dt} sn)"
}

# ======================================================================= adimlar

step_env() {
  echo "host: $(hostname) | $(uname -a)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  else
    warn "nvidia-smi yok -- GPU'suz makinedesin. Sadece CPU duman testi anlamli."
  fi
  df -h . | tail -1
  echo "git HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
}

step_deps() {
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
  # GPU makinesinde torch'u CUDA'li kur. mamba_ssm DERLENMEZ: saf PyTorch yedegi
  # varsayilan (derleme 20+ dk surup basarisiz olabiliyor, kritik yolda olamaz).
  if command -v nvidia-smi >/dev/null 2>&1; then
    python - <<'PY' || pip install -q "torch>=2.4" --index-url https://download.pytorch.org/whl/cu121
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  fi
  python -c "import torch, numpy; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| numpy', numpy.__version__)"
  pytest -q -x || die "test paketi kirmizi -- once onu duzelt, egitim baslatma"
}

# Veri indirme. Her kaynak AYRI fonksiyon; `enabled: true` olmayan kaynak indirilmez.
# Boyutlar EGITIM-PLANI §1'de, 21 Eyl 2026'da birincil kaynaktan dogrulandi.
step_data() {
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
  mkdir -p data/raw
  python - "$CORPUS" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
on = [s for s in c.get("sources", []) if s.get("enabled")]
print(f"[data] enabled kaynak: {len(on)}")
for s in on:
    n, b = s.get("n_expected"), s.get("n_bytes_expected")
    print(f"  {s['name']:22s} n={n} bytes={b} url={s.get('url','')}")
    if s.get("reason_blocked"):
        raise SystemExit(f"HATA: {s['name']} hem enabled hem reason_blocked -- celiski")
if not on:
    print("[data] enabled kaynak yok -> indirme yok.")
PY
  # --- INDIRME HIZI OLCUMU (ONCE BU) ---
  # 21 Eyl 2026 YEREL olcum: Intel Mac 0.9 MB/s -> AMOS tek basina 7.8 saat.
  # YANI: Katman B yerel makinede INDIRILEMEZ. Lambda hizi OLCULMEDI, asagisi olcer.
  # DIKKAT: Zenodo HTTP Range'i YOK SAYIYOR (206 degil 200 donuyor) -> kismi indirme
  # ve guvenilir resume YOK; yarida kesilen indirme bastan baslar.
  if [ "${SKIP_SPEED:-0}" != "1" ]; then
    echo "[data] indirme hizi olculuyor (30 sn)..."
    local t0 sz dt
    t0=$(date +%s)
    timeout 30 curl -s -o "$OUT/speed_probe.bin" \
      https://zenodo.org/records/7155725/files/amos22.zip || true
    dt=$(( $(date +%s) - t0 ))
    sz=$(stat -f%z "$OUT/speed_probe.bin" 2>/dev/null || stat -c%s "$OUT/speed_probe.bin" 2>/dev/null || echo 0)
    rm -f "$OUT/speed_probe.bin"
    python - "$sz" "$dt" <<'PYSPEED'
import sys
b, dt = float(sys.argv[1]), max(1.0, float(sys.argv[2]))
mbs = b / dt / 1e6
print(f"[data] olculen hiz: {mbs:.1f} MB/s")
for name, gb in [("AMOS 24.2GB", 24.2), ("TS-v2 23.6GB", 23.6), ("KiTS23 42.1GB", 42.1)]:
    print(f"       {name:16s} -> {gb * 1000 / max(mbs, 0.01) / 60:.0f} dk")
if mbs < 10:
    print("[UYARI] <10 MB/s: Katman B'yi kirp (EGITIM-PLANI §8). Once AMOS, KiTS23 en son.")
PYSPEED
  fi

  # --- KAYNAK BASINA INDIRME ---
  # Sira ONEMLI: once kucuk ve hizli olan (AMOS 24 GB), en son buyuk (KiTS23 42 GB).
  # Yavas indirmede EGITIM-PLANI §8: Katman B kirpilir, ana kol beklemez.
  #
  # AMOS22 (Zenodo 7155725, 24.2 GB, CC BY 4.0)
  #   aria2c -x8 -s8 -d data/raw -o amos22.zip \
  #     https://zenodo.org/records/7155725/files/amos22.zip
  #   unzip -q data/raw/amos22.zip -d data/raw/amos22
  #
  # TotalSegmentator v2 (Zenodo 10047292, 23.58 GB, CC BY 4.0)
  #   aria2c -x8 -s8 -d data/raw -o ts_v2.zip \
  #     https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip
  #
  # KiTS23 (resmi HF, 42.1 GB, CC BY-NC-SA 4.0) -- Kaggle aynalarinin cogu KiTS19, KULLANMA
  #   pip install -q huggingface_hub && \
  #   python -c "from huggingface_hub import snapshot_download; \
  #     snapshot_download('neheller/KiTS-Challenge-Imaging', repo_type='dataset', \
  #     local_dir='data/raw/kits23', allow_patterns=['images/*'])"
  #
  # LiTS: Kaggle'da hazir (andrewmvd/liver-tumor-segmentation + -part-2, 53.6 GB)
  #   kaggle datasets download -d andrewmvd/liver-tumor-segmentation -p data/raw/LiTS --unzip
  #
  # ⛔ ASLA: FLARE22/23, AbdomenAtlas, CT-ORG, WORD-RAOS, HCC-TACE-Seg (korpus icin)
  #    Gerekce: docs/leakage-map.md L3/L7/L8/L9 + configs/pretrain_corpus.yaml reason_blocked
  warn "Indirme komutlari BILEREK yorumda. Bir kaynagi acmadan once:"
  warn "  1) configs/pretrain_corpus.yaml -> enabled: true  2) lisans/erisim teyidi"
  warn "  3) ilgili satiri bu fonksiyonda yorumdan cikar"
}

step_frozen() {
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
  # B3 (leakage-auditor): make_splits "test" split'i URETMIYOR ve gercek degerlendirme
  # seg.make_folds'tan geliyor -> on egitim korpusu degerlendirme vakalarini gorebiliyordu.
  # Cozum: TEK dondurulmus ic test kumesi; on egitim + CV geri kalanda.
  # (Kat basina ayri on egitim 5x maliyet demekti, butceye sigmiyor -- EGITIM-PLANI §5.)
  if [ -f "${EVAL_IDS:-outputs/internal_test.json}" ]; then
    echo "[frozen] zaten var: ${EVAL_IDS:-outputs/internal_test.json} (DONDURULMUS, yeniden uretilmez)"
    return 0
  fi
  python -m src.data.frozen_test \
    --ids-glob 'data/raw/LiTS/segmentation-*.nii' \
    --out "${EVAL_IDS:-outputs/internal_test.json}"
}

step_manifest() {
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
  local EX=""
  [ -f "${EVAL_IDS:-outputs/internal_test.json}" ] && \
    EX="--exclude-ids ${EVAL_IDS:-outputs/internal_test.json}"
  # 🔴 B9: make_splits K3'u uyguluyor (LiTS 28-47 -> excluded_ircadb_overlap) ama korpus
  # yaml'i LiTS'in 131'ini aliyor -> beyaz liste kontrolu HER KOSUDA kirmizi yanardi.
  # Cozum assert'i kismak DEGIL, veriyi dusurmek.
  EX="$EX --exclude-splits ${EXCLUDE_SPLITS:-excluded_ircadb_overlap,external_test}"
  # shellcheck disable=SC2086
  python -m src.data.pretrain_manifest \
    --corpus "$CORPUS" --paths configs/paths.yaml \
    --splits outputs/splits.csv $EX \
    --out outputs/pretrain_manifest.csv --audit-out outputs/split_audit.json
  # Icerik hash'i: bu gece 'pending' yazildi (kural 18), burada dolduruluyor.
  python - <<'PY'
import hashlib, pathlib, pandas as pd
p = pathlib.Path("outputs/pretrain_manifest.csv")
m = pd.read_csv(p)
todo = m.index[m["checksum"].astype(str) == "pending"]
print(f"[manifest] {len(todo)} hacmin sha256'si hesaplanacak")
for i in todo:
    f = pathlib.Path(str(m.at[i, "path"]))
    if not f.is_file():
        m.at[i, "checksum"] = "MISSING"
        continue
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    m.at[i, "checksum"] = h.hexdigest()
missing = int((m["checksum"] == "MISSING").sum())
m.to_csv(p, index=False)
print(f"[manifest] tamam. eksik dosya: {missing}")
if missing:
    raise SystemExit(f"{missing} manifest satirinin dosyasi yok -- korpus eksik, duzelt")
PY
}

step_audit() {
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
  # 🔴 Uc ayri kapi, ucu de acik olmali:
  #   --require-audited        bos kumede gecen denetim DENETLENMEMISTIR
  #   --require-sources-covered korpus kaynagi split'te hic gorunmuyorsa denetim KOR
  #   --eval-ids               make_splits "test" split'i URETMIYOR; gercek degerlendirme
  #                            katlari seg.make_folds'tan gelir -> bu dosya olmadan
  #                            pretrain∩ic_test kontrolu kordur (B3)
  # --require-content BILEREK burada yok: checksum'lar `manifest` adiminda dolduruldu,
  # ama icerik dedup'i (dedup_hash) henuz kosulmadi. Onu kosunca ekle.
  local EVAL_IDS_ARG=""
  if [ -f "${EVAL_IDS:-outputs/internal_test.json}" ]; then
    EVAL_IDS_ARG="--eval-ids ${EVAL_IDS:-outputs/internal_test.json}"
  else
    warn "outputs/internal_test.json YOK -> 'pretrain∩ic_test' DENETLENMEYECEK."
    warn "  make_splits 'test' split'i uretmiyor; degerlendirme katlari seg.make_folds'tan geliyor."
    warn "  Dondurulmus ic test kumesi olmadan 'on egitim degerlendirme vakalarini gormedi' DENEMEZ."
  fi
  # 🔴 B1a + B14 (leakage-auditor, final): bu adim eskiden URETIM manifestine
  # (`outputs/pretrain_manifest.csv`) yaziyordu ve `--exclude-ids` VERMIYORDU. Iki sonucu:
  #   (1) `step_manifest`'in dusurdugu dondurulmus test vakalarini diske GERI YAZIYORDU;
  #   (2) `step_manifest`'in hesapladigi sha256'lari `pending`'e cevirip `audited_level`i
  #       kalici olarak `name_only` yapiyordu -- hash emegi cope gidiyordu.
  # Artik AYRI bir dosyaya yaziyor ve uretim manifestine DOKUNMUYOR.
  local EX2=""
  [ -f "${EVAL_IDS:-outputs/internal_test.json}" ] && \
    EX2="--exclude-ids ${EVAL_IDS:-outputs/internal_test.json}"
  EX2="$EX2 --exclude-splits ${EXCLUDE_SPLITS:-excluded_ircadb_overlap,external_test}"
  # shellcheck disable=SC2086
  python -m src.data.pretrain_manifest \
    --corpus "$CORPUS" --paths configs/paths.yaml --splits outputs/splits.csv \
    --out "$OUT/audit_manifest.csv" --audit-out outputs/split_audit.json \
    --require-audited --require-sources-covered $EX2 $EVAL_IDS_ARG \
    || die "KURAL 10: korpus denetimi kirmizi ya da denetlenmemis. EGITIM BASLAMAZ.
   Bak: outputs/split_audit.json -> checks.*.examples
   'external_test' bossa once HCC-TACE-Seg provenance'a eklenmeli (EGITIM-PLANI §9.5).
   Assert'i DEVRE DISI BIRAKMA -- veriyi duzelt."
  echo "[audit] ✅ dort kesisim de bos VE bos kume uzerinde gecmedi"
}

step_cache() {
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
  mkdir -p "$CACHE"
  # B13 (leakage-auditor, final): burada `--labels-glob` YOKTU -> npz'lerde `lab` olmaz ->
  # `finetune_e2e.LabeledSampler` firlatir. Ayrica `|| warn` cache_build'in CorpusError'ini
  # (eksik korpus) yutuyordu. Ikisi de duzeltildi.
  #
  # ⚠️ ONBELLEK IKI AYRI KUME ICIN KURULUYOR:
  #   (a) on egitim korpusu  -> outputs/pretrain_manifest.csv (dondurulmus test DUSURULMUS)
  #   (b) etiketli tum vakalar -> ince ayar + dondurulmus test tahmini icin
  # (b) olmadan `finetune_e2e.split_ids` "test vakasinin onbellegi yok" diye firlatir.
  python -m src.jepa.cache_build \
    --manifest outputs/pretrain_manifest.csv --config "$TRAIN_CFG" --out "$CACHE" \
    --labels-glob 'data/raw/LiTS/segmentation-*.nii'
  # 🔴 Dondurulmus test onbellegi ROI'SIZ kurulur: ROI kutusu GT etiketinden turetilirse
  # tahmin GT anatomisine hapsolur ve model kutu disinda YANLIS POZITIF URETEMEZ ->
  # precision / marj ihlali / ortusme yukari yanli olur ve TS tabaniyla (tum hacim)
  # KARSILASTIRILAMAZ. (leakage-auditor final, B5)
  if [ -f "${EVAL_IDS:-outputs/internal_test.json}" ]; then
    python -m src.jepa.cache_build \
      --manifest "$OUT/manifest_all.csv" --config "$TRAIN_CFG" --out "$CACHE" \
      --labels-glob 'data/raw/LiTS/segmentation-*.nii' \
      || warn "etiketli onbellek: $OUT/manifest_all.csv yok, ince ayar adimi duser"
  fi
  du -sh "$CACHE" 2>/dev/null || true
}

step_probe() {
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
  # GPU gununun ILK olcumu: EGITIM-PLANI §2'deki VRAM/sure TAHMINLERINI olcumle degistir.
  python -m src.jepa.encoder --probe --config "$TRAIN_CFG" \
    | tee "$OUT/probe.json"
  echo "[probe] Tahminden >2x sapma varsa DUR ve profil/batch yeniden sec (EGITIM-PLANI §8)"
}

_profile() {
  if [ "$PROFILE" != "auto" ]; then echo "$PROFILE"; return; fi
  python - "$GPU_HOURS" <<'PY'
import sys, pandas as pd
from src.jepa.encoder import pick_config
n = len(pd.read_csv("outputs/pretrain_manifest.csv"))
print(pick_config(n, float(sys.argv[1])))
PY
}

step_pretrain() {
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
  local prof; prof="$(_profile)"
  echo "[pretrain] profil: $prof (korpus $(wc -l < outputs/pretrain_manifest.csv) satir)"
  for obj in $ARMS; do
    [ "$obj" = "none" ] && { echo "  none: on egitim yok (rastgele baslatma)"; continue; }
    local d="$OUT/pre_$obj"
    if [ -f "$d/encoder.pt" ]; then echo "  $obj zaten var -> $d/encoder.pt"; continue; fi
    # 🔴 --eval-ids ZORUNLU: bu bayrak olmadan `pretrain` yalnizca UYARI basip devam
    # ediyor (B1c/B10). Dosya yoksa adim DURUR -- sessizce denetimsiz egitim yok.
    [ -f "${EVAL_IDS:-outputs/internal_test.json}" ] \
      || die "outputs/internal_test.json yok -- once `frozen` adimi. --eval-ids olmadan
   on egitimin degerlendirme vakalarini gormedigi DOGRULANMAZ (kural 10)."
    python -m src.jepa.pretrain --objective "$obj" --config "$TRAIN_CFG" --profile "$prof" \
      --manifest outputs/pretrain_manifest.csv --cache "$CACHE" \
      --eval-ids "${EVAL_IDS:-outputs/internal_test.json}" \
      --seed "$SEED" --out "$d" --resume
    [ -f "$d/collapse.json" ] && warn "$obj: COKUS saptandi -> $d/collapse.json (EGITIM-PLANI §8)"
  done
}

step_finetune() {
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
  local prof; prof="$(_profile)"
  for fold in $FOLDS; do
    for arm in $ARMS; do
      local d="$OUT/ft_${arm}_f${fold}"
      [ -f "$d/done" ] && { echo "  $arm kat $fold zaten bitti"; continue; }
      local init="$OUT/pre_$arm/encoder.pt"
      [ "$arm" = "none" ] && init="random"
      # B13: burada `--fold` cagriliyordu ama boyle bir bayrak YOK; `--cache` ve
      # `--test-ids` ZORUNLU. Yani Lambda hatti uctan uca HIC kosmamisti.
      python -m src.jepa.finetune_e2e --init "$init" --config "$TRAIN_CFG" --profile "$prof" \
        --cache "$CACHE" --test-ids "${EVAL_IDS:-outputs/internal_test.json}" \
        --seed "$SEED" --out "$d" --resume
      touch "$d/done"
    done
  done
}

step_evaluate() {
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
  for arm in $ARMS; do
    for fold in $FOLDS; do
      local d="$OUT/ft_${arm}_f${fold}"
      [ -d "$d/pred" ] || { warn "$d/pred yok, atlaniyor"; continue; }
      # B10 (leakage-auditor): burada `evaluate --gt/--pred` cagriliyordu ama
      # src/metrics/evaluate.py `--pairs` ISTIYOR (required) -> argparse exit 2.
      # Adim yine de "OK" gorunuyordu, yani degerlendirme HIC KOSMADAN bitmis sayiliyordu.
      mkdir -p "$d/eval"
      python -m src.metrics.make_pairs \
        --gt-glob 'data/raw/LiTS/segmentation-*.nii' \
        --pred-glob "$d/pred/*.nii.gz" --out "$d/eval/pairs.csv"
      # LiTS GT tumor etiketi 2, bizim tahminimiz ikili (1) -- etiketler AYRI verilir
      python -m src.metrics.evaluate --pairs "$d/eval/pairs.csv" --out "$d/eval" \
        --gt-label 2 --pred-label 1
      python -m src.metrics.bootstrap --eval "$d/eval" --n-boot 2000
      python -m src.metrics.margin_sweep --pairs "$d/eval/pairs.csv" --out "$d/eval" \
        --gt-label 2 --pred-label 1 || warn "margin_sweep $arm kat $fold: atlandi"
    done
  done
}

step_report() {
  {
    echo "# LiverJEPA kosu ozeti"
    echo; echo "git: $(git rev-parse --short HEAD) | $(date -u +%FT%TZ)"
    echo; echo "## Adim sureleri"; cat "$STEPS"
    echo; echo "## Korpus denetimi"
    python - <<'PY'
import json, pathlib
p = pathlib.Path("outputs/split_audit.json")
if p.exists():
    r = json.loads(p.read_text())
    print(f"clean={r['clean']} audited={r['audited']} hacim={r['corpus']['n_volumes']}")
    for k, v in r["checks"].items():
        print(f"  {k}: n={v['n']} vacuous={v['vacuous']}")
else:
    print("split_audit.json YOK")
PY
    echo; echo "## Konfig kopyasi"; echo '```'; cat "$TRAIN_CFG" 2>/dev/null; echo '```'
  } > "$OUT/REPORT.md"
  echo "-> $OUT/REPORT.md"
  cat "$STEPS"
}

# ============================================================================ ana
log "LiverJEPA bootstrap | OUT=$OUT | profil=$PROFILE | katlar=[$FOLDS] | kollar=[$ARMS]"
[ -n "$ONLY" ] && echo "(yalniz '$ONLY' adimi)"
[ -n "$FROM" ] && echo "('$FROM' adimindan itibaren)"

for s in $ALL_STEPS; do
  run_step "$s" "step_$s"
done

log "BITTI. Ozet: $OUT/REPORT.md | Adimlar: $STEPS"
