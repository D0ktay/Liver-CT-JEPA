# Kaggle'da çalıştırma

Neden Kaggle: Mac'te GPU yok, CPU'da TotalSegmentator vaka başına dakikalar sürüyor.
Kaggle ücretsiz GPU (T4, haftada ~30 saat) veriyor ve LiTS zaten orada, indirme yok.

Not defteri: `notebooks/kaggle_lits_baseline.ipynb`

## Veri

| Kaynak | Kaggle / URL | İçerik | Lisans |
|---|---|---|---|
| LiTS parça 1 | `andrewmvd/liver-tumor-segmentation` | segmentation-0..130, volume-0..50 · 30.9 GB | CC BY-NC-ND 4.0 |
| LiTS parça 2 | `andrewmvd/liver-tumor-segmentation-part-2` | volume-51..130 · 22.7 GB | CC BY-NC-ND 4.0 |
| 3D-IRCADb-01 | not defteri indirir: `cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download` | 20 hasta, DICOM · 782 MB | CC BY-NC-ND 4.0 |

- Toplam 131 volume + 131 segmentation (Kaggle API ile sayıldı, 19 Eyl 2026). Sıkıştırılmamış `.nii`.
- Kaggle'da IRCADb yok. `prathamgrover/3d-liver-segmentation` IRCADb **değil**, yeniden
  örneklenmiş MSD Task03 (= LiTS).
- Kaggle kopyasının resmi LiTS ile bayt bayt aynı olduğu **varsayılmıyor**: not defteri
  SHA256 manifestosu çıkarıyor (`outputs/manifest/lits_kaggle_sha256.csv`).
- NoDerivatives lisansı: türetilmiş maskeleri (TS çıktıları) herkese açık yeniden dağıtma.

## Tek seferlik kurulum

1. **Kaggle hesabı telefonla doğrulanmış olmalı.** Olmazsa GPU ve Internet seçenekleri çıkmıyor.
2. **GitHub token'ı** (repo private):
   GitHub → Settings → Developer settings → Fine-grained tokens → Generate.
   Repository access: yalnızca `D0ktay/Liver-CT-JEPA`. Permissions: **Contents: Read-only**. Süre: kısa tut.
3. Kaggle → Create → **Import Notebook** → `notebooks/kaggle_lits_baseline.ipynb`.
4. Not defterinin sağ panelinde:
   - **Add-ons → Secrets → Add secret**: ad `GITHUB_TOKEN`, değer token. Bu not defterine bağla.
   - **Input → Add Input**: iki LiTS veri setini ekle (yukarıdaki tablo).
   - **Settings → Accelerator: GPU T4 x2** (tek T4 yeter) · **Internet: On**.

## Çalıştırma

1. İlk hücrede `LIMIT = 5` bırak → **Run All**. Beş vakayla uçtan uca doğrula.
2. Sonuçlar makulse `LIMIT = 0` yap → **Save Version → Save & Run All (Commit)**.
   Arka planda koşar (en fazla 12 saat), tarayıcıyı kapatabilirsin.
3. Çıktılar sürümün **Output** sekmesinde: `outputs/` altında.

Önceden hesaplanmış TS çıktısı varsa atlanır. Başka bir oturumdan devam etmek için
önceki sürümün çıktısını Input olarak ekleyip `outputs/baseline/totalsegmentator/` altına kopyala.

## Çıktılar

| Dosya | Ne |
|---|---|
| `outputs/manifest/lits_kaggle_sha256.csv` | 262 dosyanın SHA256'sı |
| `outputs/ircadb_lits_match.csv` | Her IRCADb vakası → en benzer LiTS vakası (yönden bağımsız imza + HU histogramı) |
| `outputs/baseline/totalsegmentator/lits-N.nii.gz` | TS liver_lesions tahminleri (etiket 1) |
| `outputs/pairs.csv`, `outputs/eval/` | Değerlendirme (lezyon-bazlı, hedefleme hatası, marj) |
| `outputs/eval/ci.csv` | Her sayının hasta düzeyi %95 güven aralığı (2000 bootstrap, tohum 0) |
| `outputs/eval/margin_sweep.csv/.json` | Marj ek payı eğrisi: 5–10 mm marjda ihlal oranı + ideal plana göre fazladan yakılan doku |

Mac'e geri alınacak olanlar: CSV'ler ve `outputs/eval/`.

`ci.csv` yalnızca CSV'lerden hesaplanır: koşu bittikten sonra Mac'te de üretilebilir
(`python -m src.metrics.bootstrap --eval outputs/eval`). Marj eğrisi maske ister, Kaggle'da çalışır. Tahmin maskeleri Kaggle'da kalabilir.

## Hızlandırma (hazır, 19 Eyl 2026 akşam, ilk kez sonraki koşuda çalışacak)

İlk 131 vakalık koşu vaka başına ~55 sn sürdü, bunun yalnızca ~22 sn'si GPU'da model.
Ayarlar hücresinde `FAST_TS = True` (varsayılan):

1. **İki GPU paralel:** `src.baseline.run_parallel --gpus 2` vakaları sıra atlamalı ikiye böler
   (`--shard 0/2`, `1/2`), her süreç `CUDA_VISIBLE_DEVICES` ile kendi GPU'sunu görür. Loglar `outputs/logs/ts_gpu{k}.log`.
2. **Tek süreç:** `--engine inproc` TS'in **kendi** CLI `main()`'ini aynı süreçte çağırır: argümanlar CLI ile
   birebir aynı (`ts_argv`), vaka başına Python/torch açılışı yok.
3. **Doğrulama, otomatik:** koşunun başında 3 vaka eski yolla (`ts_check_cli`) ve yeni yolla (`ts_check_fast`)
   hesaplanır, `src.baseline.compare_masks` voksel voksel karşılaştırır. Aynıysa hızlı yol; **farklıysa uyarı basıp
   eski yola döner** (sonuç hiçbir durumda değişmez). Başarılı kontrol `outputs/baseline/ts_speed_check_ok` bırakır.

Beklenen: ~2 saat → ~1 saat. Kaggle çıktısında "3/3 ayni" satırını ve süreleri kontrol et.

## Bilinen sınırlar

- TotalSegmentator'ın eğitim verisi LiTS içeriyorsa LiTS skoru **dış doğrulama değil**
  (CLAUDE.md → doğrulanmayı bekleyen). IRCADb de LiTS 28–47 olduğu için dış test değil.
- Not defteri kodu repodan çeker: pipeline modülleri push edilmeden çalışmaz (ilk hücre bunu kontrol eder).
