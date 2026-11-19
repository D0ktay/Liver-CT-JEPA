# İkinci koşu planı — kanıtlı değişiklik listesi

Yazan: liver-ct-b07 (+ liver-ct-13 inceleme/literatür) · 22 Eyl 2026 · Durum: **22 Eyl gece koşusu (v1 tarif) sürerken yazıldı.**
Kural: v1 koşusunun üç kolu (none / jepa / mae) AYNI tarifle biter; aşağıdakilerin hepsi kodda **varsayılan kapalı**.
Her madde: ne · neden (ölçüm/kaynak) · anahtar · durum.

## A. İnce ayar

| # | Değişiklik | Kanıt | Anahtar | Durum |
|---|---|---|---|---|
| A1 | liver_only vakada tümör kanalı karaciğer maskesinin N mm **dışında** negatif denetlenir | v1'de tümör kanalının uzak negatifi yalnız 96 LiTS hacminden; val'de karaciğer-dışı FP payı 0.09–0.42 arası oynadı. Deney (20k→25k): ham karaciğer Dice 0.707→**0.817**, tümör farkı gürültü içinde | `finetune.liver_only_tumor_neg_margin_mm: 15` | kod+test ✅ (`region.voxel_valid`) |
| A2 | liver_only maskede hacim düzeyinde 3B delik doldurma | TS'te ≤3.3 ml kapalı delik (kist) ölçüldü; AMOS 300 + TS 230 + 191 örnek: ≥4 ml delik 0 | `finetune.liver_only_fill_holes: true` | kod+test ✅ |
| A3 | Yama oranı tümör/karaciğer/rastgele .4/.3/.3 → .3/.2/.5 | nnU-Net 0.33 ön plan (arXiv:1904.08128; batch 2'de fiilî %50). v1'de FP'nin ~%95'i GT'den >20 mm uzakta (dalak, kalp, psoas, pelvis) | `finetune.p_tumor/p_liver` | yalnız config |
| A4 | TS-v2 liver_only önbelleği **kırpılmadan** | v1 önbelleği 60 mm ROI kırpık → ~920 hacimde pelvis/gluteal yok | `data/cache/liveronly_tsfull` (Lambda; 917 yazıldı, QC 842 geçti, 20 GB) | önbellek hazır ✅; runner'a bağlanacak |
| A5 | WAW-TACE (216 vaka, tumor_only, Varşova) | liver-ct-13 hazırladı; medyan tümör 50 cm³ (büyük HCC) → ≤2 cm'e katkı beklenmez; `complete=False` 7 vaka | `complete=False` 7 vaka manifestten DIŞLANDI (kod değişikliği gerekmez) | önbellek hazır ✅: Lambda `data/cache/wawtace` 209 hacim, QC 0 ret, 4.4 GB (`outputs/night/manifest_wawtace_complete.csv`); runner'a `--extra-cache` olarak bağlanacak |

## B. JEPA ön eğitimi

| # | Değişiklik | Kanıt | Anahtar | Durum |
|---|---|---|---|---|
| B1 | Conv bloklarından sonra **yeniden maskeleme** (seyrek conv eşdeğeri) | ConvNeXt V2 (arXiv:2301.00808) 79.3→83.7; SparK (arXiv:2301.03580): doldurma ≈ ön eğitimsiz. Bizde 4000 adımlık A/B: bkz. `docs/oturum-gunlugu.md` | `pretrain.remask_conv: true` | **v1 gece koşusunun pre_jepa koluna ALINDI** (22 Eyl 00:05, başlamadan önce; günlük EK-3) |
| B1-b | s8/s16 transformer aşamalarında maskeli token'ları diziden **düşürme** (attention-mask değil) | HySparK (arXiv:2408.05815) Tablo 5: ön eğitimsiz 78.58 · yalnız conv tutarlılığı 79.47 · conv-sparse + transformer-drop 80.67; I-JEPA encoder yalnız görünür yamaları işler. Bizde s16'nın TÜM token'ları predictor'a gidiyor; düşürme bunu da I-JEPA ile hizalar | `pretrain.drop_tokens: true` | kod+test ✅; **v1 gece koşusunun pre_jepa koluna ALINDI** (22 Eyl 01:20; sonda: günlük EK-4) |
| B2 | Aşağı akışta **EMA (hedef) encoder** | I-JEPA Ek: "We use the target-encoder for evaluation" | `encoder.pt["target_encoder"]` artık kaydediliyor; ince ayar için `load_encoder` anahtar seçimi GEREKLİ | kayıt ✅, yükleme yok |
| B3 | Örnek başına ayrı maske (v1: parti başına tek maske) | I-JEPA örnek başına; çeşitlilik 8× | — | kod yok. drop_tokens ile birlikte örnek başına değişken dizi uzunluğu → dolgu + SDPA anahtar maskesi + hedef/sorgu dolgusu gerekir (objective.step, _TransformerStage, VICReg seçimi). Kanıt yalnız 'I-JEPA böyle yapıyor'; 8 farklı görüntü zaten çeşitlilik veriyor → düşük öncelik |
| B4 | Ön eğitimde çevrimiçi encoder DropPath kapalı | I-JEPA'da yok; deterministik hedefe gürültülü tahmin | — | kod yok |
| B5 | sincos en kısa dalga boyu ≥ 2×token aralığı | s4 merkezlerinde üst kanallar sabit (sin(4πk+2π)) | — | kod yok, düşük öncelik |

## C. Beklenti (abstract çerçevesi)

CT'de güçlü tabana karşı SSL kazancı küçük: HySparK BTCV +0.4 (Wald ve ark. Ek Tablo 11), OpenMind ResEnc-L +0.44 DSC.
ft_jepa − ft_none farkının GA'sı büyük olasılıkla sıfırı kapsar → iddia Dice kazancı DEĞİL; denetlenebilir korpus + hedefleme
metriği + eşleştirilmiş fark GA'sı (`src/metrics/compare.py`). JEPA fayda göstermezse rapor cümlesi: "yoğun conv + girdi
doldurmalı JEPA" (B1 uygulanmadan genelleme yapılmaz).

## D. Literatür taraması (22 Eyl, dataset-verifier; birincil kaynaklı) — 4 madde için kanıt

Etiket: [LiTS] = LiTS'te ölçüm · [başka] = başka veri seti · [—] = birincil kaynak bulunamadı.

### D1. Yanlış pozitif azaltma (bizde 210 FP / 95 TP) — en ucuz kazanç
- Bileşen bazlı RF sınıflayıcı (şekil/yoğunluk/karaciğer sınırına uzaklık): [LiTS] FP/vaka **4.6→0.7**, recall 0.72→0.63, Dice/vaka 0.51→0.58 (Chlebus 2018, doi:10.1038/s41598-018-33860-7). Eğitimsiz değil ama ucuz; **test dışı vakalarla** eğitilir.
- Maks. olasılık < 0.8 bileşeni at (Han 2017, arXiv:1704.07239; ISBI birincisi) — ablasyon sayısı yok. Bizim val taramamız (EK-1): filtreler precision 0.76→0.84, 6 lezyon kaybı, Dice değişmedi.
- TS `liver_lesions` kuralı: min hacim 0.06 ml + vaka başına ≤15 lezyon; LiTS'te FP/vaka 0.73, PPV %80 (doi:10.1007/s10278-025-01716-y).
- ⚠️ Karaciğer maskesini dilate etmek "yalnız kötüleştirdi" (arXiv:1905.03639) → bizim **3 mm dilatasyon** post-proc'u val'de ablasyonla sınanmalı.
- ⚠️ Bağlam: ikinci uzman LiTS'te 78 etiketsiz lezyon buldu (Chlebus Tablo 1); okuyucular arası duyarlılık %64.7 (TS makalesi) → FP'lerin bir kısmı gerçek olabilir; abstract'ta "referansla eşleşmeyen bileşen" dili.
- **Kural:** filtre seçimi YALNIZ val/eğitim vakalarında; 27 test bir kez.

### D2. Ensemble — +2–3 tümör Dice, kalibrasyon kazancı
- [LiTS] nnU-Net konfigürasyon ensemble'ı: tümör Dice 0.637→0.656 / 0.625→0.654 (arXiv:1904.08128 Tablo F.6/F.28).
- [başka] Aynı mimari farklı başlatma, M≥5: Dice +3–7, ECE %13→%4 (Mehrtash 2020, arXiv:1911.13273). Karşı örnek: BraTS-PEDs 2024 lezyon-bazlı Dice düştü (arXiv:2410.14020).
- MAE/JEPA/sıfırdan ensemble'ı literatürde YOK → kendi ölçümümüz ilk olur. Araç hazır: `src/jepa/ensemble.py` (last.pt, eşit ağırlık).
- LiTS 5-kat SD %1.7–6.6 → +2 Dice tek koşuda gürültüden ayrılmaz (kural 8).

### D3. Küçük lezyon (≤10 mm recall 0.21; nnU-Net LiTS küçük F1 0.239 — aynı bant)
- **blob loss** (Dice bileşen başına): [LiTS] instance sensitivity 0.801→0.861, F1 0.623→0.657, DSC ≈ aynı (Kofler arXiv:2205.08209; α/β makale içi çelişkili).
- Tversky (β_FP 0.7): [LiTS test] vaka Dice 0.45→0.57 (arXiv:1905.03639); [MS MR] küçük recall 0.80→0.83, F1 bedeli (arXiv:2604.08015).
- Çözünürlük: nnU-Net LiTS hedef spacing **1×0.77×0.77** (izotropik 1 mm değil); 1 mm iso in-plane 1.3× küçültür. Doğrudan ablasyon [—].
- Deep-supervision ağırlığı / boyut-katmanlı oversampling ablasyonu [—].
- Dalak yutma: nnU-Net cascade gerekçesi tam bu ("spleen vs liver hard when zoomed in"); 112³@1 mm = 11 cm bağlam. v2 tarifi + kırpılmamış TS daha ucuz çözüm; lowres cascade alternatif.
- **>20 mm recall 0.77 düşük** (nnU-Net F1 0.915) → büyük lezyon kaçırma ayrıca incelenmeli (val'de).

### D4. WAW-TACE ekleme
- LiTS+WAW-TACE birlikte eğitim raporu [—]. Tek-set nnU-Net: LiTS 0.688 · HCC-TACE 0.761 · WAW-TACE 0.782 (74 vaka) (doi:10.1038/s41598-025-26019-8) → WAW en kolay, küçük lezyona katkı beklenmez.
- Domain kayması: HCC-TACE ile eğitilen 2D nnU-Net dışta Dice 0.24 (PMC13374201). Örnekleme ağırlığı ∝ 1/√n (MultiTalent arXiv:2303.14444).
- Maskelerin hangi fazda çizildiği makalede yok [—] (liver-ct-13: faz kodları doğrulanamadı).
- Öneri: 1/√n ağırlık + PV faz eşleştirme + ayrı satır; ana kazanç beklentisi düşük.

### D5. CLAUDE.md "Taşıyıcı sayılar" için düzeltme adayları (claim-checker'a; CLAUDE.md liver-ct-13'te)
1. "LiTS birincisi precision 0.156 (14.)" → Bilic Tablo 7: Zou (Dice 0.702) precision **0.148, 13/15**; 0.156 C. Wang'ın.
2. RadGPT ≤2 cm %39.6 → **hasta/organ düzeyi** LLM-etiket duyarlılığı (AA3 IID), per-lezyon değil; bizim recall'la kıyaslanamaz.
3. "LiTS medyan tümör hacmi 2.49 cm³" → Bilic Tablo 3 vaka başına medyan 16.11 (eğitim) / 34.78 (test) cm³; per-lezyon 2.49 kaynağı bulunamadı.

### Önerilen sıra (ikinci koşu)
1. Val'de FP filtresi ablasyonu (maks-olasılık, min hacim, 3 mm dilatasyon kaldır) → precision/recall eğrisi; test bir kez.
2. none+jepa(+mae) ensemble (hazır araç), SD ile birlikte.
3. v2 tarifi + kırpılmamış TS (dalak sorunu).
4. blob loss / Tversky ile yeniden eğitim (küçük recall), 1×0.77×0.77 spacing denemesi.
5. WAW-TACE 1/√n ağırlıklı, ayrı satır.
