# EĞİTİM PLANI — LiverJEPA

> **Tek kaynak doğruluk belgesi.** Ön eğitimden abstract cümlesine kadar her karar burada.
> Yazıldı: 21 Eylül 2026 gece nöbeti (`liver-ct-7a`). Uygulama: `src/jepa/`, `configs/train.yaml`.
> **Ölçülmemiş her sayı "⚠️ tahmin" diye işaretli.** Tahminle ölçüm karıştırılmaz.

---

## §0 Model

**Ad: `LiverJEPA-S`** (S = small; `cok_kucuk`/`kucuk`/`orta` profilleri var, `buyuk` eğitilmiyor).

Ağırlıkları bizim eğittiğimiz, mimarisini bizim seçtiğimiz, ön eğitim korpusunu bizim
derlediğimiz 3D CT temsil modeli. İki kendi-kendine-denetimli hedef **aynı kod yolundan**
seçilebiliyor (`--objective jepa|mae|none`), böylece "JEPA mı işe yaradı, yoksa sadece ön eğitim mi"
sorusu kontrollü olarak yanıtlanabiliyor. Hazır ağırlık (Rad-JEPA 3D, OpenMind-MAE) yalnızca
**karşılaştırma kolu**; ana kolda indirilmiş checkpoint yok.

### Neden kendi modelimiz — bu, projenin ikinci ve muhtemelen daha güçlü iddiası

Kamu abdomen CT setlerinde vaka tekrarı **%59.1** (46 set / 50.256 çalışma, arXiv:2508.13626).
Hazır bir encoder kullandığımız anda o encoder'ın ön eğitim verisini denetleyemeyiz — ve
denetleyemediğimiz için **kendi sızıntı iddiamızı kendimiz çürütürüz.** Somut örnekler:

| Hazır model | Ön eğitiminde | Sonuç |
|---|---|---|
| CT-FM | hcc_tace_seg 477 + tcga_lihc 284 tarama | dış testimizle (HCC-TACE-Seg) çakışır |
| VoCo-10k | FLARE23 (⊃ HCC-TACE-Seg) | aynı zincir |
| SuPreM (supervised) | AbdomenAtlas 1.1 ⊃ LiTS | LiTS testi kirli |
| Rad-JEPA 3D | M3D-Cap ~120k CT (Radiopaedia) | ad düzeyinde temiz, **içerik düzeyinde denetlenemez** |

Kendi modelimizde tersi geçerli: **ön eğitim korpusunu vaka düzeyinde biz biliyoruz ve
test setlerinden ayrıklığını denetleyip raporluyoruz** (`outputs/pretrain_manifest.csv` +
`outputs/split_audit.json`). Bildiğimiz kadarıyla 3D CT temel modellerinde yaygın olarak raporlanmıyor (tarama referansı yok).
"Küçük veriyle eğittik" bir zayıflık değil, **denetlenebilir olmanın bedeli.**

⚠️ **Bugünkü kanıtın seviyesi:** isim/yol düzeyinde (`audited_level: name_only`). Tüm
checksum'lar `pending`, içerik dedup'ı koşulmadı. "Ayrıklık **kanıtlandı**" cümlesi ancak
`audited_level: hash_confirmed` olduktan sonra kurulabilir — `--require-content` bunu
zorlar. Bugün kurulabilecek cümle: "ayrıklık isim düzeyinde denetlendi ve raporlandı".

### MVE — Minimum Yaşayabilir Deney (≤8 GPU-saat, Kaggle T4)

Profil `kucuk` (2.75M encoder), yama 64³, korpus = LiTS'in yalnızca eğitim katlarındaki
görüntüleri. Üç kol: `jepa` · `mae` · `none` (rastgele başlatma), aynı tohum / aynı adım /
aynı örnekleme. Amacı bir "sonuç" üretmek değil, **hattın gerçek GPU'da çalıştığını kanıtlamak
ve çöküş eşiklerini kalibre etmek.** ⚠️ Süre tahmini ~1.5 GPU-saat, ölçülmedi.

---

## §1 Ön eğitim korpusu

Yapılandırma: `configs/pretrain_corpus.yaml` · Kod: `src/data/pretrain_manifest.py` (21 test)

### Kural 10 — pazarlığa kapalı

> Test ya da dış test hastasının **hiçbir taraması, etiketsiz bile olsa**, ön eğitim korpusuna
> giremez. Kırmızıysa eğitim başlamaz.

`outputs/split_audit.json` **altı** kontrolü yazar: `eğitim∩doğrulama`, `eğitim∩test`,
`pretrain∩test`, `pretrain∩dış_test`, `pretrain∩eğitim_dışı`, `pretrain∩iç_test`.

**Kimlik eşleştirmesi (düzeltildi, 21 Eyl):** `pretrain_manifest` `resolve_canonical`'ı
çağırmaz — kanonik kimliği aynı *biçimde* üretir. Bu tek başına yetmiyordu:
`resolve_canonical` union-find kökünü **sözlük sırasında küçük olanı** seçtiği için, dedup
onaylı bir LiTS↔AbdomenCT-1K birleşmesinde split'in kimliği `ABDOMENCT_1K::123`, korpusunki
`LITS|MSD_TASK03::38` oluyor ve kesişim **sessizce boş** çıkıyordu. Kesişimin sağ tarafı artık
`canonical_members` ile genişletiliyor (kök keyfi, **üyeler** değil).

**Üç ayrı bayrak, üçü de Lambda'da zorunlu:**
| bayrak | neyi engeller |
|---|---|
| `--require-audited` | boş kümede geçen denetimin "temiz" sayılması |
| `--require-sources-covered` | korpus kaynağının split'te hiç görünmemesi (kesişim yapısal olarak kör) |
| `--eval-ids` + `--exclude-ids` | değerlendirme vakalarının korpusta olması (§3, dondurulmuş iç test) |

**`clean` ≠ `audited` ≠ `audited_level`.** `clean`: ihlal bulunmadı. `audited`: denetim boş küme
üzerinde geçmedi. `audited_level`: `name_only` (bugün — tüm checksum'lar `pending`) ya da
`hash_confirmed`. **Bugün elimizdeki kanıt isim düzeyindedir**; içerik hash'i Lambda'da dolacak.

### İki ayrı savunma, ikisi de gerekli

1. **İçerme grafiği** (`contains:`, geçişli kapanış) — kaynak *adıyla* yakalanan zincir.
   FLARE23 ⊃ HCC-TACE-Seg, AbdomenAtlas ⊃ FLARE23 ⊃ … Korunan kaynağa ulaşan her aday reddedilir.
2. **Açık yasak listesi** (`reason_blocked:`) — adla yakalanamayan örtüşme.
   🚨 **CT-ORG bu yüzden var:** TCIA'nın kendi koleksiyon sayfası hacimlerin çoğunun LiTS'ten
   "ödünç alındığını" yazıyor, ama dosyalar anonim ve LiTS'e eşleştiren tablo yok →
   kanonik kimlikleri `CT_ORG::N` olur, `LITS|MSD_TASK03::N` ile eşleşmez →
   `pretrain∩test` denetimi bir LiTS **test** hastasını CT-ORG içinde **göremez**.
   Denetimin kör olduğu yerde tek doğru davranış kaynağı hiç almamaktır.

### Aday tablo (21 Eyl 2026, `dataset-verifier`, birincil kaynak)

| Kaynak | Vaka | Boyut | Lisans | Durum | Not |
|---|---|---|---|---|---|
| **LiTS** | 131 | 53.6 GB *(Kaggle, açılmış NIfTI; zip ~16.7 GB)* | ⚠️ çelişkili (BY-NC-ND / BY-SA) | ✅ **açık** (Katman A) | Kaggle'da hazır, indirme yok. Dondurulmuş iç test vakaları `--exclude-ids` ile korpustan **düşürülür** (§3) |
| TotalSegmentator v2 | 1228 | 23.58 GB | CC BY 4.0 | 🟡 Katman B | Basel rutin klinik. ⚠️ TS `liver_lesions` eğitim setiyle (Zenodo 20272572, 842) **aynı değil**. Kaçı abdomen FOV — doğrulanmadı |
| AMOS22 | 500 CT | 24.2 GB (CT+MRI) | CC BY 4.0 | 🟡 Katman B | Shenzhen. Kaggle aynası `jvora25/amos-22` boyutu eşleşiyor, iç formatı doğrulanmadı |
| KiTS23 | 489 *(eğitim; toplam 599, 110'u test)* | ⚠️ 42.1 GB doğrulanmadı | CC BY-NC-SA 4.0 | 🟡 Katman B | Resmî HF kaynağı. Kaggle aynalarının çoğu KiTS19 |
| WORD (çekirdek) | 150 *(Med Image Anal 2022 makalesi; README'den teyit edilemedi)* | ⚠️ ölçülmedi | GPL v3.0 | 🟡 Katman B | Form kalktı. **Yalnız çekirdek 150**; genişletilmiş/RAOS yasak (L9) |
| TCIA Pancreas-CT | **80** kullanılıyor *(sayfa 82 diyor; #25 ve #70 = #2'nin aynı taraması)* | 9.95 GB | CC BY 3.0 | 🟡 Katman B | ⚠️ 82 vs 80 tutarsızlığı **primer kaynağın kendi içinde** |
| CHAOS (CT) | 40 | ⚠️ ölçülmedi | ⚠️ belirsiz | 🟡 Katman B | Form gerekir. Sağlıklı donör adayları → LiTS popülasyonuyla örtüşmesi **beklenmez** (hash ile teyit edilmedi) |
| MSD Task07/08/09/10 | 1114 | ⚠️ ölçülmedi | CC BY-SA 4.0 | 🔴 **bloke** | Task08 = LiTS hipotezi **isim/kurum düzeyinde** çürütüldü (MSK NY vs IRCAD Strasbourg, PMC9287542); içerik dedup'ı yapılmadı; ama dördü de MSK → **kendi aralarında dedup şart** (L10) |
| CT-ORG | 140 | 16.9 GB | CC BY 3.0 | ⛔ **yasak** | L8: TCIA "LiTS'ten ödünç" diyor, eşleştirme tablosu yok → denetim kör |
| FLARE22/23, AbdomenAtlas | — | — | — | ⛔ **yasak** | L7/L3: dış testimizi öldürür |
| HCC-TACE-Seg | 105 hasta | — | CC BY 4.0 | 🧊 **dondurulmuş dış test** | Korpusa asla girmez |

**Bugünkü etkin korpus: 88 hacim.** Hesap: 131 − **27** dondurulmuş iç test
− **16** IRCADb-örtüşmeli (K3: LiTS 28–47'nin dondurulmuş kümede olmayan 16'sı) = **88**.

K3 neden ön eğitim korpusuna da uygulanıyor: `make_splits` o vakaları
`excluded_ircadb_overlap`'a taşıyor ve yeni `pretrain∩eğitim_dışı` kontrolü
(beyaz liste: yalnız `train` ve `val`) onları korpusta görürse **kırmızı yanar**.
Not: `pick_config` kararı 88'de de `kucuk`; değişen yalnız dürüstlük.

Lambda'da Katman B açılırsa: 88 + 1228 (TS-v2) + 500 (AMOS) + 489 (KiTS23) + 150 (WORD)
+ **80** (Pancreas-CT) = **2535**, CHAOS 40 da eklenirse **2575** hacim.
⚠️ Pancreas-CT'de **80** kullanıldı, 82 değil: TCIA'nın kendi notuna göre vaka #25 ve #70,
vaka #2'nin aynı taraması — 82 yazmak **bilinen bir kopyayı iki kez saymak** olurdu ve
projenin ana iddiasıyla çelişirdi.

### L7 çelişkisinin çözümü — karar

FLARE23 hem büyük bir etiketsiz havuz hem de HCC-TACE-Seg'i içeriyor. İkisi birden olamaz.
**Karar: FLARE23 korpustan çıkarıldı, dış test korundu.** Gerekçe: donmuş, kaynak-düzeyinde ayrık
bir dış test bu projenin ana iddiasının (denetlenebilirlik) taşıyıcısı; 500–2000 ek etiketsiz hacmin
yerine konabilecek başka kaynaklar var (TS-v2, AMOS, KiTS23), donmuş dış testin yerine konabilecek
bir şey yok. Bu karar kodda `protected_sources: [HCC_TACE_SEG]` olarak uygulanıyor ve
FLARE23 `enabled: true` yapılırsa `CorpusError` fırlatılıyor.

### Checksum

Bu gece `pending` (kural 18: >5 GB indirme yok). `scripts/lambda_bootstrap.sh` indirmeden sonra
sha256'ları doldurur ve denetimi **yeniden** koşar. Manifest bugün yalnız metadata'dan üretilmiştir.

---

## §2 Mimari

### Kararlar

| # | Karar | Gerekçe |
|---|---|---|
| Girdi | **1.0×1.0×1.0 mm izotropik**, HU (-200, 250) → [-1, 1] | Hedefleme metriğimiz mm cinsinden; anizotropi yayılımı kural 6 ihlallerinin kaynağı. Rad-JEPA'nın (1,32,256,256) kısıtı artık yok |
| Encoder | **Hibrit:** ConvNeXt-3D konv gövde (stride 1→2→4) + transformer (stride 8, 16), `F.scaled_dot_product_attention` | Dikkat yalnız 1728 ve 216 token'da → dizi uzunluğu sorunu yok; konv stride-4 haritası **4 mm token** verir; `mamba_ssm`/flash-attn gerekmez; Touchstone dersi (CNN omurga) korunur |
| Elenen: Mamba | — | `finetune.py:check_mamba_impl` zaten söylüyor: saf PyTorch Mamba geri yayılım için kullanılamaz; GPU'da `mamba_ssm` garantisi yok |
| Elenen: saf ViT patch 8³ | — | 96³ girdide token 8 mm → medyan lezyon 1.25 token = **Rad-JEPA hatasının tekrarı** |
| JEPA hedefi | stride-4 ızgarasında 3D blok maskeleme; 4 hedef bloğu (%3–8 her biri, birleşim %12–32); bağlam = ızgaranın %85–100'ü eksi hedef; predictor 4 blok / dim 256 / 8 kafa / **yalnız çapraz-dikkat** (sorgu = maskeli stride-4 konumları + mm tabanlı sin-cos; anahtar = bağlamın stride-16 token'ları); EMA 0.996→1.0 doğrusal; kayıp LayerNorm'lu hedefte L1 | **Çapraz-ölçekli predictor global token'ları ince yapıyı yerelleştirmeye ZORLAR** — Rad-JEPA'da eksik olan tam buydu |
| MAE hedefi | **Aynı encoder, aynı predictor, aynı maske, aynı tohum, aynı örnekleme.** Tek fark: çıkış boyutu 4³=64, hedef blok-normalize HU, kayıp MSE, EMA yok | OpenMind (arXiv:2412.17041): 3D tıbbi SSL'de MAE hâlâ önde → MAE kolu **zorunlu** kontrol |
| Decoder | Yeni `decoder.py`: 4 seviyeli U-Net tarzı, trilineer + 2 conv/seviye, skip {s1,s2,s4,s8}, genişlik (128,96,64,32), ~2.8M, derin denetim stride 1/2/4 ağırlık (1, .5, .25)/1.75 | `seg.py` donmuş 4×16×16 token içindi ve **bozulmuyor**; uçtan uca eğitimde token-altı ≤5 mm lezyonları taşıyan tek yol tam çözünürlüklü skip |
| Kayıp | Ana kol **Dice+CE**. Ek (bayrakla): Tversky(.3,.7), Focal(γ=2), marj-farkında CE — ağırlık `distance_transform_edt(sampling=spacing)` ile (**kural 6**) | Ek kollar ana kolu riske atmaz |

### ⭐ "Medyan tümör kaç token" — projenin dersi, sayıyla

**1 token = 4×4×4 voksel = 4×4×4 mm = 64 mm³** (stride-4, 1 mm izotropik girdi).

| Ölçü | Kaynak | Eşdeğer çap | Hacim | **Token** | Token/eksen |
|---|---|---|---|---|---|
| LiTS medyan tümör | Bilic ve ark. (CLAUDE.md taşıyıcı) | 16.82 mm | 2490 mm³ | **38.9** | 4.21 |
| **Bizim ölçtüğümüz medyan** | `outputs/kaggle131/lits_gt_lesions.csv`, 908 bileşen / 118 tümörlü vaka | **10.03 mm** | 528 mm³ | **8.26** | **2.51** |
| Aynı lezyon, **Rad-JEPA** ızgarasında | token 8×16×16 voksel @ (2.5, .8, .8) mm = 3277 mm³ | 10.03 mm | 528 mm³ | **0.16** | — |

→ **Token hacmi 3277 mm³ → 64 mm³, yani 51× daha ince** (bu bir **tasarım gerekçesidir,
ölçüm değil**; başarıya çevrildiği HENÜZ ÖLÇÜLMEDİ). Medyan lezyon 0.16 token'dan 8.26 token'a.

Bu, Rad-JEPA'nın donmuş token'larının ölçülebilir katkı yapmamasıyla (`dbeb686`) **tutarlı
bir açıklama adayıdır** — kanıtı değil. ⚠️ Dikkat: o deneydeki kontrol kolu (JEPA'sız aynı CNN)
**aynı kaba ızgarayı** kullanıyordu; token hacmi "her iki kolun da küçük lezyonda kötü olmasını"
açıklar, "JEPA'nın kontrole katkı yapmamasını" açıklamaz. Hipotezin sınanması `LiverJEPA`
kolunun işidir.

**Dürüst alt sınır:** 5 mm lezyon = 65 mm³ = **1.02 token**. Ölçülen dağılımda çapı ≤5 mm olan
lezyonlar **%13.7** (124/908). Bunlar ön eğitim hedefinde hâlâ token-altıdır; onları decoder'ın
stride-1/2 skip yolu taşır ve o yol her kolda rastgele başlatılır.
→ **Raporda: "ön eğitimden ≤5 mm lezyonlarda kazanç BEKLENMİYOR; iddia 10–20 mm bandında kurulacak."**

Kabul kuralı teste gömüldü: `tokens_per_axis(10.03, 1.0, stride=4) >= 2.0`.
Biri stride'ı 8'e çıkarırsa test kırmızı olur.

### Profiller (⚠️ param sayıları hesaplanmış tahmin, `count_params` ile doğrulanıyor)

| | `cok_kucuk` | `kucuk` (MVE) | `orta` (tam) | `buyuk` |
|---|---|---|---|---|
| yama | 48³ | 64³ | 96³ | 128³ |
| konv genişlik (s1,s2,s4) | 8, 16, 32 | 16, 32, 64 | 32, 64, 128 | 48, 96, 192 |
| s8 (dim × blok) | 96×2 | 160×3 | 320×6 | 512×8 |
| s16 (dim × blok) | 128×2 | 192×3 | 384×6 | 640×8 |
| token sayısı s4/s8/s16 | 1728/216/27 | 4096/512/64 | 13824/1728/216 | 32768/4096/512 |
| **encoder param** (ölçüldü) | **0.804M** | **2.762M** | **19.904M** | ⚠️ hesaplanmadı |
| predictor (atılır) ⚠️ tahmin | ~0.25M | ~0.49M | ~4.0M | — |
| decoder (ölçüldü) | 2.28M | **2.84M** | 4.29M | — |
| **seg modeli toplam** (ölçüldü) | 3.08M | **5.60M** | **24.19M** | — |
| kullanım | CPU duman testi | **Kaggle MVE** | Lambda tam deney | ⛔ `pick_config` döndürmez |

**Ölçek kuralı (`pick_config(n_vol, gpu_h)`, elle seçilmez):**
`n<300` → `kucuk` · `300≤n<1000` → `orta` · `n≥1000 ve gpu_h≥24` → `orta` · `gpu_h<8` → `kucuk`.
**Bugün korpus 131 → kural `kucuk` diyor.** `orta`'ya geçmek için TS-v2 / AMOS indirilmiş ve
denetlenmiş olmalı. Gerekçe: 88 eğitim hacmiyle 42M parametre ezberler.

### ⚠️ FLOP / VRAM / süre — hepsi TAHMİN, hiçbiri ölçülmedi

| Koşu | VRAM ⚠️ | Donanım |
|---|---|---|
| `orta` ön eğitim 96³ B=16 | ~10–12 GB | H100'de B=48'e yer var |
| `orta` ince ayar 128³ B=2 | ~4–6 GB | decoder stride-1 baskın |
| `kucuk` ön eğitim 64³ B=16 | ~1.5–2 GB | T4 16 GB ✅ |
| `kucuk` ince ayar 96³ B=2 | ~2–3 GB | T4 16 GB ✅ |

FLOP ⚠️: `orta`@96³ ≈ 89 GFLOP ileri; ön eğitim adımı ≈ 356 GFLOP/örnek. `kucuk`@64³ ≈ 42 GFLOP/örnek.
Süre ⚠️: H100'de `orta` 200 epoch ≈ 2.8 sa; T4'te `kucuk` 150 epoch ≈ 1.5 sa.

🔴 **Darboğaz hesap DEĞİL, korpus büyüklüğü ve disk I/O.** Bu yüzden ön eğitim verisi 1 mm
izotropik olarak bir kez `.npz` önbelleğine yazılır.
**GPU gününün İLK işi:** `python -m src.jepa.encoder --probe` — tahminleri ölçümle değiştir.

---

## §3 Etiketli veri ve split

| | Sayı | Kaynak |
|---|---|---|
| LiTS vakası | 131 | `andrewmvd/liver-tumor-segmentation` (+part-2) |
| Tümörlü / tümörsüz | **118 / 13** | `outputs/kaggle131/lits_gt_lesions.csv` |
| Lezyon bileşeni | **908** | aynı |
| Lezyon / vaka (medyan, ort.) | 3, **6.9** (131 vaka paydası) | aynı |
| Eşdeğer çap p5/25/50/75/95 (mm) | 2.72 / 6.37 / **10.03** / 17.13 / 41.64 | aynı |
| ≤10 mm / 10–20 mm / >20 mm | %49.8 / %29.1 / %21.1 | aynı |
| ≤5 mm | %13.7 (124) | aynı |
| 3D-IRCADb | 20 vaka = **LiTS 28–47** | L1; ayrı veri değil |
| Dış test (dondurulmuş) | HCC-TACE-Seg, 105 hasta | ⚠️ provenance'a henüz **yüklenmedi** → K2 fiilen denetlenmiyor |

### 🔴 Dondurulmuş iç test — `make_splits` "test" split'i ÜRETMİYOR

`src/data/make_splits.py` yalnız `train` / `val` / `external_test` /
`excluded_ircadb_overlap` üretir. Gerçek değerlendirme ise `src/jepa/seg.py:make_folds` ile
vaka listesinden kurulur ve `splits.csv`'yi **hiç okumaz**. Bunun iki sonucu vardı
(21 Eyl, `leakage-auditor` bulgusu B3):

1. `pretrain∩test` kontrolü **yapısal olarak** daima boş küme üzerinde geçiyordu;
2. korpus 131 LiTS hacminin tamamıysa, her CV katının değerlendirme vakaları ön eğitimde
   **görülmüş** oluyordu — projenin başkalarını suçladığı şeyin aynısı.

Kökü şu: 5 katlı CV'de tüm katların değerlendirme kovalarının **birleşimi** tüm vaka
kümesidir; "korpus = tüm vakalar" ile hiçbir kat temiz kalamaz. İki çözüm vardı —
(A) kat başına ayrı ön eğitim: **5× maliyet**, §5 bütçesine sığmıyor;
(B) **tek dondurulmuş iç test kümesi**, ön eğitim + CV geri kalanda. **(B) seçildi.**

| | |
|---|---|
| Üretim | `python -m src.data.frozen_test --ids-glob 'data/raw/LiTS/segmentation-*.nii'` |
| Yöntem | `seg.make_folds` ile **aynı fonksiyon** (iki yerde iki farklı bölme olmasın), hasta düzeyi, tümörlü/tümörsüz tabakalı, tohum 0, kat 0 |
| Büyüklük | **27 vaka** (24 tümörlü / 3 tümörsüz). 131/5 katları 27/26/26/26/26, kat 0 = 27 |
| Koruma | `--exclude-ids` korpustan **düşürür**; `--eval-ids` denetim **doğrular**. İkisi aynı dosya |
| Dondurulmuş | `frozen_test` çıktı dosyası varsa **yeniden üretmez** — sonucu gördükten sonra test kümesini değiştirmenin kapısı kapalı |
| ⚠️ Bedeli | Rapor edilen sayı **27 vakaya** dayanır → **güven aralıkları geniş** olacak. Bugünkü referans sayılar bunların **25'inde** ölçüldü: TS koşusunda `lits-48` ve `lits-49` yok (o koşu 131 → 126 düşmüştü) |

**Split kuralları** (`src/data/make_splits.py`, assert'li):
K1 kanonik hasta düzeyi · K2 dış test kaynak veri seti düzeyinde ayrık ·
K3 IRCADb'ye karşılık gelen LiTS 28–47 eğitimden çıkarılır.
5 katlı CV: `src/jepa/seg.py:make_folds` — hasta düzeyi, tümörlü/tümörsüz **tabakalı**, tohum 0.
Model seçimi / erken durdurma **yok**: `finetune_e2e` sabit adım sayısıyla eğitir ve tahminleri
en sonda tek seferde üretir. Bu, "test katına bakılmadı" güvencesinin en basit hâlidir.

⚠️ **Sonucu:** `splits.csv`'deki `val` kolu şu an **kullanılmıyor** — ince ayar, dondurulmuş
test dışındaki tüm vakalarda eğitiliyor. Erken durdurma, öğrenme oranı arama ya da eşik
ayarı eklenirse `val` kolu **bağlanmak zorunda**; aksi halde ayar dolaylı olarak test
kümesinden gelir. Bugün hiçbir hiperparametre veri üzerinde aranmadı (hepsi
`configs/train.yaml`'da sabit), tur sayısı **1**.

🔴 **Açık:** `external_test` boş olduğu sürece K2 boş kümede geçer → **denetlenmemiştir**.
`split_audit.json` bunu `vacuous: true` ile raporlar. HCC-TACE-Seg provenance'a eklenmeden
"dış testte doğrulandı" cümlesi kurulamaz.

---

## §4 Ön işleme

| Adım | Karar |
|---|---|
| Yön | Tüm hacimler kanonik **RAS**; dönüşüm (`ornt`) loglanır ve geri dönüşte tersi uygulanır (`src/jepa/data.py:canonical_ras`, `inverse_ornt`) |
| Spacing | 1.0 mm izotropik. **`resample` istenen değil ULAŞILAN spacing'i döndürür** — şekil yuvarlandığı için ikisi aynı değildir (7×3.0 mm → 2.0 mm istenince 10 kesit, gerçek aralık 2.1 mm, %5 sapma). İstenen değeri taşımak mm hatasını sessizce büyütür |
| Görüntü / maske | Görüntü trilineer (order=1); maske **en yakın komşu (order=0)**. `is_mask=True` ile order>0 **fırlatır** — maskede ara değer = bozuk hat |
| HU | (-200, 250), `configs/paths.yaml`'dan **okunur, kopyalanmaz**. Tek tanım (`data.py:hu_window`); eğitim ve çıkarımda birebir aynı |
| ROI | Karaciğer kutusu + 10 mm pay. Ön eğitimde etiket yok → TS `liver` maskesi ya da tüm hacim. **Bu bir varsayımdır, raporda yazılır** |
| Önbellek | 1 mm izotropik `.npz`, spacing + affine + ornt birlikte. ⚠️ Boyut/süre ölçülmedi |
| İndirme | 🔴 **Yerel Mac 0.9 MB/s ölçüldü** (21 Eyl 2026, Zenodo AMOS zip'inden 12 sn'lik ölçüm) → AMOS tek başına **7.8 saat**, Katman B toplamı ~32 saat. **Katman B yerel makinede indirilemez, Lambda'da indirilecek.** ⚠️ Lambda veri merkezi hızı ÖLÇÜLMEDİ — `lambda_bootstrap.sh` `data` adımı 30 sn ölçüp tahminleri yeniler |
| ⚠️ Zenodo tuzağı | **Zenodo HTTP `Range` başlığını yok sayıyor** — `curl -r 0-...` isteğine 206 değil **200** dönüyor, yani parça indirme ve güvenilir resume YOK. Yarıda kesilen indirme baştan başlar. `aria2c -x8` çoklu bağlantı kurar ama aynı sınıra tabidir. (Bu, 24 GB'lık bir dosyayı "birazını indirip bakayım" diye test etmenin mümkün olmadığı anlamına gelir — L6 oturumunda `remotezip` ile yaşanan sorunun kardeşi) |
| Artırma | CT'de fizyolojik olanlar: küçük dönme/ölçek, gürültü, düşük çözünürlük simülasyonu, ayna (sol-sağ DEĞİL — organ asimetrisi). **HU semantiğini bozanlar yasak** (rastgele parlaklık/kontrast HU'yu anlamsızlaştırır). JEPA'da artırma × maskeleme birlikte |

---

## §5 Eğitim, GPU ve bütçe

### Hiperparametreler (`configs/train.yaml`)

AdamW lr 5e-4, warmup %10, kosinüs → 1e-5, weight decay 0.05, grad clip 1.0.
AMP: GPU'da fp16 (bf16 yerelde yok), CPU'da kapalı. Sabit tohum. Checkpoint/resume zorunlu.
EMA momentum 0.996 → 1.0 doğrusal. Kayan pencere %50 örtüşme + Gauss ağırlık (σ = 0.125·yama).

### Lambda envanteri — **21 Eylül 2026'da çekilen fiyatlar** (lambda.ai/service/gpu-cloud)

| Örnek | VRAM | $/saat | Not |
|---|---|---|---|
| **1× A6000** | 48 GB | **$1.09** | ⭐ en iyi değer; hesap darboğaz değil |
| 1× A10 | 24 GB | $1.29 | A6000'den pahalı ve küçük → seçme |
| 1× A100 PCIe / SXM | 40 GB | $1.99 | |
| 1× GH200 | 96 GB | $2.29 | |
| 1× H100 PCIe | 80 GB | $3.29 | |
| 1× H100 SXM | 80 GB | $4.29 | |
| 8× H100 SXM | 80 GB | $3.99/GPU ⇒ $31.92/sa | DDP gerekirse |

### Bütçe dağılımı — ⚠️ saatler tahmin, fiyatlar ölçüm

| Senaryo | İş | Saat ⚠️ | A6000 ($1.09) | H100 PCIe ($3.29) |
|---|---|---|---|---|
| **Asgari** (abstract'ı taşır) | ön eğitim jepa + mae (2×) · **2 kat** × 3 kol ince ayar · önbellek + değerlendirme | ~18 | **~$20** | ~$59 |
| **Tam** | ön eğitim jepa + mae · **5 kat** × 3 kol · ablasyonlar | ~30 | **~$33** | ~$99 |
| **Bütçede YOK** | veri indirme + önbellek (instance indirirken de ücretleniyor), depolama | ⚠️ ölçülmedi, 1–4 sa? | +$1–5 | +$3–13 |

⚠️ **Saat sütunu donanımdan bağımsız varsayıldı** (§2: darboğaz hesap değil disk I/O). Bu bir
varsayımdır ve §2'deki "H100'de `orta` 200 epoch ≈ 2.8 sa" tahminiyle birebir uyuşmaz;
`--probe` ve ilk gerçek epoch ölçüsünden sonra tablo yeniden yazılmalı.

🔴 **Bütçe sığmazsa ön eğitim kısalır, kontrol kolları kısalmaz.** (a) rastgele ve (b) MAE
olmadan "JEPA işe yaradı" cümlesi kurulamaz; kısa ön eğitimle kurulabilir.
Çoklu GPU gerekirse DDP yazılır — şu an gerekmiyor (tek A6000 yetiyor).

### Geriye takvim — SIR 24 Eylül

| Ne zaman | Şart mı | İş |
|---|---|---|
| 22 Eyl sabah | ✅ şart | `lambda_bootstrap.sh` → ortam + veri + `--probe` (tahminleri ölçümle değiştir) + önbellek + **manifest denetimi yeşil** |
| 22 Eyl gün | ✅ şart | Ön eğitim: `jepa` ve `mae`. Çöküş metrikleri izlenir |
| 22 Eyl akşam | ✅ şart | Kat 0 × 3 kol ince ayar → `evaluate.py` → **ilk sayılar** |
| 23 Eyl | ✅ şart | Kalan katlar + bootstrap %95 GA + hedefleme/marj metrikleri |
| 23 Eyl akşam | ⬜ opsiyonel | Ablasyonlar (1.0 vs 1.5 mm izotropik, maske oranı) |
| 24 Eyl | ✅ şart | Abstract sayıları yerine koy → `claim-checker` → gönder |

---

## §6 Kontrol kolları — gerçek, çalışan komutlar

Hepsi tek komutla: `bash scripts/lambda_bootstrap.sh`. Aşağıdakiler o betiğin çağırdığı
adımların tek tek hâli (hata ayıklama ve ablasyon için).

**Önce (bir kez):** dondurulmuş iç test + önbellek

```bash
python -m src.data.frozen_test --ids-glob 'data/raw/LiTS/segmentation-*.nii' \
    --out outputs/internal_test.json                      # 27 vaka, DONDURULUR

python -m src.jepa.cache_build --manifest outputs/manifest_all.csv \
    --out data/cache/iso1mm --labels-glob 'data/raw/LiTS/segmentation-*.nii' \
    --roi-glob 'data/raw/LiTS/segmentation-*.nii'         # 1 mm izo + karaciğer ROI

python -m src.data.pretrain_manifest --splits outputs/splits.csv \
    --exclude-ids outputs/internal_test.json --eval-ids outputs/internal_test.json \
    --require-audited --require-sources-covered           # 🔴 kırmızıysa eğitim BAŞLAMAZ
```

**Ön eğitim** — iki hedef, **aynı kod yolu, aynı tohum, aynı örnekleme, tek fark hedef**

```bash
for OBJ in jepa mae; do
  python -m src.jepa.pretrain --objective $OBJ --profile kucuk \
      --manifest outputs/pretrain_manifest.csv --cache data/cache/iso1mm \
      --eval-ids outputs/internal_test.json \
      --out outputs/liverjepa/pre_$OBJ --steps 8000 --seed 0 --resume
done
# --objective none çağrılırsa açık mesajla REDDEDER (ön eğitim yok demektir)
```

**İnce ayar** — üç kol, tek fark `--init`

```bash
# (a) rastgele başlatma — kontrol kolu
python -m src.jepa.finetune_e2e --init random \
    --cache data/cache/iso1mm --test-ids outputs/internal_test.json \
    --out outputs/liverjepa/ft_none --profile kucuk --steps 4000 --seed 0

# (b) MAE — zorunlu kontrol (OpenMind: 3D tıbbi SSL'de MAE hâlâ önde)
python -m src.jepa.finetune_e2e --init outputs/liverjepa/pre_mae/encoder.pt \
    --cache data/cache/iso1mm --test-ids outputs/internal_test.json \
    --out outputs/liverjepa/ft_mae --profile kucuk --steps 4000 --seed 0

# (bizim ana kol) JEPA
python -m src.jepa.finetune_e2e --init outputs/liverjepa/pre_jepa/encoder.pt \
    --cache data/cache/iso1mm --test-ids outputs/internal_test.json \
    --out outputs/liverjepa/ft_jepa --profile kucuk --steps 4000 --seed 0
```

**Değerlendirme** (her kol için; TS tabanıyla **aynı kod**, etiketler AYRI)

```bash
python -m src.metrics.make_pairs --gt-glob 'data/raw/LiTS/segmentation-*.nii' \
    --pred-glob 'outputs/liverjepa/ft_jepa/pred/*.nii.gz' \
    --out outputs/liverjepa/ft_jepa/eval/pairs.csv
python -m src.metrics.evaluate --pairs outputs/liverjepa/ft_jepa/eval/pairs.csv \
    --out outputs/liverjepa/ft_jepa/eval --gt-label 2 --pred-label 1
python -m src.metrics.bootstrap --eval outputs/liverjepa/ft_jepa/eval --n-boot 2000
```

**Dış referans kolları** (bizim encoder'a YÜKLENEMEZ, ayrı koldur)

```bash
# (c) Rad-JEPA donmuş token — mevcut kol, sonuçları var (dbeb686)
python -m src.jepa.train --feats outputs/jepa_feats --variant jepa --fold 0 --out outputs/jepa_seg
# ⚠️ RAPORDA ZORUNLU NOT: Rad-JEPA repo'sunun seg/ kolu M3D-Seg ile eğitiyor;
#    M3D-Seg MSD-Liver (=LiTS) + 3D-IRCADb içerir → o yolla eğitilen seg modeli SIZINTILI.
#    Biz yalnız DONMUŞ ENCODER'ı kullandık, onların seg checkpoint'ini değil.

# (c) ve (d)'yi dondurulmuş kümeye indirgeme (GPU/veri gerekmez)
python scripts/frozen_baseline.py --eval <eski eval klasörü> --out outputs/frozen/<ad> \
    --intersect outputs/frozen/{ts_liver_lesions,radjepa,radjepa_nojepa}/per_case.csv
```

⚠️ **`load_encoder(strict=True)` bilerek:** dış bir ağırlığı (Rad-JEPA) bizim encoder'a
kısmen yüklemek, hangi katmanın eğitimli hangisinin rastgele olduğunu belirsiz bırakır ve
karşılaştırmayı anlamsızlaştırır.

---

## §7 Değerlendirme

**Metrikler** (hepsi mevcut, `src/metrics/`): Dice (boş maskede `nan`, 1.0 değil) ·
lezyon-bazlı P/R/F1 (IoU≥0.5, boyut katmanlı ≤10 / 10–20 / >20 mm) ·
⭐ **hedefleme hatası (mm)** ve **ablasyon marj ihlali** (IoU eşiği YOK; planlama maskesi =
GT ile kesişen tüm tahmin bileşenleri) · marj ek payı eğrisi 5–10 mm + fazla hacim ·
hasta düzeyi bootstrap %95 GA.

**"Başarı" tanımı — önceden yazılıyor ki sonradan uydurulmasın:**
> `jepa` kolunun `none` koluna göre üstünlüğü, **kat-arası SD'nin üstünde** ve bootstrap %95
> güven aralıkları **örtüşmeyecek** şekilde görülürse "ölçülebilir katkı" denir. Aksi halde
> "aralıklar örtüşüyor → ölçülebilir katkı saptanmadı" denir. Ara form yok.

**İddia EDİLMEYECEKLER:** "SOTA'yı geçtik" · küçük Dice farkı (LiTS fold-arası SD %1.7–6.6,
arXiv:2404.09556) · "world model" · karaciğer **organ** segmentasyonu · ≤5 mm lezyonlarda
ön eğitim kazancı (§2: token-altı) · Seg-JEPA'nın %1-etiket/91.9 sayısı (2D göğüs röntgeni).

### TS `liver_lesions` tabanı — **dondurulmuş iç testte** (21 Eyl gece, ölçüldü)

Eski sayılar (Dice 0.749, recall 0.478) **126 vakada** ölçülmüştü; `LiverJEPA` sonuçları
**27 vakalık dondurulmuş iç testte** raporlanacak. Farklı vaka kümelerindeki sayılar
karşılaştırılamaz, bu yüzden taban aynı alt kümeye kısıtlandı:

```bash
python scripts/frozen_baseline.py \
  --eval outputs/kaggle/fix2847/outputs/fix2847/all131 \
  --out  outputs/frozen/ts_liver_lesions
```

| Ölçüt | Nokta | %95 GA | payda |
|---|---|---|---|
| Dice medyan (tümörlü vaka) | **0.752** | 0.662 – 0.797 | 22 |
| Lezyon precision | 0.450 | 0.381 – 0.524 | 202 |
| Lezyon recall | 0.489 | 0.425 – 0.577 | 186 |
| Örtüşme yakalama (tümü) | 0.774 | 0.702 – 0.839 | 186 |
| **≤10 mm lezyon recall** | **0.239** | 0.131 – 0.359 | 88 |
| ≤10 mm örtüşme yakalama | 0.568 | 0.400 – 0.711 | 88 |
| 10–20 mm lezyon recall | 0.603 | 0.452 – 0.771 | 58 |
| >20 mm lezyon recall | 0.875 | 0.778 – 0.971 | 40 |
| 5 mm marj ihlali (tümü) | 0.505 | 0.393 – 0.654 | 186 |
| 5 mm marj ihlali (≤10 mm) | 0.523 | 0.377 – 0.683 | 88 |
| Merkez hatası medyan (mm) | 1.177 | 0.987 – 1.527 | 144 |

⚠️ **Üç uyarı, üçü de rapora girmeli:**
1. Dondurulmuş küme **27 vaka**, ama TS koşusunda **lits-48 ve lits-49 yok** (o koşu 131 → 126
   düşmüştü; düşen beşli lits-48…52). Taban fiilen **25 vakada**. `LiverJEPA` 27 vakada
   ölçülürse **karşılaştırma yine kaymış olur** → ya TS bu ikisinde yeniden koşturulmalı,
   ya iki kol da 25 ortak vakada raporlanmalı. Karar sabah verilmeli.
2. `specificity_case` paydası **3** (tümörsüz vaka sayısı) → GA [0, 1], anlamsız.
   Tümörsüz vaka analizi bu alt kümede yapılamaz; 126 vakalık koşudan raporlanmalı.
3. Dice medyanı 0.752, 126 vakalık 0.749 ile tutarlı → alt küme sistematik olarak kolay
   ya da zor görünmüyor. Bu bir teselli, kanıt değil (n=25).

**Bu tablo `LiverJEPA`'nın geçmesi gereken çıtadır.** Abstract Sürüm A ancak bu aralıkların
dışına çıkılırsa kurulabilir (EGITIM-PLANI §7 "başarı" tanımı).

### Üç referans kolun tamamı, **aynı 25 vakada** (21 Eyl gece, ölçüldü)

Dondurulmuş iç test 27 vaka, ama TS koşusunda `lits-48` ve `lits-49` yok (o koşu 131 → 126
düşmüştü). Farklı vaka kümelerindeki sayılar karşılaştırılamaz → üç kol da **25 ortak vakaya**
indirildi.

```bash
python scripts/frozen_baseline.py --eval <eski eval klasörü> --out outputs/frozen/<ad> \
  --intersect outputs/frozen/{ts_liver_lesions,radjepa,radjepa_nojepa}/per_case.csv
```

| Ölçüt | **(d) TS `liver_lesions`** | (c) Rad-JEPA donmuş token | (c-kontrol) JEPA'sız aynı CNN |
|---|---|---|---|
| Dice medyan | **0.752** [0.662–0.797] | 0.436 [0.184–0.632] | 0.479 [0.158–0.537] |
| Dice **ortalama** | 0.637 | 0.412 | 0.395 |
| Dice = 0 olan vaka | **2 / 22** (lits-59, lits-116) | — | — |
| Lezyon recall | **0.489** [0.425–0.577] | 0.188 [0.122–0.287] | 0.172 [0.117–0.255] |
| Lezyon precision | **0.450** [0.381–0.524] | 0.136 [0.071–0.217] | 0.125 [0.072–0.188] |
| Örtüşme yakalama | **0.774** [0.702–0.839] | 0.468 [0.380–0.596] | 0.478 [0.384–0.639] |
| **≤10 mm lezyon recall** | **0.239** [0.131–0.359] | 0.011 [0.000–0.042] | 0.000 [0.000–0.000] |
| 5 mm marj ihlali (düşük iyi) | **0.505** [0.393–0.654] | 0.774 [0.669–0.856] | 0.839 [0.742–0.909] |
| Merkez hatası medyan (mm) ⚠️ | **1.177** [0.987–1.527] | 2.196 [1.799–2.725] | 2.337 [1.972–3.496] |
| ⤷ paydası (yakalanan / tüm GT lezyon) | **144 / 186** | 87 / 186 | 89 / 186 |

**Üç okuma:**
1. **Çıta TS tabanıdır (Dice medyan 0.752 / ortalama 0.637), Rad-JEPA kolu değil.**
   `LiverJEPA` bu aralıkların dışına çıkmazsa abstract **Sürüm B** ile yazılır (§7 "başarı" tanımı).
   ⚠️ **İki zorunlu çekince, ikisi de rapora girmeli:**
   (a) **Eşit veri karşılaştırması DEĞİL.** TS `liver_lesions` Basel iç verisiyle (842 vaka)
   eğitildi ve LiTS'te *alan dışı* çalışıyor; LiTS katlarında eğitilen `LiverJEPA`'nın onu
   geçmesi beklenen bir **alan avantajıdır**, model üstünlüğü değil.
   (b) **L6 kapanmadı:** TS'in LiTS'i görüp görmediği denetlenmedi (denetlenen kısımda kopya
   saptanmadı, %92 kapsam; xy yeniden örneklenmiş kopyalar kör). Bu çıtayı **her iki yönde** bozar.
2. **Alt küme, yayımlanmış sonucu yeniden üretiyor:** Rad-JEPA donmuş token vs JEPA'sız
   kontrol, **altı ölçütte aralıklar geniş biçimde örtüşüyor**. ≤10 mm'de kontrol kolunun
   aralığı **dejenere** ([0.000–0.000], 0/88) ve Rad-JEPA'nın 0.011'i onun dışında kalıyor →
   o bantta karşılaştırma anlamsız. Sonuç: **ölçülebilir katkı saptanmadı** — bu bir
   **eşdeğerlik kanıtı değildir** (n=25'te güç yok). `dbeb686` ile tutarlı; alt kümenin
   bozuk olmadığının işareti.
3. **≤10 mm bandı:** TS 0.239, Rad-JEPA 0.011, kontrol **0.000**. §2'deki token hesabının
   tam olarak öngördüğü yer — ve bizim tasarımın iddiasının kurulacağı bant.

⚠️ `specificity_case` paydası 3 (tümörsüz vaka) → tümörsüz analizi bu alt kümede **yapılamaz**;
126 vakalık koşudan raporlanmalı.

### Abstract — iki sonuca da dayanıklı, ikisi de şimdiden yazılı

**Sürüm A — modelimiz tabanı geçerse:**
> Sızıntısı vaka düzeyinde denetlenmiş bir korpusla sıfırdan ön eğitilen `LiverJEPA-S`,
> eşit hesap ve eşit veri altında hedefleme hatasını [X] mm'den [Y] mm'ye indirdi
> ([%95 GA]); 5 mm ablasyon marjı ihlali [A]→[B]. JEPA hedefi, MAE hedefine karşı [Z]
> fark verdi. Kazanç 10–20 mm lezyon bandında yoğunlaştı; ≤5 mm'de fark ölçülmedi.

**Sürüm B — geçemezse (bu da yayımlanabilir bir bulgudur):**
> Kamu abdomen CT setlerinde %59.1 vaka tekrarı varken, mevcut 3D CT temel modellerinin
> ön eğitim korpusları vaka düzeyinde denetlenemiyor. Ön eğitim korpusunu test setlerinden
> **vaka kimliği düzeyinde denetleyip raporlayarak** sıfırdan eğittiğimiz `LiverJEPA-S` ile, kontrollü
> koşullarda (aynı tohum, aynı örnekleme, tek fark hedef) JEPA · MAE · rastgele başlatma
> kollarını karşılaştırdık: [sonuç]. **Denetlenebilir korpusla elde edilen bu taban,
> denetlenemeyen ön eğitimle bildirilen sayıların ne kadarının genelleme olduğunu
> sorgulamak için bir referans noktasıdır.**

Her iki sürümde de taşıyıcı cümle **korpus denetlenebilirliği**, model kalitesi değil.

---

## §8 Risk tablosu — her satırda somut komut

| Başarısızlık | Belirti | Komut / müdahale |
|---|---|---|
| **Temsil çöküşü** | Kayıp düşer ama `rankme/D < 0.05` veya `std_dim < 0.1/√D`, 3 ardışık log | Otomatik `CollapseError` + `collapse.json`. Sıra: EMA başı 0.996→0.99 · maske oranı ↑ · `--var-reg 0.1` · `lr ×0.5` |
| **Hedef bağlamda sızıyor** | `visible_frac` beklenenden yüksek, kayıp anormal düşük | `test_baglam_ve_hedef_ayrik` + logda `n_target_tokens`. Maske 4'lük ızgaraya hizalı mı kontrol |
| **OOM** | CUDA OOM | `--batch` yarıya · `--grad-checkpointing` · profil bir kademe küçült (`orta`→`kucuk`). **Kontrol kollarını kesme** |
| **NaN** | kayıp NaN | grad clip 1.0 zaten açık → `--lr 2.5e-4`, AMP kapat, son checkpoint'ten `--resume` |
| **İndirme yavaş** | `data` adımı <10 MB/s ölçüyor | Katman B'yi kırp: önce **AMOS (24 GB)**, sonra TS-v2. KiTS23 (42 GB) en son. Manifest `n_expected` uyarısı hangi kaynağın eksik olduğunu söyler. 🔴 Zenodo resume DESTEKLEMİYOR → indirmeyi `tmux`/`nohup` içinde başlat, SSH kopması baştan başlatır |
| **Model tabandan kötü** | `jepa` < TS `liver_lesions` | Kes: ablasyonlar → `orta` profil → ön eğitim uzatma. **Kesilmeyecek:** `none` ve `mae` kolları. Abstract Sürüm B'ye geçer |
| **Süre taşıyor** | 23 Eyl akşamı kat 0 bitmemiş | Vazgeçilme sırası: (1) ablasyonlar (2) 5 kat → 2 kat (3) ön eğitim epoch'u yarıya (4) `orta` → `kucuk` |
| **Manifest denetimi kırmızı** | `CorpusError` | **Eğitim başlamaz.** Assert'i değil veriyi düzelt: hangi kaynak, `split_audit.json` → `checks.*.examples` |
| **`--probe` tahminden >2× sapıyor** | VRAM ölçümü | Dur, batch/profil yeniden seç; §2 tablosunu ölçümle güncelle |

---

## §9 Bilinmeyenler — ölçülemeyen, varsayılan, sabah doğrulanacak

1. **⚠️ VRAM / FLOP / süre tahminlerinin hiçbiri ölçülmedi.** GPU gününün ilk işi `--probe`.
2. **⚠️ VARSAYIM: 1.0 mm izotropik en iyi geometri.** LiTS'te z-spacing 0.45–6.0 mm; 5 mm'den
   1 mm'ye yükseltmek yapay detay üretir. Doğrulama: 1.0 vs 1.5 mm tek ablasyon (aynı tohum),
   ana kolu bloklamaz.
3. **⚠️ VARSAYIM: çöküş eşikleri** (`rankme/D < 0.05`, `std_dim < 0.1/√D`) literatürden değil,
   RankMe tanımından türetilmiş sezgisel değerler. MVE'nin ilk 500 adımında rastgele-başlangıç
   taban değeri ölçülüp `configs/train.yaml` güncellenecek.
4. **⚠️ VARSAYIM: yarınki etkin korpus 88–2575 arası.** `pick_config` gerçek sayıyla çağrılır,
   elle seçilmez.
5. **🔴 AÇIK: HCC-TACE-Seg provenance'a yüklenmedi** → `external_test` boş → K2 ve
   `pretrain∩dış_test` **boş kümede geçiyor, denetlenmiyor.** "Dış testte doğrulandı" denemez.
6. **🔴 AÇIK: L6 kapanmadı.** "TS `liver_lesions` LiTS görmemiştir" denemez; denetlenen kısımda
   kopya saptanmadı (%92 kapsam, xy yeniden örneklenmiş kopyalar kör). `Ay6NSmY8ie4fXIqK ↔ lits-106`
   iki koşuda da işaretli, elle bakılmadı.
7. **⚠️ AÇIK: Lambda'nın indirme hızı ölçülmedi.** Yerel Mac 0.9 MB/s (ölçüldü) →
   yerelde imkânsız. Lambda veri merkezinde 10–100× daha hızlı olması beklenir ama
   **bu bir beklentidir, ölçüm değil.** `bootstrap data` adımı ilk 30 sn'de ölçer;
   <10 MB/s çıkarsa Katman B kırpılır ve MVE korpusu (LiTS) ile devam edilir.
8. **⚠️ AÇIK: Katman B kaynaklarının hiçbiri içerik hash'i ile teyit edilmedi** — kurumsal ayrım
   argümanı isim düzeyindedir. Lambda'da `dedup_hash` zorunlu.
9. **⚠️ AÇIK: CT-ORG'un LiTS'ten aldığı kesin hacim sayısı** (~130/140 ikincil kaynak). Kenar kesin,
   sayı değil. Korpusa zaten alınmıyor.
10. **⚠️ AÇIK: MSD Task07/08/09/10 kendi aralarında dedup edilmedi** (L10) → korpusa alınamaz.
11. **⚠️ Ön eğitimde karaciğer ROI'si** etiketsiz hacimlerde TS `liver` maskesinden gelecek —
    bu bir model çıktısına bağımlılıktır, raporda yazılır.
