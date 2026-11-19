# Kaggle koşusu 1 — TotalSegmentator liver_lesions × LiTS 131

> 19 Eyl 2026 · Kaggle T4 · TS `liver_lesions` (eğitimsiz, hazır model) · LiTS (Kaggle kopyası, SHA manifestolu)
> Ham çıktı: `outputs/kaggle_run1/` (gitignore'lu). GA: `outputs/kaggle_run1/eval/ci.csv`
> (hasta düzeyi bootstrap, 2000 tekrar, tohum 0).

## 🚨 Düzeltme gerekiyor — LiTS 28–47 başlık spacing'i sahte (liver-ct-51, 19 Eyl gece)

LiTS 28–47'nin (= IRCADb, 20 vaka) NIfTI başlığında spacing tam 1×1×1 mm (sform 2, qform 0); görüntü içeriği aynı,
yalnız başlık bozuk. Gerçek değerler IRCADb'de: düzlem içi 0.56–0.87 mm, z 1.0–4.0 mm. Sonuçları:

- Bu 20 vakada **tüm mm metrikleri yanlış**: çap katmanı (≤10 / 10–20 / >20), merkez hatası, 5 mm marj ihlali, hacimler.
  Aşağıdaki tablolar bu 20 vakayı içeriyor → **abstract'a sayı almadan önce yeniden değerlendirilmeli.**
- TS de muhtemelen yanlış spacing'le koştu. 28–47'ye karşı kalan 106 vaka: lezyon recall 0.17 / 0.49,
  overlap hit 0.40 / 0.76, Dice medyan 0.63 / 0.77. Nedenin spacing mi yoksa IRCAD merkezi mi olduğu
  henüz ayrılmadı (liver-ct-51 20 vakayı doğru spacing'le yeniden koşturuyor).
- 20 vaka düzeltilince genel sayıların hangi yöne ve ne kadar değişeceği henüz bilinmiyor.
- Açık soru: literatürde biliniyor mu? Nicoli 2025'teki LiTS dış testi (Dice 0.593) de bundan etkilenmiş olabilir.

## ⚠️ Nasıl okunmalı

- **Dış doğrulama DEĞİL.** TS eğitim seti (Zenodo 20272572, 842 vaka) LiTS içeriyor mu henüz bilinmiyor (L6).
  Bu sayılar "ölçüm yöntemi 131 vakada uçtan uca çalışıyor" kanıtı ve JEPA kolu için taban.
- Tek model, kıyas yok → fark iddiası yok (CLAUDE.md kural 7).
- Etiketler: GT tümör = 2, TS lezyon = 1. Tümörsüz vakalar Dice'a katılmadı (kural 5).
- Tespit eşleşmesi IoU ≥ 0.5; hedefleme eşleşmesi = GT ile kesişen tüm tahmin bileşenleri (kural 8).
  Marj 5 mm, `distance_transform_edt(sampling=spacing)` (kural 6).

## Kapsam

| | Sayı |
|---|---|
| LiTS vakası | 131 (118 tümörlü, 13 tümörsüz) |
| Değerlendirilen | **126** (113 tümörlü, 13 tümörsüz) |
| Düşen | 5: lits-48, 49, 50, 51, 52 (39 GT lezyonu) |
| GT lezyonu (değerlendirilen) | 869 / 908 |
| GT lezyon çapı medyanı | 10.0 mm — ≤10 mm: 452 · 10–20 mm: 264 · >20 mm: 192 (131 vakanın tamamı) |

Düşen 5 vaka — **Kaggle logundan teyit edildi (20 Eyl, `liver-lits.log`):** GT maskesi ile tahmin arasında
spacing uyuşmuyor, `evaluate` reddetmiş:

```
[HATA] lits-48: spacing farkli (1.0, 1.0, 1.0) vs (0.9765620231628418, 0.9765620231628418, 2.5)
[HATA] lits-49: spacing farkli (1.0, 1.0, 1.0) vs (0.955078125, 0.955078125, 2.5)
[HATA] lits-50: spacing farkli (1.0, 1.0, 1.0) vs (0.90625, 0.90625, 2.5)
[HATA] lits-51: spacing farkli (1.0, 1.0, 1.0) vs (0.90234375, 0.90234375, 3.0)
[HATA] lits-52: spacing farkli (1.0, 1.0, 1.0) vs (0.8359379768371582, 0.8359379768371582, 2.5)
```

→ **Sahte 1×1×1 mm başlığı 28–47 ile sınırlı değil; 48–52'de de var.** Fark şu: 48–52'de yalnız GT maskesinin
başlığı sahte, hacminki gerçek → uyuşmazlık yakalandı ve vaka düştü. 28–47'de ikisi de 1×1×1 → uyuşmazlık yok,
hata sessizce metriklere geçti. Kaç vakanın etkilendiği 131 dosyanın başlıkları taranarak çıkarılmalı.

## Bulgu 1 — 3D-IRCADb = LiTS 28–47 (teyit)

`ircadb_lits_match.csv` (yönden bağımsız imza + HU histogramı):

- **20/20** IRCADb vakası beklenen LiTS vakasına eşleşti; sıra birebir: `3Dircadb1.N → lits-(27+N)`.
- En iyi kosinüs ≥ 0.99999, HU histogram farkı (TV) **0.000** (20/20), ikinci aday ≤ 0.78.
- Sınır: imza + histogram eşleşmesi, bayt düzeyi SHA değil. Histogram farkının tam sıfır olması piksel verisinin aynı olduğunu gösteriyor.

→ CLAUDE.md "Doğrulanmayı bekleyen" maddesi "LiTS 28–47 ↔ IRCADb" kapatılabilir (`/hafiza` ile işlenmedi).

## Bulgu 2 — Voksel ve lezyon düzeyi

| Metrik | Değer | %95 GA | Payda |
|---|---|---|---|
| Dice medyan (tümörlü) | 0.731 | 0.715–0.785 | 113 vaka |
| Dice ortalama | 0.630 | — | 113 vaka |
| Lezyon precision | 0.427 | 0.379–0.473 | 914 tahmin |
| Lezyon recall | 0.449 | 0.385–0.514 | 869 lezyon |
| Lezyon F1 | 0.437 | 0.389–0.484 | — |
| Tümörsüz vakada FP | 11 / 13 | özgüllük 0.15 (0–0.38) | 13 vaka |
| Tümörsüz vakada ort. FP lezyon | 2.9 | — | 13 vaka |

- 5 vakalık provadaki Dice medyanı 0.86 **tam koşuda tekrarlanmadı**. Ortalama 0.630, yayımlanmış TS-on-LiTS
  Dice 0.593'e yakın (tanımlar birebir aynı olmayabilir). → **Dice'tan sızıntı sinyali yok**; soru L6'ya kalıyor.
- Tümörsüz n=13 → GA çok geniş, özgüllük iddiası kurulamaz.

## Bulgu 3 — Boyuta göre tespit

| Boyut | n | Recall (IoU≥0.5) | Dokunma oranı |
|---|---|---|---|
| ≤10 mm | 427 | **0.213** (0.157–0.274) | 0.522 (0.414–0.625) |
| 10–20 mm | 256 | 0.590 (0.487–0.683) | 0.844 (0.750–0.920) |
| >20 mm | 186 | 0.796 (0.723–0.862) | 0.962 (0.919–0.994) |

## Bulgu 4 — Hedefleme ve 5 mm ablasyon marjı (novelty)

| | Tümü | ≤10 mm | 10–20 mm | >20 mm |
|---|---|---|---|---|
| Marj ihlali, tüm lezyonlar | **0.524** (0.451–0.601) | 0.534 (0.438–0.631) | 0.410 (0.316–0.514) | 0.656 (0.554–0.761) |
| Marj ihlali, bulunanlar | 0.330 (0.277–0.394) | 0.108 | 0.301 | **0.642** |
| Bulunan lezyon | 618 | 223 | 216 | 179 |
| Merkez hatası medyan (mm) | 1.03 (0.93–1.18) | 0.83 | 1.02 | 1.38 |
| Merkez hatası p90 (mm) | 5.4 | 4.0 | 3.5 | 7.8 |
| Fazla hacim medyan (mL, bulunan) | 0.21 | 0.04 | 0.20 | 1.77 |
| Plan/GT hacim oranı medyan | 0.94 | 0.87 | 0.94 | 0.96 |
| Paylaşılan plan (n) | 64 | 24 | 13 | 27 |

Boyut katmanlı "bulunanlar" ihlal oranlarının GA'sı yok (bootstrap bu kırılımı üretmiyor).

**Yorum:** Merkez hatası küçük (~1 mm). Sorun sınırda: büyük tümörde TS lezyonu bulsa bile vakaların yaklaşık üçte ikisinde
5 mm marj tutmuyor. Dice bunu göstermiyor; hedefleme metriğinin ana argümanı bu.

## Görseller

`outputs/kaggle_run1/figures/` (74 vaka). Yeşil = GT, kırmızı = TS. Paneller: en büyük tümör kesiti /
en küçük lezyon / kaçırılan / uydurulan. Rapor sayfasında seçilen örnekler:

- **lits-101:** büyük kitle yakalanmış ama sınır taşıyor; 17 lezyonun 8'i kaçırılmış (çoğu 1 voksel); böbrekte 16 mm yanlış pozitif.
- **lits-0:** 5.9 mm bulunmuş, 7.7 mm kaçırılmış (11 lezyonun 2'si kaçırılmış).
- **lits-100:** 10 cm'lik tümörde sınır uyumu iyi; 3.5 mm kaçırılmış; karaciğer dışında 2.7 mm uydurulmuş.

## Açık kalanlar

- [ ] lits-48…52 neden düştü → Kaggle logu
- [ ] Marj ek payı eğrisi (5–10 mm): bu koşuda yok (eski not defteri sürümü); GT maskesi gerektirdiği için Kaggle'da koşmalı
- [ ] L6: TS eğitim seti LiTS içeriyor mu → bu sonuçların "dış test" sayılıp sayılamayacağını belirler
- [ ] Boyut katmanlı "bulunanlarda ihlal" için GA
- [ ] Hızlı TS yolu (FAST_TS) bu koşuda kullanılmadı → "3/3 aynı" teyidi hâlâ bekliyor

---

# Koşu 2 — 28–47 düzeltildi (liver-ct-51, 20 Eyl 2026 gece)

> Kaggle işi `liver-fix2847` (bitiş 00:53). Ham çıktı: `outputs/kaggle/fix2847/` (gitignore'lu).
> Yukarıdaki "🚨 Düzeltme gerekiyor" uyarısı **kapandı**; aşağıdaki sayılar geçerli olanlar.

## A/B/C — sahte başlık ölçümü mü bozdu, modeli mi?

Bunu ayırmak için üç kol koşuldu (LiTS 28–47: **20 vaka, 117 GT lezyonu; güven aralığı
hesaplanmadı**):

| Kol | Ne yapıldı | Dice medyan | Lezyon recall | Marj ihlali | Merkez hatası |
|---|---|---|---|---|---|
| **A** | Eski tahminler, sahte başlıkla ölçüldü | 0.631 | 0.171 | 0.889 | 1.47 mm |
| **B** | Eski tahminler, **gerçek** spacing ile ölçüldü | 0.631 | 0.171 | 0.829 | 1.97 mm |
| **C** | TS **yeniden koşuldu**, gerçek spacing'li hacimlerle | 0.697 | **0.385** | 0.607 | 0.94 mm |

- **A = B** → yalnız cetveli düzeltmek Dice/recall'ı değiştirmiyor; değişen sadece mm türevli ölçüler.
- **B → C** → modeli doğru veriyle yeniden koşturmak recall'ı **iki katından fazla** artırıyor (0.171 → 0.385).
  ⚠️ Buradaki Dice farkı (0.631 → 0.697) **iddia edilmez**: LiTS'in kat-arası SD'si %1.7–6.6
  (CLAUDE.md kural 7) ve bu kolda GA hesaplanmadı. Taşıyan bulgu recall ve marj ihlali.

→ **Sahte başlık bir ölçüm hatası değil, girdi hatasıydı.** TotalSegmentator girdiyi header spacing'ine göre
yeniden örneklediği için 1×1×1 mm yazan başlık modelin kendisini bozmuş. Bu, header'a güvenen her
hazır-model değerlendirmesinde sessizce olabilecek bir tuzak.

## Düzeltilmiş genel sonuç (126 vaka)

`outputs/kaggle/fix2847/outputs/fix2847/all131/` · GA: hasta düzeyi bootstrap, 2000 tekrar

| Metrik | Koşu 1 (hatalı) | **Koşu 2 (geçerli)** | %95 GA | Payda |
|---|---|---|---|---|
| Dice medyan (tümörlü) | 0.731 | **0.749** | 0.718–0.790 | 113 vaka |
| Lezyon precision | 0.427 | **0.446** | 0.401–0.487 | 931 tahmin |
| Lezyon recall | 0.449 | **0.478** | 0.439–0.525 | 869 lezyon |
| Lezyon F1 | 0.437 | **0.461** | 0.423–0.498 | — |
| Dokunma oranı | — | 0.756 | 0.709–0.799 | 869 lezyon |
| **5 mm marj ihlali (tümü)** | 0.524 | **0.486** | 0.428–0.546 | 869 lezyon |
| 5 mm marj ihlali (bulunanlar) | 0.330 | **0.320** | 0.269–0.379 | 657 lezyon |
| Merkez hatası medyan | 1.03 mm | **0.99 mm** | 0.91–1.14 | 657 lezyon |
| Tümörsüz vakada özgüllük | 0.15 | 0.154 | 0–0.385 | 13 vaka |
| **Dice ortalama** | 0.630 | **0.639** | — | 113 vaka |

⚠️ **Dice medyanı tek başına kullanılmamalı.** Dağılım çarpık: ortalama 0.639, medyan 0.749.
113 tümörlü vakanın **7'sinde Dice tam 0** (model o vakada hiçbir tümörü yakalamamış),
11'inde <0.1, 27'sinde <0.5, 39'unda ≥0.8. Abstract'ta "Dice 0.749" geçecekse yanında
ortalama ve sıfır Dice'lı vaka sayısı da geçmeli.

Düzeltme genel sayıları yukarı çekti ama tabloyu değiştirmedi: ana iddia aynı kalıyor.
⚠️ Koşu 1 ↔ Koşu 2 farkları (Dice 0.731→0.749, recall 0.449→0.478) **güven aralıkları içinde
kalıyor** — aynı 126 vakanın yeniden ölçümü olduğu için rapor içi karşılaştırma olarak
verilmiştir, **abstract'a bir "iyileşme" olarak girmez.**

## Boyuta göre — düzeltilmiş

| Boyut | n | Recall (IoU≥0.5) | Dokunma | Marj ihlali (tümü) |
|---|---|---|---|---|
| ≤10 mm | 423 | **0.222** (0.174–0.277) | 0.560 (0.471–0.640) | 0.504 (0.429–0.585) |
| 10–20 mm | 263 | 0.646 (0.569–0.720) | 0.920 (0.876–0.955) | 0.338 (0.265–0.422) |
| >20 mm | 183 | **0.825** (0.757–0.884) | 0.973 (0.936–0.995) | **0.656** (0.552–0.760) |

**Taşıyıcı cümle:** >20 mm tümörlerin %82.5'i bulunuyor, ama aynı katmanda 5 mm marj lezyonların
%65.6'sında ihlal ediliyor. **Genel** Dice medyanı 0.749 (tüm katmanların ortak medyanı;
ortalama 0.639, 7 vakada 0) "iyi" görünürken ablasyon planı tutmuyor.

## Marj ek payı eğrisi (koşu 1'de eksikti)

| Planlama marjı | İhlal (tüm lezyon) | İhlal (bulunanlar) | **İdeal ablasyona göre** fazla hacim, medyan | Plan / ideal hacim |
|---|---|---|---|---|
| **5 mm** | 0.486 | 0.320 | 0.32 mL | 0.88× |
| 6 mm | 0.414 | 0.225 | 1.20 mL | 1.12× |
| 7 mm | 0.379 | 0.178 | 2.70 mL | 1.40× |
| 8 mm | 0.364 | 0.158 | 4.75 mL | 1.74× |
| 9 mm | 0.349 | 0.139 | 7.49 mL | 2.13× |
| **10 mm** | 0.331 | 0.116 | 10.23 mL | **2.59×** |

**Yorum — bu tablo ana argümanı tamamlıyor:** Marjı 5'ten 10 mm'ye çıkarmak ihlali yalnız
%48.6'dan %33.1'e indiriyor, buna karşılık ablasyon hacmi **ideal hacmin 2.59 katına** çıkıyor
(medyan 10.2 mL fazla doku). Kalan ihlallerin çoğu *hiç bulunamamış* lezyonlar olduğu için
marj genişletmek onları kurtaramıyor. Yani "emniyet payını artıralım" yaklaşımı, sağlam
karaciğer dokusunu 2.6 kat fazla yakma pahasına sorunun ancak üçte birini çözüyor.

## L6 sonucu — TS eğitim setinde LiTS kopyası

Kaggle işi `liver-l6` (CPU, ~7 saat, bitiş 01:31). Zenodo 20272572 (38.9 GB, md5 doğrulandı) indirildi,
842 eğitim görüntüsünün her birinden 3 yama çıkarılıp 131 LiTS hacminde arandı.

| | |
|---|---|
| Birebir (tamsayı) eşleşme | **0 / 842** |
| `no_match` | 692 — ama NCC adayı olan, yani **gerçekten denetlenen: 240** |
| `ncc_partial` (elle bakılacaktı) | 150 |

150 kısmi aday `src/provenance/l6_verdict.py` ile ayrıldı. Ölçüt mutlak NCC değil **özgüllük**:
gerçek bir kopyada "kendi hastası" ile "başka hasta" arasında belirgin fark olmalı. İki kural birlikte
kullanılıyor, çünkü sabit eşik tek başına yanıltıcı:

- **Sabit eşik 0.027** — akran oturumun kontrolünden türetildi (set dışı yama 0.906, gerçek kopya ~0.99).
- **Ampirik aykırı değer (robust z ≥ 5)** — ⚠️ bu veri kümesinde gürültü tabanı 0.906 değil **~0.98**
  (ikinci en iyi skorların medyanı 0.984). Yani gerçek bir kopyanın farkı burada ~0.01 olurdu ve
  sabit eşik onu **kaçırırdı**. Bu yüzden farkın kendi dağılımına göre de bakılıyor.

| | |
|---|---|
| **Elle incelenmesi gereken** | **5** (sabit eşikle 0, aykırı değer kuralıyla 5) |
| Özgül değil (her karaciğere eşit benzeyen yama) | 67 |
| Karşılaştırılamaz (tek aday, ikinci skor yok) | 78 |
| Özgüllük farkı: medyan / en büyük | 0.0017 / 0.0148 |
| En iyi ↔ ikinci en iyi NCC korelasyonu | 0.948 |

En yüksek üç aday (`outputs/kaggle/l6/ts_vs_lits_karar.csv`):

| sorgu | en iyi | ikinci | fark | robust z |
|---|---|---|---|---|
| wNR1uduDGYoEeaak | 0.9739 lits-22 | 0.9591 lits-48 | 0.0148 | 8.8 |
| XnXTQTcmNAekhNbJ | 0.9760 lits-106 | 0.9621 lits-108 | 0.0139 | 8.2 |
| YrCXQQSJWd6I6teg | 0.9702 lits-21 | 0.9584 lits-25 | 0.0118 | 6.8 |

## L6 kapsamı — en önemli sınır

| | |
|---|---|
| **Gerçekten denetlenen** | **312 / 842 (%37)** |
| Kör: NCC adayı yok (xy yeniden örnekleme körlüğü) | 452 |
| Kör: karşılaştırılacak ikinci hacim yok | 78 |

→ **Abstract'ta "TS, LiTS içermiyor" DENEMEZ.** Denebilecek olan: *"denetlenen %37'de doğrulanmış
kopya bulunmadı; 5 aday elle incelenmeyi bekliyor."* Ayrıca kör noktalar açık: xy'de yeniden
örneklenmiş kopya görülmüyor, kısmi satırların tamamı `integer_valued=False` olduğu için birebir
arama kanalı o satırlarda zaten kördü, ve referans havuzu yalnız LiTS eğitim seti (131 hacim).

**Ek boşluk (20 Eyl, liver-ct-51):** lits-28…47'nin başlığı sahte 1×1×1 mm olduğu için NCC aday
filtresi (xy aralığı uyumu) bu **20 vakayı hiç aday olarak seçmemiş** — CSV'de bu aralıktan tek bir
referans yok (`ref_id`, `ncc_best_ref`, `ncc_second_ref` üçünde de 0 satır). Yani IRCADb vakaları
için yalnız birebir arama yapıldı, yeniden örneklenmiş kopya araması hiç yapılmadı. Bu 20 vaka
halka açık olarak en çok yeniden dağıtılan kümedir → `configs/lits_2847_true_spacing.csv` ile
düzeltilmiş başlıklarla arama tekrarlanmalı (CPU işi, GPU kotası harcamaz).

## L6 — işaretlenen 5 aday şans eseri mi? (boş hipotez testi)

İşaretlenen 5 adayın hepsinde aday havuzu yalnız **2 hacim**; yani "ikinci en iyi" tek bir
rastgele hacim ve fark doğal olarak gürültülü. 72 ölçümün en büyüğünün medyandan birkaç MAD
uzakta çıkması **beklenen** bir şey olabilir. İki boş hipotez testi yapıldı:

| Boş model | Beklenen en büyük robust z | Gözlenen | p |
|---|---|---|---|
| Yarı-normal (ölçek alt %90'dan) | medyan 3.32 · %99 dilim 6.13 | **8.84** | <0.0001 |
| Ampirik (alt 67 gözlem, yerine koyarak) | medyan 4.72 · %99 dilim 7.76 | **8.84** | **0.0056** |

→ En güçlü adayın farkı, **farkların kendi dağılımına göre aykırı** — boş modelde bu kadar
büyük bir fark beklenmiyor. Gözlenen ilk beş z (5.19, 5.40, 6.81, 8.23, 8.84) boş modelin
beklenen ilk beşinin (2.09–3.46) belirgin üstünde. ⚠️ Bu **kopya kanıtı değildir**: test
yalnız "bu satır diğerlerine benzemiyor" der, "bu bir kopyadır" demez.

⚠️ İkinci test kısmen döngüsel (boş dağılım en büyük 5 çıkarılarak kuruldu), bu yüzden p
değeri iyimser okunmamalı. Yine de **en az bir satır (wNR1uduDGYoEeaak ↔ lits-22, z 8.84)
doğrudan incelenmeyi hak ediyor.**

Kontrol: lits-106'nın iki kez çıkması tek başına dikkat çekici değil — 72 ölçülebilir satırın
referans dağılımı çarpık (lits-22 9 kez, lits-61 7 kez), yani tekrar beklenen bir şey.

**Sıradaki adım:** bu 5 sorgunun yamaları (`outputs/kaggle/l6/outputs/l6/probes_ts/<id>.npz`)
ile karşılık gelen LiTS hacimlerinin doğrudan görsel/piksel karşılaştırması. LiTS hacimleri
yerelde yok → küçük bir CPU Kaggle işi gerekiyor.

## L6 eki — IRCADb'ye karşı arama (20 Eyl ~03:20, yerel, liver-ct-51)

Yukarıdaki boşluğu kapatmak için aynı 842 yama, **gerçek başlıklı** IRCADb hacimlerine karşı
yerelde arandı (`data/raw/3Dircadb1/*/ct.nii.gz`, 20 vaka, 7 dakika).
Çıktı: `outputs/l6_ircad/ts_vs_ircad{,_karar}.csv`

| | |
|---|---|
| Birebir eşleşme | 0 |
| `ncc_partial` | 22 → elle bakılacak **0**, özgül değil 2, karşılaştırılamaz 20 |
| Gerçekten denetlenen | **30 / 842 (%4)** |
| NCC adayı çıkmayan | 792 |

**Nüans — 792 "aday yok" ne anlama geliyor?** İki okuması var ve ikisi de yazılmalı:

- *Kırpma senaryosu için kanıt:* Kırpmak spacing'i değiştirmez. Eğer TS'in eğitim görüntüleri
  IRCADb hacimlerinin kırpılmışı olsaydı, xy spacing'leri **birebir** tutardı. 792 sorgu
  aday filtresini geçemedi (filtre üç koşulu birden arar: xy spacing ≤1e-3 tolerans, xy uzanım
  ve z uzanım referansa sığmalı — `src/provenance/match_crop.py`) → bu 792 görüntü,
  IRCADb'nin yeniden örneklenmemiş kırpıntısı **değil**.
- *Yeniden örnekleme senaryosu için körlük:* xy'de yeniden örneklenmiş bir kopyanın spacing'i
  zaten değişir; bu filtre onu eler. O senaryoda 792 sorgu denetlenmemiş sayılır.

Yani "denetlenen %4" rakamı en kötü durum okumasıdır; kırpma senaryosu için kapsam çok daha geniş.
Hangi okumanın rapora gireceği sızıntı oturumunun kararı — `docs/sizinti-devir.md`'ye not düşüldü.

## L6 — birebir kanal açık, LiTS 131 (20 Eyl 06:02, `liver-l6-round`)

Birebir arama kanalı ilk koşuda **842/842 sorguda kördü** (float32 gösterim hatası). Sorguların
416'sı toleransla tamsayı; yuvarlanıp yeniden koşuldu. Birebir arama **spacing filtresi
kullanmaz**, yani 131 hacmin tamamını tarar.

| | İlk koşu (842) | **Yuvarlanmış (416)** |
|---|---|---|
| Birebir eşleşme | kanal kapalı | **0 / 416** |
| Gerçekten denetlenen | 312 / 842 (%37) | **381 / 416 (%92)** |
| Elle bakılacak | 5 | **1** |
| En büyük özgüllük farkı | 0.0148 | 0.0097 |

→ **Bu 416 görüntü için ne söylenebilir:** herhangi bir LiTS hacminin **yeniden
örneklenmemiş, tamsayı değerli** kopyası olsalardı birebir arama bulurdu; bulmadı.
Yani bu alt küme için *kırpma tipi* kopya dışlanmıştır. Yeniden örneklenmiş ya da yoğunluk
ölçeklenmiş bir kopya bu kanaldan yine kaçar; `docs/leakage-map.md` L6'yı 🟡 (kısmen
denetlendi) olarak işaretli tutuyor.

⚠️ Kalan 426 görüntü (tamsayı olmayanlar) için durum değişmedi: yalnız NCC kanıtı var,
denetlenen oran düşük ve ilk koşuda işaretlenen 4 aday hâlâ açık.

**Israrlı tek aday:** `Ay6NSmY8ie4fXIqK ↔ lits-106` her iki koşuda da işaretlendi
(NCC 0.9943 vs ikinci 0.9846, z 6.9). Doğrudan incelenmeli.

**Abstract için yazılabilecek cümle:** *"TotalSegmentator'ın 842 vakalık eğitim setinde
içerik tabanlı, kırpmaya dayanıklı arama yapıldı; **denetlenen kısımda doğrulanmış LiTS
kopyası saptanmadı.** Tamsayı değerli 416 görüntülük alt kümede (842'nin %49'u) birebir
arama %92 kapsamla negatif sonuç verdi; 5 aday elle incelenmeyi bekliyor. Yöntem xy'de
yeniden örneklenmiş kopyaları göremez, dolayısıyla sızıntı kesin olarak dışlanamaz."*

## JEPA karşılaştırması — NİHAİ (20 Eyl 06:30, `liver-jepa-train-ras`)

5 kat hasta düzeyi çapraz doğrulama, 131 vaka → kat-dışı tahmin; iki kol **aynı** CNN, **aynı**
tohum, **aynı** adım sayısı (4000) ve örnekleme, tek fark JEPA token yolu. Yön düzeltmeli
özellikler (`meta['ornt']` kontrolü ile zorunlu). TS ile **ortak 126 vaka** üzerinde ölçüldü.

| Ölçüt | TS tabanı | **JEPA** | **JEPA'sız** | JEPA ↔ JEPA'sız? |
|---|---|---|---|---|
| Dice medyan | 0.749 [0.718–0.790] | 0.354 [0.201–0.487] | 0.295 [0.172–0.441] | örtüşüyor |
| Lezyon bulma | 0.478 [0.439–0.525] | 0.138 [0.106–0.176] | 0.139 [0.108–0.177] | örtüşüyor |
| Lezyon kesinliği | 0.446 [0.401–0.487] | 0.111 [0.084–0.142] | 0.106 [0.079–0.135] | örtüşüyor |
| Dokunma oranı | 0.756 [0.709–0.799] | 0.412 [0.365–0.466] | 0.428 [0.382–0.488] | örtüşüyor |
| 5 mm marj ihlali | 0.486 [0.428–0.546] | 0.855 [0.815–0.889] | 0.853 [0.814–0.885] | örtüşüyor |
| ≤10 mm bulma | 0.222 [0.174–0.277] | 0.007 [0.000–0.015] | 0.012 [0.003–0.023] | örtüşüyor |
| Merkez hatası (mm) | 0.99 [0.91–1.14] | 2.57 [2.20–3.08] | 2.47 [2.09–3.15] | örtüşüyor |

**Sonuç: dondurulmuş Rad-JEPA token'ları ölçülebilir katkı yapmıyor.** Yedi ölçütün
hepsinde güven aralıkları örtüşüyor (CLAUDE.md kural 7: fark iddia edilmez).

**Neden — mimari değil, çözünürlük.** Rad-JEPA'nın yayımlanan kontrol noktası 32×256×256
girdiyi 4×16×16 = 1024 token'a indiriyor; bir token **8 dilim × 16 × 16 voksel** kapsıyor.
LiTS'in medyan lezyonu, bu 869 lezyonluk kümede **eşdeğer çap medyanı 10.3 mm**
(≈ 12×12×4 voksel) — yani **tek bir token'dan küçük** (token 20 × 12.8 × 12.8 mm).
⚠️ CLAUDE.md'deki taşıyıcı sayı farklı bir şeyi ölçüyor: Bilic ve ark. **hacim** medyanı
2.49 cm³ ≈ 1.7 cm çap. İkisi karıştırılmamalı (bizimki 869 lezyonun eşdeğer çap medyanı,
bizim hacim medyanımız 0.57 mL). 17 mm alınsaydı lezyon xy'de token'dan büyük olurdu ve
argüman zayıflardı; abstract'ta **hangi medyanın kullanıldığı açıkça yazılmalı.**
Lezyon ölçeğindeki bilgi temsile hiç girmiyor. ≤10 mm bulma oranının her iki kolda da
%1'in altında kalması bunu doğruluyor.

**Yön düzeltmesinin etkisi:** JEPA kolu v1 → RAS: Dice 0.335 → 0.354, bulma 0.119 → 0.138,
dokunma 0.377 → 0.412. Hepsi doğru yönde ama **aralıklar örtüşüyor** — iyileşme iddia edilemez.
Yön hatası gerçek bir hataydı ve düzeltilmesi doğruydu; ama bu ölçekte etkisi ölçülemiyor.

### ⚠️ TS ile JEPA kolları aynı koşullarda değil

Tablo aynı cetveli kullanıyor ama iki taraf aynı avantajlara sahip değil. Bunlar yazılmadan
karşılaştırma okunamaz:

| | TS tabanı | JEPA / JEPA'sız kolları |
|---|---|---|
| Karaciğer bölgesi | kendi buluyor | **GT karaciğer etiketinden alınıyor** (ROI oracle, `src/jepa/prep.py`) |
| LiTS'i görmüş mü | hayır (hazır model) | **evet** — 5 kat CV, her katta 104 hastayla eğitildi (in-domain) |
| Model boyutu | tam nnU-Net, 842 vakayla eğitilmiş | ~1M parametreli küçük CNN, 4000 adım |

İlk iki satır JEPA kollarının **lehine**, üçüncüsü aleyhine. Yani "JEPA kolları TS'in altında"
sonucu, bu avantajlara rağmen elde edilmiştir — negatif sonucu zayıflatmaz, güçlendirir.
JEPA ↔ JEPA'sız karşılaştırması ise **iki kol da aynı avantajlara sahip olduğu için adildir**;
raporun taşıyan sonucu odur.

**İki kol da TS tabanının belirgin altında.** Beklenen: başlığımız ~1M parametreli küçük bir
CNN, 131 vakayla 4000 adım eğitildi; TS 842 vakayla eğitilmiş tam bir nnU-Net. Abstract'ta
"JEPA nnU-Net'i geçti" iddiası kurulamaz ve kurulmayacak.

**Abstract'a girecek dürüst cümle:** *"Dondurulmuş bir 3B CT temel modelinin (Rad-JEPA)
token'ları, aynı segmentasyon başlığına eklendiğinde karaciğer tümörü tespitinde ölçülebilir
kazanç sağlamadı; token ızgarası (8×16×16 voksel) medyan lezyondan kaba olduğu için lezyon
ölçeğinde bilgi taşımıyor."*

## Açık kalanlardan kapananlar

- [x] lits-48…52 neden düştü → GT başlığı sahte, hacminki gerçek (yukarıda)
- [x] Marj ek payı eğrisi → yukarıdaki tablo
- [x] L6 → kopya kanıtı bulunamadı (sınırlarıyla birlikte)
- [ ] JEPA kolu: `liver-jepa-train-ras` koşuyor (bitiş ~06:20) → TS ile ortak vaka kümesinde kıyas
- [ ] Boyut katmanlı "bulunanlarda ihlal" için GA
- [ ] 78 "karşılaştırılamaz" L6 adayı: ikinci aday üretecek daha geniş NCC bütçesiyle tekrar
