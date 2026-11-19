# Sızıntı haritası — çalışan belge

Durum kolonu: 🔴 doğrulanmadı · 🟡 literatürden biliniyor, hash ile teyit bekliyor · 🟢 kendi verimizde teyit edildi

| # | Sızıntı | Durum | Etki | Aksiyon |
|---|---|---|---|---|
| L1 | 3D-IRCADb-01 ⊂ LiTS eğitim seti (kayıt 28–47) | 🟢 | IRCADb dış test olarak kullanılamaz; ön eğitim korpusuna da giremez | ✅ **Kendi verimizde teyit edildi (21 Eyl 2026):** `outputs/kaggle131/ircadb_lits_match.csv` → **20/20** vaka eşleşti, kosinüs 0.999996–1.000004, HU histogram farkı **0.0**, aynı şekil, ikinci aday ≤0.78. Eşleşme birebir sıralı: `3Dircadb1.N → lits-(27+N)`, yani `configs/paths.yaml → lits_ircadb_indices` [28…47] **doğrulandı**. ⚠️ İmza + histogram yöntemi, SHA değil. `IRCADB` kaynağı `pretrain_corpus.yaml`'da `reason_blocked` |
| L2 | MSD Task03_Liver = LiTS (aynı görüntüler) | 🟢 | İki set değil, tek set | `equivalent_sources`. ⚠️ `liver_N`↔`volume-N` **numara** eşliği varsayım — dedup ile teyit et (MSD CT'leri artık dedup'a giriyor) |
| L3 | AbdomenAtlas 3.0 = kamu veri setlerinin derlemesi (makale 17 der; ID eşlemesinde ~30 kaynak öneki — bkz. L18) | 🟢 | AbdomenAtlas'ta eğitip LiTS'te test etmek kirli | provenance tablosu; sayılar L18 |
| L4 | Aynı tarama AbdomenAtlas içinde 2–3 farklı BDMAP_ID ile | 🟡 | Hazır IID split'i kendi içinde kirli | `dedup_hash.py` → elle onay → `resolve_canonical.py` (canonical_id) → split. K3 bileşen üyeleri üzerinden |
| L5 | FLARE23 ve AbdomenCT-1K zaten LiTS içeriyor | 🟡 | İkinci katman dolaylı sızıntı | provenance; `equivalent_sources`'ta YOK → isimden yakalanmaz, yalnız dedup yakalar. Bunlar dış test seçilirse K2 LiTS örtüşmesini görmez |
| L6 | TotalSegmentator liver_lesions eğitim verisi | 🔴 | LiTS içeriyorsa baseline dış sayı değil | Makale (PMC13481620): Basel iç verisi, LiTS'i dış test sayıyor. Ama Zenodo seti 842 CT (makale 414 CT), dosya adları anonim. ✅ **Kullandığımız model bu 842'yle eğitilmiş:** TS 2.18.0 `map_tasks_config.py` → `liver_lesions` task_id 591 → `Dataset591_ct_liver_lesions_842subj` (v2.5.0-weights); Zenodo zip'i `Dataset591_liver_lesions.zip`, 842 vaka (19 Eyl 2026, iki oturum ayrı ayrı gördü). Yani L6 taraması benchmark modelinin eğitim setinin tamamı. Aynı yapılandırma: çıkarımda karaciğere kırp + 0.75×0.75×1.0 mm'ye yeniden örnekle — Zenodo başlıklarında xy aralıkları çeşitli (0.6992, 0.7871…) → ham veri yeniden örneklenmemiş görünüyor, exact arama uygun. Başlık taraması kör (görüntüler kırpılmış). **Yöntem hazır:** `src/provenance/match_crop.py` (3 yama/görüntü → LiTS'te birebir + NCC; gerçek IRCADb kırpıntılarıyla doğrulandı: kırpma, z yeniden örnekleme, HU kayması yakalanıyor; xy yeniden örnekleme KÖR, `ncc_candidates=0` ile raporlanır). **Aksiyon:** `notebooks/kaggle_leakage_l6.ipynb` → `outputs/l6/ts_vs_lits.csv`; aynı yamalar yerelde IRCADb ve HCC-TACE-Seg'e karşı |
| L7 | HCC-TACE-Seg (dış test adayı) başka derlemelerin içinde: FLARE23 (kaynak listesinde), FLARE PancancerCTSeg (74 vaka), FLARE23 üzerinden AbdomenAtlas 1.1'e dolaylı | 🟡 | FLARE23 / PancancerCTSeg / AbdomenAtlas ile eğitilmiş model için HCC-TACE-Seg dış test DEĞİL. TS liver_lesions için temiz görünüyor: Nicoli'nin "172 TACE" kohortu Basel iç verisi (XML'de TCIA/HCC-TACE geçmiyor) | arXiv:2508.13626 Tablo I, HF FLARE-MedFM/PancancerCTSeg. AbdomenAtlas'a geçen vaka sayısı doğrulanmadı → AbdomenAtlas kullanılırsa HCC-TACE-Seg ile içerik dedup'u. TS için kesin cevap: L6 yamaları HCC-TACE-Seg'e karşı |
| L8 | **CT-ORG (TCIA) ⊃ LiTS** — TCIA'nın kendi koleksiyon sayfası: "Birçok görüntü, LiTS yarışmasından ödünç alınmıştır" | 🟡 | CT-ORG ön eğitim korpusuna giremez. 🚨 **İçerme grafiği bunu yakalamaz:** dosyalar anonim, kanonik kimlik `CT_ORG::N` ≠ `LITS|MSD_TASK03::N` → `pretrain∩test` denetimi kör | `configs/pretrain_corpus.yaml` → `reason_blocked` (açık yasak). Kesin sayı (~130/140, Rister ve ark.) primer kaynakla **doğrulanmadı**. Kaggle aynası `thotakuldeep/ct-org-tcia` tek-kesit DICOM → tuzak |
| L9 | **WORD'ün genelleme değerlendirmesi 20 MSD Liver (=LiTS) vakası içeriyor** (⚠️ "RAOS ⊃ LiTS" doğrudan doğrulanamadı; RAOS deposu yalnız WORD ile örtüşme yazıyor) | 🟡 | Çekirdek WORD 150 vaka temiz; genişletilmiş set yasak | WORD GitHub README. Ayrı kaynak adı `WORD_RAOS`, yasak listesinde |
| L10 | **MSD Task07/08/09/10 dördü de Memorial Sloan Kettering kaynaklı → kendi aralarında hasta tekrarı olabilir** | 🔴 | Dördünü birlikte korpusa alırsak aynı hasta birden çok kez girer; LiTS ile ilgisi yok ama korpus sayımı ve hasta düzeyi ayrıklık bozulur | Task07↔08↔09↔10 arası `dedup_hash` koşulmadan korpusa alınmaz (`MSD_MSK.blocked_until`). ⚠️ Task08 = LiTS hipotezi **primer kaynakla çürütüldü** (PMC9287542: MSK NY vs IRCAD Strasbourg) |
| L11 | **Kanonik kök kayması** — `resolve_canonical` union-find kökünü sözlük sırasında küçük olanı seçiyor; dedup onaylı LiTS↔X birleşmesinde split `X::n` taşırken korpus `LITS|MSD_TASK03::n` taşıyor | 🟢 kendi kodumuzda gösterildi | `pretrain∩test` kesişimi **sessizce boş** çıkıyordu: bir LiTS test hastası korpustayken rapor "temiz ve denetlendi" diyordu | ✅ Kapatıldı 21 Eyl: kesişimin sağ tarafı `canonical_members` ile genişletiliyor (`pretrain_manifest._expand_members`). `make_splits.apply_ircadb_rule` K3'te zaten böyle yapıyordu |
| L12 | **CV fold değerlendirme vakaları ön eğitim korpusunda** — `make_splits` `"test"` split'i üretmiyor, `seg.make_folds` `splits.csv`'yi hiç okumuyor | 🟢 kendi kodumuzda gösterildi | `pretrain∩test` yapısal olarak daima boş; korpus 131 LiTS'in tamamıysa her katın değerlendirme vakaları ön eğitimde görülmüş olurdu | ✅ Kapatıldı 21 Eyl: `src/data/frozen_test.py` ile tek dondurulmuş iç test kümesi (~26 vaka); `--exclude-ids` korpustan düşürür, `--eval-ids` denetler. Bedeli: rapor 26 vakaya dayanır, GA geniş |
| L14 | **Kimlik kuralı ikiliği:** `build_provenance` İLK sayı grubunu, `pretrain_manifest.canonical_for` SONDAKİNİ alıyor | 🔴 | `3Dircadb1.17` → provenance `IRCADB::3`, korpus `IRCADB::17`; provenance 20 IRCADb vakasını **tek kimliğe çökertir**, korpus 20 ayrı kimlik üretir → kesişim **19 vakayı hiç görmez**. Aynı tuzak `04-12-2000-…-12345` (TCIA), `Task03_Liver_liver_43`, `amos22_0507`'de de var | 🟡 Ayrışma artık **saptanıp reddediliyor** (`assert_id_rule_agrees`, `--strict-id-rule`); **birleştirme yapılmadı** — `build_provenance`'a dokunmayı gerektiriyor. Katman B veya IRCADb açılmadan önce tek fonksiyona indirilmeli + sıfır dolgusu normalize edilmeli |
| L15 | **GT'den türetilmiş ROI ile değerlendirme** — tahmin GT karaciğer/tümör kutusuna hapsoluyor, `reconstruct` kutu dışını sıfırlıyor | 🟢 kendi kodumuzda gösterildi | Model GT anatomisi dışında **yanlış pozitif üretemez** → lezyon precision, örtüşme yakalama ve marj ihlali **yukarı yanlı**; tüm hacimde koşan TS tabanıyla **karşılaştırılamaz** | ✅ Kapatıldı 21 Eyl: dondurulmuş **test** vakaları ROI'siz (tüm hacim) önbellekleniyor; eğitim vakalarında ROI yalnız karaciğer etiketinden (`--roi-label 1`), tümör etiketi kutuyu belirlemiyor |
| L16 | **lits-48…52: segmentasyon başlıkları sahte spacing (1,1,1)** taşıyor, CT'ler gerçek değeri (ör. 0.9766/0.9766/2.5) | 🟢 kendi verimizde ölçüldü (Kaggle MVE v2 logu) | Bu beş vakanın eski TS koşusunda **neden düştüğünün kesin açıklaması** — günlük "muhtemelen header farkı, TEYİT EDİLMEDİ" diyordu. Sessizce devam etmek mm metriklerini geçersiz kılar | `cache_build --trust-ct-spacing` (açık bayrak, her vaka için uyarı basar, şekiller de farklıysa yine fırlatır). ⚠️ LiTS 28–47'deki sahte spacing bunun **tersi**: orada HACİM başlığı bozuktu (`configs/lits_2847_true_spacing.csv`) |
| L13 | **Korpus kaynak adı split'te hiç görünmüyor** (`KITS23` vs provenance `KITS`, `TCIA_PANCREAS` vs `PANCREAS_CT`, `TOTALSEGMENTATOR_V2` vs `TOTALSEGMENTATOR`) | 🟢 kendi kodumuzda gösterildi | O kaynakların test/dış-test vakaları kesişimde **hiç görünmez**; denetim "temiz" der ama bakmamıştır | 🟡 Raporlanıyor (`sources_not_in_splits` + `--require-sources-covered` reddediyor), ama ad sözlükleri **birleştirilmedi**. Katman B açılmadan önce yapılmalı |
| L17 | **AbdomenAtlas 3.0 `liver_lesion.nii.gz` maskeleri DiffTumor önerisi + radyolog düzeltmesi; DiffTumor MSD Task03 (= LiTS) ile eğitilmiş** — ETİKET sızıntısı (görüntü değil) | 🟡 birincil kaynak (RadGPT arXiv:2501.04678v2 Ek C: "tumor segmentation masks were suggested by AI and radiologists individually verified and corrected them"; DiffTumor README: eğitim `Task03_Liver.tar.gz`) | LiTS vakaları çıkarılsa bile AA3 tümör maskeleriyle eğitilen model LiTS-türevi bilgi taşır → LiTS-27'de test kirli; "kendi modelimiz" iddiası da zayıflar. İnsan düzeltmesi yapılan vaka sayısı makalede YOK; meta CSV'de `radiologist note` yalnız 117 satırda dolu | AA3 tümör etiketi eğitimde KULLANILMAZ. Yalnız etiketsiz korpus / karaciğer organ etiketi adayı, L18 ön koşuluyla |
| L18 | **AbdomenAtlas 3.0 bileşenleri — tam eşlemeden sayıldı** (`tumor-vqa/DeepTumorVQA_1.0/AbdomenAtlas_ID_mapping.csv`, 9.262 satır = 9.262 benzersiz BDMAP; iki bağımsız sayım aynı) | 🟢 kendi sayımımız (21 Eyl 2026, liver-ct-13) | **LiTS ailesi 239 satır**: LiTS 58 + MSD-Liver 78 + Decathlon_liver 39 + CT-ORG 64 (LiTS toplam ~201 tarama → tekrar kesin). AbdomenCT-1K 399 + FLARE23Val 100 içindeki gizli LiTS kopyaları ADLA ayıklanamaz. HCC-TACE adıyla **0** satır (FLARE23Val'in alt kökeni DOĞRULANAMADI). Korpusumuzla çakışan: TotalSegmentator 485, AMOS 200, KiTS19/21/23 665, MSD-/Decathlon HepaticVessel 300, MSD pankreas+kolon ~490. Korpusta olmayan: Trauma 4.067, autoPET 916, TCIAColon 615, CPTAC/TCGA ~225, NIH-Lymph 55, LDCT 25. ⚠️ "17 kamu seti" eksik sayım: eşlemede ~30 kaynak öneki var | AA'dan etiketsiz alt küme alınacaksa: 239 + 399 + 100 + korpusla çakışan kaynaklar ÇIKARILIR; AbdomenCT-1K içerik dedup'sız alınmaz. `AbdomenAtlas3.0MiniWithMeta.csv` (37 sütun) `original id` TAŞIMIYOR; `TrainTestIDS/` kökenden bağımsız → güvenilmez |
| L19 | **FLARE PancancerCTSeg (FLARE24/25) karaciğer kısmı ⊃ {MSD-Liver (=LiTS), HCC-TACE-SEG, WAW-TACE, CRLM, MSD-HepaticVessel, TS liver_lesions}** | 🟡 HF veri kartı | Hem iç testimizi (LiTS) hem dış testimizi (HCC-TACE-Seg) içerir | Yasak; hiçbir amaçla alınmaz. ⚠️ WAW-TACE ya da CRLM'i eğitime alırsak PancancerCTSeg ile eğitilmiş modellerle aynı veriyi paylaşırız (kıyasta not düş) |
| L20 | **ULS23 ⊃ LiTS** (Part 2) + MSD Task06/07/10, KiTS21, LIDC, DeepLesion | 🟡 Zenodo 10035161 açıklaması | ULS23 karaciğer lezyon kırpıntıları LiTS test vakalarını içerebilir | Yasak |
| L21 | **TCIA CRLM (197, MSKCC) ↔ MSD Task08 (MSKCC) hasta örtüşmesi ne kanıtlandı ne dışlandı** — Simpson ve ark. (PMC10847495) tam metninde "Decathlon/MSD/hepatic vessel" hiç geçmiyor; ikisi de MSK, ikisi de Scout yarı-otomatik | 🔴 | Test sızıntısı DEĞİL (ikisi de eğitim tarafı). Risk: çift sayım + Task08'den doğrulama ayrılırsa eğitim↔doğrulama sızıntısı | MSK setleri (Task07/08/10 + CRLM) TEK kaynak grubu; doğrulama MSK dışından; alınmadan önce içerik dedup (L10 ile aynı iş) |
| L22 | **lits-48…52 segmentasyon başlıkları yalnız spacing değil YÖN de bozuk** — etiket kendi affine'iyle RAS'a çevrilince CT'ye göre AYNALANIYORDU (QC lits-51: karaciğer maskesi altında medyan −29 HU) | 🟢 kendi verimizde (liver-ct-7a, Lambda, commit `b30e91d`) | **MVE bu 5 vakayı aynalı etiketle kullandı (3'ü eğitimde).** L16'nın devamı: L16 spacing'i, L22 yönü anlatır | ✅ `src/jepa/cache_build.py:load_on_grid` — dizi ızgarası aynıysa CT'nin `ornt`'u uygulanır; QC 131/131. Değerlendirmede GT başlıkları `src/data/gt_fixed.py` ile düzeltiliyor → TS tabanı yeniden ölçülürse AYNI GT kullanılmalı |

## Ön eğitimli modeller (JEPA kolu için) — 19 Eyl 2026

Kaynak: [04a554] + `dataset-verifier` (Rad-JEPA HF API ile ayrıca teyit edildi). Ad düzeyinde kontrol; görüntü düzeyinde denetlenmedi.

| # | Model | Durum | Etki |
|---|---|---|---|
| P1 | Rad-JEPA 3D encoder (HF `huyquoctrinh/Rad-Jepa-3D`, apache-2.0, 155 MB) | 🟡 temiz görünüyor | Ön eğitim M3D-Cap/Radiopaedia ~120k CT; adıyla LiTS/MSD/AbdomenAtlas yok. ⚠️ Repo'nun `seg/` eğitimi **M3D-Seg** kullanıyor: MSD-Liver (=LiTS), 3D-IRCADb, SLIVER07 içerir → o kodla eğitilen / yayımlanan seg modeli LiTS/IRCADb testinde sızıntılı |
| P2 | CT-FM (arXiv:2501.09001, HF `project-lighter/ct_fm_feature_extractor`) | 🟡 kirli (dış test için) | Ön eğitimde **hcc_tace_seg 477** + tcga_lihc 284 tarama (makale Ek Tablo 3) → HCC-TACE-Seg dış testiyle çakışır |
| P3 | VoCo-10k | 🟡 kirli (HCC-TACE-Seg) | FLARE23 içeriyor → L7 zinciri. LiTS içerip içermediği doğrulanmadı |
| P4 | SuPreM supervised checkpoint'leri | 🔴 kullanılamaz | AbdomenAtlas 1.1 ile eğitilmiş → LiTS/MSD içerir (L3) |
| P5 | OpenMind MAE (MIC-DKFZ ResEncL / PrimusM, CC-BY-4.0) | 🟡 temiz | Yalnız beyin MRI ön eğitimi; CT sızıntısı yok (alan farkı ayrı konu) |

## L4 kanıtı

`huggingface.co/datasets/tumor-vqa/DeepTumorVQA_1.0` → `AbdomenAtlas_ID_mapping.csv`
formatı `original id,AbdomenAtlas_id`. 1.516 satırlık alt kümede üç ayrı karaciğer öneki:

```
LiTS_liver_38,BDMAP_00000091
LiTS_liver_43,BDMAP_00000918
Decathlon_liver_43,BDMAP_00001114     <-- ayni indeks, farkli BDMAP_ID
MSD-Liver_liver_148,BDMAP_00000064
```

`Decathlon` = `MSD-Liver` = MSD Task03_Liver = LiTS. Aynı havuzun üç adı.

## Alan çapında bağlam

*State of Abdominal CT Datasets* (arXiv:2508.13626): 46 veri seti / 50.256 çalışma,
**%59.1 vaka tekrarı**, %75.3 Kuzey Amerika+Avrupa. LiTS/KiTS/BTCV "hub node" —
oradan AbdomenCT-1K, FLARE23, AbdomenAtlas'a yayılıyor.

## Çözüm protokolü

1. Provenance tablosu (önek tabanlı) → `src/provenance/build_provenance.py`
2. İçerik hash'i (öneklerin kaçırdıkları) → `src/provenance/dedup_hash.py`
3. Fiziksel tarama başına tek kanonik ID
4. Split kanonik hasta + kaynak veri seti düzeyinde ayrık → `src/data/make_splits.py`
5. Dondurulmuş dış test: HCC-TACE-Seg (TCIA) birinci aday

## L6 güncellemesi — 20 Eyl 2026 (liver-ct-51)

**Durum: 🔴 → 🟡 kısmen denetlendi** (🟢 değil).

| Koşu | Kapsam | Birebir eşleşme | Elle bakılacak |
|---|---|---|---|
| `liver-l6` (842 sorgu) | 312/842 (%37) | kanal **kapalıydı** | 5 |
| `liver-l6-round` (416 tamsayı sorgu) | **381/416 (%92)** | **0** | 1 |

Birebir arama kanalı ilk koşuda `p == rint(p)` katı testi yüzünden 842/842 sorguda kördü
(TS görüntüleri float32 gösterim hatası taşıyor, ~5e-4). Sorguların 416'sı toleransla tamsayı;
yuvarlanıp yeniden koşuldu (`src/provenance/l6_verdict.py`, `kaggle_jobs/l6_round/`).
`match_crop.py` DEĞİŞTİRİLMEDİ.

**Denebilecek:** "denetlenen kısımda LiTS kopyası saptanmadı."
**Denemeyecek:** "TS liver_lesions LiTS içermiyor."

Açık: (a) xy yeniden örneklenmiş kopya körlüğü, (b) 426 tamsayı olmayan görüntü,
(c) `Ay6NSmY8ie4fXIqK ↔ lits-106` (z 6.9, iki koşuda da işaretli), (d) 35–78 "karşılaştırılamaz"
satır. Veri: `outputs/kaggle/l6round/`, `outputs/l6_ircad/`, `outputs/l6_round/`.
