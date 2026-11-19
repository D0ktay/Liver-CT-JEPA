"""On egitim onbellegi: ham NIfTI -> 1 mm izotropik, HU pencereli, ROI kirpilmis .npz.

    python -m src.jepa.cache_build --manifest outputs/pretrain_manifest.csv \
        --config configs/train.yaml --out data/cache/iso1mm

NEDEN ONBELLEK: olculen darbogaz hesap degil disk I/O. Her epoch'ta 500 MB'lik ham
NIfTI'yi acip yeniden ornekleme yapmak GPU'yu bos bekletir. Bir kez yazilir, cok kez okunur.

GERI DONUS BILGISI KAYBOLMAZ. Her .npz `meta` tasir: ULASILAN spacing, hedef spacing,
orijinal affine, RAS yon donusumu (`ornt`), kirpma kutusu, orijinal sekil, HU penceresi.
Tahmin bunlarla orijinal izgaraya doner ve `src/metrics/evaluate.py` ayni kodla olcer.

uint8 KAYBI, bilerek: HU penceresi (-200, 250) = 450 HU, 256 seviyeye bolunuyor ->
seviye basina 1.76 HU. Hacim basina 4x yer kazandiriyor (1 bayt/voksel).
`src/jepa/prep.py` de ayni secimi yapiyor. Daha ince gerekirse `--dtype float16`.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from src.jepa.data import GeometryError, canonical_ras, hu_window as _hu, resample


class CacheError(RuntimeError):
    """Onbellek uretilemedi. Sessizce atlamak yerine firlat."""


def load_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, ...], np.ndarray, list, tuple[int, ...]]:
    """NIfTI -> (RAS dizi, RAS spacing, orijinal affine, ornt, ORIJINAL dizi sekli).

    Son eleman bilerek ayri dondurulur: RAS'a cevirdikten sonraki sekli "orijinal"
    sanmak, geri donuste eksenleri takas edilmis bir maske uretir ve
    `src.metrics.evaluate` bunu "sekil uyusuyor" diye kabul eder (41,51,61 vs
    51,41,61 ayni sayilar, farkli eksenler). Sessiz ve olumcul.

    Bozuk/okunamayan dosya sessizce atlanmaz -- korpus eksik kalirsa bunu bilmeliyiz.
    """
    import nibabel as nib
    p = Path(path)
    try:
        img = nib.load(str(p))
        arr = np.asanyarray(img.dataobj)
    except Exception as e:                                   # noqa: BLE001
        raise CacheError(f"{p} okunamadi: {type(e).__name__}: {e}") from e
    if arr.ndim != 3:
        raise CacheError(f"{p}: 3B degil, sekil {arr.shape}")
    aff = np.asarray(img.affine, float)
    sp = tuple(float(x) for x in img.header.get_zooms()[:3])
    src_shape = tuple(int(n) for n in arr.shape)
    ras, sp_ras, ornt = canonical_ras(np.asarray(arr), aff, sp)
    return ras, sp_ras, aff, ornt, src_shape


def load_on_grid(path: str | Path, ornt, src_shape) -> tuple[np.ndarray, tuple[float, ...]]:
    """Etiket/ROI dizisini CT'NIN yon donusumuyle RAS'a cevir (kendi affine'iyle DEGIL).

    🔴 22 Eyl 2026 (kalite kontrol yakaladi, lits-51): LiTS'te etiket ile CT AYNI VOKSEL
    IZGARASINDADIR (yarisma voksel voksel olcer), ama lits-48..52'nin segmentasyon BASLIKLARI
    bozuk (spacing 1x1x1 ve farkli yon). Etiketi KENDI affine'iyle RAS'a cevirmek onu CT'ye gore
    AYNALIYORDU: karaciger maskesi midenin ustune dusuyordu (maske altinda medyan -29 HU).
    Dogru olan: dizi izgarasi ayniysa CT'nin `ornt`'unu uygula.
    -> (RAS dizi, etiketin KENDI basligindaki spacing'in RAS sirasi -- yalniz karsilastirma icin)
    """
    import nibabel as nib
    from nibabel.orientations import apply_orientation
    p = Path(path)
    try:
        img = nib.load(str(p))
        arr = np.asanyarray(img.dataobj)
    except Exception as e:                                   # noqa: BLE001
        raise CacheError(f"{p} okunamadi: {type(e).__name__}: {e}") from e
    if tuple(arr.shape) != tuple(src_shape):
        raise CacheError(f"etiket dizisi {arr.shape} != CT dizisi {tuple(src_shape)} ({p}) -- "
                         "ayni voksel izgarasinda degil, CT yonu uygulanamaz")
    o = np.asarray(ornt, float)
    zo = [float(z) for z in img.header.get_zooms()[:3]]
    sp = [0.0, 0.0, 0.0]
    for i, (t, _) in enumerate(o):
        sp[int(t)] = zo[i]
    return apply_orientation(arr, o), tuple(sp)


def label_orientation_differs(label_path: str | Path, ct_ornt) -> bool:
    """Etiketin KENDI affine'i CT'ninkinden farkli bir yon mu soyluyor? (rapor icin)"""
    import nibabel as nib
    from nibabel.orientations import io_orientation
    return not np.array_equal(io_orientation(nib.load(str(label_path)).affine),
                              np.asarray(ct_ornt, float))


def roi_bbox(mask: np.ndarray, spacing, margin_mm: float = 10.0) -> tuple[slice, ...]:
    """Maskenin kutusu + mm cinsinden pay. Pay VOKSEL degil MM (kural 6 ruhu):
    anizotropik vokselde 5 mm, 1 mm eksende 5 voksel, 5 mm eksende 1 vokseldir."""
    idx = np.argwhere(np.asarray(mask) > 0)
    if idx.size == 0:
        raise CacheError("ROI maskesi bos -> kutu tanimsiz")
    pad = [math.ceil(float(margin_mm) / float(s)) for s in spacing]
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + 1 + np.asarray(pad), mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def build_one(ct_path: str | Path, target=(1.0, 1.0, 1.0), hu_window=(-200.0, 250.0),
              label_path: str | Path | None = None, roi_path: str | Path | None = None,
              tumor_label: int = 2, roi_label: int = 0, margin_mm: float = 10.0,
              min_voxels: int = 1000, dtype: str = "uint8",
              trust_ct_spacing: bool = False, liver_label: int | None = None,
              hu_offset_auto: bool = False, true_spacing=None) -> dict:
    """Tek hacim -> onbellek sozlugu. Goruntu trilineer, ETIKET EN YAKIN KOMSU."""
    ct, sp, aff, ornt, src_shape_t = load_volume(ct_path)
    src_shape = list(src_shape_t)
    if true_spacing is not None:
        # LiTS 28-47: baslik sahte 1x1x1, gercek deger IRCADb DICOM'undan
        # (configs/lits_2847_true_spacing.csv). fix2847 dersi: sahte baslik olcumu degil
        # GIRDIYI bozuyor -- duzeltilmezse bu vakalar modele ~1.4x buyutulmus anatomi olarak
        # girer (0.7 mm vokseller 1 mm sanilir). MVE bunu YAPMIYORDU.
        ts, tshape = true_spacing
        if tuple(src_shape) != tuple(int(n) for n in tshape):
            raise CacheError(f"{ct_path}: sekil {src_shape} != tablo {tuple(tshape)} (yanlis vaka?)")
        new_sp = [0.0, 0.0, 0.0]
        for i, (t, _) in enumerate(ornt):                # dosya ekseni i -> RAS ekseni t
            new_sp[int(t)] = float(ts[i])
        if np.allclose(sp, 1.0, atol=1e-3):
            print(f"[UYARI] {Path(ct_path).name}: sahte baslik {sp} -> gercek spacing "
                  f"{tuple(new_sp)} (configs/lits_2847_true_spacing.csv)", flush=True)
            sp = tuple(new_sp)
        elif not np.allclose(sp, new_sp, rtol=0.02):
            # ne sahte (1,1,1) ne de tablodaki deger: hangisinin dogru oldugu BILINMIYOR
            raise CacheError(f"{ct_path}: baslik spacing {sp} ne 1x1x1 ne de tablo {tuple(new_sp)}")
    if ct.size < min_voxels:
        raise CacheError(f"{ct_path}: {ct.size} voksel, en az {min_voxels} bekleniyor")
    # HU OFSET KORUMASI: abdomen CT'sinde govde disi HAVA (~-1000 HU) hemen her zaman
    # vardir. En dusuk deger >= 0 ise dosya buyuk olasilikla +1024 ofsetle (isaretsiz)
    # saklanmistir; pencere (-200, 250) o zaman tum dokuyu ust uca yapistirir ve hacim
    # SESSIZCE bombos bir goruntuye doner. Otomatik duzeltme bir VARSAYIMDIR -> acik
    # bayrak + vaka basina uyari; bayrak yoksa FIRLAT.
    hu_shift = 0.0
    if float(np.min(ct)) >= 0.0 and float(np.max(ct)) > 1200.0:
        if not hu_offset_auto:
            raise CacheError(f"{ct_path}: en dusuk deger {float(np.min(ct)):.0f} >= 0 -- HU ofsetli "
                             "(isaretsiz) saklanmis gorunuyor. --hu-offset-auto ile 1024 cikarilir.")
        hu_shift = -1024.0
        print(f"[UYARI] {Path(ct_path).name}: min {float(np.min(ct)):.0f} >= 0 -> 1024 cikarildi "
              "(--hu-offset-auto, VARSAYIM)", flush=True)
        ct = np.asarray(ct, np.float32) + hu_shift

    lab = None
    label_ornt_fixed = False
    if label_path is not None:
        lab_raw, lab_sp = load_on_grid(label_path, ornt, src_shape_t)
        if lab_raw.shape != ct.shape:
            raise CacheError(f"etiket {lab_raw.shape} != CT {ct.shape} ({label_path})")
        if label_orientation_differs(label_path, ornt):
            label_ornt_fixed = True
            print(f"[UYARI] {Path(label_path).name}: etiket basligi CT'den FARKLI yon soyluyor -> "
                  "CT'nin yonu uygulandi (ayni voksel izgarasi)", flush=True)
        if not np.allclose(lab_sp, sp, rtol=0, atol=1e-3):
            # GERCEK BULGU (Kaggle MVE v2, 21 Eyl 2026): LiTS lits-48..52'nin
            # segmentasyon basliklari (1,1,1) tasiyor, CT'ler gercek degeri
            # (orn. 0.9766, 0.9766, 2.5). Eski TS kosusunda bu 5 vakanin neden
            # dustugunun kesin aciklamasi budur (gunluk "teyit edilmedi" diyordu).
            # Sekiller ayniysa etiket izgarasi ZATEN CT izgarasidir ve iki basliktan
            # biri yanlistir; ama hangisinin dogru oldugu bir VARSAYIMDIR ->
            # sessizce degil, acik bayrakla ve uyariyla.
            if not trust_ct_spacing:
                raise CacheError(
                    f"etiket spacing {lab_sp} != CT spacing {sp} ({label_path}). "
                    "Sekiller ayniysa --trust-ct-spacing ile CT basligini kullanabilirsin "
                    "(VARSAYIM, log'a yazilir).")
            print(f"[UYARI] {Path(label_path).name}: etiket spacing {lab_sp} yerine CT "
                  f"spacing'i {sp} kullaniliyor (--trust-ct-spacing)", flush=True)
        # tumor_label < 0 -> bu kaynakta TUMOR etiketi YOK (TS-v2, AMOS: yalniz karaciger).
        # 'lab' YAZILMAZ; egitimde o vakanin tumor kanali kaybi KAPATILIR ("tumor yok" diye
        # ogretilmez -- etiketlenmemis tumor olabilir).
        has_tumor_label = int(tumor_label) >= 0
        lab = (np.asarray(lab_raw) == tumor_label).astype(np.uint8)
        # KARACIGER BOLGESI = karaciger ∪ tumor (bolge-tabanli egitim, kanal 0).
        # 22 Eyl teshisi: model yalniz karaciger kutusunda egitilip TUM GOVDEDE test
        # edilince tahminlerin cogu karacigerin DISINA dustu. Cozum: tam hacimde egit,
        # karacigeri de OGREN, tumoru modelin KENDI karaciger tahminiyle filtrele.
        # liver_label=None -> bu kaynakta karaciger etiketi YOK (orn. MSD Task08);
        # 'liver' yazilmaz ve egitimde o kanalin kaybi bu vaka icin kapatilir.
        if liver_label is not None:
            ids = (int(liver_label), int(tumor_label)) if has_tumor_label else (int(liver_label),)
            liv = np.isin(np.asarray(lab_raw), ids).astype(np.uint8)
            if not liv.any():
                raise CacheError(f"{label_path}: karaciger etiketi ({liver_label}) bos -- "
                                 "yanlis etiket haritasi? (LiTS: 1=karaciger, 2=tumor)")

    # --- ROI: ayri bir maske verildiyse ondan, yoksa tum hacim ---
    ras_shape = tuple(ct.shape)          # ROI kirpmadan ONCEKI RAS sekli
    bbox = tuple(slice(0, n) for n in ct.shape)
    if roi_path is not None:
        roi, roi_sp = load_on_grid(roi_path, ornt, src_shape_t)
        roi_sp = sp                                   # kutu payi CT'nin (dogrulanmis) spacing'iyle
        if roi.shape != ct.shape:
            raise CacheError(f"ROI {roi.shape} != CT {ct.shape} ({roi_path})")
        # roi_label=0 -> "sifirdan buyuk her sey" (GT tumoru DE kutuyu belirler).
        # roi_label=1 -> yalniz karaciger. Ikisi ayri sey: birincisinde
        # "tumor etiketi on egitimde kullanilmiyor" iddiasi YANLIS olur.
        roi_m = (np.asarray(roi) > 0) if int(roi_label) == 0 else (np.asarray(roi) == int(roi_label))
        bbox = roi_bbox(roi_m, roi_sp, margin_mm)
    ct = ct[bbox]
    if lab is not None:
        lab = lab[bbox]
        if liver_label is not None:
            liv = liv[bbox]

    img_f, got = resample(np.asarray(ct, np.float32), sp, target, order=1)
    out: dict = {"spacing": tuple(float(x) for x in got)}
    norm = _hu(img_f, hu_window, out_range=(0.0, 1.0))
    if dtype == "uint8":
        out["img"] = np.round(norm * 255).astype(np.uint8)
    elif dtype == "float16":
        out["img"] = norm.astype(np.float16)
    else:
        raise CacheError(f"bilinmeyen dtype: {dtype}")

    if lab is not None:
        lab_r, lab_got = resample(lab, sp, target, order=0, is_mask=True)
        if lab_r.shape != out["img"].shape:
            raise CacheError(f"etiket {lab_r.shape} != goruntu {out['img'].shape} "
                             "-- yeniden ornekleme ayristi")
        if not np.allclose(lab_got, got, rtol=0, atol=1e-9):
            raise CacheError(f"etiket spacing {lab_got} != goruntu spacing {got}")
        if has_tumor_label:
            out["lab"] = lab_r.astype(np.uint8)
        elif liver_label is None:
            raise CacheError(f"{label_path}: ne tumor ne karaciger etiketi istendi -- etiket dosyasi bos yere verilmis")
        if liver_label is not None:
            liv_r, _ = resample(liv, sp, target, order=0, is_mask=True)
            if liv_r.shape != out["img"].shape:
                raise CacheError(f"karaciger {liv_r.shape} != goruntu {out['img'].shape}")
            out["liver"] = liv_r.astype(np.uint8)

    out["meta"] = {
        "src_path": str(ct_path), "src_shape": src_shape,
        # geri donus icin sart: RAS dizi sekli, RAS spacing'i ve onbellek sekli
        "ras_shape": list(ras_shape), "src_spacing_ras": [float(x) for x in sp],
        "res_shape": list(out["img"].shape),
        "spacing": list(out["spacing"]), "target": list(target),
        "affine": np.asarray(aff, float).tolist(), "ornt": list(ornt),
        "bbox": [[s.start, s.stop] for s in bbox],
        "crop_shape": list(ct.shape), "hu_window": list(hu_window),
        "dtype": dtype, "tumor_label": tumor_label if (label_path and int(tumor_label) >= 0) else None,
        "liver_label": liver_label if label_path else None, "hu_shift": hu_shift,
        "label_ornt_fixed": label_ornt_fixed,
    }
    return out


def write_cache(d: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"img": d["img"], "meta": np.array(json.dumps(d["meta"]))}
    if "lab" in d:
        arrays["lab"] = d["lab"]
    if "liver" in d:
        arrays["liver"] = d["liver"]
    np.savez_compressed(p, **arrays)
    return p


def _job(job) -> tuple[str, str | None]:
    """Tek vaka (surec havuzu icin ust duzey fonksiyon). Hata METNI doner, yutulmaz."""
    cid, dst, kw = job
    try:
        write_cache(build_one(**kw), dst)
        return cid, None
    except (CacheError, GeometryError) as e:
        return cid, str(e)


# ------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    import yaml

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels-glob", default="", help="etiketli kol icin, orn. 'data/raw/LiTS/segmentation-*.nii'")
    ap.add_argument("--roi-glob", default="",
                    help="ROI maskesi (TS liver); BOS BIRAK -> tum hacim. "
                         "🔴 Dondurulmus TEST vakalarinda bos birakilmali: GT'den turetilmis "
                         "bir kutu tahmini GT anatomisine hapseder, model kutu disinda yanlis "
                         "pozitif uretemez ve precision/marj sayilari yukari yanli cikar.")
    ap.add_argument("--roi-label", type=int, default=0,
                    help="0 -> maskenin >0 olan her yeri; 1 -> yalniz karaciger "
                         "(GT tumor etiketi kutuyu BELIRLEMEZ)")
    ap.add_argument("--dtype", choices=["uint8", "float16"], default="uint8")
    ap.add_argument("--trust-ct-spacing", action="store_true",
                    help="etiket basligi CT'den farkliysa (sekil ayni) CT'ninkini kullan. "
                         "LiTS lits-48..52 bunu gerektiriyor; VARSAYIMDIR, log'a yazilir.")
    ap.add_argument("--tumor-label", type=int, default=2, help="LiTS/MSD Task08: 2")
    ap.add_argument("--liver-label", type=int, default=-1,
                    help="karaciger etiketi (LiTS: 1). -1 -> kaynakta karaciger etiketi YOK "
                         "(MSD Task08: 1=DAMAR, karaciger degil!) -> 'liver' yazilmaz")
    ap.add_argument("--roi-margin-mm", type=float, default=-1.0,
                    help="ROI kutusu payi (mm). <0 -> config geometry.roi_margin_mm (10)")
    ap.add_argument("--hu-offset-auto", action="store_true",
                    help="min>=0 olan (isaretsiz saklanmis) hacimden 1024 cikar; VARSAYIM, loglanir")
    ap.add_argument("--allow-errors", type=float, default=0.0,
                    help="ETIKETSIZ korpus icin: bu ORANA kadar bozuk hacim atlanir (errors.csv'ye "
                         "yazilir). Etiketli kolda 0 birak. Atlananlar manifest'ten de dusurulmeli.")
    ap.add_argument("--spacing-table", type=Path, default=None,
                    help="basligi sahte olan vakalar icin gercek spacing (LiTS 28-47): "
                         "configs/lits_2847_true_spacing.csv")
    ap.add_argument("--workers", type=int, default=1,
                    help="paralel surec. Olculen: tam hacim ~33 sn/vaka tek cekirdekte")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(a.config.read_text()) if a.config.exists() else {}
    paths_cfg = yaml.safe_load(a.paths.read_text()) if a.paths.exists() else {}
    # HU penceresi TEK YERDE tanimli: configs/paths.yaml. Kopyalanmaz, okunur.
    hu = tuple(paths_cfg.get("metrics", {}).get("hu_window", (-200.0, 250.0)))
    target = tuple(cfg.get("geometry", {}).get("spacing_mm", (1.0, 1.0, 1.0)))
    margin = (a.roi_margin_mm if a.roi_margin_mm >= 0
              else float(cfg.get("geometry", {}).get("roi_margin_mm", 10.0)))
    print(f"[cache] hedef spacing {target} | HU {hu} | pay {margin} mm | dtype {a.dtype}")

    man = pd.read_csv(a.manifest)
    if a.limit:
        man = man.head(a.limit)

    from src.data.case_id import case_id_from_path, glob_paths
    labels = {case_id_from_path(p): p for p in glob_paths(a.labels_glob)} if a.labels_glob else {}
    rois = {case_id_from_path(p): p for p in glob_paths(a.roi_glob)} if a.roi_glob else {}
    if a.labels_glob and not labels:
        print(f"[UYARI] --labels-glob '{a.labels_glob}' hicbir dosya bulmadi")

    a.out.mkdir(parents=True, exist_ok=True)
    ok, skip, errs, t0 = 0, 0, [], time.time()
    if a.labels_glob:
        eksik = [str(v) for v in man["volume_id"].astype(str) if v not in labels]
        if eksik:
            # Etiketli kol icin etiketsiz vaka = sessizce kuculen egitim kumesi.
            raise CacheError(f"{len(eksik)} vakanin etiketi bulunamadi (orn. {eksik[:5]}) -- "
                             "--labels-glob ile case_id kurali uyusmuyor")
    table = {}
    if a.spacing_table:
        from src.baseline.respacing import load_spacing_table
        table = load_spacing_table(a.spacing_table)
    jobs = []
    for _, row in man.iterrows():
        cid = str(row["volume_id"])
        dst = a.out / f"{cid}.npz"
        if dst.exists() and not a.overwrite:
            skip += 1
            continue
        jobs.append((cid, str(dst), dict(
            ct_path=str(row["path"]), target=target, hu_window=hu,
            label_path=labels.get(cid), roi_path=rois.get(cid), roi_label=a.roi_label,
            margin_mm=margin, dtype=a.dtype, trust_ct_spacing=a.trust_ct_spacing,
            tumor_label=a.tumor_label, hu_offset_auto=a.hu_offset_auto,
            true_spacing=((table[cid]["spacing"], table[cid]["shape"]) if cid in table else None),
            liver_label=None if a.liver_label < 0 else a.liver_label)))
    if a.workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        try:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                it = ex.map(_job, jobs)
                for i, (cid, err) in enumerate(it, 1):
                    ok, errs = (ok + 1, errs) if err is None else (ok, errs + [(cid, err)])
                    if i % 20 == 0:
                        print(f"  {i + skip}/{len(man)} ({time.time() - t0:.0f} sn)", flush=True)
        except BrokenProcessPool as e:
            # Buyuk hacimde isci bellekten oldurulurse (OOM) havuz kirilir. Yazilanlar diskte;
            # ayni komut --workers dusurulerek yeniden calistirilinca kaldigi yerden surer.
            raise CacheError(f"isci sureci oldu (buyuk olasilikla RAM): {e}. "
                             f"--workers {max(1, a.workers // 2)} ile yeniden calistir "
                             "(yazilmis .npz'ler atlanir).") from e
    else:
        for i, j in enumerate(jobs, 1):
            cid, err = _job(j)
            ok, errs = (ok + 1, errs) if err is None else (ok, errs + [(cid, err)])
            if i % 20 == 0:
                print(f"  {i + skip}/{len(man)} ({time.time() - t0:.0f} sn)", flush=True)

    total = sum(f.stat().st_size for f in a.out.glob("*.npz"))
    print(f"[cache] {ok} yazildi, {skip} atlandi, {len(errs)} hata | "
          f"{total / 1e9:.2f} GB | {time.time() - t0:.0f} sn")
    if errs:
        # Sessizce eksik korpusla egitmek, sizinti kadar sinsi bir hatadir.
        err_file = a.out / "errors.csv"
        err_file.write_text("case_id,error\n" + "\n".join(f'{c},"{m}"' for c, m in errs))
        print(f"[cache] HATALAR -> {err_file}")
        for c, m in errs[:5]:
            print(f"  {c}: {m}")
        if len(errs) <= a.allow_errors * max(1, len(man)):
            print(f"[UYARI] {len(errs)} hacim ATLANDI (--allow-errors {a.allow_errors}); "
                  f"manifest'ten DUSURULMELI: {err_file}")
            return 0
        raise CacheError(f"{len(errs)} hacim onbellege alinamadi -- korpus EKSIK. "
                         f"Duzelt ya da manifest'ten cikar: {err_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def reconstruct(prob: np.ndarray, meta: dict) -> np.ndarray:
    """Onbellek uzayindaki tahmini ORIJINAL diziye dondur (build_one'un tam tersi).

    Sira: (1 mm, kirpilmis, RAS) -> kirpma sekline geri ornekle -> ROI kutusuna
    yerlestir (disi 0) -> RAS'tan hacmin ORIJINAL dizi yonune don.

    Bu adim atlanirsa `src.metrics.evaluate` tahmini GT ile hizalayamaz; sekil
    tutsa bile yon/kirpma kaydigi icin mm metrikleri (hedefleme hatasi, marj
    ihlali) SESSIZCE yanlis cikar. fix2847 dersi: sahte baslik olcumu degil
    GIRDIYI bozuyordu.
    """
    from nibabel.orientations import apply_orientation
    prob = np.asarray(prob, np.float32)
    if prob.ndim != 3:
        raise CacheError(f"tahmin 3B olmali: {prob.shape}")
    want = tuple(int(n) for n in meta["res_shape"])
    if prob.shape != want:
        raise CacheError(f"tahmin sekli {prob.shape} onbellek sekli {want} degil -- "
                         "yanlis vakanin meta'si ya da kayan pencere kirpmasi bozuk")
    crop = tuple(int(n) for n in meta["crop_shape"])

    back, _ = resample(prob, meta["spacing"], _crop_spacing(meta), order=1)
    if back.shape != crop:
        # yuvarlama kaymasini kirp/doldur -- sessiz kayma yerine acik duzeltme
        sl = tuple(slice(0, min(a, b)) for a, b in zip(back.shape, crop))
        fixed = np.zeros(crop, np.float32)
        fixed[sl] = back[sl]
        back = fixed

    ras_shape = tuple(int(n) for n in meta["ras_shape"])
    out = np.zeros(ras_shape, np.float32)
    out[tuple(slice(a, b) for a, b in meta["bbox"])] = back
    ornt = np.asarray(meta["ornt"], float)
    from src.jepa.data import inverse_ornt
    out = apply_orientation(out, inverse_ornt(ornt))
    src_shape = tuple(int(n) for n in meta["src_shape"])
    if out.shape != src_shape:
        raise CacheError(f"geri donus sekil tutmuyor: {out.shape} != orijinal {src_shape}")
    return out


def _crop_spacing(meta: dict) -> tuple[float, ...]:
    """Kirpma sonrasi, yeniden ornekleme ONCESI spacing (RAS eksenlerinde)."""
    return tuple(float(s) for s in meta["src_spacing_ras"])


# ----------------------------------------- onbellek -> model girdisi (TEK YOL)
def _unit(img: np.ndarray) -> np.ndarray:
    """Onbellek dizisini [0, 1]'e cevir. uint8 ve float16 disi dtype REDDEDILIR."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype in (np.float16, np.float32):
        return img.astype(np.float32)
    raise CacheError(f"onbellek dtype'i beklenmedik: {img.dtype} (uint8 ya da float16 olmali)")


def to_model_input(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Onbellek dizisi -> modelin bekledigi aralik (`cfg["out_range"]`).

    🔴 metrics-reviewer bulgusu 1a (21 Eyl 2026): onbellek `out_range=(0, 1)` ile
    yaziliyor (uint8'e nicemlemek icin), `predict_volume` ise girdiyi
    `cfg["out_range"]` = (-1, 1) ile olcekliyordu. Arada donusum yapan hicbir kod
    yoktu -> egitim ve cikarim SESSIZCE ayrisiyordu. HU = [-200, 0, 25, 100, 250]
    icin onbellek [0, .444, .5, .667, 1] verirken model (-1, -.111, 0, .333, 1)
    bekliyordu; fark -1.0 ile 0.0 arasi ve DOGRUSAL DEGIL, kaydirmali.

    Egitim tarafi bu fonksiyonu, cikarim tarafi `to_hu` + `predict_volume`'u kullanir;
    ikisi birebir ayni sonucu verir (testle sabit).
    """
    a, b = (float(cfg["out_range"][0]), float(cfg["out_range"][1]))
    return (_unit(img) * (b - a) + a).astype(np.float32)


def to_hu(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Onbellek dizisi -> sozde HU (pencere icinde). `predict_volume` ham HU bekler
    ve `hu_window`'u kendisi uygular; bu fonksiyon nicemlemeyi geri acar.
    Pencere disi bilgi onbellege yazilirken zaten kirpildi -- geri gelmez."""
    lo, hi = (float(cfg["hu_window"][0]), float(cfg["hu_window"][1]))
    return (_unit(img) * (hi - lo) + lo).astype(np.float32)
