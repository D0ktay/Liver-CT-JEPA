"""UCTAN UCA HAT: onbellek -> on egitim -> ince ayar -> evaluate, GERCEK CLI'larla.

Veri gerektirmez (sentetik NIfTI). Yavas ama kritik: birim testleri modulleri ayri
ayri dogruluyor, bu test MODULLER ARASINDAKI sozlesmeleri dogruluyor -- normalizasyon
araligi, spacing tasinmasi, geri donus sekli, ckpt yukleme, evaluate uyumu.
Bu hat kirilirsa Kaggle/Lambda kosusu saatler harcayip bos doner.
"""
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFIL = "cok_kucuk"          # yama 48^3, 0.8M param -> CPU'da saniyeler


def _run(*args, cwd=ROOT):
    r = subprocess.run([sys.executable, "-m", *map(str, args)], cwd=cwd,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{args[0]} basarisiz:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def _case(tmp: Path, n: int, shape=(56, 56, 56), seed=0):
    """LiTS benzeri sentetik vaka: CT + segmentasyon (karaciger 1, tumor 2)."""
    rng = np.random.default_rng(seed + n)
    ct = rng.normal(30, 25, shape).astype(np.float32)
    seg = np.zeros(shape, np.uint8)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c = np.array(shape) // 2
    seg[((zz - c[0])**2 + (yy - c[1])**2 + (xx - c[2])**2) <= 18**2] = 1       # karaciger
    tc = c + rng.integers(-6, 7, 3)
    tum = ((zz - tc[0])**2 + (yy - tc[1])**2 + (xx - tc[2])**2) <= 5**2
    seg[tum & (seg == 1)] = 2                                                  # tumor
    ct[seg == 1] += 60.0
    ct[seg == 2] += 130.0
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(ct, aff), tmp / f"volume-{n}.nii")
    nib.save(nib.Nifti1Image(seg, aff), tmp / f"segmentation-{n}.nii")


@pytest.mark.slow
def test_onbellek_onegitim_inceayar_evaluate(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    n_case = 6
    for i in range(n_case):
        _case(raw, i)

    # --- 1) manifest (split olmadan denetim yapilamaz -> minik bir splits.csv) ---
    import pandas as pd
    man = pd.DataFrame([{
        "volume_id": f"lits-{i}", "canonical_id": f"LITS|MSD_TASK03::{i}",
        "source": "LITS", "source_family": "LITS|MSD_TASK03",
        "path": str(raw / f"volume-{i}.nii"), "n_bytes": 1, "checksum": "pending",
        "checksum_kind": "sha256", "canonical_status": "name_only",
        "license": "", "url": "",
    } for i in range(n_case)])
    cache = tmp_path / "cache"
    man_p = tmp_path / "manifest.csv"

    # dondurulmus ic test: son iki vaka
    test_ids = [f"lits-{i}" for i in (4, 5)]
    frozen = tmp_path / "internal_test.json"
    frozen.write_text(json.dumps({
        "case_ids": test_ids,
        "canonical_ids": [f"LITS|MSD_TASK03::{i}" for i in (4, 5)]}))

    # KURAL 10: test vakalari on egitim korpusunda OLMAMALI
    man[~man["volume_id"].isin(test_ids)].to_csv(man_p, index=False)
    # ince ayar onbellegi TUM vakalari icerir (egitim + dondurulmus test)
    man.to_csv(tmp_path / "manifest_all.csv", index=False)

    # --- 2) onbellek (etiketli, ROI = GT karaciger etiketi) ---
    _run("src.jepa.cache_build", "--manifest", tmp_path / "manifest_all.csv",
         "--out", cache, "--labels-glob", str(raw / "segmentation-*.nii"),
         "--liver-label", "1", "--workers", "2")
    assert len(list(cache.glob("*.npz"))) == n_case
    z = np.load(next(iter(sorted(cache.glob("*.npz")))))
    assert "liver" in z and z["liver"].sum() > z["lab"].sum() > 0     # bolge = karaciger ∪ tumor

    # --- 3) on egitim (jepa), degerlendirme sizintisi kontrolu acik ---
    pre = tmp_path / "pre_jepa"
    out = _run("src.jepa.pretrain", "--objective", "jepa", "--manifest", man_p,
               "--cache", cache, "--out", pre, "--profile", PROFIL,
               "--eval-ids", frozen, "--steps", 6, "--batch", 1,
               "--log-every", 3, "--ckpt-every", 1000, "--device", "cpu")
    assert "hicbiri korpusta yok" in out
    assert (pre / "encoder.pt").exists()
    import torch as _t
    _ck = _t.load(pre / "encoder.pt", map_location="cpu")
    # EMA (hedef) encoder EK anahtarla saklanir; "encoder" (cevrimici) ince ayarin okudugu anahtar olarak kalir
    assert set(_ck["target_encoder"]) == set(_ck["encoder"])
    assert not (pre / "collapse.json").exists(), "sentetik veride cokus tetiklenmemeli"

    # --- 4) ince ayar: jepa kolu + rastgele kontrol kolu, AYNI tohum/adim ---
    for arm, init in (("jepa", pre / "encoder.pt"), ("none", "random")):
        ft = tmp_path / f"ft_{arm}"
        _run("src.jepa.finetune_e2e", "--init", init, "--cache", cache,
             "--test-ids", frozen, "--out", ft, "--profile", PROFIL,
             "--steps", 4, "--batch", 1, "--log-every", 2, "--val-n", 1, "--val-every", 2,
             "--device", "cpu", "--gt-dir", raw)
        assert (ft / "val.jsonl").exists() and (ft / "last.pt").exists()
        d = json.loads((ft / "done.json").read_text())
        assert d["checkpoint_used"] == "last" and len(d["val_ids"]) == 1
        assert not set(d["val_ids"]) & set(test_ids)
        preds = sorted(ft.glob("pred/*.nii.gz"))
        assert len(preds) == len(test_ids), f"{arm}: {len(preds)} tahmin"

        # tahmin GT ile AYNI sekil ve affine tasimali (evaluate bunu sart kosuyor)
        for p in preds:
            i = int(p.name.split("-")[1].split(".")[0])
            gt = nib.load(str(raw / f"segmentation-{i}.nii"))
            pr = nib.load(str(p))
            assert pr.shape == gt.shape
            assert np.allclose(pr.affine, gt.affine)

        # --- 5) src.metrics.evaluate AYNI kodla olcsun (--gt-label 2 / --pred-label 1) ---
        ev = ft / "eval"
        _run("src.metrics.make_pairs", "--gt-glob", str(raw / "segmentation-*.nii"),
             "--pred-glob", str(ft / "pred" / "*.nii.gz"), "--out", ev / "pairs.csv")
        _run("src.metrics.evaluate", "--pairs", ev / "pairs.csv", "--out", ev,
             "--gt-label", "2", "--pred-label", "1")
        s = json.loads((ev / "summary.json").read_text())
        assert s["n_cases"] == len(test_ids)
        assert "dice_median" in s["voxel"]


@pytest.mark.slow
def test_onegitim_degerlendirme_vakasi_korpustaysa_BASLAMAZ(tmp_path):
    """KURAL 10 -- bu testin kirmizi olmasi, hattin sessizce sizintili kosmasindan iyidir."""
    raw = tmp_path / "raw"; raw.mkdir()
    for i in range(3):
        _case(raw, i)
    import pandas as pd
    man_p = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "volume_id": f"lits-{i}", "canonical_id": f"LITS|MSD_TASK03::{i}",
        "source": "LITS", "source_family": "LITS|MSD_TASK03",
        "path": str(raw / f"volume-{i}.nii"), "n_bytes": 1, "checksum": "pending",
        "checksum_kind": "sha256", "canonical_status": "name_only", "license": "", "url": "",
    } for i in range(3)]).to_csv(man_p, index=False)
    cache = tmp_path / "cache"
    _run("src.jepa.cache_build", "--manifest", man_p, "--out", cache)
    frozen = tmp_path / "internal_test.json"
    frozen.write_text(json.dumps({"case_ids": ["lits-2"],
                                  "canonical_ids": ["LITS|MSD_TASK03::2"]}))
    r = subprocess.run([sys.executable, "-m", "src.jepa.pretrain", "--objective", "jepa",
                        "--manifest", str(man_p), "--cache", str(cache),
                        "--out", str(tmp_path / "pre"), "--profile", PROFIL,
                        "--eval-ids", str(frozen), "--steps", "2", "--batch", "1",
                        "--device", "cpu"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0, "sizintili korpusla on egitim BASLAMAMALIYDI"
    assert "KURAL 10" in (r.stdout + r.stderr)
