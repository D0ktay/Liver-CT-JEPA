"""On egitim CLI yardimcilari -- VERI OLMADAN calisir."""
import json

import numpy as np
import pytest

from src.jepa.cache_build import CacheError
from src.jepa.pretrain import assert_cache_spacing, cache_paths, lr_at


def _npz(tmp_path, name, spacing, shape=(8, 8, 8)):
    p = tmp_path / f"{name}.npz"
    np.savez_compressed(p, img=np.zeros(shape, np.uint8),
                        meta=np.array(json.dumps({"spacing": list(spacing)})))
    return p


def _man(tmp_path, ids):
    import pandas as pd
    p = tmp_path / "man.csv"
    pd.DataFrame([{"volume_id": i, "canonical_id": f"X::{i}"} for i in ids]).to_csv(p, index=False)
    return p


# ------------------------------------------------------------------- spacing
def test_onbellek_spacingi_hedeften_cok_sapiyorsa_firlatir():
    """On egitim spacing'i modele iletmiyor (parti tek spacing ister) -> hepsinin
    hedefe yakin oldugu ASSERT EDILMELI. Aksi halde mm tabanli konum kodlamasi
    sessizce yanlis olculer kullanir."""
    with pytest.raises(CacheError, match="spacing"):
        assert_cache_spacing([{"spacing": [1.0, 1.0, 2.5]}], target=(1.0, 1.0, 1.0), rtol=0.01)


def test_yuvarlama_kaynakli_kucuk_sapma_kabul_edilir():
    """512 x 0.700000 mm -> ulasilan 1.001117 mm. Bu GERCEK LiTS geometrisi ve
    reddedilmemeli (metrics-reviewer bulgusu 2 ile ayni tuzak)."""
    assert_cache_spacing([{"spacing": [1.0, 1.001117, 1.001117]}],
                         target=(1.0, 1.0, 1.0), rtol=0.005)


def test_hata_mesaji_hangi_vakayi_soyler():
    with pytest.raises(CacheError, match="kotu-vaka"):
        assert_cache_spacing([{"spacing": [1.0, 1.0, 1.0], "id": "iyi"},
                              {"spacing": [1.0, 1.0, 3.0], "id": "kotu-vaka"}],
                             target=(1.0, 1.0, 1.0), rtol=0.005)


# ------------------------------------------------------------------ manifest
def test_eksik_onbellek_sessizce_atlanmaz(tmp_path):
    _npz(tmp_path, "a", (1.0, 1.0, 1.0))
    with pytest.raises(CacheError, match="onbellegi yok"):
        cache_paths(_man(tmp_path, ["a", "b"]), tmp_path)


def test_tum_onbellek_varsa_liste_doner(tmp_path):
    for n in ("a", "b"):
        _npz(tmp_path, n, (1.0, 1.0, 1.0))
    assert len(cache_paths(_man(tmp_path, ["a", "b"]), tmp_path)) == 2


# ------------------------------------------------------------------------ lr
def test_warmup_ve_kosinus():
    cfg = {"optim": {"lr": 5e-4, "min_lr": 1e-5, "warmup_frac": 0.1}}
    assert lr_at(0, 1000, cfg) < lr_at(99, 1000, cfg)          # warmup ciksi
    assert lr_at(99, 1000, cfg) == pytest.approx(5e-4)          # tepe
    assert lr_at(999, 1000, cfg) == pytest.approx(1e-5, abs=1e-6)
    assert all(lr_at(i, 1000, cfg) >= lr_at(i + 1, 1000, cfg) for i in range(100, 999))
