"""Topluluk: olasilik ortalamasi + sizinti kurallari (last.pt zorunlu, kol klasorune yazmaz)."""
import numpy as np
import pytest

from src.jepa.ensemble import average_probs, main


def test_ortalama_esit_agirlikli_ve_sekil_uyusmazsa_firlatir():
    a = np.zeros((2, 4, 4, 4), np.float32); b = np.ones((2, 4, 4, 4), np.float32)
    m = average_probs([a, b])
    assert m.shape == a.shape and float(m.mean()) == 0.5
    with pytest.raises(ValueError):
        average_probs([a, np.ones((2, 4, 4, 5), np.float32)])
    with pytest.raises(ValueError):
        average_probs([])


def test_best_pt_ile_topluluk_kurulamaz_last_pt_sart(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    (tmp_path / "a" / "best.pt").write_bytes(b"x"); (tmp_path / "b" / "best.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="last.pt yok"):
        main(["--arms", str(tmp_path / "a"), str(tmp_path / "b"), "--cache", str(tmp_path),
              "--test-ids", str(tmp_path / "ids.json"), "--gt-dir", str(tmp_path), "--out", str(tmp_path / "ens")])


def test_tek_kol_reddedilir(tmp_path):
    with pytest.raises(SystemExit, match="en az 2"):
        main(["--arms", str(tmp_path), "--cache", str(tmp_path), "--test-ids", str(tmp_path / "i.json"),
              "--gt-dir", str(tmp_path), "--out", str(tmp_path / "e")])
