"""TotalSegmentator hizlandirma testleri -- TS CALISTIRMADAN.

Hedeflenen hatalar:
  - Parcalara bolerken bir vaka iki GPU'da ya da hicbirinde kosar (eksik tahmin sessizce).
  - --limit parcalamadan SONRA uygulanir -> 'ilk 5 vaka' iki GPU'da farkli 10 vaka olur.
  - Hizli yol (ayni surec) CLI'dan farkli ayarla TS cagirir -> farkli maske.
  - Bir GPU sureci coker, digeri biter, basarili gorunur.
  - Maske karsilastiricisi 'ayni' der ama aslinda grid/affine farkli.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from src.baseline.run_totalsegmentator import parse_shard, select_cases, ts_argv


def test_parcalar_ayrik_ve_birlesimi_tum_vakalar():
    cases = [(f"lits-{i}", Path(f"v{i}")) for i in range(131)]
    parts = [select_cases(cases, limit=0, shard=(k, 2)) for k in range(2)]
    ids = [c for p in parts for c, _ in p]
    assert sorted(ids) == sorted(c for c, _ in cases) and len(ids) == len(set(ids))
    assert abs(len(parts[0]) - len(parts[1])) <= 1          # yuk dengeli


def test_limit_parcalamadan_once_uygulanir():
    cases = [(f"c{i:03d}", Path(f"v{i}")) for i in range(50)]
    parts = [select_cases(cases, limit=5, shard=(k, 2)) for k in range(2)]
    assert sorted(c for p in parts for c, _ in p) == [f"c{i:03d}" for i in range(5)]


@pytest.mark.parametrize("bad", ["2/2", "-1/2", "0/0", "a/b", "1", "1/2/3"])
def test_gecersiz_parca_firlatir(bad):
    with pytest.raises(ValueError):
        parse_shard(bad)


def test_parca_ayristirma():
    assert parse_shard("0/1") == (0, 1) and parse_shard("1/2") == (1, 2)


def test_iki_motor_ayni_ts_argumanlarini_kullanir(tmp_path, monkeypatch):
    """Hizli yol TS'in KENDI CLI main()'ini ayni surecte cagirir: ayarlar tanim geregi
    ayni. Burada iki yolun TS'e verdigi argv'nin birebir ayni oldugu sinanir."""
    from src.baseline import run_totalsegmentator as rt
    seen = {}

    def fake_run(cmd, check):
        seen["cli"] = cmd[1:]
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"x")

    def fake_main():
        seen["inproc"] = sys.argv[1:]
        Path(sys.argv[sys.argv.index("-o") + 1]).write_bytes(b"x")

    monkeypatch.setattr(rt.subprocess, "run", fake_run)
    monkeypatch.setattr(rt, "_ts_main", lambda: fake_main)
    ct = tmp_path / "volume-3.nii"
    rt.run_case(ct, tmp_path / "a.nii.gz", device="gpu", engine="cli")
    rt.run_case(ct, tmp_path / "a.nii.gz", device="gpu", engine="inproc")
    assert seen["cli"] == seen["inproc"] == ts_argv(ct, tmp_path / "a.nii.gz", "liver_lesions",
                                                    False, "gpu")
    assert sys.argv[1:] != seen["inproc"]                   # sys.argv geri yuklendi


def test_ayni_surec_motoru_cikti_yazmazsa_firlatir(tmp_path, monkeypatch):
    from src.baseline import run_totalsegmentator as rt
    monkeypatch.setattr(rt, "_ts_main", lambda: (lambda: None))
    with pytest.raises(FileNotFoundError):
        rt.run_case(tmp_path / "v.nii", tmp_path / "o.nii.gz", engine="inproc")


# ---------------------------------------------------------------- paralel baslatici
def test_paralel_komutlar_her_gpuya_bir_parca():
    from src.baseline.run_parallel import build_jobs
    jobs = build_jobs(["--in-glob", "x", "--limit", "5"], n_gpus=2, log_dir=Path("L"))
    assert [j["env"]["CUDA_VISIBLE_DEVICES"] for j in jobs] == ["0", "1"]
    assert [j["cmd"][j["cmd"].index("--shard") + 1] for j in jobs] == ["0/2", "1/2"]
    assert all(j["cmd"][:3] == [sys.executable, "-m", "src.baseline.run_totalsegmentator"]
               for j in jobs)
    assert all("--limit" in j["cmd"] and "--device" in j["cmd"] for j in jobs)
    assert jobs[0]["log"] != jobs[1]["log"]


def test_paralel_tek_is_cokerse_basarisiz_doner(tmp_path):
    from src.baseline.run_parallel import run_jobs
    ok = {"cmd": [sys.executable, "-c", "print('ok')"], "env": {}, "log": tmp_path / "a.log"}
    bad = {"cmd": [sys.executable, "-c", "import sys; sys.exit(3)"], "env": {},
           "log": tmp_path / "b.log"}
    assert run_jobs([ok, ok | {"log": tmp_path / "c.log"}], poll_s=0.05) == 0
    assert run_jobs([ok, bad], poll_s=0.05) != 0
    assert "ok" in (tmp_path / "a.log").read_text()


def test_paralel_parca_ve_cihaz_disaridan_verilemez():
    from src.baseline.run_parallel import build_jobs
    for bad in (["--shard", "0/2"], ["--device", "cpu"]):
        with pytest.raises(ValueError):
            build_jobs(["--in-glob", "x", *bad], n_gpus=2, log_dir=Path("L"))


# ---------------------------------------------------------------- maske karsilastirici
def _save(path, arr, aff=None):
    import nibabel as nib
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), np.eye(4) if aff is None else aff), str(path))


def test_maske_karsilastirici(tmp_path):
    from src.baseline.compare_masks import compare_dirs
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    m = np.zeros((8, 8, 8)); m[2:5, 2:5, 2:5] = 1
    _save(a / "c1.nii.gz", m); _save(b / "c1.nii.gz", m)                  # ayni
    m2 = m.copy(); m2[0, 0, 0] = 1
    _save(a / "c2.nii.gz", m); _save(b / "c2.nii.gz", m2)                 # 1 voksel farkli
    aff = np.eye(4); aff[0, 0] = -1
    _save(a / "c3.nii.gz", m); _save(b / "c3.nii.gz", m, aff)             # veri ayni, yon farkli
    _save(a / "c4.nii.gz", m)                                              # b'de yok
    rows = {r["case_id"]: r for r in compare_dirs(a, b)}
    assert rows["c1"]["identical"] and rows["c1"]["n_diff_voxels"] == 0
    assert not rows["c2"]["identical"] and rows["c2"]["n_diff_voxels"] == 1
    assert not rows["c3"]["identical"] and not rows["c3"]["affine_equal"]
    assert not rows["c4"]["identical"] and rows["c4"]["missing"] == "b"


def test_karsilastirici_bos_klasorde_basarisiz(tmp_path, monkeypatch):
    """Iki bos klasor 'hepsi ayni' demek degil: hicbir sey karsilastirilmadi."""
    from src.baseline import compare_masks
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    monkeypatch.setattr(sys, "argv", ["x", "--a", str(tmp_path / "a"), "--b", str(tmp_path / "b")])
    assert compare_masks.main() != 0


def test_gercek_ts_cli_argumanlarimizi_kabul_eder_ve_ayni_ayarla_cagirir(tmp_path, monkeypatch):
    """Model CALISTIRMADAN: TS'in gercek main()'i argv'mizi ayristirir; yalniz en sondaki
    segmentasyon cagrisi yakalanir. liver_lesions + --ml + cihaz dogru gecmeli."""
    ts_bin = pytest.importorskip("totalsegmentator.bin.TotalSegmentator")
    from src.baseline import run_totalsegmentator as rt
    calls = []

    def fake_ts(inp, out, ml, *args, **kw):
        calls.append({"input": str(inp), "output": str(out), "ml": ml, "args": args, **kw})
        Path(out).write_bytes(b"x")

    monkeypatch.setattr(ts_bin, "totalsegmentator", fake_ts)
    out = tmp_path / "lits-3.nii.gz"
    rt.run_case(tmp_path / "volume-3.nii", out, device="cpu", engine="inproc")
    assert len(calls) == 1
    c = calls[0]
    assert c["ml"] is True and c["device"] == "cpu" and c["output"] == str(out)
    assert "liver_lesions" in c["args"]                       # gorev konumsal gecer


def test_ts_arguman_hatasi_donguyu_oldurmez(tmp_path, monkeypatch):
    """argparse hatasi SystemExit firlatir; ayni surecte bu tum kosuyu durdururdu."""
    from src.baseline import run_totalsegmentator as rt

    def exits():
        raise SystemExit(2)
    monkeypatch.setattr(rt, "_ts_main", lambda: exits)
    with pytest.raises(RuntimeError):
        rt.run_case(tmp_path / "v.nii", tmp_path / "o.nii.gz", engine="inproc")
