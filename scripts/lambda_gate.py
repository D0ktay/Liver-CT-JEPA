#!/usr/bin/env python3
"""Gece egitimi KAPISI -- Lambda'da, yerel makineden BAGIMSIZ calisir.

    tmux new -d -s kapi "python3 scripts/lambda_gate.py --budget-hours 36"

Uc sinama da GECERSE `lambda_night.py --from bench`'i baslatir; biri bile gecmezse HICBIR SEY
baslatmaz ve nedenini outputs/night/GATE.json'a yazar. Esikler ONCEDEN yazildi (21 Eyl 2026):
  K1 veri      : hazirlik tmux'u bitti VE outputs/night/state/qc.done var (QC kirmiziysa olusmaz)
  K2 ince ayar : 5000 adimlik gercek-veri sinamasinin SON dogrulamasinda
                 karaciger Dice >= 0.80 VE tumor Dice >= 0.10  (MVE'de tumor 0.05'ti)
  K3 JEPA      : 2000 adimlik sinama hatasiz bitti, kayip sonlu ve baslangictan dusuk,
                 hicbir logda RankMe < 0.10 * D (D = 128)
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

OUT = Path("outputs/night")


def tmux_alive(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def jl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=36.0)
    ap.add_argument("--max-wait-min", type=float, default=240.0)
    a = ap.parse_args()
    t0 = time.time()
    while any(tmux_alive(s) for s in ("hazirlik", "sanity", "jepa")):
        if (time.time() - t0) / 60 > a.max_wait_min:
            break
        time.sleep(30)
    res = {"time": time.strftime("%F %T"), "checks": {}}
    res["checks"]["K1_veri"] = {"ok": (OUT / "state/qc.done").exists() and not tmux_alive("hazirlik")}
    v = jl(OUT / "sanity/val.jsonl")
    last = v[-1] if v else {}
    res["checks"]["K2_ince_ayar"] = {
        "ok": bool(v) and (OUT / "sanity/done.json").exists()
        and last.get("liver_dice", 0) >= 0.80 and last.get("tumor_dice", 0) >= 0.10,
        "egri": [(r["step"], round(r["liver_dice"], 3), round(r["tumor_dice"], 3),
                  round(r["outside_liver_frac"], 3)) for r in v]}
    lg = jl(OUT / "sanity_jepa/log.jsonl")
    txt = (OUT / "sanity_jepa.log").read_text(errors="ignore") if (OUT / "sanity_jepa.log").exists() else ""
    res["checks"]["K3_jepa"] = {
        "ok": bool(lg) and (OUT / "sanity_jepa/done.json").exists() and "Traceback" not in txt
        and lg[-1]["loss"] == lg[-1]["loss"] and lg[-1]["loss"] < lg[0]["loss"]
        and min(r["rankme"] for r in lg) >= 0.10 * lg[0]["dim"],
        "kayip_ilk_son": [lg[0]["loss"], lg[-1]["loss"]] if lg else None,
        "rankme_min_son": [min(r["rankme"] for r in lg), lg[-1]["rankme"]] if lg else None}
    res["go"] = all(c["ok"] for c in res["checks"].values())
    (OUT / "GATE.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    if not res["go"]:
        print("[kapi] ❌ EGITIM BASLATILMADI", flush=True)
        return 1
    cmd = (f"PATH=$HOME/.local/bin:$PATH .venv/bin/python scripts/lambda_night.py --from bench --ram-gb 300 "
           f"--budget-hours {a.budget_hours} 2>&1 | tee -a outputs/night/night.log")
    subprocess.run(["tmux", "new", "-d", "-s", "gece", cmd], check=True)
    print("[kapi] ✅ gece egitimi baslatildi (tmux: gece)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
