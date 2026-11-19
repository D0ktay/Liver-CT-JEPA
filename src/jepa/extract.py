"""Rad-JEPA ozellik cikarma: LiTS -> pencereler (src.jepa.prep) -> donmus encoder -> npz.

Vaka basina outputs/jepa_feats/<case>.npz:
  img (n,32,256,256) uint8 | lab ayni sekil (tumor) | tok_l4, tok_l8, tok_out (n,1024,384) float16
  meta (json dizgisi: geri donus bilgisi, src.jepa.prep.reconstruct)
Ara katmanlar (4. ve 8. blok) ileri kancayla yakalanir; encoder yalniz son katmani tam verir.
Spacing: basliktan; LiTS 28-47 icin configs/lits_2847_true_spacing.csv (baslikta sahte 1x1x1).

    python -m src.jepa.extract --ckpt encoder.pt --repo radjepa3d --out outputs/jepa_feats
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def spacing_for(case_id: str, header_zooms, table: dict) -> tuple[float, float, float]:
    if case_id in table:
        return tuple(float(s) for s in table[case_id]["spacing"])
    z = tuple(float(s) for s in header_zooms[:3])
    if np.allclose(z, 1.0, atol=1e-4):
        raise ValueError(f"{case_id}: baslik spacing'i 1x1x1 (sahte olabilir) ve tabloda yok")
    return z


@torch.no_grad()
def encode_tiles(enc, img_u8: np.ndarray, device: str = "cuda", batch: int = 4,
                 layers=(3, 7)) -> dict[str, np.ndarray]:
    grabbed: dict[int, list] = {i: [] for i in layers}
    hooks = [enc.blocks[i].register_forward_hook(
        lambda m, a, o, i=i: grabbed[i].append(o.detach().float().cpu())) for i in layers]
    outs = []
    try:
        for s in range(0, len(img_u8), batch):
            x = torch.from_numpy(img_u8[s:s + batch]).to(device).float().div_(255.0)[:, None]
            outs.append(enc(x, indices=None).detach().float().cpu())
    finally:
        for h in hooks:
            h.remove()
    res = {f"l{i + 1}": torch.cat(grabbed[i]).numpy().astype(np.float16) for i in layers}
    res["out"] = torch.cat(outs).numpy().astype(np.float16)
    return res


def main() -> int:
    import nibabel as nib
    from src.baseline.respacing import load_spacing_table
    from src.data.case_id import case_id_from_path, glob_paths
    from src.jepa.prep import prepare_case
    from src.jepa.radjepa import load_radjepa_encoder
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vol-glob", default="data/raw/LiTS/volume-*.nii")
    ap.add_argument("--seg-glob", default="data/raw/LiTS/segmentation-*.nii")
    ap.add_argument("--table", type=Path, default=Path("configs/lits_2847_true_spacing.csv"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/jepa_feats"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force-ref", action="store_true", help="mamba_ssm olsa bile saf PyTorch Mamba")
    a = ap.parse_args()
    table = load_spacing_table(a.table)
    vols = {case_id_from_path(p): p for p in glob_paths(a.vol_glob)}
    segs = {case_id_from_path(p): p for p in glob_paths(a.seg_glob)}
    ids = sorted(set(vols) & set(segs), key=lambda c: int(c.split("-")[1]))
    if set(vols) != set(segs):
        raise SystemExit(f"hacim/segmentasyon kumeleri farkli: {sorted(set(vols) ^ set(segs))[:5]}")
    if a.limit:
        ids = ids[:a.limit]
    enc = load_radjepa_encoder(a.ckpt, a.repo, device=a.device, force_ref=a.force_ref)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, cid in enumerate(ids, 1):
        f = a.out / f"{cid}.npz"
        if f.exists():
            continue
        vi, si = nib.load(str(vols[cid])), nib.load(str(segs[cid]))
        sp = spacing_for(cid, vi.header.get_zooms(), table)
        c = prepare_case(np.asarray(vi.dataobj), np.asarray(si.dataobj).astype(np.uint8), sp,
                         affine=vi.affine)
        tok = encode_tiles(enc, c["img"], a.device, a.batch)
        c["meta"]["case_id"] = cid
        c["meta"]["mamba_impl"] = getattr(enc, "mamba_impl", "?")
        np.savez_compressed(f, img=c["img"], lab=c["lab"], tok_l4=tok["l4"], tok_l8=tok["l8"],
                            tok_out=tok["out"], meta=json.dumps(c["meta"]))
        print(f"[{k}/{len(ids)}] {cid}: {len(c['img'])} pencere, spacing {sp}, "
              f"{time.time() - t0:.0f} sn", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
