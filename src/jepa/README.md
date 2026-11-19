# The JEPA arm

## JEPA is not a generative model

It produces no images. There is no diffusion and no GAN. It predicts the **representation** of a masked region, not its pixels. What is being learned is a representation, not a generator.

## Roadmap

> **The original roadmap below was superseded on 21 September 2026.** The project now pre-trains its own model from scratch (`LiverJEPA-S`). The current plan is in [`docs/EGITIM-PLANI.md`](../../docs/EGITIM-PLANI.md).
> Reason: the pre-training data of an off-the-shelf encoder cannot be audited, so relying on one would undermine the project's own leakage claim. Off-the-shelf weights are now only a **comparison arm**. The table and the "no pre-training from scratch" rule below are history; they stay here only for the verified facts about the candidate encoders.

| GPU | (old roadmap) |
|---|---|
| None | Feature extraction with an off-the-shelf encoder on a very small subset. No training. |
| 24 GB | Off-the-shelf JEPA encoder + fine-tuning a light segmentation decoder. nnU-Net ResEnc-M as reference. |
| 40 to 80 GB | The above + an MAE comparison arm + ResEnc-L. |

~~**No JEPA pre-training from scratch under any scenario.**~~ (Removed on 21 September 2026.)
Rad-JEPA 3D was pre-trained on about 120,000 CTs. We do not repeat that; we train our own model on a **much smaller but auditable** corpus (`docs/EGITIM-PLANI.md` §0).

## Candidates (weights verified on 19 September 2026 without downloading: HF API + HEAD)

| Candidate | Weights | Notes |
|---|---|---|
| **Rad-JEPA 3D** (arXiv:2607.26196) | ✅ `huggingface.co/huyquoctrinh/Rad-Jepa-3D` `encoder/encoder.pt`, 155.6 MB, apache-2.0, not gated | 3D CT, H-Mamba, 18.3M parameters. Input (1,32,256,256), normalised to [0,1], **no spacing / mm information**. The token grid is **4x16x16, which is coarse** (a structural limit for small lesions). Needs `mamba_ssm` (not verified on a Kaggle T4). Pre-trained on about 120k CTs from M3D-Cap (Radiopaedia), so LiTS is not in it by name. **There is no segmentation checkpoint**; the `seg/` code in that repository trains with M3D-Seg, and M3D-Seg contains MSD-Liver (= LiTS) and IRCADb, so a segmentation model trained that way is LEAKY. **The only realistic candidate for a five-day experiment.** |
| Seg-JEPA (ACIIDS 2026) | ❌ no code, no weights | "91.9 Dice with 1% of labels (MAE 76.3)" is a **2D chest X-ray (TB)** result, not CT. It cannot support a CT claim. |
| MedJEPA (`prthmmkhija1/MedJEPA`) | ❌ code only (MIT), no release | The 3D part covers BraTS / MSD Task02 / Task09. |
| V-JEPA 2 / 2.1 (`facebookresearch/vjepa2`) | ✅ 2.1 ViT-B, 1.66 GB, MIT | Natural video. On CT only a "slice = frame" baseline, with no mm geometry. |

## Implementation status (19 to 20 September 2026)

- `mamba_ref.py` (pure-PyTorch Mamba) equals the real `mamba_ssm`: on a Kaggle T4 the largest difference in Rad-JEPA output is 5.1e-6, cosine 1.0 (`outputs/jepa_feats_run/.../mamba_ref_vs_mamba_ssm.json`). The same code runs on a Mac.
- `prep.py` + `extract.py`: 131 LiTS cases in 23 minutes (Kaggle `liver-jepa-feats`); tokens from the 4th, 8th and last layers.
- `seg.py` + `train.py`: frozen tokens + a small CNN; a control arm without JEPA; 5 folds (Kaggle `liver-jepa-train`).

## The mandatory comparison arm

OpenMind benchmark (ICCV 2025, arXiv:2412.17041): in 3D medical self-supervised learning, **MAE is still ahead**. Nobody can say "JEPA is better" without comparing against MAE.

MAE weights (19 September 2026): OpenMind `MIC-DKFZ/ResEncL-OpenMind-MAE` (491 MB) and `PrimusM-OpenMind-MAE` (788 MB), CC-BY-4.0. They are pre-trained on **brain MRI only**: no CT leakage, but there is an MRI-to-CT domain gap.
Unusable or to be handled with care: CT-FM (pre-training includes 477 HCC-TACE-Seg scans), VoCo-10k (FLARE23), SuPreM supervised (AbdomenAtlas 1.1, which contains LiTS).

## A note on terminology

A JEPA encoder is **not a world model**. A world model needs p(s_t+1 | s_t, a_t): state, action and transition. Here there is neither an action nor a time axis.
The honest name is "a JEPA-based 3D CT representation".
