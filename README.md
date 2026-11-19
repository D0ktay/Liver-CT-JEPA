# Liver-CT-JEPA

Liver tumour detection on CT: a leakage-free benchmark, a targeting-error metric in millimetres, and a JEPA-based 3D CT representation trained on a corpus that can be audited.

> **All rights reserved.** This repository is public for reading and reference only. You may not copy, reuse, modify or redistribute the code, documents or results, or use them to train models, without written permission. See [`LICENSE`](LICENSE).

## The two claims

**1. The benchmarks are contaminated.** Public abdominal CT datasets repeat the same scans: a published audit of 46 datasets found 59.1% case duplication (arXiv:2508.13626). 3D-IRCADb sits inside LiTS. MSD Task03 is LiTS. AbdomenAtlas 3.0 compiles all of them and holds the same scan two or three times under different IDs. Train on one and test on another, and the "external" test is not external.
`src/provenance/` maps where each scan comes from, and `src/data/make_splits.py` produces a clean split.

**2. Dice asks the wrong question.** The winner of the LiTS MICCAI 2017 challenge ranked 1st on tumour Dice (0.702) and 13th of 15 on lesion precision (0.148). The clinically useful question is: if an ablation had been planned from this mask, how many millimetres off would it have been, and would a 5 mm margin have held?
`src/metrics/targeting.py` measures that.

Neither claim needs training. Both produce full results without a GPU.

## Quick start

```bash
bash scripts/00_setup.sh     # virtualenv + dependencies + test suite
```

The tests need no data. If they fail, the environment is broken; fix that before touching any data. Python 3.10 or later is required.

## Pipeline

```bash
bash scripts/00_setup.sh                 # environment + tests
bash scripts/01_download_min.sh          # ~30 GB; the commands are commented out on purpose
bash scripts/02_audit_leakage.sh         # leakage audit. Run this first
bash scripts/03_baseline_and_metrics.sh  # TotalSegmentator baseline + metrics
```

### Why the leakage audit comes first

Every number produced on a contaminated split is worthless, and everything downstream has to be redone if this step is skipped. The audit runs in two passes. The first builds the provenance table and content-based duplicate candidates, then stops: `outputs/dedup/candidate_duplicates.csv` has to be reviewed **by hand**. The second pass builds the split from the reviewed file.

### Baseline

TotalSegmentator `liver_lesions`: one `pip install` and one command, Apache-2.0, runs on CPU (5 to 15 minutes per case). Published figures: 75.8% sensitivity and Dice 0.658 on CT. It is a realistic floor to compare against.

### JEPA

See [`src/jepa/README.md`](src/jepa/README.md). JEPA is not a generative model and produces no images; it predicts the representation of a masked region, not its pixels. The project pre-trains its own small model (`LiverJEPA-S`) on a corpus whose contents can be checked against the test set, because an off-the-shelf encoder with unauditable pre-training data would undermine the leakage claim. Off-the-shelf weights are kept only as comparison arms, and a masked-autoencoder arm is mandatory before saying anything about JEPA being better.

## Rules the code enforces

Code comments that mention `CLAUDE.md kural N` refer to rule N in this list. (`CLAUDE.md` was an internal working file and is not part of the public repository.)

1. **3D-IRCADb is not an external test set.** It is records 28 to 47 of the LiTS training set. `configs/paths.yaml` enforces this.
2. **MSD Task03_Liver is LiTS.** Same images, two names.
3. **The pre-training corpus obeys the split too.** No scan of a test patient may enter the corpus, even without labels. `src/data/pretrain_manifest.py` asserts the intersections are empty. Two separate defences are needed: the containment graph (FLARE23 contains HCC-TACE-Seg) and an explicit deny list (CT-ORG contains LiTS, but its files are anonymised, so canonical IDs do not match).
4. **AbdomenAtlas 3.0 is a compilation of public sets, not new data.** Its tumour masks derive from a model trained on LiTS and are not used as tumour labels. Do not trust a ready-made IID split; build your own with `src/data/make_splits.py`.
5. **Splits are made at canonical patient level**, and the external test is disjoint at source-dataset level. `make_splits.py` asserts this. If an assert fires, fix the data, not the assert.
6. **Dice on an empty mask is `nan`, not 1.0.** Tumour-free cases are reported through specificity and false positives.
7. **A millimetre margin on anisotropic voxels is not `binary_dilation(iterations=N)`.** Use `distance_transform_edt(sampling=spacing)`, as in `src/metrics/targeting.py:_dilate_mm`.
8. **Do not claim small Dice differences.** Fold-to-fold standard deviation on LiTS is 1.7 to 6.6 points (arXiv:2404.09556).
9. **The targeting metric uses no IoU threshold.** The planning mask is every predicted component that touches the ground truth. IoU ≥ 0.5 is used only for detection precision and recall.
10. **TotalSegmentator labels lesions as 1; LiTS labels tumour as 2.** `evaluate` takes `--gt-label` and `--pred-label` separately.

## Repository layout

| Path | Contents |
|---|---|
| `configs/paths.yaml` | All paths and leakage constants. No hard-coded paths in code |
| `src/provenance/` | Provenance table, content-based duplicate detection, canonical ID resolution |
| `src/data/` | Case IDs, leakage-free splits, pre-training manifest, dataset converters |
| `src/baseline/` | TotalSegmentator runner (no training, runs on CPU) |
| `src/metrics/lesionwise.py` | Lesion-wise precision, recall and F1, stratified by size |
| `src/metrics/targeting.py` | Targeting error (mm) and ablation-margin violation |
| `src/metrics/evaluate.py` | Full evaluation in one command |
| `src/jepa/` | Encoder, pre-training objective, decoder, fine-tuning |
| `src/viz/` | Overlays and the comparison panel |
| `kaggle_jobs/`, `notebooks/` | Kaggle kernels and notebooks |
| `scripts/` | Setup, download, audit, baseline and cloud-GPU run scripts |
| `tests/` | Test suite; needs no data |

## Running on Kaggle

The Kaggle account name has been replaced with the placeholder `YOUR_KAGGLE_USERNAME` in `kaggle_jobs/*/kernel-metadata.json`. Put your own username there, and set `KAGGLE_USER` for the helper scripts:

```bash
KAGGLE_USER=<your-username> bash scripts/kaggle_push.sh <job-name>
```

Details are in [`docs/kaggle.md`](docs/kaggle.md).

## Documentation

The working documents are written in Turkish.

- [`docs/leakage-map.md`](docs/leakage-map.md): every known leakage path between the datasets, with its verification status
- [`docs/sonuclar-kaggle-kosu1.md`](docs/sonuclar-kaggle-kosu1.md): results of the baseline and frozen-encoder runs on LiTS
- [`docs/EGITIM-PLANI.md`](docs/EGITIM-PLANI.md): training plan for `LiverJEPA-S`, from pre-training to evaluation
- [`docs/IKINCI-KOSU-PLANI.md`](docs/IKINCI-KOSU-PLANI.md): evidence-backed change list for the second run
- [`docs/kaggle.md`](docs/kaggle.md): running the notebooks on Kaggle

## Data

No imaging data is stored in this repository. Each dataset has its own licence, and several (LiTS, 3D-IRCADb, AbdomenAtlas) are restricted to non-commercial use. Check the terms before you download anything.

## Licence

Copyright © 2026 D0ktay. All rights reserved. The repository is source-available for viewing only; see [`LICENSE`](LICENSE).
