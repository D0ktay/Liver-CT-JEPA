# jepa_train_ras

The same job as `jepa_train` (`run.py` is a copy). It runs under a separate Kaggle name with the orientation-corrected features (commit `8d0bfc7`, `liver-jepa-feats` v2).

The first run (`liver-jepa-train` v1) used features produced in raw array orientation, and LiTS mixes LAS, RAS and LPS. That run is kept as the "effect of the orientation bug" comparison. The final result is **this** job.
