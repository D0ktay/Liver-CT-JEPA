"""Dogrusal sonda: etiketleme ve olcut, veri/GPU gerektirmez."""
import numpy as np
import torch

from src.jepa.probe import balanced_accuracy, token_labels


def test_token_etiketi_tumor_karaciger_arka_plan_ve_belirsiz():
    liv = torch.zeros(1, 1, 8, 8, 8); tum = torch.zeros(1, 1, 8, 8, 8)
    liv[..., 0:4, 0:4, 0:4] = 1                 # token (0,0,0): tam karaciger
    tum[..., 4:8, 4:8, 4:8] = 1                 # token (1,1,1): tam tumor (karaciger etiketi olmasa da bolge)
    liv[..., 0:1, 4:8, 0:4] = 1                 # token (0,1,0): %25 karaciger -> belirsiz
    lab = token_labels(liv, tum)[0]
    assert lab[0, 0, 0] == 1 and lab[1, 1, 1] == 2 and lab[0, 1, 0] == -1 and lab[1, 0, 0] == 0


def test_dengeli_dogruluk_sinif_dengesizliginden_etkilenmez():
    y = np.array([0] * 98 + [2] * 2); pred = np.zeros(100, int)
    ba, rec = balanced_accuracy(pred, y)
    assert rec[0] == 1.0 and rec[2] == 0.0 and np.isnan(rec[1]) and abs(ba - 0.5) < 1e-9
