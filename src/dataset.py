"""PyTorch Dataset for CircuitNet-N14 thermal map prediction.

Implements DATA-02 (alignment), DATA-07 (loading), DATA-08 (augmentation).
Honors locked decisions D-09 (per-channel norm from train split only),
D-10 (raw temperature labels), D-11 (H+V flip, train-only, paired with label).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class ThermalDataset(Dataset):
    """Loads (floorplan, power, label) tuples for thermal map regression.

    Args:
        index_path: path to data/splits/{train,val,test}.json (list of dicts with
            "floorplan", "power", "label" string paths).
        stats_path: path to data/normalization_stats.json with per-channel
            {"floorplan": {"mean", "std"}, "power": {"mean", "std"}}.
            If mean/std values are null, normalization is skipped (pass-through).
        training: if True, apply horizontal+vertical flip augmentation (D-11);
            applied IDENTICALLY to input and label (Pitfall 4).

    __getitem__ returns:
        x:    torch.float32 tensor, shape (2, 256, 256). Channel 0 = normalized
              floorplan, channel 1 = normalized power.
        label: torch.float32 tensor, shape (1, 256, 256). Raw temperature (D-10).
    """

    def __init__(self, index_path: str, stats_path: str, training: bool = False):
        with open(index_path) as f:
            self.index: List[dict] = json.load(f)
        with open(stats_path) as f:
            stats = json.load(f)

        # Null-safe: if stats not yet computed (null placeholders), skip normalization
        fp_mean = stats["floorplan"]["mean"]
        fp_std = stats["floorplan"]["std"]
        pw_mean = stats["power"]["mean"]
        pw_std = stats["power"]["std"]

        self.fp_mean: Optional[float] = float(fp_mean) if fp_mean is not None else None
        self.fp_std: Optional[float] = float(fp_std) if fp_std is not None else None
        self.pw_mean: Optional[float] = float(pw_mean) if pw_mean is not None else None
        self.pw_std: Optional[float] = float(pw_std) if pw_std is not None else None
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        entry = self.index[idx]
        fp = np.load(entry["floorplan"]).astype(np.float32)
        pw = np.load(entry["power"]).astype(np.float32)
        lb = np.load(entry["label"]).astype(np.float32)

        # DATA-02: alignment assertion (defensive)
        assert fp.shape == pw.shape == lb.shape, (
            f"Shape mismatch at idx={idx} design={entry.get('design')}: "
            f"fp={fp.shape} pw={pw.shape} label={lb.shape}"
        )

        fp_t = torch.from_numpy(fp).unsqueeze(0)  # (1, H, W)
        pw_t = torch.from_numpy(pw).unsqueeze(0)
        lb_t = torch.from_numpy(lb).unsqueeze(0)

        # D-09 per-channel normalization, train-split stats only.
        # Skip normalization when stats are null (not yet computed — Pitfall 5 guard).
        if self.fp_mean is not None and self.fp_std is not None:
            fp_t = (fp_t - self.fp_mean) / (self.fp_std + 1e-8)
        if self.pw_mean is not None and self.pw_std is not None:
            pw_t = (pw_t - self.pw_mean) / (self.pw_std + 1e-8)

        x = torch.cat([fp_t, pw_t], dim=0)  # (2, H, W)

        # D-11: identical random flip on x and label, train-only, no augmentation in eval
        if self.training:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])      # horizontal (last axis = W)
                lb_t = torch.flip(lb_t, dims=[2])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])      # vertical (axis = H)
                lb_t = torch.flip(lb_t, dims=[1])

        return x.contiguous(), lb_t.contiguous()
