"""Unit tests for ThermalDataset covering DATA-02, DATA-07, DATA-08."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import ThermalDataset


def _write_design(root: Path, name: str, fp: np.ndarray, pw: np.ndarray, lb: np.ndarray):
    proc_dir = root / "processed" / name
    proc_dir.mkdir(parents=True, exist_ok=True)
    np.save(proc_dir / "floorplan.npy", fp.astype(np.float32))
    np.save(proc_dir / "power.npy", pw.astype(np.float32))
    lbl_dir = root / "labels" / name
    lbl_dir.mkdir(parents=True, exist_ok=True)
    np.save(lbl_dir / "thermal.npy", lb.astype(np.float32))
    return {
        "design": name,
        "floorplan": str(proc_dir / "floorplan.npy"),
        "power": str(proc_dir / "power.npy"),
        "label": str(lbl_dir / "thermal.npy"),
    }


@pytest.fixture
def tmp_dataset(tmp_path):
    """Three synthetic designs at 256x256, with distinctive patterns so flips are detectable."""
    rng = np.random.default_rng(0)
    entries = []
    for i in range(3):
        fp = rng.random((256, 256), dtype=np.float32)
        pw = rng.random((256, 256), dtype=np.float32)
        # Asymmetric label so a flip is observable
        lb = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        entries.append(_write_design(tmp_path, f"design_{i}", fp, pw, lb))

    # Identity-normalization stats so eval output equals raw .npy (within fp32 precision)
    stats = {
        "floorplan": {"mean": 0.0, "std": 1.0},
        "power": {"mean": 0.0, "std": 1.0},
    }
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.json").write_text(json.dumps(entries))
    (tmp_path / "normalization_stats.json").write_text(json.dumps(stats))

    return {
        "root": tmp_path,
        "index": str(splits_dir / "train.json"),
        "stats": str(tmp_path / "normalization_stats.json"),
        "entries": entries,
    }


def test_getitem_shapes(tmp_dataset):
    """DATA-07: returns (2,256,256) float32 input and (1,256,256) float32 label."""
    ds = ThermalDataset(tmp_dataset["index"], tmp_dataset["stats"], training=False)
    assert len(ds) == 3
    x, y = ds[0]
    assert x.shape == (2, 256, 256), f"input shape: {x.shape}"
    assert y.shape == (1, 256, 256), f"label shape: {y.shape}"
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_channel_alignment(tmp_dataset):
    """DATA-02: floorplan, power, label all share spatial dims."""
    ds = ThermalDataset(tmp_dataset["index"], tmp_dataset["stats"], training=False)
    x, y = ds[0]
    assert x.shape[1:] == y.shape[1:], (
        f"channels not aligned: x.HW={x.shape[1:]}, y.HW={y.shape[1:]}"
    )


def test_normalization_identity_in_eval(tmp_dataset):
    """DATA-09: with mean=0/std=1 stats, eval output equals raw npy contents."""
    ds = ThermalDataset(tmp_dataset["index"], tmp_dataset["stats"], training=False)
    x, y = ds[0]
    raw_fp = torch.from_numpy(np.load(tmp_dataset["entries"][0]["floorplan"]))
    raw_pw = torch.from_numpy(np.load(tmp_dataset["entries"][0]["power"]))
    raw_lb = torch.from_numpy(np.load(tmp_dataset["entries"][0]["label"]))
    assert torch.allclose(x[0], raw_fp, atol=1e-5)
    assert torch.allclose(x[1], raw_pw, atol=1e-5)
    assert torch.allclose(y[0], raw_lb, atol=1e-5)


def test_eval_is_deterministic(tmp_dataset):
    """Eval mode never applies augmentation -> repeated __getitem__(0) is identical."""
    ds = ThermalDataset(tmp_dataset["index"], tmp_dataset["stats"], training=False)
    x1, y1 = ds[0]
    x2, y2 = ds[0]
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_augmentation_paired(tmp_dataset):
    """DATA-08 + Pitfall 4: when a flip is drawn, input AND label flip together.

    Strategy: monkeypatch torch.rand to force-flip on both axes, then verify that
    the returned tensors are exactly the deterministic flips of the eval-mode tensors.
    This catches augmentation desync regressions deterministically (no flaky retries).
    """
    ds_eval = ThermalDataset(tmp_dataset["index"], tmp_dataset["stats"], training=False)
    x_eval, y_eval = ds_eval[0]

    ds_train = ThermalDataset(tmp_dataset["index"], tmp_dataset["stats"], training=True)

    # Force both flip branches to take (rand -> 0.0 < 0.5) and rotation to k=0.
    import src.dataset as ds_mod

    orig_rand = torch.rand
    orig_randint = torch.randint

    def always_flip(*args, **kwargs):
        return torch.zeros(1)

    def no_rotation(*args, **kwargs):
        return torch.zeros(1, dtype=torch.long)

    ds_mod.torch.rand = always_flip
    ds_mod.torch.randint = no_rotation
    try:
        x_train, y_train = ds_train[0]
    finally:
        ds_mod.torch.rand = orig_rand
        ds_mod.torch.randint = orig_randint

    expected_x = torch.flip(torch.flip(x_eval, dims=[2]), dims=[1])
    expected_y = torch.flip(torch.flip(y_eval, dims=[2]), dims=[1])
    assert torch.equal(x_train, expected_x), "input flip mismatch"
    assert torch.equal(y_train, expected_y), "label flip mismatch (desync regression)"


def test_dataloader_batches(tmp_dataset):
    """End-to-end smoke: torch.utils.data.DataLoader yields stacked batches."""
    from torch.utils.data import DataLoader
    ds = ThermalDataset(tmp_dataset["index"], tmp_dataset["stats"], training=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    assert x.shape == (2, 2, 256, 256)
    assert y.shape == (2, 1, 256, 256)
