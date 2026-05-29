import math
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.unet import UNet
from src.train import train


def _make_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n, 2, 64, 64)
    T_gt = torch.randn(n, 1, 64, 64)
    return DataLoader(TensorDataset(x, T_gt), batch_size=batch_size)


def test_checkpoint_created():
    loader = _make_loader()
    with tempfile.TemporaryDirectory() as tmp:
        train(UNet(), loader, loader, epochs=2, device="cpu", checkpoint_dir=tmp)
        import os
        assert os.path.isfile(f"{tmp}/best.pt")


def test_checkpoint_contents():
    loader = _make_loader()
    with tempfile.TemporaryDirectory() as tmp:
        train(UNet(), loader, loader, epochs=2, device="cpu", checkpoint_dir=tmp)
        ckpt = torch.load(f"{tmp}/best.pt", weights_only=True)
        assert "epoch" in ckpt
        assert "model_state" in ckpt
        assert "val_loss" in ckpt
        assert math.isfinite(ckpt["val_loss"])


def test_val_loss_finite():
    loader = _make_loader()
    with tempfile.TemporaryDirectory() as tmp:
        train(UNet(), loader, loader, epochs=1, device="cpu", checkpoint_dir=tmp)
        ckpt = torch.load(f"{tmp}/best.pt", weights_only=True)
        assert math.isfinite(ckpt["val_loss"])


def test_log_fn_called_each_epoch(tmp_path):
    """log_fn is called once per epoch with (epoch, train_metrics, val_metrics, lr)."""
    x = torch.randn(4, 2, 64, 64)
    y = torch.randn(4, 1, 64, 64)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    calls = []

    def log_fn(epoch, tr, vl, lr):
        calls.append((epoch, tr, vl, lr))

    model = UNet(base_channels=4)
    train(model, loader, loader, epochs=3, lam_phys=0.0,
          checkpoint_dir=tmp_path, device="cpu", log_fn=log_fn)

    assert len(calls) == 3, f"expected 3 calls, got {len(calls)}"
    assert calls[0][0] == 1
    assert calls[2][0] == 3
    assert "mse" in calls[0][1]
    assert "mse" in calls[0][2]
    assert isinstance(calls[0][3], float)
