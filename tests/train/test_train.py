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
