import pytest
import torch
import torch.nn.functional as F
from src.models.unet import DoubleConv, PixelShuffleUp, UNet, PhysicsLoss


class TestDoubleConv:
    def test_output_shape(self):
        x = torch.randn(2, 3, 64, 64)
        out = DoubleConv(3, 16)(x)
        assert out.shape == (2, 16, 64, 64)

    def test_preserves_spatial_dims(self):
        x = torch.randn(1, 64, 32, 32)
        out = DoubleConv(64, 128)(x)
        assert out.shape == (1, 128, 32, 32)

    def test_no_nan(self):
        x = torch.randn(2, 8, 16, 16)
        out = DoubleConv(8, 8)(x)
        assert not torch.isnan(out).any()
