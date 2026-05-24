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


class TestPixelShuffleUp:
    def test_doubles_spatial(self):
        x = torch.randn(2, 64, 16, 16)
        out = PixelShuffleUp(64, 32)(x)
        assert out.shape == (2, 32, 32, 32)

    def test_no_nan(self):
        x = torch.randn(2, 128, 8, 8)
        out = PixelShuffleUp(128, 64)(x)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        out = PixelShuffleUp(64, 32)(x)
        out.sum().backward()
        assert x.grad is not None
