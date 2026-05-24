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


class TestUNet:
    def test_output_shape(self):
        x = torch.randn(2, 2, 256, 256)
        out = UNet()(x)
        assert out.shape == (2, 1, 256, 256)

    def test_small_input(self):
        x = torch.randn(1, 2, 64, 64)
        out = UNet()(x)
        assert out.shape == (1, 1, 64, 64)

    def test_no_activation_on_output(self):
        x = torch.randn(2, 2, 64, 64)
        out = UNet()(x)
        assert out.min().item() < 0

    def test_no_nan(self):
        x = torch.randn(2, 2, 64, 64)
        out = UNet()(x)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        x = torch.randn(1, 2, 64, 64, requires_grad=True)
        out = UNet()(x)
        out.sum().backward()
        assert x.grad is not None


class TestPhysicsLoss:
    def test_loss_is_scalar(self):
        T = torch.randn(2, 1, 64, 64)
        Q = torch.zeros(2, 1, 64, 64)
        loss = PhysicsLoss()(T, Q)
        assert loss.shape == ()

    def test_zero_loss_when_residual_is_zero(self):
        # construct Q = -laplacian(T) so residual is exactly zero
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        T = torch.randn(1, 1, 16, 16)
        Q = -F.conv2d(T, kernel, padding=1)
        loss = PhysicsLoss(k=1.0)(T, Q)
        assert loss.item() < 1e-5

    def test_no_nan(self):
        T = torch.randn(2, 1, 32, 32)
        Q = torch.randn(2, 1, 32, 32)
        loss = PhysicsLoss()(T, Q)
        assert not torch.isnan(loss)

    def test_gradient_flows_through_T(self):
        T = torch.randn(1, 1, 16, 16, requires_grad=True)
        Q = torch.randn(1, 1, 16, 16)
        PhysicsLoss()(T, Q).backward()
        assert T.grad is not None

    def test_k_scales_loss(self):
        T = torch.randn(1, 1, 16, 16)
        Q = torch.zeros(1, 1, 16, 16)
        l1 = PhysicsLoss(k=1.0)(T, Q).item()
        l2 = PhysicsLoss(k=2.0)(T, Q).item()
        assert abs(l2 - 4 * l1) < 1e-4
