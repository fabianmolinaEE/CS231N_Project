import pytest
import torch
from src.models.baseline_cnn import PlainCNN
from src.models.encoder_decoder import EncoderDecoder


class TestPlainCNN:
    def test_output_shape(self):
        x = torch.randn(2, 2, 64, 64)
        assert PlainCNN()(x).shape == (2, 1, 64, 64)

    def test_large_input(self):
        x = torch.randn(1, 2, 256, 256)
        assert PlainCNN()(x).shape == (1, 1, 256, 256)

    def test_no_activation_on_output(self):
        x = torch.randn(2, 2, 64, 64)
        assert PlainCNN()(x).min().item() < 0

    def test_no_nan(self):
        x = torch.randn(2, 2, 64, 64)
        assert not torch.isnan(PlainCNN()(x)).any()

    def test_gradient_flows(self):
        x = torch.randn(1, 2, 64, 64, requires_grad=True)
        PlainCNN()(x).sum().backward()
        assert x.grad is not None


class TestEncoderDecoder:
    def test_output_shape(self):
        x = torch.randn(2, 2, 64, 64)
        assert EncoderDecoder()(x).shape == (2, 1, 64, 64)

    def test_large_input(self):
        x = torch.randn(1, 2, 256, 256)
        assert EncoderDecoder()(x).shape == (1, 1, 256, 256)

    def test_no_activation_on_output(self):
        x = torch.randn(2, 2, 64, 64)
        assert EncoderDecoder()(x).min().item() < 0

    def test_no_nan(self):
        x = torch.randn(2, 2, 64, 64)
        assert not torch.isnan(EncoderDecoder()(x)).any()

    def test_gradient_flows(self):
        x = torch.randn(1, 2, 64, 64, requires_grad=True)
        EncoderDecoder()(x).sum().backward()
        assert x.grad is not None
