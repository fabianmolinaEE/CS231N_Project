import torch
from src.evaluate import rmse, ssim, hotspot_iou


class TestRMSE:
    def test_zero_on_identical(self):
        T = torch.randn(2, 1, 32, 32)
        assert rmse(T, T) == 0.0

    def test_positive_on_different(self):
        T_pred = torch.zeros(2, 1, 32, 32)
        T_gt = torch.ones(2, 1, 32, 32)
        assert rmse(T_pred, T_gt) > 0


class TestSSIM:
    def test_perfect_on_identical(self):
        T = torch.rand(2, 1, 64, 64)
        assert abs(ssim(T, T) - 1.0) < 1e-4

    def test_in_range(self):
        T_pred = torch.randn(2, 1, 64, 64)
        T_gt = torch.randn(2, 1, 64, 64)
        s = ssim(T_pred, T_gt)
        assert -1.0 <= s <= 1.0


class TestHotspotIoU:
    def test_perfect_on_identical(self):
        T = torch.rand(2, 1, 32, 32)
        assert hotspot_iou(T, T) == 1.0

    def test_zero_on_disjoint(self):
        T_pred = torch.zeros(1, 1, 4, 4)
        T_gt = torch.zeros(1, 1, 4, 4)
        # pred hotspot in top-left, gt hotspot in bottom-right
        T_pred[0, 0, 0, 0] = 1.0
        T_gt[0, 0, 3, 3] = 1.0
        # frac=1/16 → k=1 pixel each, guaranteed disjoint
        assert hotspot_iou(T_pred, T_gt, frac=1 / 16) == 0.0

    def test_in_range(self):
        T_pred = torch.rand(2, 1, 32, 32)
        T_gt = torch.rand(2, 1, 32, 32)
        iou = hotspot_iou(T_pred, T_gt)
        assert 0.0 <= iou <= 1.0
