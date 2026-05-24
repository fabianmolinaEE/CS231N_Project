from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def rmse(T_pred: torch.Tensor, T_gt: torch.Tensor) -> float:
    return torch.sqrt(F.mse_loss(T_pred, T_gt)).item()


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    return (kernel / kernel.sum()).view(1, 1, size, size)


def ssim(
    T_pred: torch.Tensor,
    T_gt: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    kernel = _gaussian_kernel(kernel_size, sigma).to(T_pred.device)
    pad = kernel_size // 2

    mu_x = F.conv2d(T_pred, kernel, padding=pad)
    mu_y = F.conv2d(T_gt, kernel, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2 = F.conv2d(T_pred ** 2, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(T_gt ** 2, kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(T_pred * T_gt, kernel, padding=pad) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (num / den).mean().item()


def hotspot_iou(
    T_pred: torch.Tensor,
    T_gt: torch.Tensor,
    frac: float = 0.05,
) -> float:
    n_pixels = T_pred.numel()
    k = max(1, int(n_pixels * frac))
    flat_pred = T_pred.flatten()
    flat_gt = T_gt.flatten()
    thresh_pred = flat_pred.topk(k).values[-1]
    thresh_gt = flat_gt.topk(k).values[-1]
    mask_pred = flat_pred >= thresh_pred
    mask_gt = flat_gt >= thresh_gt
    intersection = (mask_pred & mask_gt).sum().item()
    union = (mask_pred | mask_gt).sum().item()
    return intersection / union if union > 0 else 0.0


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    preds, gts = [], []
    for x, T_gt in loader:
        x = x.to(device)
        preds.append(model(x).cpu())
        gts.append(T_gt)
    T_pred_all = torch.cat(preds)
    T_gt_all = torch.cat(gts)
    return {
        "rmse": rmse(T_pred_all, T_gt_all),
        "ssim": ssim(T_pred_all, T_gt_all),
        "hotspot_iou": hotspot_iou(T_pred_all, T_gt_all),
    }
