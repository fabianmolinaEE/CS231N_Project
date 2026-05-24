import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two Conv3×3 → BN → ReLU blocks. Preserves spatial dimensions."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Placeholder classes for future implementation
class PixelShuffleUp(nn.Module):
    pass


class UNet(nn.Module):
    pass


class PhysicsLoss(nn.Module):
    pass
