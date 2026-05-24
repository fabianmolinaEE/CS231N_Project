import torch
import torch.nn as nn

from src.models.unet import DoubleConv

_DILATIONS = [1, 1, 2, 2, 4, 4, 8, 8]


class PlainCNN(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        layers: list[nn.Module] = [DoubleConv(in_channels, base_channels, dilation=_DILATIONS[0])]
        for d in _DILATIONS[1:]:
            layers.append(DoubleConv(base_channels, base_channels, dilation=d))
        layers.append(nn.Conv2d(base_channels, out_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
