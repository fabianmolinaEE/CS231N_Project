import torch
import torch.nn as nn

from src.models.unet import DoubleConv, PixelShuffleUp


class EncoderDecoder(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = PixelShuffleUp(b * 16, b * 8)
        self.dec4 = DoubleConv(b * 8, b * 8)
        self.up3 = PixelShuffleUp(b * 8, b * 4)
        self.dec3 = DoubleConv(b * 4, b * 4)
        self.up2 = PixelShuffleUp(b * 4, b * 2)
        self.dec2 = DoubleConv(b * 2, b * 2)
        self.up1 = PixelShuffleUp(b * 2, b)
        self.dec1 = DoubleConv(b, b)
        self.out_conv = nn.Conv2d(b, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(self.up4(b))
        d3 = self.dec3(self.up3(d4))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))
        return self.out_conv(d1)
