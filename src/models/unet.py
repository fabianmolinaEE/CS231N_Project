import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two Conv3×3 → BN → ReLU blocks. Preserves spatial dimensions."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PixelShuffleUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class UNet(nn.Module):
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
        self.dec4 = DoubleConv(b * 16, b * 8)
        self.up3 = PixelShuffleUp(b * 8, b * 4)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = PixelShuffleUp(b * 4, b * 2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = PixelShuffleUp(b * 2, b)
        self.dec1 = DoubleConv(b * 2, b)
        self.out_conv = nn.Conv2d(b, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class PhysicsLoss(nn.Module):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', kernel)

    def forward(self, T_pred: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        lap = F.conv2d(T_pred, self.lap_kernel, padding=1)
        return (self.k * lap + Q).pow(2).mean()
