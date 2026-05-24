# U-Net + Physics Loss Design

**Date:** 2026-05-24  
**Project:** CS231N Chip Thermal Map Prediction  
**Status:** Approved

## Problem

Predict full-chip thermal maps from 2-channel image (floorplan + power density).
Physics-constrained training enables OOD generalization on unseen die parameters.

## Scope

`src/models/unet.py` only. Dataset and training loop are separate milestones.

## Architecture

4-level U-Net with Pixel Shuffle upsampling.

**Encoder:** DoubleConv blocks + MaxPool2d(2)
- enc1: 2 → 64     [H×W]
- enc2: 64 → 128   [H/2×W/2]
- enc3: 128 → 256  [H/4×W/4]
- enc4: 256 → 512  [H/8×W/8]
- bottleneck: 512 → 1024  [H/16×W/16]

**Decoder:** PixelShuffleUp + skip concat + DoubleConv
- up4: 1024→512 upsample, concat enc4 skip, DoubleConv(1024→512)
- up3: 512→256 upsample, concat enc3 skip, DoubleConv(512→256)
- up2: 256→128 upsample, concat enc2 skip, DoubleConv(256→128)
- up1: 128→64 upsample, concat enc1 skip, DoubleConv(128→64)
- out: Conv1×1(64→1), no activation

## Physics Loss

```
residual = k * laplacian(T_pred) + Q
L_phys = mean(residual^2)
```

Laplacian computed via fixed depthwise Conv2d with 3×3 kernel [0,1,0;1,-4,1;0,1,0], padding=1.
`k` is a constructor argument (default 1.0 for normalized units).

## Public API

```python
from src.models.unet import UNet, PhysicsLoss

model = UNet(in_channels=2, out_channels=1, base_channels=64)
T_pred = model(x)                          # x: (B, 2, H, W)

phys = PhysicsLoss(k=1.0)
L = phys(T_pred, x[:, 1:2])               # Q = power density channel
```

## Key Decisions

- Pixel Shuffle preferred over bilinear: cleaner gradients for spatial regression
- k as constructor arg: enables normalized vs. physical unit sweep without code changes
- PhysicsLoss as separate nn.Module in unet.py: clean separation from architecture
- No activation on output: linear regression head
