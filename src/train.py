from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.unet import PhysicsLoss

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    phys_loss: PhysicsLoss,
    lam: float,
    device: str,
) -> dict[str, float]:
    model.train()
    mse_fn = nn.MSELoss()
    total = mse_total = phys_total = 0.0
    for x, T_gt in loader:
        x, T_gt = x.to(device), T_gt.to(device)
        T_pred = model(x)
        mse = mse_fn(T_pred, T_gt)
        phys = phys_loss(T_pred, x[:, 1:2])
        loss = mse + lam * phys
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = x.size(0)
        total += loss.item() * n
        mse_total += mse.item() * n
        phys_total += phys.item() * n
    N = len(loader.dataset)
    return {"loss": total / N, "mse": mse_total / N, "phys": phys_total / N}


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    phys_loss: PhysicsLoss,
    lam: float,
    device: str,
) -> dict[str, float]:
    model.eval()
    mse_fn = nn.MSELoss()
    total = mse_total = phys_total = 0.0
    for x, T_gt in loader:
        x, T_gt = x.to(device), T_gt.to(device)
        T_pred = model(x)
        mse = mse_fn(T_pred, T_gt)
        phys = phys_loss(T_pred, x[:, 1:2])
        loss = mse + lam * phys
        n = x.size(0)
        total += loss.item() * n
        mse_total += mse.item() * n
        phys_total += phys.item() * n
    N = len(loader.dataset)
    return {"loss": total / N, "mse": mse_total / N, "phys": phys_total / N}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    lam_phys: float = 0.1,
    checkpoint_dir: str | Path = "checkpoints",
    device: str = "cuda",
    log_fn: Callable[[int, dict, dict, float], None] | None = None,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    phys_loss = PhysicsLoss().to(device)

    best_val_loss = math.inf
    w = len(str(epochs))

    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, phys_loss, lam_phys, device)
        vl = val_epoch(model, val_loader, phys_loss, lam_phys, device)
        scheduler.step(vl["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:{w}d}/{epochs} | "
            f"train {tr['loss']:.4f} (mse {tr['mse']:.4f} phys {tr['phys']:.4f}) | "
            f"val {vl['loss']:.4f} (mse {vl['mse']:.4f} phys {vl['phys']:.4f}) | "
            f"lr {current_lr:.2e}"
        )

        if log_fn is not None:
            log_fn(epoch, tr, vl, current_lr)

        if vl["mse"] < best_val_loss:
            best_val_loss = vl["mse"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                checkpoint_dir / "best.pt",
            )
