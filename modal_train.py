"""
Train U-Net with physics loss on Modal GPU.

Reads split JSONs and labels from the circuitnet-n14 volume.
Saves checkpoints to /data/checkpoints/{run_name}/ on the volume.

Usage:
    modal run modal_train.py                          # default: lam_phys=0.1
    modal run modal_train.py --lam-phys 0.0           # MSE-only baseline
    modal run modal_train.py --lam-phys 1.0 --epochs 100
"""
import modal
from pathlib import Path

volume = modal.Volume.from_name("circuitnet-n14", create_if_missing=False)
MOUNT = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "numpy",
        "scipy",
        "scikit-image",
        "wandb",
        "matplotlib",
    )
    .add_local_dir("src", remote_path="/app/src")
)

app = modal.App("unet-thermal-train", image=image)


@app.function(
    volumes={MOUNT: volume},
    gpu="A10G",
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb")],
)
def train_unet(
    lam_phys: float = 0.1,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 8,
    base_channels: int = 32,
):
    import sys
    sys.path.insert(0, "/app")

    import json
    import torch
    from torch.utils.data import DataLoader

    from src.dataset import ThermalDataset
    from src.models.unet import UNet
    from src.train import train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  lam_phys={lam_phys}  |  epochs={epochs}  |  base_ch={base_channels}")

    volume.reload()

    index_train = f"{MOUNT}/splits/train.json"
    index_val   = f"{MOUNT}/splits/val.json"
    stats_path  = f"{MOUNT}/splits/normalization_stats.json"

    train_ds = ThermalDataset(index_train, stats_path, training=True)
    val_ds   = ThermalDataset(index_val,   stats_path, training=False)
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(base_channels=base_channels)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    run_name = f"unet_lam{lam_phys}_ep{epochs}_b{base_channels}"
    ckpt_dir = Path(MOUNT) / "checkpoints" / run_name

    import wandb
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    wandb.init(
        project="gpu-thermal-prediction",
        name=run_name,
        config={
            "lam_phys": lam_phys,
            "epochs": epochs,
            "base_channels": base_channels,
            "lr": lr,
            "batch_size": batch_size,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_params": n_params,
        },
    )

    # Pre-load one val sample for image previews — avoids spawning/leaking
    # DataLoader workers on every image-logging epoch (WR-02).
    _x0, _y0 = val_ds[0]
    _x0 = _x0.unsqueeze(0)   # (1, 2, H, W)
    _y0 = _y0.unsqueeze(0)   # (1, 1, H, W)

    def _tensor_to_pil(t, vmin, vmax, cmap="inferno"):
        """Render tensor as a PIL Image using a shared [vmin, vmax] scale (WR-03)."""
        arr = t.squeeze().cpu().float().numpy()
        arr = (arr - vmin) / (vmax - vmin + 1e-8)
        arr = arr.clip(0, 1)
        colored = (plt.get_cmap(cmap)(arr)[:, :, :3] * 255).astype(np.uint8)
        return PILImage.fromarray(colored)

    def log_fn(epoch, tr, vl, lr):
        log_dict = {
            "epoch": epoch,
            "train/loss": tr["loss"],
            "train/mse": tr["mse"],
            "train/physics": tr["phys"],
            "val/loss": vl["loss"],
            "val/mse": vl["mse"],
            "val/physics": vl["phys"],
            "lr": lr,
        }
        if epoch % 10 == 0:
            model.eval()
            try:
                with torch.no_grad():
                    x_s = _x0.to(device)
                    y_s = _y0.to(device)
                    pred_s = model(x_s)
            finally:
                model.train()  # always restore, even if inference fails (WR-01)
            gt_arr = y_s[0].squeeze().cpu().float().numpy()
            vmin, vmax = float(gt_arr.min()), float(gt_arr.max())
            log_dict["val/sample"] = [
                wandb.Image(_tensor_to_pil(x_s[0, 0], 0.0, 1.0, cmap="gray"), caption="Cell Density"),
                wandb.Image(_tensor_to_pil(pred_s[0], vmin, vmax, cmap="inferno"), caption="Predicted"),
                wandb.Image(_tensor_to_pil(y_s[0],    vmin, vmax, cmap="inferno"), caption="Ground Truth"),
            ]
        wandb.log(log_dict, step=epoch)

    try:
        train(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            lam_phys=lam_phys,
            checkpoint_dir=ckpt_dir,
            device=device,
            log_fn=log_fn,
        )
    finally:
        wandb.finish()  # always mark run complete, even on crash (WR-04)
    volume.commit()
    print(f"Checkpoint saved to {ckpt_dir}/best.pt")


@app.local_entrypoint()
def main(
    lam_phys: float = 0.1,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 8,
    base_channels: int = 32,
):
    train_unet.remote(
        lam_phys=lam_phys,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        base_channels=base_channels,
    )
