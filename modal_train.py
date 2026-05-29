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

    def _tensor_to_pil(t, cmap="inferno"):
        """Convert (1, H, W) or (H, W) tensor to a PIL Image using a colormap."""
        arr = t.squeeze().cpu().float().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
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
            with torch.no_grad():
                x_sample, T_gt_sample = next(iter(val_loader))
                x_sample = x_sample[:1].to(device)
                T_gt_sample = T_gt_sample[:1].to(device)
                T_pred_sample = model(x_sample)
            model.train()
            log_dict["val/sample"] = [
                wandb.Image(_tensor_to_pil(x_sample[0, 0], cmap="gray"),  caption="Cell Density"),
                wandb.Image(_tensor_to_pil(T_pred_sample[0], cmap="inferno"), caption="Predicted"),
                wandb.Image(_tensor_to_pil(T_gt_sample[0],   cmap="inferno"), caption="Ground Truth"),
            ]
        wandb.log(log_dict, step=epoch)

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

    wandb.finish()
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
