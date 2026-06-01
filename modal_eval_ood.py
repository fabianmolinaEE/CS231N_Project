"""Evaluate all λ checkpoints on in-distribution val set + OOD conditions.

Produces a results table (RMSE in Kelvin, SSIM, Hotspot-IoU) for each
(model, condition) pair and saves results to /data/eval_ood_results.json.

Usage:
    modal run modal_eval_ood.py
"""
import json
import modal
from pathlib import Path

volume = modal.Volume.from_name("circuitnet-n14", create_if_missing=False)
MOUNT = "/data"

LAM_VALUES = [0.0, 0.01, 0.1, 1.0]
EPOCHS = 250
BASE_CHANNELS = 32

# Tags must match modal_ood.py: _t_chip_tag() and _r_convec_tag()
OOD_THICKNESS_TAGS = ["t75um", "t100um", "t200um", "t300um"]
OOD_HTC_TAGS = ["rconv0p01", "rconv0p05", "rconv0p20", "rconv0p50"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "numpy",
        "scipy",
        "scikit-image",
    )
    .add_local_dir("src", remote_path="/app/src")
)

app = modal.App("unet-eval-ood", image=image)


def _ckpt_name(lam: float) -> str:
    return f"unet_lam{lam}_ep{EPOCHS}_b{BASE_CHANNELS}"


def _build_ood_index(val_index: list, tag: str, ood_root: Path) -> list:
    """Replace label paths with OOD labels; skip instances where label is missing."""
    ood_entries = []
    for entry in val_index:
        family = entry["family"]
        stem = Path(entry["label"]).parent.name
        ood_label = ood_root / tag / family / stem / "thermal.npy"
        if ood_label.exists():
            ood_entries.append({**entry, "label": str(ood_label)})
    return ood_entries


@app.function(
    volumes={MOUNT: volume},
    gpu="A10G",
    timeout=3600,
)
def eval_all():
    import os
    import sys
    import tempfile
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, "/app")

    from torch.utils.data import DataLoader, Dataset
    from src.models.unet import UNet
    from src.dataset import ThermalDataset
    from src.evaluate import ssim, hotspot_iou

    volume.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load normalization stats
    stats_path = f"{MOUNT}/splits/normalization_stats.json"
    with open(stats_path) as f:
        norm_stats = json.load(f)
    label_stats = norm_stats["label"]

    # Load val index
    with open(f"{MOUNT}/splits/val.json") as f:
        val_index = json.load(f)
    print(f"Val set: {len(val_index)} instances")

    # Build OOD indices
    ood_root = Path(MOUNT) / "ood_labels"
    ood_tags = OOD_THICKNESS_TAGS + OOD_HTC_TAGS
    ood_indices = {}
    for tag in ood_tags:
        entries = _build_ood_index(val_index, tag, ood_root)
        ood_indices[tag] = entries
        print(f"  OOD [{tag}]: {len(entries)} instances with labels")

    # Dataset that also yields family name for per-sample denormalization
    class FamilyDataset(Dataset):
        def __init__(self, base_ds):
            self.ds = base_ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            x, lb = self.ds[idx]
            return x, lb, self.ds.index[idx]["family"]

    def make_loader(index):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(index, tmp)
            tmp_path = tmp.name
        try:
            ds = ThermalDataset(tmp_path, stats_path, training=False, label_stats=label_stats)
            return DataLoader(FamilyDataset(ds), batch_size=8, shuffle=False, num_workers=2)
        finally:
            os.unlink(tmp_path)

    @torch.no_grad()
    def eval_loader(model, loader):
        """Compute RMSE (K), SSIM, Hotspot-IoU; denormalize to Kelvin per family."""
        model.eval()
        preds_K, gts_K = [], []
        for x, T_norm, families in loader:
            x = x.to(device)
            pred_norm = model(x).cpu()
            for i, fam in enumerate(families):
                std = label_stats[fam]["std"]
                mean = label_stats[fam]["mean"]
                preds_K.append(pred_norm[i] * std + mean)
                gts_K.append(T_norm[i] * std + mean)

        T_pred = torch.stack(preds_K)  # (N, 1, H, W)
        T_gt = torch.stack(gts_K)
        data_range = float(T_gt.max() - T_gt.min())
        return {
            "rmse_K": float(torch.sqrt(F.mse_loss(T_pred, T_gt)).item()),
            "ssim": ssim(T_pred, T_gt, data_range=data_range),
            "hotspot_iou": hotspot_iou(T_pred, T_gt),
            "n": len(preds_K),
        }

    results = {}

    for lam in LAM_VALUES:
        run_name = _ckpt_name(lam)
        ckpt_path = Path(MOUNT) / "checkpoints" / run_name / "best.pt"
        if not ckpt_path.exists():
            print(f"MISSING checkpoint: {ckpt_path}")
            continue

        model = UNet(base_channels=BASE_CHANNELS).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        saved_epoch = ckpt.get("epoch", "?")
        saved_loss = ckpt.get("val_loss", float("nan"))
        print(f"\n=== lam={lam} (epoch {saved_epoch}, best_val_mse={saved_loss:.5f}) ===")

        lam_results = {}

        # In-distribution val
        m = eval_loader(model, make_loader(val_index))
        lam_results["val_id"] = m
        print(f"  val (in-dist): RMSE={m['rmse_K']:.3f}K  SSIM={m['ssim']:.4f}  IoU={m['hotspot_iou']:.4f}  n={m['n']}")

        # OOD conditions
        for tag in ood_tags:
            if not ood_indices[tag]:
                print(f"  [{tag}] skipped (no labels found)")
                continue
            m = eval_loader(model, make_loader(ood_indices[tag]))
            lam_results[tag] = m
            print(f"  [{tag}]: RMSE={m['rmse_K']:.3f}K  SSIM={m['ssim']:.4f}  IoU={m['hotspot_iou']:.4f}  n={m['n']}")

        results[str(lam)] = lam_results

    out_path = Path(MOUNT) / "eval_ood_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nResults saved to {out_path}")
    return results


@app.local_entrypoint()
def main():
    results = eval_all.remote()

    conditions = ["val_id"] + OOD_THICKNESS_TAGS + OOD_HTC_TAGS

    def print_table(metric_key, label):
        print(f"\n=== {label} ===")
        header = f"{'Condition':<18}" + "".join(f"  {'λ='+str(l):<10}" for l in LAM_VALUES)
        print(header)
        for cond in conditions:
            row = f"{cond:<18}"
            for lam in LAM_VALUES:
                m = results.get(str(lam), {}).get(cond)
                if m:
                    row += f"  {m[metric_key]:>8.4f}  "
                else:
                    row += "  —         "
            print(row)

    print_table("rmse_K", "RMSE (Kelvin)")
    print_table("ssim", "SSIM")
    print_table("hotspot_iou", "Hotspot IoU (top 5%)")

    with open("eval_ood_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved locally to eval_ood_results.json")
