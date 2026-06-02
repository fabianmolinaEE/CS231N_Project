"""Evaluate all checkpoints on combined val+test set + OOD conditions.

Models evaluated:
  - UNet with λ = 0.0, 0.01, 0.1, 1.0  (base_channels=32)
  - PlainCNN with λ = 0.0               (base_channels=64)
  - EncoderDecoder with λ = 0.0         (base_channels=64)

Metrics: RMSE (Kelvin), SSIM, Hotspot-IoU (top 5%)
Results saved to /data/eval_ood_results.json and locally.

Usage:
    modal run modal_eval_ood.py
"""
import json
import modal
from pathlib import Path

volume = modal.Volume.from_name("circuitnet-n14", create_if_missing=False)
MOUNT = "/data"

# Tags must match modal_ood.py: _t_chip_tag() and _r_convec_tag()
OOD_THICKNESS_TAGS = ["t75um", "t100um", "t200um", "t300um"]
OOD_HTC_TAGS = ["rconv0p01", "rconv0p05", "rconv0p20", "rconv0p50"]

# All model runs to evaluate: (display_name, model_type, lam, base_channels)
MODEL_CONFIGS = [
    ("unet_lam0.00",     "unet",            0.0,  32),
    ("unet_lam0.01",     "unet",            0.01, 32),
    ("unet_lam0.10",     "unet",            0.1,  32),
    ("unet_lam1.00",     "unet",            1.0,  32),
    ("plain_cnn_lam0.00","plain_cnn",       0.0,  64),
    ("plain_cnn_lam0.01","plain_cnn",       0.01, 64),
    ("plain_cnn_lam0.10","plain_cnn",       0.1,  64),
    ("plain_cnn_lam1.00","plain_cnn",       1.0,  64),
    ("enc_dec_lam0.00",  "encoder_decoder", 0.0,  64),
    ("enc_dec_lam0.01",  "encoder_decoder", 0.01, 64),
    ("enc_dec_lam0.10",  "encoder_decoder", 0.1,  64),
    ("enc_dec_lam1.00",  "encoder_decoder", 1.0,  64),
]

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


def _ckpt_name(model_type: str, lam: float, base_channels: int) -> str:
    return f"{model_type}_lam{lam}_ep250_b{base_channels}"


def _build_ood_index(base_index: list, tag: str, ood_root: Path) -> list:
    """Replace label paths with OOD labels; skip instances where label is missing."""
    entries = []
    for entry in base_index:
        family = entry["family"]
        stem = Path(entry["label"]).parent.name
        ood_label = ood_root / tag / family / stem / "thermal.npy"
        if ood_label.exists():
            entries.append({**entry, "label": str(ood_label)})
    return entries


def _load_model(model_type: str, base_channels: int, ckpt_path: Path, device: str):
    import torch
    if model_type == "unet":
        from src.models.unet import UNet
        model = UNet(base_channels=base_channels)
    elif model_type == "plain_cnn":
        from src.models.baseline_cnn import PlainCNN
        model = PlainCNN(base_channels=base_channels)
    elif model_type == "encoder_decoder":
        from src.models.encoder_decoder import EncoderDecoder
        model = EncoderDecoder(base_channels=base_channels)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan"))


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
    from src.dataset import ThermalDataset
    from src.evaluate import ssim, hotspot_iou

    volume.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    stats_path = f"{MOUNT}/splits/normalization_stats.json"
    with open(stats_path) as f:
        norm_stats = json.load(f)
    label_stats = norm_stats["label"]

    # Combine val + test for evaluation
    with open(f"{MOUNT}/splits/val.json") as f:
        val_index = json.load(f)
    with open(f"{MOUNT}/splits/test.json") as f:
        test_index = json.load(f)
    eval_index = val_index + test_index
    print(f"Eval set: {len(val_index)} val + {len(test_index)} test = {len(eval_index)} total")

    # Build OOD indices
    ood_root = Path(MOUNT) / "ood_labels"
    ood_tags = OOD_THICKNESS_TAGS + OOD_HTC_TAGS
    ood_indices = {}
    for tag in ood_tags:
        entries = _build_ood_index(eval_index, tag, ood_root)
        ood_indices[tag] = entries
        print(f"  OOD [{tag}]: {len(entries)} instances")

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
        T_pred = torch.stack(preds_K)
        T_gt = torch.stack(gts_K)
        data_range = float(T_gt.max() - T_gt.min())
        return {
            "rmse_K": float(torch.sqrt(F.mse_loss(T_pred, T_gt)).item()),
            "ssim": ssim(T_pred, T_gt, data_range=data_range),
            "hotspot_iou": hotspot_iou(T_pred, T_gt),
            "n": len(preds_K),
        }

    results = {}
    ckpt_root = Path(MOUNT) / "checkpoints"

    for display_name, model_type, lam, base_ch in MODEL_CONFIGS:
        ckpt_name = _ckpt_name(model_type, lam, base_ch)
        ckpt_path = ckpt_root / ckpt_name / "best.pt"
        if not ckpt_path.exists():
            print(f"MISSING: {ckpt_path} — skipping {display_name}")
            continue

        model, epoch, val_loss = _load_model(model_type, base_ch, ckpt_path, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {display_name} (epoch {epoch}, best_val_mse={val_loss:.5f}, params={n_params:,}) ===")

        run = {}

        m = eval_loader(model, make_loader(eval_index))
        run["eval_id"] = m
        print(f"  eval (in-dist): RMSE={m['rmse_K']:.3f}K  SSIM={m['ssim']:.4f}  IoU={m['hotspot_iou']:.4f}  n={m['n']}")

        for tag in ood_tags:
            if not ood_indices[tag]:
                print(f"  [{tag}] skipped (no labels)")
                continue
            m = eval_loader(model, make_loader(ood_indices[tag]))
            run[tag] = m
            print(f"  [{tag}]: RMSE={m['rmse_K']:.3f}K  SSIM={m['ssim']:.4f}  IoU={m['hotspot_iou']:.4f}  n={m['n']}")

        results[display_name] = run

    out_path = Path(MOUNT) / "eval_ood_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nResults saved to {out_path}")
    return results


@app.local_entrypoint()
def main():
    results = eval_all.remote()

    model_names = list(results.keys())
    conditions = ["eval_id"] + OOD_THICKNESS_TAGS + OOD_HTC_TAGS

    def print_table(metric_key, label, fmt):
        print(f"\n=== {label} ===")
        col_w = 13
        header = f"{'Condition':<18}" + "".join(f"  {n:>{col_w}}" for n in model_names)
        print(header)
        for cond in conditions:
            row = f"{cond:<18}"
            for name in model_names:
                m = results.get(name, {}).get(cond)
                row += f"  {fmt(m):>{col_w}}" if m else f"  {'—':>{col_w}}"
            print(row)

    print_table("rmse_K",      "RMSE (Kelvin)",         lambda m: f"{m['rmse_K']:.3f}K")
    print_table("hotspot_iou", "Hotspot IoU (top 5%)",  lambda m: f"{m['hotspot_iou']:.4f}")
    print_table("ssim",        "SSIM",                  lambda m: f"{m['ssim']:.4f}")

    with open("eval_ood_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved locally to eval_ood_results.json")
