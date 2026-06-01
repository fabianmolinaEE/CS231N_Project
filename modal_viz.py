"""Visualize per-family predictions from a saved checkpoint."""
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

app = modal.App("unet-thermal-viz", image=image)


@app.function(
    volumes={MOUNT: volume},
    gpu="A10G",
)
def visualize_per_family(run_name: str = "unet_lam0.0_ep250_b32") -> bytes:
    import sys
    sys.path.insert(0, "/app")

    import json
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    from src.dataset import ThermalDataset
    from src.models.unet import UNet

    volume.reload()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats_path = f"{MOUNT}/splits/normalization_stats.json"
    val_path   = f"{MOUNT}/splits/val.json"

    with open(stats_path) as f:
        norm_stats = json.load(f)
    label_stats = norm_stats["label"]

    with open(val_path) as f:
        val_index = json.load(f)

    # First val sample index per family
    family_idx = {}
    for i, entry in enumerate(val_index):
        fam = entry["family"]
        if fam not in family_idx:
            family_idx[fam] = i
        if len(family_idx) == len(label_stats):
            break

    val_ds = ThermalDataset(val_path, stats_path, training=False, label_stats=label_stats)

    ckpt_path = Path(MOUNT) / "checkpoints" / run_name / "best.pt"
    model = UNet(base_channels=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    families = sorted(family_idx.keys())
    n_fam = len(families)
    cols = ["Cell Density", "Power", "GT (K)", "Predicted (GT scale)", "Predicted (autoscale)"]
    fig, axes = plt.subplots(n_fam, 5, figsize=(20, 4 * n_fam))
    fig.suptitle(f"Per-family predictions — {run_name}", fontsize=14, y=1.01)

    for row, fam in enumerate(families):
        idx = family_idx[fam]
        x, y = val_ds[idx]
        x_t = x.unsqueeze(0).to(device)
        y_t = y.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_t)

        lmean = label_stats[fam]["mean"]
        lstd  = label_stats[fam]["std"]
        gt_K   = (y_t[0].squeeze().cpu().float() * lstd + lmean).numpy()
        pred_K = (pred[0].squeeze().cpu().float() * lstd + lmean).numpy()
        density = x_t[0, 0].cpu().float().numpy()
        power   = x_t[0, 1].cpu().float().numpy()

        gt_vmin, gt_vmax = gt_K.min(), gt_K.max()
        print(f"{fam}: GT [{gt_vmin:.1f}–{gt_vmax:.1f} K]  pred [{pred_K.min():.1f}–{pred_K.max():.1f} K]")

        panels = [
            (density, 0,              1,              "gray",    ""),
            (power,   power.min(),    power.max(),    "hot",     ""),
            (gt_K,    gt_vmin,        gt_vmax,        "inferno", f"[{gt_vmin:.1f}–{gt_vmax:.1f} K]"),
            (pred_K,  gt_vmin,        gt_vmax,        "inferno", f"[{pred_K.min():.1f}–{pred_K.max():.1f} K]"),
            (pred_K,  pred_K.min(),   pred_K.max(),   "inferno", f"[{pred_K.min():.1f}–{pred_K.max():.1f} K]"),
        ]

        for col, (arr, vmin, vmax, cmap, note) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(cols[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(fam, fontsize=9)
            ax.set_xlabel(note, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@app.local_entrypoint()
def main(run_name: str = "unet_lam0.0_ep250_b32", out: str = "per_family_preds.png"):
    png = visualize_per_family.remote(run_name=run_name)
    Path(out).write_bytes(png)
    print(f"Saved → {out}")


FAMILIES = ["Vortex-small", "Vortex-large", "nvdla-large"]
THICKNESSES = [
    (75,  "ood_labels/t75um"),
    (100, "ood_labels/t100um"),
    (150, "labels"),           # baseline training thickness
    (200, "ood_labels/t200um"),
    (300, "ood_labels/t300um"),
]


@app.function(volumes={MOUNT: volume})
def visualize_ood_thicknesses() -> bytes:
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    from pathlib import Path

    volume.reload()

    val_path = f"{MOUNT}/splits/val.json"
    with open(val_path) as f:
        val_index = json.load(f)

    family_stem = {}
    for entry in val_index:
        fam = entry["family"]
        if fam not in family_stem:
            family_stem[fam] = entry["instance"]
        if len(family_stem) == len(FAMILIES):
            break

    n_fam = len(FAMILIES)
    n_thick = len(THICKNESSES)
    fig, axes = plt.subplots(n_fam, n_thick, figsize=(4 * n_thick, 4 * n_fam))
    fig.suptitle("Thermal maps across chip thickness conditions", fontsize=14, y=1.01)

    for row, fam in enumerate(FAMILIES):
        stem = family_stem[fam]

        maps = []
        for _, subdir in THICKNESSES:
            p = Path(MOUNT) / subdir / fam / stem / "thermal.npy"
            maps.append(np.load(p) if p.exists() else None)

        valid = [m for m in maps if m is not None]
        vmin = min(m.min() for m in valid)
        vmax = max(m.max() for m in valid)

        for col, ((thickness, _), arr) in enumerate(zip(THICKNESSES, maps)):
            ax = axes[row, col]
            if arr is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_facecolor("#222")
            else:
                im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap="inferno", origin="upper")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xlabel(f"[{arr.min():.0f}–{arr.max():.0f} K]", fontsize=8)
            label = f"{thickness} µm" + (" ★" if thickness == 150 else "")
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(fam, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@app.local_entrypoint()
def viz_ood(out: str = "ood_thickness_comparison.png"):
    import subprocess
    png = visualize_ood_thicknesses.remote()
    Path(out).write_bytes(png)
    print(f"Saved → {out}")
    subprocess.run(["open", out])


INSTANCE_PAIRS = [
    ("Vortex-small", "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap"),
    ("Vortex-small", "Vortex-small_freq_200_mp_1_fpu_55_fpa_1.0_p_1_fi_ap"),
    ("Vortex-large", "Vortex-large_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap"),
    ("Vortex-large", "Vortex-large_freq_200_mp_1_fpu_55_fpa_1.0_p_1_fi_ap"),
]


@app.function(volumes={MOUNT: volume})
def visualize_instances_by_thickness() -> bytes:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    from pathlib import Path

    volume.reload()

    n_inst = len(INSTANCE_PAIRS)
    n_thick = len(THICKNESSES)
    fig, axes = plt.subplots(n_inst, n_thick, figsize=(4 * n_thick, 3.5 * n_inst))
    fig.suptitle("Thermal maps — same instances, different chip thicknesses", fontsize=13, y=1.01)

    for row, (fam, stem) in enumerate(INSTANCE_PAIRS):
        maps = []
        for _, subdir in THICKNESSES:
            p = Path(MOUNT) / subdir / fam / stem / "thermal.npy"
            maps.append(np.load(p) if p.exists() else None)

        valid = [m for m in maps if m is not None]
        vmin = min(m.min() for m in valid)
        vmax = max(m.max() for m in valid)

        short = stem.replace(fam + "_", "")
        for col, ((thickness, _), arr) in enumerate(zip(THICKNESSES, maps)):
            ax = axes[row, col]
            if arr is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_facecolor("#222")
            else:
                im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap="inferno", origin="upper")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xlabel(f"[{arr.min():.0f}–{arr.max():.0f} K]", fontsize=7)
            if row == 0:
                label = f"{thickness} µm" + (" ★" if thickness == 150 else "")
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{fam}\n{short}", fontsize=7.5)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@app.local_entrypoint()
def viz_instances(out: str = "instance_thickness_comparison.png"):
    import subprocess
    png = visualize_instances_by_thickness.remote()
    Path(out).write_bytes(png)
    print(f"Saved → {out}")
    subprocess.run(["open", out])
