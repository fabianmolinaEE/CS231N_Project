"""Visualize thermal.npy label files from a local directory.

Usage:
    python scripts/view_labels.py ./smoke_labels
    python scripts/view_labels.py ./smoke_labels --save smoke_labels.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def find_labels(root: Path) -> list[Path]:
    return sorted(root.rglob("thermal.npy"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to this path (default: <labels_dir>.png)")
    args = parser.parse_args()

    paths = find_labels(args.labels_dir)
    if not paths:
        print(f"No thermal.npy files found under {args.labels_dir}")
        return 1
    print(f"Found {len(paths)} labels")

    n = len(paths)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax, path in zip(axes, paths):
        arr = np.load(path)
        # Label is relative to ambient — show ΔT from min
        delta = arr.max() - arr.min()
        # family/stem from path: .../labels/<family>/<stem>/thermal.npy
        parts = path.parts
        try:
            label_idx = parts.index("labels")
            title = f"{parts[label_idx+1]}\n{parts[label_idx+2]}"
        except (ValueError, IndexError):
            title = path.parent.name
        im = ax.imshow(arr, cmap="plasma", interpolation="nearest")
        ax.set_title(f"{title}\nΔT={delta:.1f}K  min={arr.min():.1f}K", fontsize=7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()

    save_path = args.save or str(args.labels_dir).rstrip("/") + ".png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
