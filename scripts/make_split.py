"""Build train/val/test JSON split + normalization stats from local label files.

Local equivalent of modal_pipeline.make_split_and_stats.
Run after labels exist locally in data/labels/{family}/{stem}/thermal.npy.

Decisions: D-07 (80/10/10, seed=42), D-08 (random split, family leakage deferred),
           D-09 (per-channel mean/std from train only).

Usage:
    python scripts/make_split.py

    # With explicit paths:
    python scripts/make_split.py \\
        --raw-dir data/raw/CircuitNet-N14 \\
        --labels-dir data/labels \\
        --splits-dir data/splits \\
        --stats-path data/normalization_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

RANDOM_SEED = 42
GPU_FAMILIES = ["Vortex-small", "Vortex-large", "nvdla-large"]


def iter_instances(raw_root: Path, labels_root: Path):
    """Yield entry dicts for all instances that have floorplan, power, and label.

    Intersects floorplan and power npz stems per family; skips missing labels.
    Verified family counts (docs/dataset-structure.md):
      Vortex-small: 96, Vortex-large: 61, nvdla-large: 32 -> total 189 matched
    """
    for family in GPU_FAMILIES:
        fp_dir = raw_root / "routability_features" / family / family / "macro_region"
        pw_dir = raw_root / "IR_drop_features" / family / family / "power_all"
        if not fp_dir.exists() or not pw_dir.exists():
            continue
        for fp in sorted(fp_dir.glob("*.npz")):
            pw = pw_dir / fp.name
            label = labels_root / family / fp.stem / "thermal.npy"
            if pw.exists() and label.exists():
                yield {
                    "instance": fp.stem,
                    "family": family,
                    "floorplan": str(fp),
                    "power": str(pw),
                    "label": str(label),
                }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build train/val/test splits and normalization stats from local labels"
    )
    p.add_argument("--raw-dir", default="data/raw/CircuitNet-N14",
                   help="Root of raw CircuitNet-N14 data (default: data/raw/CircuitNet-N14)")
    p.add_argument("--labels-dir", default="data/labels",
                   help="Root of generated HotSpot label files (default: data/labels)")
    p.add_argument("--splits-dir", default="data/splits",
                   help="Output directory for split JSON files (default: data/splits)")
    p.add_argument("--stats-path", default="data/normalization_stats.json",
                   help="Output path for normalization stats JSON")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help=f"Random seed for shuffle (default: {RANDOM_SEED})")
    args = p.parse_args()

    entries = list(iter_instances(Path(args.raw_dir), Path(args.labels_dir)))
    if not entries:
        print("ERROR: no complete instances found (need floorplan + power + label)")
        print(f"  raw-dir:    {args.raw_dir}")
        print(f"  labels-dir: {args.labels_dir}")
        return 1
    print(f"Found {len(entries)} complete instances")

    # Deterministic 80/10/10 split (D-07, D-08)
    rng = random.Random(args.seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, int(n * 0.10))
    n_tst = max(1, int(n * 0.10))
    train = shuffled[:n - n_val - n_tst]
    val   = shuffled[n - n_val - n_tst : n - n_tst]
    test  = shuffled[n - n_tst:]
    print(f"Split (random_state={args.seed}): train={len(train)} val={len(val)} test={len(test)}")

    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        out = splits_dir / f"{name}.json"
        out.write_text(json.dumps(data, indent=2))
        print(f"  Wrote {out}")

    # Normalization from TRAIN only (D-09 — prevents val/test leakage)
    fp_vals, pw_vals = [], []
    for e in train:
        fp_vals.append(np.load(e["floorplan"])["data"].ravel().astype(np.float64))
        pw_vals.append(np.load(e["power"])["data"].ravel().astype(np.float64))
    fp_all = np.concatenate(fp_vals)
    pw_all = np.concatenate(pw_vals)
    stats = {
        "floorplan": {"mean": float(fp_all.mean()), "std": float(fp_all.std())},
        "power":     {"mean": float(pw_all.mean()), "std": float(pw_all.std())},
    }
    Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_path).write_text(json.dumps(stats, indent=2))
    print(f"Norm stats written to {args.stats_path}: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
