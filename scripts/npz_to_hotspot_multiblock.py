"""Convert CircuitNet-N14 npz files to HotSpot multi-block .flp + .ptrace.

Multi-block approach: coarsens the power_all tile grid into NxM rectangular blocks,
each covering a group of CircuitNet tiles. Each block gets its own power entry in the
.ptrace. This gives HotSpot spatially varying power input, producing thermal maps that
reflect the actual power distribution across the chip — unlike the single-block approach
(npz_to_hotspot.py) which gives uniform temperature distribution.

HotSpot constraint: number of blocks is limited only by memory/speed, but the .ptrace
format uses named blocks, so keep to a practical number (64-1024).

Default coarsening: 16x16 blocks (256 total). Each block covers ~29x28 CircuitNet tiles
for Vortex-small (459x456). Runtime is similar to single-block.

Power scaling: same --max-total-power-w cap as npz_to_hotspot.py. Per-block power is
proportional to that block's fraction of total power_all sum.

CLI usage:
    python scripts/npz_to_hotspot_multiblock.py \\
        data/raw/CircuitNet-N14/routability_features/Vortex-small/Vortex-small/macro_region/INST.npz \\
        data/raw/CircuitNet-N14/IR_drop_features/Vortex-small/Vortex-small/power_all/INST.npz \\
        data/labels/INST_multiblock/

    # Custom block grid (32x32 = 1024 blocks):
    python scripts/npz_to_hotspot_multiblock.py FP PW OUT --block-rows 32 --block-cols 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TILE_SIZE_M = 2.25e-6  # N14: 2.25 um per tile


def load_npz_data(path: Path) -> np.ndarray:
    npz = np.load(path, allow_pickle=False)
    if "data" not in npz.files:
        raise KeyError(f"Key 'data' not in {path}. Available: {list(npz.files)}")
    return npz["data"]


def coarsen_power(pw_arr: np.ndarray, block_rows: int, block_cols: int) -> np.ndarray:
    """Sum power_all into a block_rows x block_cols grid.

    Each block covers ceil(rows/block_rows) x ceil(cols/block_cols) tiles.
    Uses numpy slice-based summation (no loops over blocks).

    Returns shape (block_rows, block_cols) float64.
    """
    rows, cols = pw_arr.shape
    # Pad to make evenly divisible if needed
    pad_r = (-rows) % block_rows
    pad_c = (-cols) % block_cols
    if pad_r > 0 or pad_c > 0:
        pw_arr = np.pad(pw_arr, ((0, pad_r), (0, pad_c)), mode="constant")

    h = (rows + pad_r) // block_rows
    w = (cols + pad_c) // block_cols
    # Reshape into (block_rows, h, block_cols, w) then sum over tile dims
    coarse = pw_arr.reshape(block_rows, h, block_cols, w).sum(axis=(1, 3))
    return coarse.astype(np.float64)


def write_flp(out_path: Path, chip_rows: int, chip_cols: int,
              block_rows: int, block_cols: int,
              tile_size_m: float) -> None:
    """Write multi-block floorplan.

    Blocks are laid out in a regular grid, row-major (row 0 at bottom in HotSpot coords).
    Block naming: b{row:03d}_{col:03d}

    Format: name<TAB>width<TAB>height<TAB>left_x<TAB>bottom_y
    """
    chip_w = chip_cols * tile_size_m
    chip_h = chip_rows * tile_size_m
    block_w = chip_w / block_cols
    block_h = chip_h / block_rows

    with open(out_path, "w") as f:
        for br in range(block_rows):
            for bc in range(block_cols):
                name = f"b{br:03d}_{bc:03d}"
                left_x = bc * block_w
                # HotSpot uses (0,0) at bottom-left; row 0 of our array → top of chip
                # Map array row br to physical bottom_y (br=0 → top → bottom_y = chip_h - block_h)
                bottom_y = chip_h - (br + 1) * block_h
                f.write(f"{name}\t{block_w:.9f}\t{block_h:.9f}\t{left_x:.9f}\t{bottom_y:.9f}\n")


def write_ptrace(out_path: Path, coarse_power: np.ndarray,
                 total_cap_w: float) -> None:
    """Write multi-block power trace.

    Header line: space-separated block names
    Data line: space-separated power values in Watts

    Each block's power is proportional to its fraction of total power_all,
    scaled so total = total_cap_w.
    """
    block_rows, block_cols = coarse_power.shape
    raw_total = coarse_power.sum()

    if raw_total <= 0 or not np.isfinite(raw_total):
        # Fallback: uniform distribution
        per_block = total_cap_w / (block_rows * block_cols)
        scaled = np.full_like(coarse_power, per_block)
    else:
        scaled = coarse_power * (total_cap_w / raw_total)

    names = [f"b{br:03d}_{bc:03d}"
             for br in range(block_rows)
             for bc in range(block_cols)]
    powers = [scaled[br, bc]
              for br in range(block_rows)
              for bc in range(block_cols)]

    with open(out_path, "w") as f:
        f.write(" ".join(names) + "\n")
        f.write(" ".join(f"{p:.9f}" for p in powers) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CircuitNet npz to HotSpot multi-block .flp + .ptrace"
    )
    parser.add_argument("floorplan_npz")
    parser.add_argument("power_npz")
    parser.add_argument("out_dir")
    parser.add_argument("--block-rows", type=int, default=16,
                        help="Number of block rows in coarsened grid (default: 16)")
    parser.add_argument("--block-cols", type=int, default=16,
                        help="Number of block cols in coarsened grid (default: 16)")
    parser.add_argument("--tile-size-m", type=float, default=TILE_SIZE_M)
    parser.add_argument("--max-total-power-w", type=float, default=10.0,
                        help="Cap total chip power in Watts (default: 10.0)")
    args = parser.parse_args()

    fp_path = Path(args.floorplan_npz)
    pw_path = Path(args.power_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_arr = load_npz_data(fp_path)
    pw_arr = load_npz_data(pw_path)

    assert fp_arr.shape == pw_arr.shape, (
        f"Shape mismatch: floorplan={fp_arr.shape}, power={pw_arr.shape}"
    )
    if fp_arr.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got ndim={fp_arr.ndim}")

    rows, cols = fp_arr.shape
    coarse = coarsen_power(pw_arr, args.block_rows, args.block_cols)

    raw_total = float(pw_arr.sum())
    if raw_total > args.max_total_power_w and args.max_total_power_w > 0:
        print(
            f"WARNING: raw total_power={raw_total:.3f} W exceeds "
            f"--max-total-power-w={args.max_total_power_w:.3f} W; scaling down."
        )
    total_w = min(raw_total, args.max_total_power_w) if args.max_total_power_w > 0 else raw_total
    if not np.isfinite(total_w) or total_w <= 0:
        total_w = 1e-6

    write_flp(out_dir / "design.flp", rows, cols, args.block_rows, args.block_cols,
              args.tile_size_m)
    write_ptrace(out_dir / "design.ptrace", coarse, total_w)

    chip_w = cols * args.tile_size_m
    chip_h = rows * args.tile_size_m
    n_blocks = args.block_rows * args.block_cols
    print(f"Wrote {out_dir/'design.flp'}: {n_blocks} blocks "
          f"({args.block_rows}x{args.block_cols}), "
          f"chip {chip_w:.4e}m x {chip_h:.4e}m")
    print(f"Wrote {out_dir/'design.ptrace'}: total_power={total_w:.6f} W "
          f"(min_block={coarse.min()*total_w/coarse.sum():.6f}, "
          f"max_block={coarse.max()*total_w/coarse.sum():.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
