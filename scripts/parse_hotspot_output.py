"""Parse HotSpot .grid.steady output -> resized float32 .npy.

Format (verified from HotSpot examples):
  Each non-header line: <col_idx> <row_idx> <temperature_kelvin>
  Some lines may be comments or headers (non-numeric) - those are skipped.
  Row indices may run max->0; orientation is documented after Plan 05 visualization.

CLI usage:
    python scripts/parse_hotspot_output.py \\
        data/labels/{design}/design.grid.steady \\
        data/labels/{design}/thermal.npy \\
        --rows 256 --cols 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


def parse_grid_steady(path: Path, rows: int, cols: int) -> np.ndarray:
    """Parse HotSpot .grid.steady file into a float32 numpy array.

    Args:
        path: Path to .grid.steady file.
        rows: Number of grid rows used in the HotSpot simulation.
        cols: Number of grid cols used in the HotSpot simulation.

    Returns:
        float32 array of shape (rows, cols) with temperature in Kelvin.
    """
    grid = np.zeros((rows, cols), dtype=np.float32)
    seen = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            try:
                col_idx = int(parts[0])
                row_idx = int(parts[1])
                temp = float(parts[2])
            except ValueError:
                continue
            if 0 <= row_idx < rows and 0 <= col_idx < cols:
                grid[row_idx, col_idx] = temp
                seen += 1
    if seen == 0:
        raise RuntimeError(f"No grid values parsed from {path}")
    return grid


def resize_to(arr: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    """Resize array to target dimensions using bilinear interpolation."""
    if arr.shape == (target_rows, target_cols):
        return arr.astype(np.float32)
    zh = target_rows / arr.shape[0]
    zw = target_cols / arr.shape[1]
    return zoom(arr.astype(np.float32), (zh, zw), order=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse HotSpot .grid.steady into a resized float32 .npy file"
    )
    parser.add_argument("grid_steady_path", help="Path to HotSpot .grid.steady file")
    parser.add_argument("output_npy_path", help="Output path for .npy thermal label")
    parser.add_argument(
        "--rows",
        type=int,
        default=256,
        help="Grid rows used in HotSpot simulation (default: 256)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=256,
        help="Grid cols used in HotSpot simulation (default: 256)",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=256,
        help="Target output rows (default: 256)",
    )
    parser.add_argument(
        "--target-cols",
        type=int,
        default=256,
        help="Target output cols (default: 256)",
    )
    args = parser.parse_args()

    grid = parse_grid_steady(Path(args.grid_steady_path), args.rows, args.cols)
    final = resize_to(grid, args.target_rows, args.target_cols)
    assert final.shape == (args.target_rows, args.target_cols)
    assert final.dtype == np.float32
    Path(args.output_npy_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy_path, final)
    print(
        f"Wrote {args.output_npy_path} shape={final.shape} dtype={final.dtype} "
        f"min={final.min():.3f} max={final.max():.3f} mean={final.mean():.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
