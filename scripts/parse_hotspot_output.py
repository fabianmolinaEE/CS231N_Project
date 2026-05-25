"""Parse HotSpot .grid.steady output -> resized float32 .npy.

Verified format (Plan-02, HotSpot v6 grid mode):
  Layer 0:
  0<TAB>382.46
  1<TAB>382.46
  ...
  65535<TAB>319.58
  Layer 1:
  0<TAB>...
  ...

  - File contains 4 layers (die, spreader, sink, ...) separated by "Layer N:" headers.
  - Each layer has rows*cols entries in row-major order: idx = row * cols + col.
  - We extract Layer 0 (the chip die layer).
  - Temperature values are in Kelvin.

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


def parse_grid_steady(path: Path, rows: int, cols: int, layer: int = 0) -> np.ndarray:
    """Parse HotSpot .grid.steady file into a float32 numpy array.

    Actual format (verified Plan-02):
        Layer 0:
        <idx>\\t<temp_kelvin>
        ...
        Layer 1:
        ...

    Indices are sequential (row-major): row = idx // cols, col = idx % cols.

    Args:
        path: Path to .grid.steady file.
        rows: Number of grid rows used in the HotSpot simulation.
        cols: Number of grid cols used in the HotSpot simulation.
        layer: Which layer to extract (default 0 = chip die).

    Returns:
        float32 array of shape (rows, cols) with temperature in Kelvin.
    """
    grid = np.zeros((rows, cols), dtype=np.float32)
    n_cells = rows * cols
    in_target_layer = False
    seen = 0

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # Detect layer header lines: "Layer N:"
            if stripped.startswith("Layer "):
                try:
                    layer_num = int(stripped.split()[1].rstrip(":"))
                    in_target_layer = (layer_num == layer)
                except (IndexError, ValueError):
                    in_target_layer = False
                continue
            if not in_target_layer:
                continue
            # Data line: "<idx>\t<temperature>"
            parts = stripped.split()
            if len(parts) != 2:
                continue
            try:
                idx = int(parts[0])
                temp = float(parts[1])
            except ValueError:
                continue
            if 0 <= idx < n_cells:
                row_idx = idx // cols
                col_idx = idx % cols
                grid[row_idx, col_idx] = temp
                seen += 1

    if seen == 0:
        raise RuntimeError(
            f"No grid values parsed from {path} for layer={layer}. "
            f"Check that HotSpot ran in grid mode and the file is not empty."
        )
    if seen < n_cells:
        import warnings
        warnings.warn(
            f"Only {seen}/{n_cells} cells parsed from {path} layer={layer}. "
            f"Missing cells default to 0.0 K."
        )
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
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="HotSpot layer index to extract (default: 0 = chip die)",
    )
    args = parser.parse_args()

    grid = parse_grid_steady(Path(args.grid_steady_path), args.rows, args.cols, args.layer)
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
