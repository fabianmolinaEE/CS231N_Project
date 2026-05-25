"""Convert one CircuitNet-N14 design's npz features to HotSpot .flp + .ptrace.

Single-block aggregate approach (per RESEARCH.md Pitfall 2):
  - Chip is a single block in .flp (thermal monolith).
  - .ptrace contains total watts derived from power_all tiles.
  - HotSpot grid mode distributes block power uniformly across chip area; this
    is the most natural mapping when only tile-level power density is available.

Verified Plan-01 default key: all npz files use key 'data' (not the feature name).
See docs/dataset-structure.md for verified structure.

Power units note (Plan-02 finding):
  CircuitNet power_all values appear to represent relative power density
  (dimensionless or per-unit), NOT absolute Watts. Raw sums (~1.6e6) passed
  directly to HotSpot cause the solver to diverge (T_chip > 160,000 K).
  The --max-total-power-w flag (default 10.0 W) caps the total chip power to a
  physically realistic value for the Vortex-small design family (~1mm^2 chip at
  200 MHz). The relative spatial distribution of heat is preserved.
  Revisit with actual physical calibration before final label generation (Plan 03).

CLI usage (two separate file paths):
    python scripts/npz_to_hotspot.py \\
        data/raw/CircuitNet-N14/routability_features/Vortex-small/Vortex-small/macro_region/Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz \\
        data/raw/CircuitNet-N14/IR_drop_features/Vortex-small/Vortex-small/power_all/Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz \\
        data/labels/Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# N14 tile = 2.25 um x 2.25 um (RESEARCH.md CircuitNet N14 Specifics)
TILE_SIZE_M = 2.25e-6


def load_npz_data(path: Path) -> np.ndarray:
    """Load array from npz file using verified key 'data'."""
    npz = np.load(path, allow_pickle=False)
    if "data" not in npz.files:
        raise KeyError(f"Key 'data' not in {path}. Available: {list(npz.files)}")
    return npz["data"]


def write_flp(out_path: Path, width_m: float, height_m: float) -> None:
    """Single-block floorplan: name<TAB>width<TAB>height<TAB>left_x<TAB>bottom_y."""
    with open(out_path, "w") as f:
        f.write(f"chip\t{width_m:.6f}\t{height_m:.6f}\t0.000000\t0.000000\n")


def write_ptrace(out_path: Path, total_power_w: float) -> None:
    """Header row of block names + one data row of watts."""
    with open(out_path, "w") as f:
        f.write("chip\n")
        f.write(f"{total_power_w:.6f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CircuitNet-N14 npz files to HotSpot .flp + .ptrace"
    )
    parser.add_argument(
        "floorplan_npz",
        help="Path to floorplan npz (routability_features/.../macro_region/{inst}.npz)",
    )
    parser.add_argument(
        "power_npz",
        help="Path to power npz (IR_drop_features/.../power_all/{inst}.npz)",
    )
    parser.add_argument(
        "out_dir",
        help="Output dir for design.flp and design.ptrace",
    )
    parser.add_argument(
        "--tile-size-m",
        type=float,
        default=TILE_SIZE_M,
        help="Physical tile size in meters (default: 2.25e-6 for N14)",
    )
    parser.add_argument(
        "--max-total-power-w",
        type=float,
        default=10.0,
        help=(
            "Cap total chip power (Watts) to this value. "
            "CircuitNet power_all values are relative/dimensionless; raw sums "
            "cause HotSpot to diverge. Set to 0 to disable capping. "
            "Default: 10.0 W (realistic for Vortex-small at 200 MHz)."
        ),
    )
    args = parser.parse_args()

    fp_path = Path(args.floorplan_npz)
    pw_path = Path(args.power_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_arr = load_npz_data(fp_path)
    pw_arr = load_npz_data(pw_path)

    # DATA-02 + A7: shape alignment assertion
    assert fp_arr.shape == pw_arr.shape, (
        f"Shape mismatch: floorplan={fp_arr.shape}, power={pw_arr.shape}"
    )
    if fp_arr.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got ndim={fp_arr.ndim}")

    rows, cols = fp_arr.shape
    height_m = rows * args.tile_size_m
    width_m = cols * args.tile_size_m

    total_power_w = float(np.asarray(pw_arr, dtype=np.float64).sum())
    if not np.isfinite(total_power_w) or total_power_w <= 0:
        total_power_w = 1e-6  # guard against zero/NaN; HotSpot would otherwise NaN

    # Cap to realistic chip power if requested (see docstring for units explanation).
    # Raw sums from CircuitNet power_all are ~1.6e6 "units" which causes HotSpot's
    # steady-state solver to diverge (T > 160,000 K). We cap to --max-total-power-w
    # to produce physically meaningful temperatures while preserving relative scaling.
    if args.max_total_power_w > 0 and total_power_w > args.max_total_power_w:
        print(
            f"WARNING: raw total_power={total_power_w:.3f} W exceeds "
            f"--max-total-power-w={args.max_total_power_w:.3f} W; "
            f"capping to prevent HotSpot solver divergence."
        )
        total_power_w = args.max_total_power_w

    write_flp(out_dir / "design.flp", width_m, height_m)
    write_ptrace(out_dir / "design.ptrace", total_power_w)

    print(f"Wrote {out_dir / 'design.flp'} (chip {width_m:.6e}m x {height_m:.6e}m)")
    print(f"Wrote {out_dir / 'design.ptrace'} (total_power={total_power_w:.6f} W)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
