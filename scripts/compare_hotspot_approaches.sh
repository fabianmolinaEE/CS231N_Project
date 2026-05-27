#!/usr/bin/env bash
# Compare single-block vs multi-block HotSpot power injection on one design.
# Produces thermal.npy for both, prints stats, and shows spatial correlation
# between the power_all input and thermal output for each approach.
#
# Usage:
#   bash scripts/compare_hotspot_approaches.sh [POWER_NPZ] [FLOORPLAN_NPZ]
#
# Defaults to the first Vortex-small instance.
set -euo pipefail

HOTSPOT_BIN="${HOTSPOT_BIN:-third_party/HotSpot/hotspot}"
CONFIG="${CONFIG:-hotspot.config}"
GRID_ROWS="${GRID_ROWS:-256}"
GRID_COLS="${GRID_COLS:-256}"
BLOCK_ROWS="${BLOCK_ROWS:-16}"
BLOCK_COLS="${BLOCK_COLS:-16}"

POWER_NPZ="${1:-data/raw/CircuitNet-N14/IR_drop_features/Vortex-small/Vortex-small/power_all/Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz}"
FP_NPZ="${2:-data/raw/CircuitNet-N14/routability_features/Vortex-small/Vortex-small/macro_region/Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz}"

OUTDIR_SINGLE="data/labels/compare_single"
OUTDIR_MULTI="data/labels/compare_multi"

echo "=== Comparing HotSpot approaches ==="
echo "Design: $(basename ${POWER_NPZ} .npz)"
echo ""

# --- Single-block ---
echo "--- Single-block (npz_to_hotspot.py) ---"
.venv/bin/python scripts/npz_to_hotspot.py "$FP_NPZ" "$POWER_NPZ" "$OUTDIR_SINGLE"

"$HOTSPOT_BIN" -c "$CONFIG" \
  -f "${OUTDIR_SINGLE}/design.flp" \
  -p "${OUTDIR_SINGLE}/design.ptrace" \
  -steady_file "${OUTDIR_SINGLE}/design.steady" \
  -model_type grid \
  -grid_rows "$GRID_ROWS" -grid_cols "$GRID_COLS" \
  -grid_steady_file "${OUTDIR_SINGLE}/design.grid.steady"

.venv/bin/python scripts/parse_hotspot_output.py \
  "${OUTDIR_SINGLE}/design.grid.steady" \
  "${OUTDIR_SINGLE}/thermal.npy" \
  --rows "$GRID_ROWS" --cols "$GRID_COLS"

echo ""

# --- Multi-block ---
echo "--- Multi-block ${BLOCK_ROWS}x${BLOCK_COLS} (npz_to_hotspot_multiblock.py) ---"
.venv/bin/python scripts/npz_to_hotspot_multiblock.py \
  "$FP_NPZ" "$POWER_NPZ" "$OUTDIR_MULTI" \
  --block-rows "$BLOCK_ROWS" --block-cols "$BLOCK_COLS"

"$HOTSPOT_BIN" -c "$CONFIG" \
  -f "${OUTDIR_MULTI}/design.flp" \
  -p "${OUTDIR_MULTI}/design.ptrace" \
  -steady_file "${OUTDIR_MULTI}/design.steady" \
  -model_type grid \
  -grid_rows "$GRID_ROWS" -grid_cols "$GRID_COLS" \
  -grid_steady_file "${OUTDIR_MULTI}/design.grid.steady"

.venv/bin/python scripts/parse_hotspot_output.py \
  "${OUTDIR_MULTI}/design.grid.steady" \
  "${OUTDIR_MULTI}/thermal.npy" \
  --rows "$GRID_ROWS" --cols "$GRID_COLS"

echo ""

# --- Comparison stats ---
echo "=== Stats comparison ==="
.venv/bin/python - <<'PY'
import numpy as np
from scipy.stats import pearsonr

single = np.load("data/labels/compare_single/thermal.npy")
multi  = np.load("data/labels/compare_multi/thermal.npy")
pw     = np.load("data/raw/CircuitNet-N14/IR_drop_features/Vortex-small/Vortex-small/power_all/Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz")["data"]

from scipy.ndimage import zoom
pw_256 = zoom(pw.astype(np.float32), (256/pw.shape[0], 256/pw.shape[1]), order=1)

print(f"Single-block thermal: min={single.min():.3f}K  max={single.max():.3f}K  "
      f"mean={single.mean():.3f}K  std={single.std():.4f}K  "
      f"delta_T={single.max()-single.min():.4f}K")
print(f"Multi-block thermal:  min={multi.min():.3f}K  max={multi.max():.3f}K  "
      f"mean={multi.mean():.3f}K  std={multi.std():.4f}K  "
      f"delta_T={multi.max()-multi.min():.4f}K")
print()

r_single, _ = pearsonr(pw_256.ravel(), single.ravel())
r_multi,  _ = pearsonr(pw_256.ravel(), multi.ravel())
print(f"Pearson r(power_all, thermal):")
print(f"  single-block: {r_single:.4f}")
print(f"  multi-block:  {r_multi:.4f}")
print()
print("Interpretation:")
print(f"  Higher |r| means thermal map better tracks power distribution.")
print(f"  Single-block delta_T={single.max()-single.min():.4f}K (almost flat)")
print(f"  Multi-block  delta_T={multi.max()-multi.min():.4f}K")
if abs(r_multi) > abs(r_single) + 0.05:
    print("  => Multi-block produces more spatially informative labels.")
elif multi.max() - multi.min() < 1.0:
    print("  => Both approaches produce nearly flat thermal maps.")
    print("     Consider: power_all may not have meaningful spatial variation,")
    print("     or 10W total is too low to see lateral gradients.")
else:
    print("  => Similar spatial correlation; check visual output.")
PY
