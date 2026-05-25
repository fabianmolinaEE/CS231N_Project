#!/usr/bin/env bash
# Prototype: run HotSpot grid mode on N sample GPU designs (default 5).
# Verifies DATA-04 before parallelizing in Plan 03.
#
# Actual directory structure (verified Plan 01):
#   data/raw/CircuitNet-N14/
#     routability_features/{family}/{family}/macro_region/{instance}.npz
#     IR_drop_features/{family}/{family}/power_all/{instance}.npz
#   All npz files use key 'data'.
#
# Usage:
#   FAMILY=Vortex-small SAMPLE_N=5 bash scripts/run_hotspot_sample.sh
set -euo pipefail

HOTSPOT_BIN="${HOTSPOT_BIN:-third_party/HotSpot/hotspot}"
CONFIG="${CONFIG:-hotspot.config}"
RAW_DIR="${RAW_DIR:-data/raw/CircuitNet-N14}"
LABELS_DIR="${LABELS_DIR:-data/labels}"
FAMILY="${FAMILY:-Vortex-small}"       # Vortex-small is smallest/fastest
SAMPLE_N="${SAMPLE_N:-5}"
GRID_ROWS="${GRID_ROWS:-256}"          # power of 2 required (Pitfall 3)
GRID_COLS="${GRID_COLS:-256}"

POWER_DIR="${RAW_DIR}/IR_drop_features/${FAMILY}/${FAMILY}/power_all"
FLOORPLAN_DIR="${RAW_DIR}/routability_features/${FAMILY}/${FAMILY}/macro_region"

mkdir -p "${LABELS_DIR}"

# Collect instance stems from power_all (authoritative list per Plan 01 intersection)
# Note: mapfile (readarray) requires bash >= 4; macOS ships bash 3.2, so use a while loop.
POWER_FILES=()
while IFS= read -r f; do
    POWER_FILES+=("$f")
done < <(ls "${POWER_DIR}"/*.npz 2>/dev/null | head -n "${SAMPLE_N}")

if [ "${#POWER_FILES[@]}" -eq 0 ]; then
    echo "ERROR: No npz files found under ${POWER_DIR}" >&2
    exit 1
fi

echo "=== HotSpot prototype: family=${FAMILY}, n=${#POWER_FILES[@]}, grid=${GRID_ROWS}x${GRID_COLS} ==="

mkdir -p docs
TIMINGS_FILE="docs/hotspot-prototype-timings.tsv"
: > "${TIMINGS_FILE}"
printf "design\twall_seconds\tthermal_min_k\tthermal_max_k\tthermal_mean_k\n" >> "${TIMINGS_FILE}"

for power_npz in "${POWER_FILES[@]}"; do
    # Derive instance name (stem without .npz)
    instance=$(basename "${power_npz}" .npz)
    floorplan_npz="${FLOORPLAN_DIR}/${instance}.npz"

    if [ ! -f "${floorplan_npz}" ]; then
        echo "WARNING: floorplan npz not found for ${instance}, skipping." >&2
        continue
    fi

    outdir="${LABELS_DIR}/${instance}"
    mkdir -p "${outdir}"

    echo "--- ${instance} ---"
    start=$(date +%s)

    # Convert npz -> .flp + .ptrace
    .venv/bin/python scripts/npz_to_hotspot.py \
        "${floorplan_npz}" \
        "${power_npz}" \
        "${outdir}"

    # Run HotSpot grid mode
    "${HOTSPOT_BIN}" \
        -c "${CONFIG}" \
        -f "${outdir}/design.flp" \
        -p "${outdir}/design.ptrace" \
        -steady_file "${outdir}/design.steady" \
        -model_type grid \
        -grid_rows "${GRID_ROWS}" -grid_cols "${GRID_COLS}" \
        -grid_steady_file "${outdir}/design.grid.steady"

    # Parse .grid.steady -> thermal.npy
    .venv/bin/python scripts/parse_hotspot_output.py \
        "${outdir}/design.grid.steady" \
        "${outdir}/thermal.npy" \
        --rows "${GRID_ROWS}" --cols "${GRID_COLS}"

    end=$(date +%s)
    elapsed=$((end - start))

    # Extract stats from the saved npy
    stats=$(.venv/bin/python - <<PY
import numpy as np
a = np.load("${outdir}/thermal.npy")
print(f"{a.min():.3f}\t{a.max():.3f}\t{a.mean():.3f}")
PY
    )
    printf "%s\t%d\t%s\n" "${instance}" "${elapsed}" "${stats}" >> "${TIMINGS_FILE}"
    echo "Done: ${instance} in ${elapsed}s"
done

echo ""
echo "=== All done. Timings written to ${TIMINGS_FILE} ==="
cat "${TIMINGS_FILE}"
