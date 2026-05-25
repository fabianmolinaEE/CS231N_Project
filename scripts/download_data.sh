#!/usr/bin/env bash
# Download CircuitNet-N14 GPU/accelerator subset and extract to data/raw/.
#
# Actual HF repo structure (discovered 2026-05-24):
#   CircuitNet-N14/{feature_type}/{design_name}.tar.gz
# GPU/accelerator designs: Vortex-small, Vortex-large, nvdla-large (note: lowercase nvdla)
# Feature types downloaded: IR_drop_features, routability_features
#
# Usage: HF_TOKEN=hf_xxx ./scripts/download_data.sh
#        source .venv/bin/activate first — hf CLI lives in the venv
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN env var (Hugging Face read token) before running}"

DATA_DIR="${DATA_DIR:-data/raw}"
REPO_ID="${REPO_ID:-CircuitNet/CircuitNet}"
SUBDIR="CircuitNet-N14"

mkdir -p "$DATA_DIR"

# Resolve CLI name: huggingface_hub>=1.0 installs as 'hf'
if command -v hf &>/dev/null; then HF_CLI="hf"
elif command -v huggingface-cli &>/dev/null; then HF_CLI="huggingface-cli"
else
    echo "ERROR: neither 'hf' nor 'huggingface-cli' found. Run: source .venv/bin/activate" >&2
    exit 1
fi

# GPU/accelerator designs (D-01, D-03). nvdla-large uses lowercase in the repo.
GPU_DESIGNS=("Vortex-small" "Vortex-large" "nvdla-large")
FEATURE_TYPES=("IR_drop_features" "routability_features")

# Build list of tar.gz files to download
FILES=()
for feature in "${FEATURE_TYPES[@]}"; do
    for design in "${GPU_DESIGNS[@]}"; do
        FILES+=("${SUBDIR}/${feature}/${design}.tar.gz")
    done
done

echo "Downloading ${#FILES[@]} tar.gz files from ${REPO_ID} → ${DATA_DIR}"
for f in "${FILES[@]}"; do echo "  $f"; done

"$HF_CLI" download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    --token "$HF_TOKEN" \
    "${FILES[@]}"

echo ""
echo "Extracting tar.gz archives..."
for feature in "${FEATURE_TYPES[@]}"; do
    for design in "${GPU_DESIGNS[@]}"; do
        archive="${DATA_DIR}/${SUBDIR}/${feature}/${design}.tar.gz"
        dest="${DATA_DIR}/${SUBDIR}/${feature}/${design}"
        if [ -f "$archive" ]; then
            echo "  Extracting ${archive} → ${dest}"
            mkdir -p "$dest"
            tar -xzf "$archive" -C "$dest" --strip-components=1 2>/dev/null \
                || tar -xzf "$archive" -C "$dest"
        else
            echo "  WARNING: $archive not found after download"
        fi
    done
done

echo ""
echo "Done. GPU design directories:"
for feature in "${FEATURE_TYPES[@]}"; do
    echo "  ${SUBDIR}/${feature}/"
    for design in "${GPU_DESIGNS[@]}"; do
        d="${DATA_DIR}/${SUBDIR}/${feature}/${design}"
        if [ -d "$d" ]; then
            count=$(find "$d" -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
            echo "    ${design}: ${count} npz files"
        else
            echo "    ${design}: NOT FOUND"
        fi
    done
done
