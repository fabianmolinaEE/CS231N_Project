#!/usr/bin/env bash
# Download CircuitNet-N14 GPU/accelerator subset (D-01, D-03).
# Usage: HF_TOKEN=hf_xxx ./scripts/download_data.sh [DESIGN_PREFIX ...]
# Default DESIGN_PREFIXes: Vortex-small Vortex-large NVDLA-large
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN env var (Hugging Face read token) before running}"

DATA_DIR="${DATA_DIR:-data/raw}"
REPO_ID="${REPO_ID:-CircuitNet/CircuitNet}"
SUBDIR="${SUBDIR:-CircuitNet-N14}"

mkdir -p "$DATA_DIR"

# Default GPU/accelerator design prefixes (D-01)
if [ "$#" -eq 0 ]; then
    set -- "Vortex-small" "Vortex-large" "NVDLA-large"
fi

# Resolve CLI name: huggingface_hub>=1.0 installs as 'hf', older versions as 'huggingface-cli'
if command -v hf &>/dev/null; then HF_CLI="hf"
elif command -v huggingface-cli &>/dev/null; then HF_CLI="huggingface-cli"
else
    echo "ERROR: neither 'hf' nor 'huggingface-cli' found. Activate the venv: source .venv/bin/activate" >&2
    exit 1
fi

# Build an include-pattern argument list
INCLUDE_ARGS=()
for prefix in "$@"; do
    INCLUDE_ARGS+=(--include "${SUBDIR}/${prefix}*/**")
done

echo "Downloading from ${REPO_ID}/${SUBDIR} → ${DATA_DIR}"
echo "Design prefixes: $*"

"$HF_CLI" download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    --token "$HF_TOKEN" \
    "${INCLUDE_ARGS[@]}"

echo "Done. Downloaded designs:"
find "$DATA_DIR/$SUBDIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null \
    || ls "$DATA_DIR/$SUBDIR" 2>/dev/null \
    || echo "(no $SUBDIR subdirectory found — verify download)"
