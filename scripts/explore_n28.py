"""Explore CircuitNet-N28 repo structure on HuggingFace.

Lists all top-level directories and GPU/accelerator-looking design families
so we know what to download before committing.

Usage:
    HF_TOKEN=hf_xxx python scripts/explore_n28.py
"""
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("Set HF_TOKEN env var first")

api = HfApi()
repo_id = "CircuitNet/CircuitNet"

print(f"Listing files in {repo_id} under CircuitNet-N28/ ...")
all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)

n28_files = [f for f in all_files if f.startswith("CircuitNet-N28/")]
print(f"Total N28 files: {len(n28_files)}")

# Extract unique (feature_type, family) combos
entries = set()
for f in n28_files:
    parts = f.split("/")
    if len(parts) >= 3:
        entries.add((parts[1], parts[2].replace(".tar.gz", "")))

feature_types = sorted({e[0] for e in entries})
families = sorted({e[1] for e in entries})

print(f"\nFeature types ({len(feature_types)}):")
for ft in feature_types:
    print(f"  {ft}")

print(f"\nDesign families ({len(families)}):")
for fam in families:
    print(f"  {fam}")

# Flag GPU/accelerator candidates
GPU_KEYWORDS = ["gpu", "vortex", "nvdla", "npu", "tpu", "accelerator", "dla"]
candidates = [f for f in families if any(k in f.lower() for k in GPU_KEYWORDS)]
print(f"\nGPU/accelerator candidates: {candidates if candidates else 'none found — check full list above'}")
