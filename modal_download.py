"""
Download CircuitNet-N14 GPU/accelerator subset into a Modal Volume.

The HF repo stores data as tar.gz archives:
  CircuitNet-N14/{feature_type}/{family}.tar.gz

We download 6 archives (2 feature types × 3 GPU families) and extract them
so modal_pipeline.py finds data at:
  /data/CircuitNet-N14/{feature_type}/{family}/{family}/{feature}/{inst}.npz

Usage:
    modal run modal_download.py

Setup (one-time):
    modal secret create huggingface HF_TOKEN=hf_your_token_here
"""
import modal

volume = modal.Volume.from_name("circuitnet-n14", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=1.0")
)

app = modal.App("circuitnet-download", image=image)

GPU_FAMILIES = ["Vortex-small", "Vortex-large", "nvdla-large"]
FEATURE_TYPES = ["IR_drop_features", "routability_features"]

# Explicit tar.gz paths within the HF repo
GPU_FILES = [
    f"CircuitNet-N14/{feat}/{family}.tar.gz"
    for feat in FEATURE_TYPES
    for family in GPU_FAMILIES
]


@app.function(
    volumes={"/data": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def download():
    import os
    import tarfile
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    token = os.environ["HF_TOKEN"]
    dest = Path("/data/CircuitNet-N14")
    dest.mkdir(parents=True, exist_ok=True)

    for repo_path in GPU_FILES:
        feat_type = repo_path.split("/")[1]   # e.g. IR_drop_features
        archive_name = repo_path.split("/")[2]  # e.g. Vortex-small.tar.gz
        family = archive_name.replace(".tar.gz", "")

        extract_dir = dest / feat_type / family / family
        # Skip if already extracted (re-run safe)
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"  SKIP {repo_path} (already extracted)")
            continue

        print(f"Downloading {repo_path}...")
        local_archive = hf_hub_download(
            repo_id="CircuitNet/CircuitNet",
            repo_type="dataset",
            filename=repo_path,
            token=token,
            local_dir="/tmp/circuitnet_archives",
        )

        print(f"  Extracting → {dest / feat_type}/")
        with tarfile.open(local_archive) as tf:
            tf.extractall(dest / feat_type)
        print(f"  Done: {family} / {feat_type}")

    volume.commit()

    # Report what landed
    total_npz = sum(1 for _ in dest.rglob("*.npz"))
    print(f"\nTotal .npz files in volume: {total_npz}")
    for feat in FEATURE_TYPES:
        for family in GPU_FAMILIES:
            d = dest / feat / family / family
            if d.exists():
                subdirs = [s.name for s in d.iterdir() if s.is_dir()]
                for sub in subdirs:
                    n = len(list((d / sub).glob("*.npz")))
                    print(f"  {feat}/{family}/{sub}: {n} instances")
            else:
                print(f"  MISSING: {feat}/{family}")


@app.local_entrypoint()
def main():
    download.remote()
