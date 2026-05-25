"""
Download CircuitNet-N14 GPU/accelerator subset into a Modal Volume.

Usage:
    modal run modal_download.py

Setup (one-time):
    modal secret create huggingface HF_TOKEN=hf_your_token_here
"""
import modal

# Persists across all Modal runs — download once, reuse everywhere
volume = modal.Volume.from_name("circuitnet-n14", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=1.0")
)

app = modal.App("circuitnet-download", image=image)

GPU_PATTERNS = [
    "CircuitNet-N14/Vortex-small*/**",
    "CircuitNet-N14/Vortex-large*/**",
    "CircuitNet-N14/NVDLA-large*/**",
]


@app.function(
    volumes={"/data": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def download():
    import os
    from huggingface_hub import snapshot_download

    print("Starting CircuitNet-N14 GPU subset download...")
    snapshot_download(
        repo_id="CircuitNet/CircuitNet",
        repo_type="dataset",
        local_dir="/data",
        allow_patterns=GPU_PATTERNS,
        token=os.environ["HF_TOKEN"],
    )
    volume.commit()

    # Report what landed
    base = "/data/CircuitNet-N14"
    if os.path.exists(base):
        designs = sorted(os.listdir(base))
        print(f"\nDownloaded {len(designs)} designs:")
        for d in designs:
            print(f"  {d}")
    else:
        print("WARNING: CircuitNet-N14 directory not found — check patterns")


@app.local_entrypoint()
def main():
    download.remote()
