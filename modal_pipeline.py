"""Modal pipeline: HotSpot multi-block label generation + split/norm stats.

Run full pipeline (label generation then split/norm):
    modal run modal_pipeline.py

Run label generation only:
    modal run modal_pipeline.py::generate_labels

Run split/norm only (after labels exist):
    modal run modal_pipeline.py::make_split_and_stats

Background:
- D-04: Full label generation runs on Modal (not local xargs -P)
- Plan 02 proved multi-block 16x16 is required: delta_T=54.84K vs 0.34K single-block
- Labels live in Modal Volume alongside raw data (see volume definition below)
- power_all values are NOT in Watts; use MAX_POWER_W=10.0 cap consistently (Plan 02)
"""
import modal
import os
from pathlib import Path

# Same volume as modal_download.py
volume = modal.Volume.from_name("circuitnet-n14", create_if_missing=False)
MOUNT = "/data"

GPU_FAMILIES = ["Vortex-small", "Vortex-large", "nvdla-large"]
RANDOM_SEED = 42  # D-07
MAX_POWER_W = 10.0  # consistent cap across all designs (power_all not in Watts)
GRID_ROWS = 256
GRID_COLS = 256

# Build container image: Python deps + HotSpot built from source
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "make", "gcc", "libsuperlu-dev")
    .pip_install("numpy>=1.26", "scipy>=1.11", "tqdm>=4.66")
    .run_commands(
        "git clone https://github.com/uvahotspot/HotSpot.git /hotspot",
        "cd /hotspot && make SUPERLU=1 || make",
    )
    # Copy local scripts into the image
    .copy_local_file("scripts/npz_to_hotspot_multiblock.py", "/scripts/npz_to_hotspot_multiblock.py")
    .copy_local_file("scripts/parse_hotspot_output.py", "/scripts/parse_hotspot_output.py")
    .copy_local_file("hotspot.config", "/hotspot.config")
)

app = modal.App("circuitnet-pipeline", image=image)


def _iter_instances(raw_root: Path):
    """Yield (family, stem, fp_path, pw_path) for all matched GPU instances.

    Takes the intersection of floorplan and power npz stems per family.
    Verified counts (docs/dataset-structure.md):
      Vortex-small: 96, Vortex-large: 61, nvdla-large: 32 -> total 189
    """
    for family in GPU_FAMILIES:
        fp_dir = raw_root / "routability_features" / family / family / "macro_region"
        pw_dir = raw_root / "IR_drop_features" / family / family / "power_all"
        if not fp_dir.exists() or not pw_dir.exists():
            continue
        fp_stems = {p.stem for p in fp_dir.glob("*.npz")}
        pw_stems = {p.stem for p in pw_dir.glob("*.npz")}
        matched = sorted(fp_stems & pw_stems)
        for stem in matched:
            yield family, stem, fp_dir / f"{stem}.npz", pw_dir / f"{stem}.npz"


def _run_one(args):
    """Run HotSpot multi-block pipeline for one instance.

    Designed for Python map() within a single Modal container (cpu=8).
    Skips instances where thermal.npy already exists (idempotent).
    """
    family, stem, fp_path, pw_path, out_dir = args
    import subprocess
    import sys
    from pathlib import Path

    out = Path(out_dir)
    label_path = out / "thermal.npy"
    if label_path.exists():
        return {"stem": stem, "status": "skipped"}

    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert npz -> .flp + .ptrace (multi-block 16x16)
    result = subprocess.run(
        [sys.executable, "/scripts/npz_to_hotspot_multiblock.py",
         str(fp_path), str(pw_path), str(out),
         "--max-total-power-w", str(MAX_POWER_W)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return {"stem": stem, "status": "failed",
                "error": result.stderr[-500:]}

    # Step 2: Run HotSpot in grid mode (256x256)
    result = subprocess.run(
        ["/hotspot/hotspot",
         "-c", "/hotspot.config",
         "-f", str(out / "design.flp"),
         "-p", str(out / "design.ptrace"),
         "-steady_file", str(out / "design.steady"),
         "-model_type", "grid",
         "-grid_rows", str(GRID_ROWS),
         "-grid_cols", str(GRID_COLS),
         "-grid_steady_file", str(out / "design.grid.steady")],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        return {"stem": stem, "status": "failed",
                "error": result.stderr[-500:]}

    # Step 3: Parse .grid.steady -> thermal.npy (float32, 256x256 Kelvin)
    result = subprocess.run(
        [sys.executable, "/scripts/parse_hotspot_output.py",
         str(out / "design.grid.steady"), str(label_path),
         "--rows", str(GRID_ROWS), "--cols", str(GRID_COLS)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return {"stem": stem, "status": "failed",
                "error": result.stderr[-500:]}

    return {"stem": stem, "status": "ok"}


@app.function(volumes={MOUNT: volume}, timeout=3600, cpu=8)
def generate_labels():
    """Generate HotSpot thermal labels for all 189 GPU instances.

    Runs multi-block 16x16 HotSpot pipeline for each instance in parallel
    using Python map() with cpu=8. Results saved to:
      /data/labels/{family}/{stem}/thermal.npy

    Idempotent: existing thermal.npy files are skipped.
    """
    raw_root = Path(MOUNT) / "CircuitNet-N14"
    instances = list(_iter_instances(raw_root))
    print(f"Processing {len(instances)} instances...")

    args_list = [
        (fam, stem, str(fp), str(pw),
         str(Path(MOUNT) / "labels" / fam / stem))
        for fam, stem, fp, pw in instances
    ]

    results = list(map(_run_one, args_list))
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = [r for r in results if r["status"] == "failed"]
    print(f"Done: {ok} ok, {skipped} skipped, {len(failed)} failed")
    for f in failed[:10]:
        print(f"  FAIL {f['stem']}: {f.get('error', '')[:200]}")

    volume.commit()
    return {"ok": ok, "skipped": skipped, "failed": len(failed)}


@app.function(volumes={MOUNT: volume}, timeout=600)
def make_split_and_stats():
    """Build 80/10/10 split JSONs and per-channel normalization stats.

    Scans /data/labels/ for complete instances (input + label present),
    shuffles with RANDOM_SEED=42, writes split JSONs and norm stats to
    /data/splits/.

    Normalization computed from train split ONLY (prevents val/test leakage).
    Per-channel global mean/std for floorplan and power_all features.
    """
    import json
    import random
    import numpy as np
    from pathlib import Path

    raw_root = Path(MOUNT) / "CircuitNet-N14"
    labels_root = Path(MOUNT) / "labels"

    entries = []
    for family, stem, fp_path, pw_path in _iter_instances(raw_root):
        label = labels_root / family / stem / "thermal.npy"
        if label.exists():
            entries.append({
                "instance": stem,
                "family": family,
                "floorplan": str(fp_path),
                "power": str(pw_path),
                "label": str(label),
            })
    print(f"Complete instances (input+label): {len(entries)}")

    rng = random.Random(RANDOM_SEED)
    shuffled = entries[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, int(n * 0.10))
    n_tst = max(1, int(n * 0.10))
    train = shuffled[:n - n_val - n_tst]
    val = shuffled[n - n_val - n_tst:n - n_tst]
    test = shuffled[n - n_tst:]
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    splits_dir = Path(MOUNT) / "splits"
    splits_dir.mkdir(exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        (splits_dir / f"{name}.json").write_text(json.dumps(data, indent=2))

    # Normalization stats from train only (D-09, prevents val/test leakage)
    fp_vals, pw_vals = [], []
    for e in train:
        fp_vals.append(np.load(e["floorplan"])["data"].ravel().astype(np.float64))
        pw_vals.append(np.load(e["power"])["data"].ravel().astype(np.float64))
    fp_all = np.concatenate(fp_vals)
    pw_all = np.concatenate(pw_vals)
    stats = {
        "floorplan": {"mean": float(fp_all.mean()), "std": float(fp_all.std())},
        "power": {"mean": float(pw_all.mean()), "std": float(pw_all.std())},
    }
    (splits_dir / "normalization_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Norm stats: {stats}")

    volume.commit()
    return {"n_train": len(train), "n_val": len(val), "n_test": len(test), "stats": stats}


@app.local_entrypoint()
def main():
    """Run full pipeline: label generation then split/norm stats."""
    print("=== Step 1: Generate HotSpot labels ===")
    result = generate_labels.remote()
    print(result)
    print("\n=== Step 2: Build split + norm stats ===")
    result = make_split_and_stats.remote()
    print(result)
