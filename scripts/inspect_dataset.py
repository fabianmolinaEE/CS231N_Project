"""Inspect downloaded CircuitNet-N14 designs.

Verifies assumptions A1 (floorplan key = macro_region), A2 (power key = power_all),
A3 (GPU filter by prefix), A4 (design count), A7 (shape consistency).
Writes a markdown summary to docs/dataset-structure.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GPU_PREFIXES = ("Vortex-small", "Vortex-large", "NVDLA-large")


def is_gpu_design(name: str) -> bool:
    return any(name.startswith(p) for p in GPU_PREFIXES)


def inspect_design(design_dir: Path) -> dict:
    """Return inspection record for one design directory."""
    record: dict = {"design": design_dir.name, "errors": []}
    # Candidate feature paths (RESEARCH.md ASSUMED keys; verify here)
    candidates = {
        "floorplan": [
            ("routability_features/macro_region.npz", "macro_region"),
            ("routability_features/cell_density.npz", "cell_density"),
        ],
        "power": [
            ("IR_drop_features/power_all.npz", "power_all"),
            ("IR_drop_features/power_i.npz", "power_i"),
            ("IR_drop_features/power_s.npz", "power_s"),
        ],
    }
    for role, options in candidates.items():
        for rel, key in options:
            p = design_dir / rel
            if not p.exists():
                continue
            try:
                npz = np.load(p, allow_pickle=False)
                keys = list(npz.files)
                if key in keys:
                    arr = npz[key]
                    record.setdefault(role, []).append({
                        "path": str(p.relative_to(design_dir.parent)),
                        "key": key,
                        "all_keys": keys,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    })
            except Exception as e:
                record["errors"].append(f"{p}: {type(e).__name__}: {e}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/CircuitNet-N14")
    parser.add_argument("--out-doc", default="docs/dataset-structure.md")
    parser.add_argument("--sample", type=int, default=5,
                        help="Inspect up to this many designs per GPU prefix")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    if not raw.exists():
        print(f"ERROR: {raw} does not exist. Run scripts/download_data.sh first.")
        return 1

    all_dirs = sorted([d for d in raw.iterdir() if d.is_dir()])
    gpu_dirs = [d for d in all_dirs if is_gpu_design(d.name)]
    print(f"Total directories under {raw}: {len(all_dirs)}")
    print(f"GPU/accelerator designs (filtered): {len(gpu_dirs)}")

    # Inspect up to args.sample per prefix
    sampled: list[Path] = []
    for prefix in GPU_PREFIXES:
        matches = [d for d in gpu_dirs if d.name.startswith(prefix)][: args.sample]
        sampled.extend(matches)
        print(f"  {prefix}: {sum(1 for d in gpu_dirs if d.name.startswith(prefix))} total, inspecting {len(matches)}")

    records = [inspect_design(d) for d in sampled]

    # Write summary doc
    out = Path(args.out_doc)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# CircuitNet-N14 Dataset Structure (verified)\n")
    lines.append(f"Source: `{raw}`\n")
    lines.append("## Design Counts\n")
    lines.append(f"- Total directories: {len(all_dirs)}")
    lines.append(f"- GPU/accelerator subset: {len(gpu_dirs)}")
    for prefix in GPU_PREFIXES:
        n = sum(1 for d in gpu_dirs if d.name.startswith(prefix))
        lines.append(f"  - {prefix}*: {n}")
    lines.append("\n## Verified NPZ Keys\n")
    lines.append("| Design | Role | Path | Key | Shape | Dtype |")
    lines.append("|--------|------|------|-----|-------|-------|")
    for r in records:
        for role in ("floorplan", "power"):
            for hit in r.get(role, []):
                lines.append(
                    f"| {r['design']} | {role} | {hit['path']} | "
                    f"{hit['key']} | {hit['shape']} | {hit['dtype']} |"
                )
    lines.append("\n## Errors\n")
    any_err = False
    for r in records:
        for e in r.get("errors", []):
            lines.append(f"- {r['design']}: {e}")
            any_err = True
    if not any_err:
        lines.append("- None")
    lines.append("\n## Filter Definition (D-01, D-03)\n")
    lines.append("GPU/accelerator subset includes designs whose name starts with one of:")
    for p in GPU_PREFIXES:
        lines.append(f"- `{p}`")
    lines.append("\n## Raw Inspection Records (JSON)\n")
    lines.append("```json")
    lines.append(json.dumps(records, indent=2))
    lines.append("```")
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
