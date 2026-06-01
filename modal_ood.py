"""OOD label generation for chip thickness sweep.

Generates HotSpot thermal labels at unseen t_chip values to evaluate
whether physics-constrained U-Net generalizes better than MSE-only.

Labels saved to /data/ood_labels/{tag}/{family}/{stem}/thermal.npy.
Existing training labels at /data/labels/ are never touched.

Smoke test (4 instances × 4 thicknesses in parallel):
    modal run modal_ood.py::smoke_ood

Full sweep (189 instances × 4 thicknesses in parallel):
    modal run modal_ood.py
"""
import modal
from pathlib import Path

volume = modal.Volume.from_name("circuitnet-n14", create_if_missing=False)
MOUNT = "/data"

GPU_FAMILIES = ["Vortex-small", "Vortex-large", "nvdla-large"]
MAX_POWER_W = 10.0
GRID_ROWS = 256
GRID_COLS = 256

# Training thickness: 150 µm. OOD conditions span below and above.
OOD_T_CHIP = [0.000075, 0.0001, 0.0002, 0.0003]  # 75, 100, 200, 300 µm

# Training r_convec: 0.1 K/W. OOD conditions span better and worse cooling.
OOD_R_CONVEC = [0.01, 0.05, 0.2, 0.5]  # K/W

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "make", "gcc", "libsuperlu-dev")
    .pip_install("numpy>=1.26", "scipy>=1.11", "tqdm>=4.66")
    .run_commands(
        "git clone https://github.com/uvahotspot/HotSpot.git /hotspot",
        "cd /hotspot && make SUPERLU=1 || make",
    )
    .add_local_file("scripts/npz_to_hotspot_multiblock.py", "/scripts/npz_to_hotspot_multiblock.py")
    .add_local_file("scripts/parse_hotspot_output.py", "/scripts/parse_hotspot_output.py")
    .add_local_file("hotspot.config", "/hotspot.config")
)

app = modal.App("circuitnet-ood", image=image)


def _iter_instances(raw_root: Path):
    """Yield (family, stem, fp_path, pw_path) for all matched GPU instances."""
    for family in GPU_FAMILIES:
        fp_dir = raw_root / "routability_features" / family / "cell_density"
        pw_dir = raw_root / "IR_drop_features" / family / "power_all"
        if not fp_dir.exists() or not pw_dir.exists():
            continue
        fp_stems = {p.stem for p in fp_dir.glob("*.npz")}
        pw_stems = {p.stem for p in pw_dir.glob("*.npz")}
        for stem in sorted(fp_stems & pw_stems):
            yield family, stem, fp_dir / f"{stem}.npz", pw_dir / f"{stem}.npz"


def _t_chip_tag(t_chip: float) -> str:
    return f"t{round(t_chip * 1e6)}um"


def _r_convec_tag(r_convec: float) -> str:
    # e.g. 0.01 → "rconv0p01", 0.5 → "rconv0p5"
    return "rconv" + f"{r_convec:.2f}".replace(".", "p")


def _run_one_ood(args):
    family, stem, fp_path, pw_path, out_dir, extra_hotspot_args = args
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    out = Path(out_dir)
    label_path = out / "thermal.npy"
    if label_path.exists():
        return {"stem": stem, "status": "skipped"}

    out.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            tmp = Path(tmpdir)

            r1 = subprocess.run(
                [sys.executable, "/scripts/npz_to_hotspot_multiblock.py",
                 str(fp_path), str(pw_path), str(tmp),
                 "--max-total-power-w", str(MAX_POWER_W)],
                capture_output=True, text=True, timeout=60,
            )
            if r1.returncode != 0:
                return {"stem": stem, "status": "failed",
                        "error": f"npz2flp: {r1.stderr[-300:]}"}

            r2 = subprocess.run(
                ["/hotspot/hotspot",
                 "-c", "/hotspot.config",
                 "-f", str(tmp / "design.flp"),
                 "-p", str(tmp / "design.ptrace"),
                 "-steady_file", str(tmp / "design.steady"),
                 "-model_type", "grid",
                 "-grid_rows", str(GRID_ROWS),
                 "-grid_cols", str(GRID_COLS),
                 "-grid_steady_file", str(tmp / "design.grid.steady")]
                + extra_hotspot_args,
                capture_output=True, text=True, timeout=2000,
            )
            if r2.returncode != 0:
                return {"stem": stem, "status": "failed",
                        "error": f"hotspot: {r2.stderr[-300:]}"}

            r3 = subprocess.run(
                [sys.executable, "/scripts/parse_hotspot_output.py",
                 str(tmp / "design.grid.steady"), str(label_path),
                 "--rows", str(GRID_ROWS), "--cols", str(GRID_COLS)],
                capture_output=True, text=True, timeout=60,
            )
            if r3.returncode != 0:
                return {"stem": stem, "status": "failed",
                        "error": f"parse: {r3.stderr[-300:]}"}

    except subprocess.TimeoutExpired as e:
        return {"stem": stem, "status": "failed", "error": f"timeout: {e}"}
    except Exception as e:
        return {"stem": stem, "status": "failed", "error": f"exception: {e}"}

    return {"stem": stem, "status": "ok"}


@app.function(volumes={MOUNT: volume}, timeout=36000, cpu=8)
def generate_ood_labels(t_chip: float, start: int = 0, limit: int = 0):
    """Generate OOD labels for all instances at a single t_chip value.

    Args:
        t_chip: chip thickness in meters (e.g. 0.0001 for 100 µm)
        start: start index into the sorted instance list
        limit: if > 0, process only this many instances from start
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    volume.reload()
    raw_root = Path(MOUNT) / "CircuitNet-N14"
    tag = _t_chip_tag(t_chip)
    extra_args = ["-t_chip", str(t_chip)]

    instances = list(_iter_instances(raw_root))
    args_list = [
        (fam, stem, str(fp), str(pw),
         str(Path(MOUNT) / "ood_labels" / tag / fam / stem),
         extra_args)
        for fam, stem, fp, pw in instances
    ]

    end = (start + limit) if limit > 0 else len(args_list)
    args_list = args_list[start:end]
    print(f"[{tag}] indices {start}–{end - 1} ({len(args_list)} instances)")

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_run_one_ood, a): a for a in args_list}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
            except Exception as e:
                stem = futures[fut][1]
                r = {"stem": stem, "status": "failed", "error": str(e)}
            results.append(r)
            if r["status"] == "failed":
                print(f"  [{tag}] FAIL [{i}] {r['stem']}: {r.get('error', '')[:300]}")
            if i % 10 == 0:
                ok = sum(1 for x in results if x["status"] == "ok")
                skip = sum(1 for x in results if x["status"] == "skipped")
                fail = sum(1 for x in results if x["status"] == "failed")
                print(f"  [{tag}] {i}/{len(args_list)}: {ok} ok, {skip} skipped, {fail} failed")
                volume.commit()

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = [r for r in results if r["status"] == "failed"]
    print(f"[{tag}] Done: {ok} ok, {skipped} skipped, {len(failed)} failed")
    for f in failed[:5]:
        print(f"  [{tag}] FAIL {f['stem']}: {f.get('error', '')[:200]}")

    volume.commit()
    return {"tag": tag, "ok": ok, "skipped": skipped, "failed": len(failed)}


@app.local_entrypoint()
def smoke_ood():
    """16 Modal jobs in parallel: 4 thicknesses × 4 instances each (limit=1 per job).

        modal run modal_ood.py::smoke_ood
    """
    handles = [
        generate_ood_labels.spawn(t_chip=t, start=i, limit=1)
        for t in OOD_T_CHIP
        for i in range(4)
    ]
    results = [h.get() for h in handles]
    for r in results:
        print(r)


@app.local_entrypoint()
def main():
    """Full sweep: 189 instances × 4 thicknesses in parallel.

        modal run modal_ood.py
    """
    handles = [
        generate_ood_labels.spawn(t_chip=t)
        for t in OOD_T_CHIP
    ]
    results = [h.get() for h in handles]
    for r in results:
        print(r)


@app.function(volumes={MOUNT: volume}, timeout=36000, cpu=8)
def generate_ood_htc_labels(r_convec: float, start: int = 0, limit: int = 0):
    """Generate OOD labels for all instances at a single r_convec value.

    Args:
        r_convec: convection resistance in K/W (training default: 0.1)
        start: start index into the sorted instance list
        limit: if > 0, process only this many instances from start
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    volume.reload()
    raw_root = Path(MOUNT) / "CircuitNet-N14"
    tag = _r_convec_tag(r_convec)
    extra_args = ["-r_convec", str(r_convec)]

    instances = list(_iter_instances(raw_root))
    args_list = [
        (fam, stem, str(fp), str(pw),
         str(Path(MOUNT) / "ood_labels" / tag / fam / stem),
         extra_args)
        for fam, stem, fp, pw in instances
    ]

    end = (start + limit) if limit > 0 else len(args_list)
    args_list = args_list[start:end]
    print(f"[{tag}] indices {start}–{end - 1} ({len(args_list)} instances)")

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_run_one_ood, a): a for a in args_list}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
            except Exception as e:
                stem = futures[fut][1]
                r = {"stem": stem, "status": "failed", "error": str(e)}
            results.append(r)
            if r["status"] == "failed":
                print(f"  [{tag}] FAIL [{i}] {r['stem']}: {r.get('error', '')[:300]}")
            if i % 10 == 0:
                ok = sum(1 for x in results if x["status"] == "ok")
                skip = sum(1 for x in results if x["status"] == "skipped")
                fail = sum(1 for x in results if x["status"] == "failed")
                print(f"  [{tag}] {i}/{len(args_list)}: {ok} ok, {skip} skipped, {fail} failed")
                volume.commit()

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = [r for r in results if r["status"] == "failed"]
    print(f"[{tag}] Done: {ok} ok, {skipped} skipped, {len(failed)} failed")
    for f in failed[:5]:
        print(f"  [{tag}] FAIL {f['stem']}: {f.get('error', '')[:200]}")

    volume.commit()
    return {"tag": tag, "ok": ok, "skipped": skipped, "failed": len(failed)}


@app.local_entrypoint()
def smoke_htc():
    """16 Modal jobs in parallel: 4 r_convec values × 4 instances each.

        modal run modal_ood.py::smoke_htc
    """
    handles = [
        generate_ood_htc_labels.spawn(r_convec=r, start=i, limit=1)
        for r in OOD_R_CONVEC
        for i in range(4)
    ]
    results = [h.get() for h in handles]
    for r in results:
        print(r)


@app.function(volumes={MOUNT: volume}, timeout=50000)
def run_htc_sweep():
    """Coordinator that runs on Modal so --detach keeps all 4 jobs alive."""
    handles = [
        generate_ood_htc_labels.spawn(r_convec=r)
        for r in OOD_R_CONVEC
    ]
    results = [h.get() for h in handles]
    for r in results:
        print(r)
    return results


@app.local_entrypoint()
def main_htc():
    """Full sweep: 189 instances × 4 r_convec values in parallel.

        modal run modal_ood.py::main_htc
    """
    run_htc_sweep.spawn()
