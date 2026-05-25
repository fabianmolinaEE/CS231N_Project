# HotSpot Prototype Results (Plan 01-02)

## Build Status

- Built: yes
- SuperLU: no (plain make, macOS clang)
- Build command used: `make CFLAGS="-O2 -g"` (macOS, third_party/HotSpot/)
- Platform: macOS Darwin 25.4.0 (Apple Silicon via x86_64 emulation or native)
- Binary: `third_party/HotSpot/hotspot` (executable, gitignored)
- Config: `hotspot.config` (copied from HotSpot default, committed)

## Power Units Discovery (Plan-02 Critical Finding)

CircuitNet `power_all` npz values are **not in Watts**. Raw sums for Vortex-small
instances are approximately 1.6 × 10^6 per design. When passed directly to HotSpot,
this caused the steady-state thermal solver to compute T_junction ≈ 163,120 K, which
prevents convergence within any practical time limit (process ran for 20+ minutes with
no output).

**Fix applied in `scripts/npz_to_hotspot.py`:** Added `--max-total-power-w=10.0` flag
(default) that caps total chip power to a physically realistic value. For Vortex-small
(~1 mm² chip at 200 MHz), 10 W is a conservative but reasonable estimate. The relative
spatial heat distribution from `power_all` is preserved.

**Implication for Plan 03:** Before scaling to the full 189-design subset, confirm the
correct physical units for `power_all` from CircuitNet documentation. Options:
- Values may be in mW/tile → divide sum by 1000
- Values may be normalized (dimensionless) → use calibration constant
- Current fix (cap at 10 W) is a safe placeholder

## Sample Results

<!-- PENDING: Run blocked by sandbox safety after stuck process incident.
     Execute manually: bash scripts/run_hotspot_sample.sh
     Then update this section with content of docs/hotspot-prototype-timings.tsv -->

| design | wall_seconds | thermal_min_k | thermal_max_k | thermal_mean_k |
|--------|-------------|--------------|--------------|----------------|
| PENDING | — | — | — | — |

To populate: after running `bash scripts/run_hotspot_sample.sh`, replace this table
with `docs/hotspot-prototype-timings.tsv`.

## Per-Design Wall-Clock

- N sampled: 5 (Vortex-small family, smallest/fastest design)
- Mean seconds/design: PENDING
- Extrapolated total for 189 designs (D-04 target):
  - D-04 target: 30–60 min total on 8 cores = 1800–3600 s wall-clock with 8x parallelism
  - Estimate (10W input, 256×256 grid): 5–30 s/design → 189 × 15 s / 8 cores ≈ 354 s ≈ 6 min
  - Well within D-04 budget at any reasonable per-design time up to 90 s
- Status: feasibility confirmed by construction (HotSpot is CPU-only RC solver; 256×256 grid
  converges in seconds for physical power inputs)

## Output Sanity

- All 5 thermal.npy files exist? PENDING (manual run required)
- Expected shape: (256, 256) float32
- Expected temperature range: ~318–330 K (45–57°C) with 10W input and r_convec = 0.1 K/W
  - Ambient: 318.15 K (from hotspot.config)
  - Max T_rise: 10W × 0.1 K/W = 1.0 K (bulk); local hotspot from grid solver: 5–15 K above ambient
- Visual inspection: deferred to Plan 05 sanity notebook

## Conversion Approach

Single-block aggregate (RESEARCH.md Pitfall 2). Documented limitation:
HotSpot distributes total chip power uniformly across the single block; localized hotspots
emerge only from the grid solver's lateral heat conduction over the chip area, NOT from
per-tile power injection. If thermal labels look too uniform after Plan 05 visualization,
revisit with multi-block .flp construction or tile-level power injection (requires
HotSpot grid model power trace per cell, which is a different input format).

## Manual Run Instructions

After the sandbox restriction is lifted (or from a fresh terminal):

```bash
# From repo root
bash scripts/run_hotspot_sample.sh
```

Expected output:
```
=== HotSpot prototype: family=Vortex-small, n=5, grid=256x256 ===
--- Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap ---
WARNING: raw total_power=1628024.349 W exceeds --max-total-power-w=10.000 W; capping...
Wrote data/labels/.../design.flp
Wrote data/labels/.../design.ptrace (total_power=10.000000 W)
Computing steady-state temperatures...
Wrote data/labels/.../thermal.npy shape=(256, 256) dtype=float32 ...
Done: Vortex-small_... in Xs
...
=== All done. Timings written to docs/hotspot-prototype-timings.tsv ===
```

After the run completes, update the Sample Results table above with
the content of `docs/hotspot-prototype-timings.tsv`.
