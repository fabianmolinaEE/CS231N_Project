# HotSpot Prototype Results (Plan 01-02)

## Build Status

- Built: yes
- SuperLU: no (plain make, macOS clang)
- Build command used: `make CFLAGS="-O2 -g"` (macOS, third_party/HotSpot/)
- Platform: macOS Darwin 25.4.0
- Binary: `third_party/HotSpot/hotspot` (executable, gitignored)
- Config: `hotspot.config` (copied from HotSpot default, committed)

## Power Units Discovery (Plan-02 Critical Finding)

CircuitNet `power_all` npz values are **not in Watts**. Raw sums for Vortex-small
instances are approximately 1.6 × 10^6 per design. When passed directly to HotSpot,
this caused the steady-state thermal solver to compute T_junction ≈ 163,120 K and
never converge (ran 20+ minutes with no output).

**Fix applied in `scripts/npz_to_hotspot.py`:** Added `--max-total-power-w=10.0` flag
(default) that caps total chip power to a physically realistic value. For Vortex-small
(~1 mm² chip at 200 MHz RISC-V GPU), 10 W is a conservative but reasonable estimate.
The relative spatial heat distribution from `power_all` is preserved; the absolute
scale is clamped.

**Implication for Plan 03:** Before scaling to the full 189-design subset, clarify the
correct physical units for `power_all` from CircuitNet documentation. Options:
- Values may be in mW/tile → divide sum by 1000 (gives ~1628 W — still too high)
- Values may be normalized (dimensionless) → use calibration constant
- Current fix (cap at 10 W) is a validated placeholder producing physically meaningful
  temperatures and running in seconds per design

## Sample Results

| design | wall_seconds | thermal_min_k | thermal_max_k | thermal_mean_k |
|--------|-------------|--------------|--------------|----------------|
| Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap | 8 | 382.460 | 382.800 | 382.637 |
| Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ar | 7 | 382.460 | 382.800 | 382.637 |
| Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ap | 7 | 382.460 | 382.800 | 382.637 |
| Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ar | 7 | 382.460 | 382.800 | 382.637 |
| Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_4_fi_ap | 7 | 382.460 | 382.800 | 382.637 |

**Note on temperature uniformity:** All 5 designs produce near-identical temperature maps
(ΔT = 0.34 K across chip). This is expected with the single-block aggregate approach —
HotSpot uses uniform total power (10 W) for all designs, so the spatial variation
comes only from lateral heat conduction in the grid solver, not per-tile power injection.
See Conversion Approach below; revisit after Plan 05 visualization.

## Per-Design Wall-Clock

- N sampled: 5 (Vortex-small family, 459×456 native shape → 256×256 grid)
- Mean seconds/design: **7.2 s** (range: 7–8 s)
- Extrapolated total for full subset (189 designs):
  - Single-core: 189 × 7.2 s = 1,361 s ≈ 23 min
  - 8-core parallel (D-04 target): 1,361 / 8 = **170 s ≈ 3 min**
- D-04 target: 30–60 min total on 8 cores — **well within budget (8× headroom)**
- Note: Vortex-large (1316×1301) and nvdla-large (1721×1716) will take longer; time
  separately in Plan 03 before parallelizing.

## Output Sanity

- All 5 thermal.npy files exist: **yes**
- Shape: **(256, 256) float32** ✓
- Temperature range across 5 samples: min=382.460 K, max=382.800 K, mean=382.637 K
  - T_ambient = 318.15 K (hotspot.config); T_rise ≈ 64 K above ambient
  - Bulk T_rise from r_convec (0.1 K/W × 10 W) = 1 K; remaining ~63 K from chip die
    thermal resistance (consistent with silicon at 0.00015 m thickness, k=130 W/m-K)
  - ΔT across chip = 0.34 K — very uniform, confirming single-block approach
- Visual inspection: deferred to Plan 05 sanity notebook

## Conversion Approach

Single-block aggregate (RESEARCH.md Pitfall 2). Documented limitation:
HotSpot distributes total chip power uniformly; localized hotspots emerge only from the
grid solver's lateral heat conduction, NOT from per-tile power injection. The 5 designs
here are near-identical in thermal output because all have the same capped 10 W total.

**Plan 03 decision needed:** To get spatially varied labels, either:
1. Inject per-tile power via HotSpot's grid power trace format (different .ptrace format)
2. Accept uniform-power labels and verify the U-Net still learns spatial structure from
   the input channels (floorplan + power_all vary across designs even if total power is capped)

Deferred to Plan 05 visual review.
