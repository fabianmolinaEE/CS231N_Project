# Phase 1 Sanity Check Notes

## Status
- Notebook executed: yes (2026-05-25, re-executed after floorplan feature fix)
- Human review completed: yes (2026-05-25)
- Resume signal from Task 2: `looks correct`

## Findings

### Power -> Thermal Correspondence
- Thermal labels show a smooth center-hot radial gradient across all 4 train samples and
  the 1 val sample. This is expected: the 5 local designs were generated with the
  **single-block HotSpot approach** (Plan 01-02 prototype), which treats the chip as one
  uniform block. Heat spreads symmetrically from the center regardless of power map layout.
- No per-pixel correspondence between power map and thermal label is observable — and none
  is expected for single-block labels. The full 189-design Modal labels use multi-block 16×16
  (ΔT ≈ 54.84 K) and will show meaningful spatial correlation.
- No misalignment, black panels, or NaN artifacts observed.

### Orientation (Assumption A8)
- Thermal map is not flipped relative to the power map. The radial gradient is symmetric
  and consistent across all samples. No corrective action needed.

### Channel Health
- Floorplan (`cell_density`): non-trivial spatial spread (normalized range -0.98 to +6.70,
  std=0.88). Cell density shows where standard cells are placed — a meaningful thermal proxy.
- Power: non-trivial spatial spread (normalized range -0.40 to +7.26, std=0.80).
- Thermal: physically reasonable range (~382.46–382.80 K, ~109°C). Narrow 0.34 K spread is
  the single-block artifact — the Modal multi-block labels will have ~54 K spread.

## Per-Channel Stats (logged from notebook Cell 3, after floorplan feature fix)
| Channel | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| x_train ch0 (cell_density, normalized) | -0.9758 | 6.7045 | -0.0017 | 0.8803 |
| x_train ch1 (power, normalized)        | -0.3998 | 7.2604 | -0.0037 | 0.7971 |
| y_train (thermal, raw K)               | 382.4600 | 382.8000 | 382.6373 | 0.0844 |

## Bugs Fixed During Plan 01-05
Three bugs found and fixed:

1. **Wrong floorplan feature** (`macro_region` → `cell_density`): The split pipeline used
   `macro_region` as the floorplan channel. For Vortex-small designs this is all zeros
   (no large macro blocks). `cell_density` is the correct feature — it shows standard-cell
   placement density and has meaningful spatial variation for all three design families.
   Fixed in `modal_pipeline.py` and `scripts/make_split.py`. **The Modal split JSONs
   (train/val/test.json) must be regenerated on Modal with this fix before training.**

2. **NpzFile `.astype()` error** in `ThermalDataset`: `np.load(npz).astype()` fails on
   NpzFile objects. Fixed with a `_load_array` helper extracting key `'data'`.

3. **Shape mismatch**: CircuitNet native resolution is ~440–459px; HotSpot labels are
   256×256. Fixed with bilinear resize (`scipy.ndimage.zoom`) in `_resize_to`.

All 44 tests pass after the fixes.

## Carry-Forward for Phase 2
- **Action required before training:** Re-run `modal_pipeline.py` on Modal to regenerate
  `train.json`, `val.json`, `test.json`, and `normalization_stats.json` using `cell_density`
  as the floorplan feature. The current Modal split JSONs point to `macro_region` paths and
  are incorrect for training.
- Thermal labels are in raw Kelvin (mean ~382.6 K for single-block; expect ~340–395 K
  range for multi-block Modal labels). Consider `(T - T_mean) / T_std` normalization for
  training stability.
- The local 5-design sanity split is for local visual review only — single-block labels,
  not suitable for training. All model training uses the Modal 189-design dataset.

## Open Items
- Full visual sanity on multi-block Modal labels not yet done (blocked on Modal run access
  during local review session). Recommend running the notebook against the Modal volume
  or downloading 10 random multi-block labels before Phase 2 training begins.
- Design-family leakage check (D-08): confirm Vortex-large / nvdla-large instances don't
  appear in both train and val/test splits before Phase 4 OOD evaluation.
