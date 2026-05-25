# Phase 1 Sanity Check Notes

## Status
- Notebook executed: yes (2026-05-25, `jupyter nbconvert --to notebook --execute`)
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
- Floorplan: all-zero (all-black panels). `macro_region` feature is zero for all 5
  Vortex-small prototype designs — these instances have no large macro blocks. Expected;
  not a bug. The full dataset (Vortex-large, nvdla-large) will have non-zero macro regions.
- Power: non-trivial spatial spread, fine-grained hotspot structure visible across all samples.
- Thermal: physically reasonable range (~382.46–382.80 K, ~109°C). Narrow 0.34 K spread is
  the single-block artifact — the Modal multi-block labels will have ~54 K spread.

## Per-Channel Stats (logged from notebook Cell 3)
| Channel | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| x_train ch0 (floorplan, normalized) | -0.5861 | -0.5861 | -0.5861 | 0.0000 |
| x_train ch1 (power, normalized)     | -0.3255 |  0.5298 | -0.2813 | 0.0890 |
| y_train (thermal, raw K)            | 382.4600 | 382.8000 | 382.6373 | 0.0844 |

## Dataset Bug Fixed During Plan 01-05
`ThermalDataset.__getitem__` had two bugs that only surface with real data (tests used
synthetic `.npy` files at 256×256):

1. `.npz` files: `np.load(path).astype()` fails on NpzFile objects. Fixed by extracting
   key `'data'` via a `_load_array` helper.
2. Shape mismatch: CircuitNet native resolution is 459×456; HotSpot labels are 256×256.
   Fixed by bilinear resize (scipy `zoom`) in `_resize_to` before the shape assertion.

Both fixes are in `src/dataset.py`. All 44 tests pass after the fix.

## Carry-Forward for Phase 2
- Thermal labels are in raw Kelvin (mean ~382.6 K for single-block; expect ~340–395 K
  range for multi-block Modal labels). Consider mean-subtracting or using
  `(T - T_mean) / T_std` normalization in the training loss for stability.
- The floorplan `macro_region` channel is all-zero for Vortex-small designs. Phase 2
  baseline training on Vortex-small only should treat ch0 as effectively a bias term.
  Vortex-large and nvdla-large will populate this channel.
- The local 5-design sanity split (`data/splits/local_train.json`,
  `data/splits/local_val.json`) is for local visual review only — single-block labels,
  not suitable for training. All model training uses the Modal 189-design dataset.

## Open Items
- Full visual sanity on multi-block Modal labels not yet done (blocked on Modal run access
  during local review session). Recommend running the notebook against the Modal volume
  or downloading 10 random multi-block labels before Phase 2 training begins.
- Design-family leakage check (D-08): confirm Vortex-large / nvdla-large instances don't
  appear in both train and val/test splits before Phase 4 OOD evaluation.
