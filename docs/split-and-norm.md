# Train/Val/Test Split and Normalization (Plan 01-03)

## Split (D-07, D-08)

- **Ratios:** 80% train / 10% val / 10% test
- **Random seed:** 42 (deterministic, `random_state=42`)
- **Split level:** design instance (each npz file = one row)
- **Total instances:** 189 matched (Vortex-small: 96, Vortex-large: 61, nvdla-large: 32)

### How to run

Full pipeline on Modal (recommended — generates labels then splits in one command):
```bash
modal run modal_pipeline.py
```

Split/norm only (after labels already exist in the Modal Volume):
```bash
modal run modal_pipeline.py::make_split_and_stats
```

Local fallback (after syncing labels to `data/labels/`):
```bash
python scripts/make_split.py
```

### Split JSON schema

Each split file is a list of entry dicts:
```json
[
  {
    "instance": "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap",
    "family": "Vortex-small",
    "floorplan": "/data/CircuitNet-N14/routability_features/Vortex-small/Vortex-small/macro_region/INST.npz",
    "power":     "/data/CircuitNet-N14/IR_drop_features/Vortex-small/Vortex-small/power_all/INST.npz",
    "label":     "/data/labels/Vortex-small/INST/thermal.npy"
  }
]
```

Paths in Modal-generated JSONs are Volume-absolute (`/data/...`).
Paths in locally generated JSONs are relative to repo root.

### Design-family leakage (D-08, deferred)

Instances from the same RTL design family (e.g., Vortex-small with different
placement seeds) may appear in both train and test. This is explicitly deferred
per decision D-08. Revisit before Phase 4 evaluation.

**Risk:** Models may overfit to placement-variant patterns without truly
generalizing to unseen circuit topologies. Mitigations to consider:
- Family-stratified split (ensure each family present in all three sets)
- Cross-family holdout (nvdla-large as test only)

## Normalization (D-09)

- **Schema:**
  ```json
  {"floorplan": {"mean": <float>, "std": <float>},
   "power":     {"mean": <float>, "std": <float>}}
  ```
- **Computed from TRAIN split only** (prevents val/test leakage into feature statistics)
- **Per-channel global statistics** (not per-pixel, not per-instance)
- **Applied at DataLoader load time** in `src/dataset.py` (Plan 04)
- Stored in `data/normalization_stats.json` (committed; placeholder until labels generated)

## Power Unit Note

CircuitNet `power_all` values are **not in Watts**. The raw sum for a Vortex-small
design is ~1,628,024 (dimensionless relative power density, likely mW/tile or
normalized). Normalization stats reflect the raw dimensionless values.

HotSpot label generation uses a 10W cap (`--max-total-power-w 10.0`) for thermal
simulation stability, applied consistently across all 189 instances. The spatial power
distribution pattern is preserved; only the absolute scale is adjusted. Absolute
temperature scale does not affect the ML task — the U-Net learns spatial thermal
gradients.

See `docs/hotspot-prototype.md` for power units discovery details (Plan 02).
