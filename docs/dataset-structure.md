# CircuitNet-N14 Dataset Structure (verified)

Source: `data/raw/CircuitNet-N14`

> **STATUS: PENDING DOWNLOAD**
> This file was created before `data/raw/CircuitNet-N14` was available.
> The NPZ key table below will be populated once the dataset is downloaded.
> Run: `HF_TOKEN=<your_token> ./scripts/download_data.sh` then
> `python scripts/inspect_dataset.py` to regenerate this document with verified keys.

## Design Counts

- Total directories: TODO (run inspect_dataset.py after download)
- GPU/accelerator subset: TODO
  - Vortex-small*: TODO
  - Vortex-large*: TODO
  - NVDLA-large*: TODO

## Verified NPZ Keys

> TODO: Populate by running `python scripts/inspect_dataset.py` after download.
> Expected keys (from RESEARCH.md, unverified):
> - Floorplan: `routability_features/macro_region.npz` → key `macro_region` (assumption A1)
> - Power: `IR_drop_features/power_all.npz` → key `power_all` (assumption A2)

| Design | Role | Path | Key | Shape | Dtype |
|--------|------|------|-----|-------|-------|
| (pending download) | - | - | - | - | - |

## Errors

- None

## Filter Definition (D-01, D-03)

GPU/accelerator subset includes designs whose name starts with one of:
- `Vortex-small`
- `Vortex-large`
- `NVDLA-large`

## Raw Inspection Records (JSON)

```json
[]
```
