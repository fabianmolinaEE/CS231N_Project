# CircuitNet-N14 Dataset Structure (verified 2026-05-24)

Source: `data/raw/CircuitNet-N14/`
HF repo: `CircuitNet/CircuitNet` (dataset)

## Directory Layout

```
data/raw/CircuitNet-N14/
  {feature_type}/           # IR_drop_features | routability_features
    {design_family}/        # Vortex-small | Vortex-large | nvdla-large
      {design_family}/      # repeated (archive extraction nests one level)
        {feature_name}/     # power_all | macro_region | etc.
          {instance}.npz    # one file per design instance
```

Each npz file contains **one key: `'data'`** (not the feature name).

## Design Counts

| Family | Instances | Native Shape | Resize needed |
|--------|-----------|--------------|---------------|
| Vortex-small | 96 | [459, 456] | → 256×256 |
| Vortex-large | 61 | [1316, 1301] | → 256×256 |
| nvdla-large | 32 | [1721, 1716] | → 256×256 |
| **Total GPU subset** | **189** | — | — |

Note: `nvdla-large` uses **lowercase** in the repo (not `NVDLA-large`).

## Verified NPZ Keys

| Design | Feature type | Feature dir | NPZ key | Shape | Dtype |
|--------|-------------|-------------|---------|-------|-------|
| Vortex-small | routability_features | macro_region | `data` | [459, 456] | float64 |
| Vortex-small | IR_drop_features | power_all | `data` | [459, 456] | float64 |
| Vortex-large | routability_features | macro_region | `data` | [1316, 1301] | float64 |
| Vortex-large | IR_drop_features | power_all | `data` | [1316, 1301] | float64 |
| nvdla-large | routability_features | macro_region | `data` | [1721, 1716] | float64 |
| nvdla-large | IR_drop_features | power_all | `data` | [1721, 1716] | float64 |

**Key finding:** All npz files use key `'data'`, not the feature directory name.

## Assumptions Status

| ID | Assumption | Status |
|----|-----------|--------|
| A1 | Floorplan = macro_region | **VERIFIED** |
| A2 | Power = power_all | **VERIFIED** |
| A3 | GPU filter by prefix | **VERIFIED** |
| A4 | Design count ~100-500 | **VERIFIED** (189 instances) |
| A7 | Shape consistency within family | **VERIFIED** |

## Filter Definition (D-01, D-03)

GPU/accelerator subset includes designs from these families:
- `Vortex-small`
- `Vortex-large`
- `nvdla-large`

## Path Helper (Python)

```python
ROOT = Path("data/raw/CircuitNet-N14")
GPU_FAMILIES = ["Vortex-small", "Vortex-large", "nvdla-large"]

def floorplan_dir(family: str) -> Path:
    return ROOT / "routability_features" / family / family / "macro_region"

def power_dir(family: str) -> Path:
    return ROOT / "IR_drop_features" / family / family / "power_all"

# Load: np.load(path, allow_pickle=False)["data"]
```

## Raw Inspection Records

```json
[
  {
    "family": "Vortex-small",
    "errors": [],
    "instance_count": {
      "floorplan (macro_region)": 96,
      "power (power_all)": 96
    },
    "floorplan (macro_region)": [
      {
        "file": "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          459,
          456
        ],
        "dtype": "float64"
      },
      {
        "file": "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ar.npz",
        "keys": [
          "data"
        ],
        "shape": [
          459,
          456
        ],
        "dtype": "float64"
      },
      {
        "file": "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          459,
          456
        ],
        "dtype": "float64"
      }
    ],
    "power (power_all)": [
      {
        "file": "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          459,
          456
        ],
        "dtype": "float64"
      },
      {
        "file": "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ar.npz",
        "keys": [
          "data"
        ],
        "shape": [
          459,
          456
        ],
        "dtype": "float64"
      },
      {
        "file": "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          459,
          456
        ],
        "dtype": "float64"
      }
    ]
  },
  {
    "family": "Vortex-large",
    "errors": [],
    "instance_count": {
      "floorplan (macro_region)": 74,
      "power (power_all)": 61
    },
    "floorplan (macro_region)": [
      {
        "file": "Vortex-large_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1316,
          1301
        ],
        "dtype": "float64"
      },
      {
        "file": "Vortex-large_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ar.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1316,
          1301
        ],
        "dtype": "float64"
      },
      {
        "file": "Vortex-large_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1316,
          1301
        ],
        "dtype": "float64"
      }
    ],
    "power (power_all)": [
      {
        "file": "Vortex-large_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1316,
          1301
        ],
        "dtype": "float64"
      },
      {
        "file": "Vortex-large_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ar.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1316,
          1301
        ],
        "dtype": "float64"
      },
      {
        "file": "Vortex-large_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1316,
          1301
        ],
        "dtype": "float64"
      }
    ]
  },
  {
    "family": "nvdla-large",
    "errors": [],
    "instance_count": {
      "floorplan (macro_region)": 68,
      "power (power_all)": 32
    },
    "floorplan (macro_region)": [
      {
        "file": "nvdla-large_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1721,
          1716
        ],
        "dtype": "float64"
      },
      {
        "file": "nvdla-large_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ar.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1721,
          1716
        ],
        "dtype": "float64"
      },
      {
        "file": "nvdla-large_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1721,
          1716
        ],
        "dtype": "float64"
      }
    ],
    "power (power_all)": [
      {
        "file": "nvdla-large_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ar.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1721,
          1716
        ],
        "dtype": "float64"
      },
      {
        "file": "nvdla-large_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ap.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1721,
          1716
        ],
        "dtype": "float64"
      },
      {
        "file": "nvdla-large_freq_200_mp_1_fpu_55_fpa_1.0_p_1_fi_ar.npz",
        "keys": [
          "data"
        ],
        "shape": [
          1642,
          1638
        ],
        "dtype": "float64"
      }
    ]
  }
]
```