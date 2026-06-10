# Physics-Constrained Thermal Map Prediction for VLSI Chip Design

CS231N Final Project — Stanford Spring 2026  
**Fabian Molina** · **Ruben Carrazco**

---

## Overview

We train a physics-constrained U-Net to predict full-chip steady-state thermal heatmaps from a 2-channel floorplan+power-density image. The model enforces the 2D heat equation as a differentiable Laplacian penalty — no simulator calls during training.

**Research question:** does a heat-equation residual constraint reduce OOD degradation when die thickness or convective resistance fall outside the training distribution?

**Short answer:** yes for moderate convection-resistance shifts (8–11% RMSE reduction), no for die-thickness extrapolation.

---

## Results

### In-Distribution Test Set

| Model | RMSE (K) ↓ | SSIM ↑ | Hotspot IoU ↑ |
|---|---|---|---|
| PlainCNN | 2.420 | 0.987 | 0.630 |
| EncoderDecoder | 1.213 | 0.999 | 0.781 |
| U-Net (λ=0, MSE-only) | 1.008 | 0.995 | 0.819 |
| **U-Net (λ=0.01)** | **0.895** | **0.996** | **0.841** |

The physics penalty at λ=0.01 reduces RMSE by 11% and improves hotspot IoU by 2.7 pp over the MSE-only baseline.

### OOD Generalization (U-Net variants)

| Condition | λ=0 RMSE (K) | λ=0.01 RMSE (K) | Δ |
|---|---|---|---|
| In-dist (150 µm, r=0.1 K/W) | 1.008 | **0.895** | −11.2% |
| r=0.01 K/W | 1.595 | **1.458** | −8.6% |
| r=0.05 K/W | 1.259 | **1.124** | −10.7% |
| r=0.20 K/W | 1.203 | **1.200** | ≈0% |
| r=0.50 K/W | **3.847** | 3.926 | +2.1% |
| 75 µm | 6.592 | **6.428** | −2.5% |
| 100 µm | 4.299 | **4.136** | −3.8% |
| 200 µm | **3.137** | 3.252 | +3.7% |
| 300 µm | **9.376** | 9.504 | +1.4% |

Die-thickness OOD degrades 9–10× at 300 µm regardless of λ; the Laplacian kernel cannot encode thickness-dependent boundary conditions.

---

## Method

**Input:** 2-channel 256×256 image — post-placement floorplan (cell density) + power-density map  
**Output:** 1-channel 256×256 steady-state thermal heatmap (Kelvin)

### Physics Loss

The 2D steady-state heat equation: `k ∇²T + Q = 0`

Laplacian approximated via a fixed 3×3 finite-difference kernel applied to the model's output. No simulator at training time.

```
L_total = L_MSE(T_pred, T_label) + λ · ‖k (K * T_pred) + Q‖²_F
```

`λ ∈ {0, 0.01, 0.1, 1.0}` swept across all three architectures (12 models total).

### Architectures

- **U-Net** (b=32, ~2M params) — encoder-decoder with lateral skip connections + PixelShuffle upsampling
- **EncoderDecoder** (b=64, ~28M params) — same structure, no skip connections
- **PlainCNN** (b=64, ~600K params) — 8 dilated DoubleConv blocks at full resolution, no downsampling

### Training

- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **Schedule:** CosineAnnealingLR (T_max=250, η_min=1e-6)
- **Epochs:** 250, no early stopping; checkpoint at lowest validation MSE
- **Hardware:** NVIDIA A10G (24 GB) via Modal
- **Logging:** W&B

---

## Dataset

**Source:** CircuitNet 2.0 — GPU/accelerator subset  
- Vortex-small (96 instances), Vortex-large (61), NVDLA-large (32) → **189 instances total**

**Labels:** HotSpot in grid mode (256×256), multi-block 16×16 floorplan decomposition  
- Training config: die thickness 150 µm, r_convec = 0.1 K/W  
- Power density capped at 10 W/tile; label generation parallelized on Modal  
- OOD labels: 4 additional thicknesses + 4 additional convection resistances = 1,512 extra labels

**Split:** 80/10/10 train/val/test (seed 42), stratified by family → 153 / 18 / 18 instances  
**Normalization:** per-family z-score on training split

---

## Repo Structure

```
CS231N_Project/
├── src/
│   ├── dataset.py              # ThermalDataset + DataLoader
│   ├── train.py                # training loop, W&B logging, cosine LR
│   ├── evaluate.py             # RMSE, hotspot IoU, SSIM
│   ├── visualize.py
│   └── models/
│       ├── unet.py             # U-Net with PixelShuffleUp
│       ├── encoder_decoder.py  # EncoderDecoder (no skip connections)
│       └── baseline_cnn.py     # PlainCNN (dilated, full-resolution)
├── scripts/
│   ├── npz_to_hotspot.py       # CircuitNet NPZ → HotSpot floorplan format
│   ├── parse_hotspot_output.py # parse .grid.steady → (256,256) numpy array
│   ├── make_split.py           # local fallback for train/val/test splits
│   └── log_eval_to_wandb.py    # log eval tables + charts to W&B
├── modal_pipeline.py           # HotSpot label generation + splits on Modal
├── modal_train.py              # remote training entrypoint
├── modal_ood.py                # OOD label generation (thickness + HTC sweep)
├── modal_eval_ood.py           # evaluate all checkpoints on OOD conditions
├── modal_viz.py                # generate result figures
├── eval_ood_results.json       # full evaluation results (n=36, 9 conditions)
├── tests/
│   ├── test_dataset.py
│   └── test_evaluate.py
├── Final Report/
│   └── CS231N_report.tex       # final paper (CVPR format)
├── CS231N_FinalPoster/
│   └── main.tex                # poster (Gemini beamer theme)
├── third_party/HotSpot/        # HotSpot thermal simulator
├── docs/                       # design notes, meeting summaries
└── requirements.txt
```

---

## Setup

```bash
git clone <repo-url>
cd CS231N_Project
pip install -r requirements.txt

# Modal (for remote training/label generation)
pip install modal
modal setup
```

Data lives in `data/` (gitignored). Download CircuitNet 2.0 GPU subset via Hugging Face:

```bash
bash scripts/download_data.sh   # requires HF_TOKEN env var
```

---

## References

- [CircuitNet 2.0](https://openreview.net/forum?id=H1z7m3Kc7S)
- [HotSpot](https://dl.acm.org/doi/10.1145/859618.859620)
- [U-Net (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [PixelShuffle (Shi et al., 2016)](https://arxiv.org/abs/1609.05158)
- [PINNs (Raissi et al., 2019)](https://arxiv.org/abs/1711.10561)
- [DeepOHeat](https://doi.org/10.1109/DAC56929.2023.10247998)
- [pix2pix](https://arxiv.org/abs/1611.07004)
- [ML-PACT](https://arxiv.org/abs/2302.08806)
