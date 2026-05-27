# CS231N Project — Physics-Constrained Chip Thermal Map Prediction

Predict full-chip thermal maps from post-placement floorplan and power-density images using a physics-constrained U-Net. The key research question: does enforcing the steady-state heat equation as a differentiable training constraint help a model generalize to chip configurations (die sizes, die thicknesses) it was never trained on?

**Team:** Fabian Molina, Ruben Carrazco
**Course:** CS231N — Deep Neural Networks for Computer Vision, Stanford Spring 2026

---

## Novelty Claim

Two threads exist separately in the literature:
- **Image-based thermal prediction** — data-driven CNN models, no physics constraints, fail outside training distribution
- **Physics-informed neural networks (PINNs)** — enforce heat equation constraints, but use raw matrix inputs, not images

**This project combines them.** The headline experiment: a physics-constrained U-Net degrades less than an MSE-only baseline when evaluated on physical parameters (die thickness, die size) outside the training sweep. This OOD generalization gap is the core contribution.

---

## Problem Statement

Given a 2-channel input image (post-placement floorplan + power-density map), predict a single-channel thermal heatmap. Framed as dense image-to-image regression.

During design-space exploration (e.g., simulated annealing with tens of thousands of steps), running a full thermal simulator at each step is prohibitive. A fast neural surrogate can replace it — but only if it stays physically plausible outside its training distribution.

**Success targets:**
- SSIM ≥ 0.80 on held-out test set
- Physics-constrained model degrades less than MSE-only baseline on OOD die parameters
- Inference time < 100 ms per design on GPU

---

## Technical Approach

### Simulator
**HotSpot** (confirmed). Open-source, block-level RC thermal simulator. Used in grid mode → dense 2D heatmap output compatible with image-to-image framing. Parameterizable die thickness and die size sweep. Parallelizable with `xargs` for large-scale label generation.

PACT investigated and ruled out: at 256×256 with few cores, HotSpot is 2.3× faster than PACT (PACT paper Fig. 18). PACT also requires Xyce + OpenMPI setup and CircuitNet→PACT format conversion. PACT and ML-PACT will be cited in related work only.

### Dataset
- **Primary:** CircuitNet 2.0 — GPU/accelerator subset (Vortex-small, Vortex-large, NVDLA-large)
- **Ablations:** Synthetic random floorplans — grid of rectangular blocks with random power values, zero setup cost, used for rapid iteration before full CircuitNet pipeline is ready
- **Supplemental (if needed):** Chipyard for architectural diversity; ISPD benchmarks for out-of-distribution validation

### Architecture
U-Net encoder-decoder with skip connections. Two training configurations:
- **Baseline:** MSE loss only
- **Primary:** Physics-informed composite loss

### Physics Loss

The 2D steady-state heat equation: `k · ∇²T + Q = 0`

Where `T` is the predicted temperature map, `Q` is the power-density input, and `k` is thermal conductivity of silicon (~149 W/m·K). Laplacian computed via a fixed 3×3 convolution applied to the predicted output:

```
Laplacian kernel:  0   1   0
                   1  -4   1
                   0   1   0
```

Training objective:

```
L_total = λ_data · L_MSE(T_pred, T_label) + λ_phys · mean(|| k · ∇²T_pred + Q ||²)
```

Fully differentiable. No simulation at training time. Penalizes physically implausible predictions independent of whether a ground-truth label exists — which is why it helps OOD generalization.

### Headline Experiment

Train all models with die thicknesses and die sizes sampled from a fixed sweep. Evaluate on physical parameters **outside** that sweep. Measure SSIM and hotspot localization degradation. Physics-constrained U-Net should degrade less than MSE-only baseline.

---

## Milestones

### Milestone 1 — Problem Definition + Related Work (DONE)
- [x] Define the problem as image-to-image regression
- [x] Identify CircuitNet 2.0 as the primary dataset
- [x] Plan HotSpot label generation workflow
- [x] Survey related work: U-Net, pix2pix, DeepOHeat, CircuitNet 2.0, HotSpot
- [x] Receive and incorporate TA feedback (add explicit baselines, ablations, validation strategy)

---

### Milestone 2 — Data Pipeline
**Owner: TBD (pipeline partner)**

- [ ] Download CircuitNet 2.0 GPU/accelerator subset (Vortex-small, Vortex-large, NVDLA-large)
- [ ] Verify spatial alignment of floorplan and power-map channels
- [ ] Run HotSpot in grid mode on a small sample (5–10 designs) to validate label-generation workflow
- [ ] Lock HotSpot grid resolution and temperature normalization strategy
- [ ] Define die thickness and die size sweep values (at least 3 die thickness values)
- [ ] Implement label generation at scale with parameterized die configs (parallelized with `xargs`)
- [ ] Define train/val/test split by design family to avoid leakage; hold out a subset of die configs for OOD eval
- [ ] Implement synthetic random floorplan generator for ablation runs
- [ ] Write PyTorch `Dataset` class that loads (input pair, thermal label, die config) tuples
- [ ] Write `DataLoader` with basic augmentation (horizontal/vertical flip)
- [ ] Sanity-check script: visualize 3–5 input/label pairs side by side

---

### Milestone 3 — Baseline Models
**Owner: TBD (model partner)**

- [ ] Implement flat CNN regressor (no skip connections) — lower bound
- [ ] Implement basic encoder-decoder without skip connections
- [ ] Implement U-Net with MSE-only loss — primary baseline
- [ ] Train all baselines; log RMSE and SSIM on validation set
- [ ] Set up training infrastructure: loss logging, checkpoint saving, learning rate scheduler
- [ ] Confirm SSIM as primary metric for model selection

---

### Milestone 4 — Physics-Constrained U-Net
**Owner: TBD (model partner)**

- [ ] Implement physics loss: fixed Laplacian kernel applied to predicted output
- [ ] Implement composite loss: `L_total = λ_data · L_MSE + λ_phys · L_physics`
- [ ] Train physics-constrained U-Net and compare against MSE-only baseline
- [ ] Sweep `λ_phys` values; log both losses separately per run
- [ ] Run ablations:
  - [ ] Skip connections on vs. off
  - [ ] λ_phys = 0 (MSE-only) vs. λ_phys > 0 (physics-constrained)
  - [ ] Physics loss in normalized vs. physical units
- [ ] Save best checkpoint and log all run configs

---

### Milestone 5 — OOD Evaluation + Analysis
**Owner: both**

- [ ] Evaluate all models on held-out die configs (thickness and size outside training sweep)
- [ ] Compute SSIM and hotspot localization degradation: physics model vs. MSE-only baseline
- [ ] Compute full test-set metrics: RMSE, MSE, SSIM on in-distribution test set
- [ ] Benchmark inference time on same hardware for all models
- [ ] Generate qualitative heatmap visualizations: input channels → predicted → ground truth
- [ ] Run GradCAM or saliency analysis to understand what layout features drive predictions
- [ ] Identify and document at least 3 failure cases with explanations
- [ ] Produce comparison table: CNN baseline, encoder-decoder, U-Net MSE-only, U-Net physics-constrained

---

### Milestone 6 — Final Report + Poster
**Owner: both**

- [ ] Write intro and motivation: design-space exploration bottleneck, OOD generalization problem
- [ ] Write related work: CircuitNet, HotSpot, U-Net, PINNs, ML-PACT (cite but don't use), DeepOHeat
- [ ] Write data section: CircuitNet subset, HotSpot workflow, die config sweep, split strategy
- [ ] Write method section: architecture diagram, physics loss derivation, training details
- [ ] Write experiments section: in-distribution metrics, OOD degradation table, ablation table, qualitative figures
- [ ] Write conclusion: what worked, what didn't, what we would do next
- [ ] Create architecture diagram: input → encoder → bottleneck → decoder → output heatmap + physics loss term
- [ ] Prepare poster for CS231N poster session

---

## Work Split (Proposed)

| Track | Owner | Milestones |
|-------|-------|------------|
| Data pipeline + HotSpot labels | TBD | M2 |
| Baselines + physics-constrained U-Net | TBD | M3, M4 |
| OOD evaluation, report, poster | Both | M5, M6 |

Assign names once you've agreed on the split.

---

## Repo Structure

```
CS231N_Project/
├── README.md
├── docs/
│   ├── project-proposal.md
│   ├── project-milestone1.md
│   ├── architecture-research.md
│   └── meetings/
├── data/                  ← gitignored; CircuitNet 2.0 lives here
│   ├── raw/
│   └── labels/
├── src/
│   ├── dataset.py         ← PyTorch Dataset + DataLoader
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   ├── encoder_decoder.py
│   │   └── unet.py        ← primary model (MSE-only + physics-constrained configs)
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── scripts/
│   └── generate_labels.sh ← HotSpot label generation (parameterized for die config sweep)
├── notebooks/
│   └── sanity_check.ipynb
└── checkpoints/           ← gitignored
```

---

## Key References

- [CircuitNet 2.0](https://openreview.net/forum?id=H1z7m3Kc7S)
- [U-Net (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [pix2pix (Isola et al., 2017)](https://arxiv.org/abs/1611.07004)
- [Encoder-Decoder Networks for Thermal/PDN Analysis](https://dl.acm.org/doi/10.1145/3526115)
- [DeepOHeat](https://doi.org/10.1109/DAC56929.2023.10247998)
- [HotSpot thermal simulator](https://dl.acm.org/doi/10.1145/859618.859620)
- [Physics-Informed Neural Networks (Raissi et al., 2019)](https://arxiv.org/abs/1711.10561)
- [PACT (Yuan et al., IEEE TCAD)](https://ieeexplore.ieee.org/document/9296639) — cited in related work; HotSpot used for label generation
- [ML-PACT](https://arxiv.org/abs/2302.08806) — cited in related work; targets transient (not steady-state)

---

## Open Questions

- [ ] How many CircuitNet GPU/accelerator designs after filtering? Enough for training?
- [ ] What specific die thickness values to sweep? (at least 3)
- [ ] Unit normalization in physics loss: normalized vs. physical units?
- [ ] λ_data and λ_phys starting values for the sweep?
- [ ] Does ML-PACT have a publicly released dataset?

---

## Compute

| Provider | Credits | Notes |
|----------|---------|-------|
| Modal | $200 (shared) | Primary — Python-native, easiest for iterative runs |
| AWS | $100 | EC2 `p3`/`g5` instances; good backup |
| Azure Students | $100 | NC-series VMs (V100/A100) |
| Google Cloud | $50 | Supplemental |

**Total: ~$450 in GPU credits.**

Modal is the default for training runs:
- GPU target: A100 or H100 via Modal's on-demand fleet
- Launch training with `modal run scripts/train_modal.py`
- Checkpoints written back to a Modal volume (persistent across runs)

## Setup

```bash
git clone <repo-url>
cd CS231N_Project

# install dependencies (fill in once decided)
pip install -r requirements.txt

# Modal CLI (for remote training)
pip install modal
modal setup
```
