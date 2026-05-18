# CS231N Project — GPU Thermal Map Prediction

Predict full-chip GPU thermal maps from post-placement floorplan and power-density images using a CNN-based image-to-image regression model. Fast inference replaces expensive thermal simulation (HotSpot/ANSYS) during early chip-design iteration.

**Team:** Fabian Molina, Ruben [last name]
**Course:** CS231N — Deep Neural Networks for Computer Vision, Stanford Spring 2026

---

## Problem Statement

Given a 2-channel input image (post-placement floorplan + power-density map), predict a single-channel thermal heatmap. Framed as dense image-to-image regression. Primary dataset: CircuitNet 2.0. Ground-truth labels: HotSpot thermal simulator in grid mode.

**Success targets:**
- SSIM ≥ 0.80 on the held-out test set
- Hotspot localization accuracy on the top 5% hottest regions
- Inference time < 100 ms per design on GPU

---

## Milestones

### Milestone 1 — Problem Definition + Related Work ✅
- [x] Define the problem as image-to-image regression
- [x] Identify CircuitNet 2.0 as the primary dataset
- [x] Plan HotSpot label generation workflow
- [x] Survey related work: U-Net, pix2pix, DeepOHeat, CircuitNet 2.0, HotSpot
- [x] Receive and incorporate TA feedback (add explicit baselines, ablations, validation strategy)

See [`docs/project-proposal.md`](docs/project-proposal.md) and [`docs/project-milestone1.md`](docs/project-milestone1.md).

---

### Milestone 2 — Data Pipeline 🔲
**Owner: TBD (pipeline partner)**

- [ ] Download CircuitNet 2.0 and inspect the data format
- [ ] Decide on the subset: all designs vs. GPU/accelerator-only filter
- [ ] Verify spatial alignment of floorplan and power-map channels
- [ ] Run HotSpot in grid mode on a small sample (5–10 designs) to validate the label-generation workflow
- [ ] Lock HotSpot grid resolution and temperature normalization strategy
- [ ] Estimate full label-generation time and storage requirements
- [ ] Implement label generation at scale (parallelized if needed)
- [ ] Define train / val / test split (consider splitting by design family to avoid leakage)
- [ ] Write a PyTorch `Dataset` class that loads (input pair, thermal label) tuples
- [ ] Write a `DataLoader` with basic augmentation (horizontal/vertical flip)
- [ ] Add a sanity-check script: visualize 3–5 input/label pairs side by side

---

### Milestone 3 — Baseline Models 🔲
**Owner: TBD (model partner)**

- [ ] Implement a simple flat CNN regressor (no skip connections, no encoder-decoder structure) as the lower bound
- [ ] Implement a basic encoder-decoder without skip connections
- [ ] Train both baselines, log RMSE and SSIM on the validation set
- [ ] (Optional) Implement a pix2pix-style image-to-image baseline using a conditional GAN
- [ ] Set up training infrastructure: loss logging, checkpoint saving, learning rate scheduler
- [ ] Define one primary metric for model selection (propose: SSIM, but decide as a team)

---

### Milestone 4 — U-Net Implementation 🔲
**Owner: TBD (model partner)**

- [ ] Implement U-Net encoder-decoder with skip connections
- [ ] Experiment with encoder options: plain CNN vs. ResNet-50 pretrained encoder
- [ ] Choose loss function: MSE only, L1 only, or a weighted MSE + SSIM combo
- [ ] Train U-Net and compare against Milestone 3 baselines on RMSE and SSIM
- [ ] Run ablations:
  - [ ] Skip connections on vs. off
  - [ ] Encoder depth (shallow vs. deep)
  - [ ] Loss function choice
- [ ] Save best checkpoint and log all run configs

---

### Milestone 5 — Evaluation + Analysis 🔲
**Owner: both**

- [ ] Compute final test-set metrics: RMSE, MSE, SSIM
- [ ] Implement hotspot localization evaluation: precision/recall on top 5% hottest pixels
- [ ] Benchmark inference time on the same hardware for all models
- [ ] Generate qualitative heatmap visualizations: input channels → predicted → ground truth
- [ ] Run saliency / GradCAM analysis to understand what layout features drive predictions
- [ ] Identify and document at least 3 failure cases with explanations
- [ ] Produce a table comparing all models: CNN baseline, encoder-decoder, U-Net, pix2pix (if run)

---

### Milestone 6 — Final Report + Poster 🔲
**Owner: both**

- [ ] Write the intro and motivation section
- [ ] Write the related work section (expand from milestone 1)
- [ ] Write the data section: CircuitNet 2.0 subset, HotSpot workflow, split strategy
- [ ] Write the method section: architecture diagram, loss, training details
- [ ] Write the experiments section: all metrics, ablation table, qualitative figures
- [ ] Write the conclusion: what worked, what didn't, what we would do next
- [ ] Create architecture diagram (input → encoder → bottleneck → decoder → output heatmap)
- [ ] Prepare poster for the CS231N poster session

---

## Work Split (Proposed)

| Track | Owner | Milestones |
|-------|-------|------------|
| Data pipeline + HotSpot labels | TBD | M2 |
| Baselines + U-Net implementation | TBD | M3, M4 |
| Evaluation, report, poster | Both | M5, M6 |

Assign names once you've agreed on the split.

---

## Repo Structure

```
CS231N_Project/
├── README.md              ← this file
├── docs/                  ← project notes (synced from wiki)
│   ├── project-proposal.md
│   └── project-milestone1.md
├── data/                  ← gitignored; CircuitNet 2.0 lives here
│   ├── raw/
│   └── labels/
├── src/
│   ├── dataset.py         ← PyTorch Dataset + DataLoader
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   ├── encoder_decoder.py
│   │   ├── unet.py
│   │   └── pix2pix.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── scripts/
│   └── generate_labels.sh ← HotSpot label generation
├── notebooks/
│   └── sanity_check.ipynb ← visualize input/label pairs
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

---

## Setup

```bash
# clone
git clone <repo-url>
cd CS231N_Project

# install dependencies (fill in once decided)
pip install -r requirements.txt
```
