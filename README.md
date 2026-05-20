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

### Milestone 1 — Problem Definition + Related Work (DONE)
- [x] Define the problem as image-to-image regression
- [x] Identify CircuitNet 2.0 as the primary dataset
- [x] Plan HotSpot label generation workflow
- [x] Survey related work: U-Net, pix2pix, DeepOHeat, CircuitNet 2.0, HotSpot
- [x] Receive and incorporate TA feedback (add explicit baselines, ablations, validation strategy)

See [`docs/project-proposal.md`](docs/project-proposal.md) and [`docs/project-milestone1.md`](docs/project-milestone1.md).

---

### Milestone 2 — Data Pipeline 
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

### Milestone 3 — Baseline Models 
**Owner: TBD (model partner)**

- [ ] Implement a simple flat CNN regressor (no skip connections, no encoder-decoder structure) as the lower bound
- [ ] Implement a basic encoder-decoder without skip connections
- [ ] Train both baselines, log RMSE and SSIM on the validation set
- [ ] (Optional) Implement a pix2pix-style image-to-image baseline using a conditional GAN
- [ ] Set up training infrastructure: loss logging, checkpoint saving, learning rate scheduler
- [ ] Define one primary metric for model selection (propose: SSIM, but decide as a team)

---

### Milestone 4 — U-Net Implementation 
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

### Milestone 6 — Final Report + Poster 
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

---

## Architecture Designs

### Task Framing
- **Input:** 2-channel image — (1) post-placement floorplan, (2) power-density map
- **Output:** 1-channel thermal heatmap (temperature per pixel)
- **Resolution:** TBD — determined by native CircuitNet 2.0 resolution (likely 256×256 or 512×512)
- **Scale:** ~10,000 training samples on Stanford HPC

---

### Design 1: Classic U-Net (Baseline)

**Architecture overview:**
```
Input (2ch)
  → Encoder:
      [Conv3x3 → BN → ReLU] × 2  → 64ch    (scale 1)
      MaxPool 2×2
      [Conv3x3 → BN → ReLU] × 2  → 128ch   (scale 2)
      MaxPool 2×2
      [Conv3x3 → BN → ReLU] × 2  → 256ch   (scale 3)
      MaxPool 2×2
      [Conv3x3 → BN → ReLU] × 2  → 512ch   (scale 4)
      MaxPool 2×2
  → Bottleneck: [Conv3x3 → BN → ReLU] × 2  → 1024ch
  → Decoder (×4 scales):
      Bilinear 2× upsample + Conv1x1
      Concat skip connection from matching encoder scale
      [Conv3x3 → BN → ReLU] × 2
  → Output: Conv1x1 → 1ch (linear activation)
```

**Parameters:** ~8M | **VRAM:** ~4–6 GB at 512×512 | **Training:** ~3–4 hrs on V100

**Pros:**
- Simplest to implement — strong baseline with well-understood behavior
- Proven on dense regression tasks with 10k-scale datasets
- Skip connections preserve fine-grained spatial detail (critical for hotspot localization)
- Fast convergence; low debugging surface area
- Directly applicable: a U-Net variant is used in [recent chip thermal prediction work](https://www.sciencedirect.com/science/article/abs/pii/S1879239125003893)

**Cons:**
- No mechanism to suppress irrelevant features in skip connections — all encoder features flow through
- MSE loss encourages prediction averaging → over-smoothed thermal gradients
- Fixed-scale skip connections; decoder doesn't see coarser context when decoding fine details
- Bottleneck may not capture full chip-level global heat context

---

### Design 2: Attention U-Net

**Architecture overview:**
Same encoder-decoder structure as Classic U-Net, but each skip connection passes through an **Attention Gate** before concatenation:

```
Attention Gate at each skip:
  encoder_feat (fine, high-res)
  decoder_feat (coarse, low-res)  ← gating signal
  → Linear(encoder_feat) + Linear(upsample(decoder_feat))
  → ReLU → Linear → Sigmoid       ← soft spatial mask α ∈ [0,1]
  → output = encoder_feat * α     ← suppress irrelevant regions
  → concat with decoder feat as usual
```

**Parameters:** ~8.5M (+5% over U-Net) | **VRAM:** ~5 GB | **Training:** ~3.5–4.5 hrs on V100

**Pros:**
- Learns to focus on thermally-relevant regions (e.g., high-power functional units)
- Suppresses irrelevant low-power regions in skip connections — useful when power density is sparse
- Minimal compute overhead (~5% extra params, same memory class as U-Net)
- Interpretable: attention maps show which regions influence each prediction
- Good regularization effect; reduces overfitting risk on 10k datasets

**Cons:**
- Empirical gains are modest (0.6–2% metric improvement reported in literature)
- Attention may collapse during training if power maps are noisy
- Still single-scale decoder path — no multi-scale context for each decoder node
- Does not solve the blurred-output problem from MSE loss

Reference: [Attention U-Net (Schlemper et al., 2018)](https://arxiv.org/abs/1804.03999)

---

### Design 3: U-Net++ — Nested Dense Skip Connections ← Recommended

**Architecture overview:**
Replaces single skip connections with a **dense grid of nested pathways**. Each decoder node `X(i,j)` receives features from all encoder and intermediate nodes at shallower depths:

```
Encoder nodes:  X(0,0)  X(1,0)  X(2,0)  X(3,0)  X(4,0)
                  ↓        ↓       ↓       ↓
Nested nodes:  X(0,1)  X(1,1)  X(2,1)  X(3,1)
                  ↓        ↓       ↓
               X(0,2)  X(1,2)  X(2,2)
                  ↓        ↓
               X(0,3)  X(1,3)
                  ↓
               X(0,4)   ← final output

Each X(i,j):
  Concat [ X(i, 0..j-1), upsample(X(i+1, j-1)) ]
  → [Conv3x3 → BN → ReLU] × 2

Deep supervision:
  Auxiliary output head at X(0,1), X(0,2), X(0,3) — each trained with full loss
  Final output: X(0,4)
```

Upsampling: **Pixel Shuffle** (sub-pixel convolution) — fewer checkerboard artifacts than transposed conv.

**Parameters:** ~10M (+25% over U-Net) | **VRAM:** ~6–7 GB | **Training:** ~5–6 hrs on V100

**Pros:**
- Multi-scale awareness: decoder sees coarse global heat distribution AND fine local hotspot detail simultaneously
- Dense skip connections bridge semantic gap between encoder and decoder → sharper thermal gradients
- Deep supervision regularizes training; each scale produces independently supervised predictions
- Faster convergence via improved gradient flow through nested pathways
- Empirically: 2.8–3.9 point gain over U-Net in dense prediction benchmarks
- Pixel shuffle upsampling reduces artifacts → smoother temperature field predictions
- Shallow pathways can be pruned at inference for ~30% speedup with minimal accuracy loss

**Cons:**
- More complex to implement than U-Net (nested indexing, multiple intermediate loss heads)
- Higher memory during training (intermediate supervision outputs stored in graph)
- Deep supervision can overfit to noisy intermediate scales on small datasets
- Dense concatenation can dilute discriminative features with irrelevant ones (no gating)

Reference: [U-Net++ (Zhou et al., 2018/2019)](https://arxiv.org/abs/1912.05074)

---

### Design 4: EfficientNet-B4 Encoder + U-Net Decoder (Pretrained)

**Architecture overview:**
Replace the from-scratch encoder with **EfficientNet-B4** pretrained on ImageNet. Attach a standard U-Net decoder trained from scratch:

```
Input (2ch)
  → Channel adapter: Conv1x1 (2ch → 3ch)       ← match ImageNet input expectation
  → EfficientNet-B4 encoder (pretrained ImageNet, all layers fine-tuned):
      MBConv blocks at 5 resolution scales
      Feature maps at: 1/2, 1/4, 1/8, 1/16, 1/32 of input
  → Bottleneck: ~1792ch feature map (8×8 spatial at 256×256 input)
  → U-Net decoder (from scratch):
      Pixel Shuffle 2× upsample
      Concat skip from matching EfficientNet scale
      [Conv3x3 → BN → ReLU] × 2
      (×5 decoder stages)
  → Output: Conv1x1 → 1ch (linear activation)
```

**Parameters:** ~19M | **VRAM:** ~2.5 GB (depthwise separable convs are VRAM-efficient) | **Training:** ~5–7 hrs

**Pros:**
- ImageNet pretraining provides robust low-level features (edges, corners, lines) that partially transfer to chip layouts
- Faster convergence: 2–3× fewer epochs vs. from-scratch on comparable dataset sizes
- EfficientNet-B4 is parameter-efficient via compound scaling (optimal depth/width/resolution balance)
- Lower VRAM than U-Net++ despite larger parameter count — depthwise separable convolutions
- Best option if domain shift turns out to be manageable

**Cons:**
- **Domain shift risk (primary concern):** ImageNet features are learned from natural photos; chip layouts are geometric/schematic with no natural objects. Negative transfer is possible.
- Decoder still trained from scratch — only encoder benefits from pretraining
- Requires a channel adapter layer (2ch → 3ch) — small architectural mismatch
- Harder to debug: pretrained encoder may encode irrelevant ImageNet patterns
- Evidence from out-of-domain transfer learning suggests ImageNet pretraining can hurt when input domain diverges significantly

**Recommended use:** Implement as an ablation alongside Design 3. If EfficientNet-B4 outperforms U-Net++, ImageNet low-level features do transfer. If not, from-scratch wins.

Reference: [TernausNet (Iglovikov & Shvets, 2018)](https://arxiv.org/abs/1801.05746), [Transfer Learning for Segmentation (2022)](https://arxiv.org/abs/2207.14508)

---

### Loss Function

**Recommended: L1 + SSIM composite**

```
Loss = 0.7 × L1(pred, target) + 0.3 × (1 − SSIM(pred, target))
```

- **L1** preserves absolute temperature accuracy (critical for thermal maps); avoids MSE-induced blurring
- **SSIM** preserves spatial structure and thermal gradients (perceptual quality of heatmap)
- Start with α = 0.7/0.3; tune as ablation

Alternatives to benchmark: pure MSE (lower bound baseline), pure L1, Smooth L1 (Huber)

---

### Upsampling Strategy

**Recommended: Pixel Shuffle (sub-pixel convolution)**

```
PixelShuffle block:
  Conv2d(in_ch, out_ch × 4, kernel=3, padding=1)
  PixelShuffle(upscale_factor=2)    ← reshape [C×4, H, W] → [C, 2H, 2W]
  Conv2d(out_ch, out_ch, kernel=3, padding=1)   ← optional refinement
```

- Fewer checkerboard artifacts vs. transposed convolution
- Sharper outputs vs. bilinear interpolation (better for regression)
- No learnable upsampling artifacts — important for smooth thermal gradient reconstruction

---

### Architecture Recommendation Summary

| Design | Params | VRAM | Key Advantage | Key Risk | Priority |
|--------|--------|------|---------------|----------|----------|
| Classic U-Net | ~8M | 4–6 GB | Simple, proven baseline | Blurred outputs, no feature selection | Implement first |
| Attention U-Net | ~8.5M | ~5 GB | Spatial focus, interpretable | Modest gains (~1%) | Optional ablation |
| **U-Net++** | **~10M** | **6–7 GB** | **Multi-scale, sharper outputs, regularized** | **More complex** | **Primary model** |
| EfficientNet-B4 + decoder | ~19M | ~2.5 GB | Fast convergence | Domain shift from ImageNet | Ablation vs U-Net++ |

**Implementation order:**
1. Classic U-Net — establish baseline metrics (SSIM, RMSE, hotspot recall)
2. U-Net++ — primary model; expect 3–5% SSIM gain over baseline
3. EfficientNet-B4 — ablation to determine whether ImageNet transfer helps
4. Attention U-Net — only if U-Net++ underperforms expectations
