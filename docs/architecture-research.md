# Architecture Research — GPU Thermal Map Prediction

## Task Framing
- **Input:** 2-channel image — (1) post-placement floorplan, (2) power-density map
- **Output:** 1-channel thermal heatmap (temperature per pixel)
- **Resolution:** TBD — determined by native CircuitNet 2.0 resolution (likely 256×256 or 512×512)
- **Scale:** ~10,000 training samples on Stanford HPC

---

## Design 1: Classic U-Net (Baseline)

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

## Design 2: Attention U-Net

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

## Design 3: U-Net++ — Nested Dense Skip Connections ← Recommended

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

## Design 4: EfficientNet-B4 Encoder + U-Net Decoder (Pretrained)

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

## Loss Function

**Primary: Physics-informed composite loss**

The 2D steady-state heat equation: `k · ∇²T + Q = 0`

Where `T` is the predicted temperature map, `Q` is the power-density input channel, and `k` is silicon thermal conductivity (~149 W/m·K). The Laplacian is computed via a fixed (non-learned) 3×3 convolution:

```
Laplacian kernel:  0   1   0
                   1  -4   1
                   0   1   0
```

Training objective:

```
L_total = λ_data · L_MSE(T_pred, T_label) + λ_phys · mean(|| k · ∇²T_pred + Q ||²)
```

Fully differentiable. No simulation at training time. The physics term penalizes predictions that violate the heat equation regardless of whether a ground-truth label exists — this is what helps OOD generalization.

**Baseline (comparison):** MSE-only (`λ_phys = 0`)

**Ablation sweep:**
- `λ_phys` values (log sweep: 0, 0.01, 0.1, 1.0)
- Physics loss in normalized vs. physical units
- L1 vs. MSE for the data term

---

## Upsampling Strategy

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

## Architecture Recommendation Summary

| Design | Params | VRAM | Key Advantage | Key Risk | Priority |
|--------|--------|------|---------------|----------|----------|
| **Classic U-Net + physics loss** | **~8M** | **4–6 GB** | **Physics constraint → OOD generalization** | **Physics loss tuning** | **Primary model** |
| Classic U-Net (MSE-only) | ~8M | 4–6 GB | Simple baseline | Blurred outputs, no physics | Implement first |
| Encoder-decoder (no skip) | ~5M | ~3 GB | Ablation: skip connection value | Weaker performance expected | Ablation |
| Flat CNN regressor | ~3M | ~2 GB | Lower bound | No spatial structure | Ablation |
| Attention U-Net | ~8.5M | ~5 GB | Spatial focus, interpretable | Modest gains | Optional if time permits |
| U-Net++ | ~10M | 6–7 GB | Multi-scale, sharper outputs | More complex, secondary concern | Deprioritized |

**Implementation order:**
1. Flat CNN regressor + encoder-decoder — lower bounds
2. Classic U-Net with MSE-only loss — primary baseline
3. Classic U-Net with physics loss (`L_total = λ_data · L_MSE + λ_phys · L_physics`) — primary model
4. OOD evaluation: compare degradation on unseen die configs between steps 2 and 3
5. Attention U-Net / U-Net++ — only if physics-constrained U-Net underperforms expectations

The novelty is the physics constraint, not the architecture complexity. Keep the backbone simple so the loss function effect is clearly attributable.
