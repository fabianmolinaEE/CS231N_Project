# Requirements: CS231N Thermal Map Prediction

**Defined:** 2026-05-17
**Core Value:** Fast, accurate hotspot localization from chip-layout images — before a designer waits for a full simulation

## v1 Requirements

### Data Pipeline

- [ ] **DATA-01**: CircuitNet 2.0 dataset is downloaded and the directory structure is documented
- [ ] **DATA-02**: Floorplan and power-map channels are verified to be spatially aligned
- [ ] **DATA-03**: A GPU/accelerator-only subset filter is defined and applied (or full dataset is used with justification)
- [ ] **DATA-04**: HotSpot label generation runs end-to-end on a sample of 5–10 designs in grid mode
- [ ] **DATA-05**: Full label generation is automated and reproducible (script in `scripts/generate_labels.sh`)
- [ ] **DATA-06**: Train / val / test split is defined, documented, and split by design family to prevent leakage
- [ ] **DATA-07**: PyTorch `Dataset` class loads (floorplan, power-map, thermal-label) tuples correctly
- [ ] **DATA-08**: `DataLoader` supports batching with horizontal/vertical flip augmentation
- [ ] **DATA-09**: Sanity-check notebook visualizes 3–5 input/label pairs side by side

### Baselines

- [ ] **BASE-01**: Flat CNN regressor (no encoder-decoder structure) is implemented, trained, and evaluated
- [ ] **BASE-02**: Basic encoder-decoder without skip connections is implemented, trained, and evaluated
- [ ] **BASE-03**: Both baselines log RMSE and SSIM on the validation set
- [ ] **BASE-04**: Training infrastructure is shared: loss logging, checkpoint saving, LR scheduler

### Model — U-Net

- [ ] **UNET-01**: U-Net encoder-decoder with skip connections is implemented in PyTorch
- [ ] **UNET-02**: Plain CNN encoder variant is supported
- [ ] **UNET-03**: ResNet-50 pretrained encoder variant is supported as an option
- [ ] **UNET-04**: Loss function is configurable: MSE, L1, or weighted MSE + SSIM combo
- [ ] **UNET-05**: U-Net is trained and beats both baselines on RMSE and SSIM on the val set

### Evaluation

- [ ] **EVAL-01**: RMSE and MSE are computed on the held-out test set for all models
- [ ] **EVAL-02**: SSIM is computed on the held-out test set for all models
- [ ] **EVAL-03**: Hotspot localization accuracy is measured — precision/recall on the top 5% hottest pixels
- [ ] **EVAL-04**: Inference time is benchmarked on the same hardware for all models
- [ ] **EVAL-05**: Qualitative heatmap visualizations are produced for all models (input → predicted → ground truth)

### Ablations

- [ ] **ABL-01**: Skip connections on vs. off ablation is run and reported
- [ ] **ABL-02**: Encoder depth (shallow vs. deep) ablation is run and reported
- [ ] **ABL-03**: Loss function choice ablation is run and reported (MSE vs. L1 vs. MSE+SSIM)

### Analysis + Report

- [ ] **RPT-01**: Saliency / GradCAM analysis identifies which layout features drive thermal predictions
- [ ] **RPT-02**: At least 3 failure cases are documented with explanations
- [ ] **RPT-03**: Architecture diagram is created (input channels → encoder → bottleneck → decoder → heatmap)
- [ ] **RPT-04**: Full metrics comparison table covers CNN baseline, encoder-decoder, U-Net, and ablations
- [ ] **RPT-05**: Final report writeup covers intro, data, method, experiments, conclusion

## v2 Requirements

### Optional Baselines

- **V2-01**: pix2pix conditional GAN baseline — strong image-to-image baseline, high implementation cost
- **V2-02**: PINN-style physics-informed baseline — requires reformulation of the problem

### Deployment / Demo

- **V2-03**: Interactive demo that takes a layout image and shows the predicted heatmap
- **V2-04**: Integration with an EDA tool for inline thermal prediction

## Out of Scope

| Feature | Reason |
|---------|--------|
| Post-silicon thermal measurement | Inputs are layout images, not hardware sensors |
| 3D-IC / multi-die thermal modeling | Scope limited to 2D single-chip layouts |
| Real-time streaming thermal prediction | Target is per-design inference, not a live stream |
| Mobile / edge deployment | Compute target is GPU workstation |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 to DATA-09 | Phase 1 | Pending |
| BASE-01 to BASE-04 | Phase 2 | Pending |
| UNET-01 to UNET-05 | Phase 3 | Pending |
| EVAL-01 to EVAL-05 | Phase 4 | Pending |
| ABL-01 to ABL-03 | Phase 4 | Pending |
| RPT-01 to RPT-05 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 28 total
- Mapped to phases: 28
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-17*
*Last updated: 2026-05-17 after initial definition*
