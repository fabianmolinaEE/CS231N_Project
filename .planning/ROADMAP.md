# Roadmap: CS231N Thermal Map Prediction

**5 phases** | **28 requirements mapped** | All v1 requirements covered ✓

## Overview

| # | Phase | Goal | Requirements | Owner |
|---|-------|------|-------------|-------|
| 1 | Data Pipeline | Working DataLoader that feeds training | DATA-01 – DATA-09 | TBD |
| 2 | Baseline Models | Lower-bound baselines trained and evaluated | BASE-01 – BASE-04 | TBD |
| 3 | U-Net | Primary model beats baselines | UNET-01 – UNET-05 | TBD |
| 4 | Evaluation + Ablations | Full test-set metrics, ablations, benchmarks | EVAL-01 – EVAL-05, ABL-01 – ABL-03 | Both |
| 5 | Analysis + Report | Final report, poster, and visualizations | RPT-01 – RPT-05 | Both |

---

## Phase 1 — Data Pipeline

**Goal:** A working PyTorch DataLoader that loads (floorplan, power-map, thermal-label) tuples from CircuitNet 2.0, with HotSpot labels generated, a clean train/val/test split, and a sanity-check notebook confirming the data looks right.

**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07, DATA-08, DATA-09

**Success criteria:**
1. `DataLoader` yields batches of (2-channel input, 1-channel label) tensors without errors
2. A sanity-check notebook shows 3+ input/label pairs where the thermal map visually corresponds to the power map
3. HotSpot label generation script runs end-to-end without manual steps
4. Train/val/test split is documented and design-family leakage is addressed
5. Dataset size, resolution, and normalization are logged and reproducible

**Owner:** TBD (pipeline partner)

---

## Phase 2 — Baseline Models

**Goal:** Two trained baselines (flat CNN and encoder-decoder without skip connections) with RMSE and SSIM logged on the val set, establishing the lower bound for the U-Net to beat.

**Requirements:** BASE-01, BASE-02, BASE-03, BASE-04

**Success criteria:**
1. Flat CNN baseline trains to convergence and val metrics are logged
2. Encoder-decoder baseline trains to convergence and val metrics are logged
3. Both baselines use the same training loop, loss function config, and checkpoint saving
4. A comparison table of both baselines exists with RMSE and SSIM

**Owner:** TBD (model partner)

**Note:** Can be started in parallel with Phase 1 using synthetic/dummy data; switch to real data once Phase 1 is complete.

---

## Phase 3 — U-Net

**Goal:** A trained U-Net with skip connections that outperforms both Phase 2 baselines on RMSE and SSIM on the val set. Supports plain CNN and ResNet-50 encoder variants.

**Requirements:** UNET-01, UNET-02, UNET-03, UNET-04, UNET-05

**Success criteria:**
1. U-Net (plain CNN encoder) trains and beats the encoder-decoder baseline on SSIM
2. ResNet-50 encoder variant trains and is compared against the plain CNN encoder variant
3. Loss function is configurable; the best combination is selected and documented
4. Best U-Net checkpoint is saved and can be loaded for evaluation
5. Training config (LR, batch size, epochs, loss weights) is logged reproducibly

**Owner:** TBD (model partner)

---

## Phase 4 — Evaluation + Ablations

**Goal:** Full test-set evaluation of all models (CNN baseline, encoder-decoder, U-Net variants), hotspot localization metrics, inference benchmarks, and ablation results — everything needed to fill the experiments section of the report.

**Requirements:** EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, ABL-01, ABL-02, ABL-03

**Success criteria:**
1. RMSE, MSE, and SSIM are computed on the held-out test set for all models
2. Hotspot localization precision/recall is computed on the top 5% hottest pixels for all models
3. Inference time is benchmarked on the same hardware for all models
4. Ablation table shows impact of: skip connections, encoder depth, loss function choice
5. Qualitative heatmaps are generated for all models on the same test designs

**Owner:** Both

---

## Phase 5 — Analysis + Report

**Goal:** A complete final report and poster covering motivation, data, method, experiments, and conclusion — with architecture diagrams, GradCAM/saliency analysis, failure case documentation, and a metrics comparison table.

**Requirements:** RPT-01, RPT-02, RPT-03, RPT-04, RPT-05

**Success criteria:**
1. Architecture diagram clearly shows the full pipeline (input → encoder → bottleneck → decoder → heatmap)
2. GradCAM or saliency maps identify which layout features drive thermal predictions
3. At least 3 failure cases are documented with explanations of why the model failed
4. Metrics comparison table covers all models and ablations
5. Final report draft covers all required sections: intro, related work, data, method, experiments, conclusion

**Owner:** Both

---

## Work Split

| Track | Phases | Assign to |
|-------|--------|-----------|
| Data pipeline + HotSpot labels | Phase 1 | TBD |
| Baselines + U-Net implementation | Phase 2, 3 | TBD |
| Evaluation, analysis, report | Phase 4, 5 | Both |

Update this table once you and Ruben agree on the split.
