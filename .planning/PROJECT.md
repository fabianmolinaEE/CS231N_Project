# CS231N Thermal Map Prediction

## What This Is

A vision-based deep learning system that predicts full-chip GPU thermal maps from post-placement floorplan and power-density images. Given a 2-channel chip-layout image, the model produces a dense temperature heatmap in milliseconds — replacing expensive thermal simulators (HotSpot, ANSYS) for use during early-stage chip-design iteration. Built as a Stanford CS231N course project by Fabian Molina and Ruben.

## Core Value

Fast, accurate hotspot localization: the model must place thermal hotspots in the right regions of the chip, reliably, before a designer has to wait for a full simulation.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Data pipeline loads CircuitNet 2.0 floorplan + power-map pairs as tensors
- [ ] HotSpot label generation is automated and reproducible at scale
- [ ] Train/val/test split is defined and prevents design-family leakage
- [ ] Simple CNN baseline is trained and evaluated (lower-bound reference)
- [ ] Basic encoder-decoder (no skip connections) is trained and evaluated
- [ ] U-Net with skip connections is trained and evaluated as the primary model
- [ ] RMSE and SSIM are computed on the held-out test set for all models
- [ ] Hotspot localization accuracy is measured on the top 5% hottest pixels
- [ ] Inference time is benchmarked on the same hardware for all models
- [ ] Qualitative heatmap visualizations are produced (input → predicted → ground truth)
- [ ] Ablation study covers skip connections, encoder depth, and loss function choice
- [ ] Final report includes architecture diagram, metrics table, ablations, and failure analysis

### Out of Scope

- Real-time / interactive thermal simulation — inference speed target is <100ms per design, not true real-time
- Post-silicon thermal measurement — inputs are layout images, not silicon temperature sensors
- 3D-IC / multi-die thermal modeling — scoped to single-chip 2D layouts
- PINN (physics-informed) baseline — formulation mismatch with image-to-image regression; defer unless time allows
- pix2pix GAN baseline — optional stretch goal only; not required for the paper

## Context

- Dataset: CircuitNet 2.0 — 10,000+ preprocessed chip designs (CPU, GPU, AI accelerator layouts), public
- Label tool: HotSpot thermal simulator in grid mode — fast enough for large-scale label generation
- Architecture template: U-Net — spatially dense prediction, skip connections preserve hotspot detail
- Closest prior work: "Encoder-Decoder Networks for Analyzing Thermal and Power Delivery Networks" (ACM 2022)
- Course feedback emphasized: explicit baselines, ablations, and a clear validation strategy
- Success targets from proposal: SSIM ≥ 0.80, hotspot localization accuracy on top 5%, inference < 100ms

## Constraints

- **Timeline**: CS231N course project — final report and poster due end of Spring 2026 quarter
- **Team size**: Two people (Fabian + Ruben) — scope must be achievable by a team of two
- **Compute**: Stanford AIP / course-provided GPU access — no private cloud budget
- **Framework**: PyTorch — standard for CV research, required by course environment
- **Reproducibility**: All experiments must be reproducible from the repo (fixed seeds, logged configs)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| U-Net as primary model | Spatially dense task; skip connections preserve hotspot detail better than flat CNN | — Pending |
| CircuitNet 2.0 as dataset | Public, large-scale, includes GPU/accelerator layouts; feasible for course timeline | — Pending |
| HotSpot for labels | Open-source, fast in grid mode, grounded in established thermal modeling practice | — Pending |
| SSIM as headline metric | Captures spatial heatmap similarity better than pixel-wise RMSE alone | — Pending |
| Image-to-image regression framing | Leverages CNN spatial inductive bias; simpler than GAN or operator-learning approaches | — Pending |

---
*Last updated: 2026-05-17 after initialization*
