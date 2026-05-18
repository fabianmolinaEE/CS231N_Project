---
title: Project Milestone 1
type: source
sources: 1
updated: 2026-05-12
---

# Project Milestone 1 — Problem + Related Work

This note fills out the CS231N milestone 1 requirements using the submitted proposal as the source of truth.

Source note: [[project-proposal]]

## Project Summary

### Problem Statement
We are predicting full-chip GPU thermal maps from post-placement floorplan and power-map images. The goal is to learn a fast vision model that approximates thermal simulation outputs well enough to support early-stage chip-design iteration.

Why this matters:
- thermal hotspots can cause throttling, reduced performance, and reliability issues
- simulation tools are accurate but slow and expensive
- a learned predictor could give designers fast feedback during iteration

Scope:
- input is a structured chip-layout image representation, not raw silicon imagery
- target domain is GPU and AI accelerator designs
- output is a dense thermal heatmap, not a single scalar temperature
- the project is framed as image-to-image regression

## Data

### Planned Dataset
- Primary dataset: CircuitNet 2.0
- Expected data types: post-placement floorplans, power-density maps, and associated chip-design metadata
- Feasibility note: CircuitNet 2.0 is public and large enough to support training + validation

### Labels
Thermal ground-truth maps will be generated with HotSpot in grid mode by pairing each floorplan/power-map input with a simulated thermal output.

### Data TODOs
- [ ] Confirm the exact CircuitNet 2.0 subset we will use
- [ ] Verify how floorplan and power-map channels are aligned spatially
- [ ] Decide whether to train on all available designs or a filtered GPU/accelerator subset
- [ ] Check train/val/test split strategy and whether splits should be by design family
- [ ] Confirm HotSpot label-generation settings and grid resolution
- [ ] Estimate total label-generation time and storage requirements

## Model / Method

### Proposed Approach
The main model is a U-Net-style encoder-decoder CNN for dense prediction.

Design ideas from the proposal:
- U-Net encoder-decoder backbone
- skip connections to preserve spatial detail
- optional ResNet-50 encoder
- image-to-image regression formulation

### Why this model fits
This is a spatially dense prediction problem, so convolutional inductive bias is a good match. U-Net is especially appealing because skip connections can preserve the fine-grained layout details that matter for hotspot prediction.

### Method TODOs
- [ ] Decide on the exact backbone: plain U-Net, ResNet-U-Net, or another encoder-decoder variant
- [ ] Decide whether to predict raw temperature values or normalized heatmaps
- [ ] Define the loss function(s): MSE, L1, SSIM loss, or a combination
- [ ] Decide if we need post-processing for hotspot localization
- [ ] Plan a saliency / interpretability visualization

## Success Criteria

### Qualitative success
- predicted heatmaps should place hotspots in the right regions
- the model should recover the overall shape of the thermal distribution
- failure cases should be understandable rather than noisy or unstable

### Quantitative success
- RMSE / MSE should improve over simpler baselines
- SSIM should be reasonably high for heatmap similarity
- hotspot localization for the top 5% hottest regions should be accurate
- inference should be much faster than full HotSpot simulation

### Success TODOs
- [ ] Define one primary metric that we will optimize for in the writeup
- [ ] Decide whether hotspot localization or SSIM is the headline metric
- [ ] Set a concrete inference-time target and measure it on the same hardware for all models

## Evaluation Plan

### Metrics
- RMSE or MSE on temperature values
- SSIM for predicted vs. true heatmaps
- hotspot localization accuracy, especially on the top 5% hottest regions
- inference time per design

### Baselines
Feedback on the proposal suggested making baselines more explicit. Good candidate baselines:  
- HotSpot simulator as the reference point for label generation / comparison
- a simpler CNN regressor
- a basic encoder-decoder without skip connections
- pix2pix-style image-to-image translation
- a physics-informed / PINN-style approach if the formulation makes sense
- a ResNet-encoder variant of U-Net

### Validation strategy TODOs
- [ ] Define the exact validation split and whether we need cross-validation
- [ ] Decide whether to compare against ablations or only against full baselines
- [ ] Define a fair benchmarking protocol for runtime comparisons
- [ ] Decide which error maps / qualitative plots to include in the final report
- [ ] Decide whether to evaluate hotspot ranking quality in addition to pixel error

## Related Work

### 1) CircuitNet and CircuitNet 2.0
CircuitNet is the dataset foundation for this project. CircuitNet 2.0 provides realistic layout representations that make the thermal-prediction task feasible. The project differs because it targets thermal regression rather than generic ML-for-CAD prediction.

### 2) Encoder-Decoder Networks for Analyzing Thermal and Power Delivery Networks
This is the closest structural predecessor. It also uses an encoder-decoder approach for thermal-related prediction, which makes it a strong baseline and comparison point.

### 3) U-Net
U-Net is the architectural template for our proposed model. The skip connections are especially relevant because they preserve spatial detail for hotspot prediction.

### 4) DeepOHeat
DeepOHeat is relevant as a fast surrogate for thermal simulation. It is similar in motivation, but uses operator learning rather than the CNN image-to-image formulation we are proposing.

### 5) Temperature-Aware Microarchitecture / HotSpot
HotSpot is the classical thermal simulator used to generate labels and serve as the main simulation baseline. It gives us the target behavior we want to approximate quickly.

## Related Articles to Read

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — the core encoder-decoder / skip-connection baseline
- [pix2pix: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — a strong image-to-image baseline suggested in feedback
- [CircuitNet 2.0: An Advanced Dataset for Promoting Machine Learning Innovations in Realistic Chip Design Environment](https://openreview.net/forum?id=H1z7m3Kc7S) — dataset context for the project
- [Encoder-Decoder Networks for Analyzing Thermal and Power Delivery Networks](https://dl.acm.org/doi/10.1145/3526115) — the closest thermal-prediction precedent
- [DeepOHeat: Operator Learning-Based Ultra-Fast Thermal Simulation in 3D-IC Design](https://doi.org/10.1109/DAC56929.2023.10247998) — an alternative fast-surrogate thermal approach
- [Temperature-Aware Microarchitecture](https://dl.acm.org/doi/10.1145/859618.859620) — background on HotSpot-style thermal modeling
- [Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561) — a useful reference if we decide to explore PINN-style baselines

## Project TODOs

- [ ] Finalize the exact problem statement in one sentence
- [ ] Lock the dataset subset and data split
- [ ] Run HotSpot label generation on a small sample first
- [ ] Implement a simple CNN baseline before U-Net
- [ ] Implement U-Net and a ResNet-encoder variant
- [ ] Compare against pix2pix if time permits
- [ ] Decide whether a PINN baseline is actually practical for this formulation
- [ ] Create a figure showing input channels → model → output heatmap
- [ ] Draft the evaluation section with concrete metrics and plots
- [ ] Collect a small set of qualitative examples for the final report

## Helpful Links

- [[project-proposal]] — source note for the submitted proposal
- [[lecture09-detection-segmentation-visualization]] — good background for dense prediction framing, segmentation-style metrics, and output structure
- [[cnn-architectures]] — useful for CNN and ResNet encoder background
- [[attention-transformers]] — only relevant if we later explore transformer-based backbones
