---
title: Project-Proposal
type: source
sources: 1
updated: 2026-05-12
---

# CS231N Project Proposal

Source: `raw/courses/cs231n/cs231n_proposal.pdf`.
Pages summarized: 1-2.

## Project Overview

The proposal describes a vision-based method for predicting full-chip GPU thermal maps from post-placement floorplan and power-map images. The core idea is to replace or accelerate expensive thermal simulation workflows with a convolutional neural network that can infer a dense temperature map directly from layout features.

The motivation is that modern GPUs and AI accelerators have very high power density, and thermal hotspots can cause throttling, reduced performance, and reliability issues. Existing tools such as ANSYS, Cadence, and detailed thermal simulators are too slow or expensive for rapid design iteration.

## Problem Definition

### What is being predicted
- Input: a two-channel image composed of a post-placement floorplan and a power-density map
- Output: a single-channel thermal heatmap over the chip layout

### Why this problem matters
- Thermal analysis is a major bottleneck in chip design
- Early prediction can help identify hotspots before expensive redesigns
- Fast inference could make thermal-aware optimization more practical during design iteration

### Scope
- The project focuses on post-placement chip layouts rather than post-silicon measured temperatures
- The target domain is GPU and AI accelerator designs, using CircuitNet-style representations
- The method is framed as image-to-image regression over structured chip-layout data

## Data Plan

### Dataset
The proposal plans to use CircuitNet 2.0, a public dataset with 10,000+ preprocessed chip designs, including CPU, GPU, and AI accelerator examples with floorplan and power-related inputs.

### Label generation
Thermal ground-truth maps will be generated with the open-source HotSpot thermal simulator in grid mode. The proposed workflow is:
1. take the floorplan and power-map inputs,
2. run HotSpot to compute thermal labels,
3. pair each input design with its corresponding thermal map.

### Data feasibility
The proposal argues that HotSpot is fast enough for large-scale label generation, making the project feasible without requiring proprietary simulation tools.

## Method

The proposed model is a U-Net-style encoder-decoder CNN for dense prediction.

### Architecture ideas
- U-Net encoder-decoder backbone
- Skip connections to preserve spatial detail
- Optional ResNet-50 encoder
- Image-to-image regression formulation

### Why this model
The task is spatially dense and benefits from convolutional inductive bias. The proposal treats thermal prediction like a structured image translation problem rather than a scalar regression problem.

## Evaluation Plan

### Quantitative metrics
The proposal lists several evaluation criteria:
- RMSE or mean squared error on temperature values
- SSIM for similarity of predicted vs. true heatmaps
- Hotspot localization accuracy, especially for the top 5% hottest regions

### Baselines
- HotSpot simulator as the accuracy/latency reference point
- Simpler CNN or encoder-decoder variants without skip connections
- Potentially different encoder choices, such as ResNet-50

### Practical metric
- Inference time target: under 100 ms per design

### Qualitative analysis
- Visual inspection of predicted heatmaps
- Saliency analysis to understand what layout features influence thermal output
- Failure-case visualization

## Related Work

### 1) CircuitNet and CircuitNet 2.0
The proposal uses CircuitNet as the main dataset foundation. CircuitNet 2.0 provides the realistic layout representations needed for this vision-style prediction task. The project is related because it builds directly on the dataset, but differs in that it focuses on thermal map prediction rather than generic ML-for-CAD tasks. ([1], [6])

### 2) Encoder-Decoder Networks for Analyzing Thermal and Power Delivery Networks
This paper is closely aligned with the proposed modeling strategy because it also uses an encoder-decoder approach for thermal-related prediction tasks. The project borrows the same broad architecture family, but applies it to a different dataset and to GPU / AI accelerator post-placement layouts. ([2])

### 3) U-Net
U-Net is the proposed model inspiration for dense thermal regression. Its skip connections are especially relevant because they help preserve fine spatial detail, which matters for localized hotspot prediction. The proposal adapts U-Net from biomedical segmentation to chip thermal mapping. ([9])

### 4) DeepOHeat
DeepOHeat explores operator-learning-based thermal simulation, which is relevant as a fast-surrogate alternative to conventional simulators. The project is similar in motivation, but differs in formulation: it uses a CNN image-to-image approach on CircuitNet-style inputs rather than operator learning for 3D-IC thermal simulation. ([7])

### 5) Temperature-Aware Microarchitecture / HotSpot
HotSpot is the classical thermal simulator used as the label source and baseline reference. It grounds the prediction target in established thermal modeling practice. The project is not trying to replace HotSpot analytically; it is trying to approximate its outputs much faster. ([10])

## Proposal Claim

The central claim is that a CNN-based image-to-image model can predict thermal maps from chip-layout representations quickly enough to be useful for thermal-aware design iteration, while maintaining meaningful accuracy on hotspot structure and temperature distribution.

## Helpful Links

- [[lecture09-detection-segmentation-visualization]] — useful for dense prediction framing, especially segmentation-style metrics and output structure.
- [[cnn-architectures]] — background for CNN backbones and residual encoders.
- [[attention-transformers]] — relevant if the project later experiments with transformer-based vision backbones.

## Notes from the Proposal

- The proposal explicitly frames the work as original CS231N project work.
- The intended venue / relevance target mentioned in the proposal is EDA-oriented publication venues such as DAC, ICCAD, or MLCAD.
- The claimed success target includes SSIM above 80% and sub-100 ms inference time.
- The proposal emphasizes that thermal hotspots are the main practical output of interest, not just overall pixel-wise error.

## Feedback on the Proposal

> "Hi Ruben and Fabian, this is a very-well motivated idea and proposal does a great job explaining the motivation! One way to make the project even stronger would be to make the baselines and validation strategy more explicit. For example, you could compare the U-Net model against simpler CNN / encoder-decoder baselines, pix2pix-style (https://arxiv.org/abs/1611.07004) image-to-image translation, or physics-informed (PINN) approaches, and then analyze where each method succeeds or fails. In addition, thinking of some ablations could improve the paper substantially. Overall great work! Excited to see how you execute this proposal!"

### Key takeaways from the feedback
- Make baselines more explicit, especially simpler CNN / encoder-decoder comparisons.
- Consider adding a pix2pix-style image-to-image translation baseline.
- Consider a physics-informed (PINN) baseline if it fits the formulation.
- Define the validation strategy more clearly.
- Include ablations to show which design choices matter most.
- The overall feedback was positive and emphasized that the motivation is already strong.
