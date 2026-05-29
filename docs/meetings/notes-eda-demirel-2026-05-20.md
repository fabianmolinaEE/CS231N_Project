---
title: Meeting Notes — Eda Deniz Demirel
date: 2026-05-20
attendees: Fabian Molina, Ruben Carrazco, Eda Deniz Demirel
type: meeting
---

# Meeting Notes — Eda Deniz Demirel (2026-05-20)

## Dataset

- Finding a suitable dataset is a core challenge in this space — most thermal datasets are not public or not clean.
- **Dennis Rich** may have sweep results from prior work, but they are not cleaned up. Worth following up to see if they are usable.

## Physics-Informed Direction

The main recommendation from Eda: consider incorporating the heat equation directly into the model rather than relying purely on data-driven regression.

### Motivation
- A physics-informed approach reduces dependency on large labeled datasets.
- The heat equation constrains the solution space — the model cannot produce physically implausible thermal maps.

### What This Could Look Like
- **Loss function:** compute loss from heat flux or temperature gradient (not just pixel-wise MSE against ground truth).
- **Die thickness compensation:** incorporate heat loss through die thickness as an additional term to compensate for model inaccuracy at boundaries.

### Novelty framing
- Challenges in thermal prediction are well-documented in the literature.
- **Our novelty:** combining image-based input representation with physics-backed loss — this is a gap in the existing work.
- Reducing scope is encouraged given time constraints — we do not need a high-fidelity simulator to make the claim.

## Ground Truth / Simulators

### HotSpot
- Still the best practical option for ground truth label generation at our scale.
- Open-source, fast, and well-validated for architecture-level thermal modeling.

### MAPDL (Ansys)
- Commercial tool, more numerically accurate.
- Models chip as stacked rectangular dielectric blocks — not an actual circuit; extracts a power map from a circuit representation.
- Overkill for CS231N scope.

### Other simulators to be aware of
- **COMSOL** — general-purpose FEM thermal simulator, high fidelity.
- **PACKED** — thermal simulator from Ayse Coskun's group (Boston University). Used in their transient thermal predictor work. Relevant if we want to compare against their approach.

## Parameters to Sweep

If using a simulator for label generation, the key parameters to vary are:
- Die thickness
- Die size
- Power dissipation magnitude and distribution

Eda's view: if the simulator is good enough (HotSpot in grid mode), sweeping these parameters across a reasonable range should produce sufficient training diversity.

Practical simplification: **assume random power distribution** across tiles — avoids needing real workload traces and is consistent with what the Ansys CoAEMLSim paper does.

## Key People / Resources

| Name | Relevance |
|---|---|
| **Erick Pop** (Stanford) | Major figure in multi-scale thermal modeling — worth reading his work for physics background |
| **Ayse Coskun** (Boston University) | Advisor on the PACKED project; her group built a transient thermal predictor using PACKED simulations |
| **Dennis Rich** | May have unsanitized sweep datasets from prior thermal work |

## Tools / Frameworks

- **Chipyard** — open-source framework for chip design; give it a design recipe and it outputs RTL. Potentially useful for generating diverse chip layouts if CircuitNet coverage is insufficient.
- **Logic synthesis** — mentioned as part of the design flow context (RTL → gate-level netlist → placement).

## 2D → 3D Extension

Eda noted that extension to 3D is a natural direction and the physics shifts in predictable ways:

- Power source expands into a volume (not a 2D plane).
- 3D chips have more material → absolute temperature values are higher but gradients are shallower (less steep hotspot peaks).
- Some tolerance parameters change between 2D and 3D.
- **Takeaway:** a 2D model is a reasonable starting point; 3D is a clear future extension with well-understood physics differences.

## Core Problem / Solution Framing

- **Problem:** the parameter space for chip thermal behavior is extensive (geometry, materials, power, cooling).
- **Solution goal:** a model that predicts thermal behavior reasonably well across that space without running a full simulation for every point.

## Action Items

- [ ] Follow up with Dennis Rich about his sweep dataset — ask if it can be shared or used for validation
- [ ] Look into PACKED simulator and Ayse Coskun's transient thermal predictor paper
- [ ] Look into Erick Pop's multi-scale thermal work for physics background
- [ ] Explore Chipyard as a potential source of additional diverse layouts
- [ ] Decide whether to add a physics-informed loss term (heat flux / temperature gradient) to the U-Net training objective
- [ ] Confirm random power distribution assumption is acceptable for label generation scope
- [ ] Define the parameter sweep range for die thickness and power dissipation in HotSpot config

## Links

- [[sources/project-milestone1]] — open TODOs still relevant after this meeting
- [[literature/eda-pre-call-notes]] — pre-call preparation and CoAEMLSim paper summary
- [[datasets/circuitnet-n14]] — primary dataset reference
