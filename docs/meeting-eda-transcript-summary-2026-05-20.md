# Meeting Transcript Summary — Eda Deniz Demirel (2026-05-20)

**Attendees:** Fabian Molina, Ruben Carrazco, Eda Deniz Demirel (Stanford PhD, Prof. Mitra group)
**Source:** Ruben's structured summary of the recorded meeting transcript.

---

## Why This Meeting

We went in to reframe scope. After reviewing the literature Eda shared, we recognized the original project framing had limited novelty. Eda's first unprompted comment: *"It's hard to find a dataset you can rely on — the capabilities of your model will depend on your dataset eventually."*

---

## Original Framing and Why It Falls Short

**Original plan:** CircuitNet layouts → HotSpot labels → CNN predicts thermal maps faster than HotSpot.

**Problem (Fabian's own words in the meeting):** *"our model would essentially just be acting like HotSpot."* This approach has already been done in the literature. Minimal novelty. HotSpot is already a fast, abstracted approximation; replacing it with a neural approximation of itself doesn't solve a compelling problem.

---

## The Novelty Gap: Physics-Informed Loss

**Eda's recommendation:** The gap is combining **image-based inputs** with **physics-informed training**. Neither exists separately in the literature in combination for this problem.

**Loss formulation:**
```
L_total = λ_data · L_MSE(T_pred, T_label) + λ_phys · L_physics(T_pred)
L_physics = || k · ∇²T_pred + Q ||²
```

Implemented via differentiable Laplacian convolution on the predicted output — no simulation at training time. This constrains predictions to be physically plausible even for configurations outside the training distribution.

---

## Reframed Problem Statement

Eda's exact quote:

> *"The problem statement can focus on the extensiveness of the input parameters and how you can tackle not being able to train on all of them, but being able to predict them reasonably."*

**Core claim:** Train on a finite parameter sweep, generalize beyond it using physics-informed constraints.

**Engineering motivation:** Design-space exploration (e.g., simulated annealing) can require ~100,000 thermal evaluation steps. A fast neural predictor with physics-backed generalization fills this gap.

---

## Input Parameter Axes

Two categories:

| Axis | Parameters |
|------|-----------|
| Circuit / electrical | Topology, frequency, power distribution |
| Thermal / physical | Die thickness, die size, HTC, BEOL materials |

**Our approach:** Primary axis = circuit diversity (CircuitNet layouts) + narrow thermal sweep (die thickness, die size). Single HTC value acceptable for CS231N scope.

---

## Dataset

- **CircuitNet 2.0:** Layout diversity, lacks BEOL stack thermal properties. Usable.
- **Dennis Rich's data:** Raw sweep outputs, not curated. He did NOT offer it in the email — low confidence on availability.
- **Random power distribution:** Eda confirmed this is a valid simplification; papers exist that train thermal predictors this way.
- **No clean public thermal dataset with actual circuitry exists.**

---

## Simulator Decision

**HotSpot confirmed.** MAPDL's geometry representation (rectangular dielectric blocks stacked to approximate a chip) is not fundamentally different from HotSpot's abstraction. Neither simulates actual transistors or wires. MAPDL's only advantage is solver numerical accuracy. Overkill for CS231N.

**Focus:** Steady-state prediction only. Transient (handled by ML-PACT) is a harder, separate problem.

---

## Key Contacts

| Name | Affiliation | Why |
|------|------------|-----|
| Eric Pop | Stanford | Multi-scale thermal modeling background |
| Ayse Coskun | Boston University | PI behind PACT and ML-PACT — closest existing reference |
| Dennis Rich | Mitra group (Stanford) | May have sweep data — low confidence |

---

## Is This Worth Doing?

**Eda: Yes.** The IC prediction space is crowded, but combining image-based input + physics-informed training is underexplored, especially for OOD generalization. This is the differentiator.

---

## Post-Meeting Problem Framing

**Original:** Replicate HotSpot faster.
**Revised:** Train an image-based predictor with physics-informed loss that generalizes beyond the training parameter range — specifically to die thicknesses not seen during training.

**Novelty:**
1. Reduces labeled-data dependency via physical constraints in the loss
2. OOD generalization — can't produce thermally implausible outputs
3. HotSpot sufficient as ground truth (no commercial solver needed)
