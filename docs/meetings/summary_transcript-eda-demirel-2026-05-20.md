---
title: Meeting Summary (Transcript-Based) — Eda Deniz Demirel
date: 2026-05-20
attendees: Fabian Molina, Ruben Carrazco, Eda Deniz Demirel
type: meeting-summary
raw-transcript: transcript-eda-demirel-2026-05-20.md
---

# Meeting Summary — Eda Deniz Demirel (2026-05-20)

Eda is a PhD student under Prof. Mitra (Stanford), working on 3D thermal simulation and BEOL material optimization using PACT. She reached out after reviewing the project proposal submitted to Prof. Mitra.

---

## Why We Called This Meeting

Fabian and Ruben went into this meeting wanting to reframe scope. After reading the paper Eda shared and the related literature, we recognized that our original framing had limited novelty. Eda's very first comment, unprompted: **"It's hard to find a dataset you can rely on — the capabilities of your model will depend on your dataset eventually."** That set the tone for the whole conversation.

---

## Original Framing and Why It Falls Short

The original plan:
1. Pull layouts + power maps from CircuitNet 2.0
2. Run HotSpot on each to generate thermal labels
3. Train a CNN to reproduce those thermal maps faster than HotSpot

**Problem:** As Fabian put it directly in the meeting — *"our model would essentially just be acting like HotSpot."* The goal would be: can we match HotSpot accuracy with lower latency? Eda confirmed this general approach has already been done in the literature (similar to papers she sent the day before). It has minimal novelty on its own.

There's a deeper issue: HotSpot is already a fast, abstracted approximation of the physics. Replacing a fast approximation with a neural approximation of the same fast approximation doesn't solve a compelling problem — especially when HotSpot is already open-source and well-calibrated. The model would learn to approximate HotSpot's approximation, with no mechanism to generalize beyond what HotSpot was trained against.

---

## The Novelty Gap: Physics-Informed Loss

**Eda's core recommendation:** The gap in the literature is combining **image-based inputs** (floorplan + power map) with **physics-informed training**. Most existing work is one or the other:

- Purely data-driven image models: no physics constraints, heavily dataset-dependent, fail outside training distribution
- Physics-informed thermal neural networks: exist, but they use raw matrix/tensor inputs and compute heat equation residuals — not image-based inputs, not a CV approach

**What "physics-informed loss" means mechanically:**
Instead of computing loss as pixel-wise MSE or L1 between predicted and ground-truth thermal maps, compute the loss (or part of it) from **heat flux or temperature gradient residuals derived from the heat equation**. The model's predicted temperature field must satisfy the heat equation — if it doesn't, that's penalized directly in the loss, independent of whether you have a ground-truth label for that configuration.

**Why this matters for generalization:**
A physics-informed loss constrains predictions to be physically plausible even for configurations the model has never seen. Concretely: if you train only on one range of die thicknesses, the heat equation still holds at other thicknesses. The physics loss propagates that constraint into the model's weights, so when you apply it to a different thickness at inference, it doesn't hallucinate thermally implausible outputs the way a pure MSE model would.

This directly addresses the failure mode of the paper Eda shared: it degraded badly when transferred to die thicknesses and HTCs outside the training distribution. A physics-informed loss is a principled way to compensate for that.

---

## Reframed Problem Statement

Eda explicitly proposed this as the right problem to tackle:

> *"The problem statement can focus on the extensiveness of the input parameters and how you can tackle not being able to train on all of them, but being able to predict them reasonably."*

The parameter space for chip thermal behavior is continuous and enormous — power distribution, die thickness, die size, HTC, material conductivities. No model can be trained across all of it. The actual problem is: **train on a finite, defensible parameter sweep and still generalize reasonably beyond it** — using physics-informed constraints to fill the extrapolation gap.

This framing is more defensible, more novel, and tied to a real engineering need: design-space exploration. Eda's example: simulated annealing during chip design can require ~100,000 evaluation steps. Running a thermal simulator at each step is prohibitive; a fast neural predictor with physics-backed generalization fills that gap.

---

## Input Parameter Space

Eda identified two distinct categories of inputs relevant to thermal prediction:

**1. Electrical / circuit parameters**
- Circuit topology (what functional blocks are where)
- Operating frequency, voltage
- Power distribution across blocks

**2. Thermal / physical parameters**
- Die thickness
- Die size
- Material conductivities (especially BEOL stack materials)
- Heat transfer coefficient (HTC) — boundary condition for cooling

Your scope choice determines which axis you sweep and which you fix:

| Strategy | Circuit side | Thermal/material side |
|----------|-------------|----------------------|
| Eda's group | Fixed (1 CPU + 1 AI accelerator) | Swept (BEOL materials, die thickness, HTC) |
| Our option A | Swept (CircuitNet / Chipyard diversity) | Fixed |
| Our option B | Narrowly swept | Narrowly swept |

Eda's guidance: pick one axis of variation, define the parameter ranges clearly, and defend why the sweep is representative. You do not have to cover everything — the scope limitation is itself part of the problem statement.

---

## Dataset Situation

### CircuitNet 2.0
Eda checked it the day before the meeting (she hadn't seen it before). It's framed as a prediction benchmark for early EDA development. It provides layout diversity but lacks BEOL stack thermal properties — not ideal if thermal/material parameter sweep is the focus. Usable for circuit-diversity-focused experiments. The 2D-only scope is a known limitation.

### Dennis Rich (PhD student, Mitra group)
Has EDA flow sweep results that include thermal simulations. However:
- Not a clean dataset — raw sweep outputs, not curated training data
- **He did not offer it in the email thread.** This is a meaningful signal: if it were easily shareable or he thought it was usable, he would have offered. Don't count on this.
- Eda herself was uncertain whether he would have time to clean it up for external use.

### Eda's Own Group's Data
They use PACT for fast design-space-exploration simulations. Their circuit space is narrow: one CPU architecture + one AI accelerator, with floor planning variations. Their focus is BEOL material optimization, not circuit diversity. The data is not publicly available and not directly applicable to our framing.

### No Clean Public Thermal Dataset
Eda is not aware of any publicly available, clean thermal dataset with actual circuitry. Her own simulation research doesn't use real circuits — it simulates materials with resistive heaters as proxies for heat sources, not actual chip designs.

### Random Power Distribution (Valid Simplification)
Eda confirmed: papers exist that **randomly assign power values to square blocks** on a layout and train thermal predictors from that. She has reviewed several such papers. This avoids needing real workload traces entirely and is acceptable for the scope of this project — consistent with what the ANSYS CoAEMLSim paper does. If circuit-level power accuracy is not the focus, random power distribution is a reasonable simplification.

---

## Simulator Comparison

| Simulator | Type | Key Detail |
|-----------|------|------------|
| **HotSpot** | Open-source, block-level | Abstracted approximation, fast. Grid mode works well for label generation at our scale. Practical and well-cited. Fine for CS231N. |
| **MAPDL (ANSYS)** | Commercial | More numerically accurate solver algorithms. But geometry representation is nearly identical to HotSpot: stacked rectangular dielectric blocks, no actual wires or logic gates. The chip is not simulated — only a block-level power map is extracted from it, then fed into rectangular geometry blocks. Main advantage is solver accuracy, not geometric fidelity. Overkill for this project. |
| **COMSOL** | Commercial (FEM) | General-purpose finite-element thermal solver, high fidelity. Stanford may have a university license. Expensive to run at scale. |
| **PACT** | Open-source (Boston University) | Eda's group uses it for fast design-space exploration. Extendable to 3D. From Ayse Coskun's group (BU). |
| **ML-PACT** | Open-source (Boston University) | ML-based thermal predictor trained on PACT simulation data. Focuses on transient thermal simulation, not steady-state. Useful to read as a reference but not directly what we're building. |

**Critical clarification on MAPDL vs HotSpot:** We went in thinking MAPDL was fundamentally more capable because it has "3D features." Eda corrected this. MAPDL's geometry representation — rectangular dielectric blocks stacked to approximate a chip — is not very different from HotSpot's abstraction. Neither tool simulates actual transistors, wires, or logic gates. MAPDL's advantage is purely in solver numerical accuracy (different algorithms for solving the heat PDE), not in richer circuit representation. This means using HotSpot is not a meaningful methodological downgrade from the paper we were comparing against.

**Steady state vs. transient:** Eda recommended focusing on **steady-state** thermal prediction. It's simpler, more tractable for a short timeline, and the more common formulation in the literature. Transient is what ML-PACT handles — a harder, separate problem.

**Eda's bottom line:** *"As long as you use a well-cited open-source or well-known commercial tool and defend your dataset's parameter coverage, the simulator choice is fine."* This is a CS/CV class. You don't need sign-off-level fidelity and you don't need to simulate an actual layout.

---

## Chipyard as a Layout Generator

Eda introduced [Chipyard](https://github.com/ucb-bar/chipyard), an open-source RISC-V architecture and design framework. You give it a recipe — e.g., "CPU core + accelerator + this much memory" — and it produces an RTL-level design that can yield a floormap. Key properties:

- Generates diverse chip architecture configurations on demand
- Does not require exact power numbers to produce a floormap
- Could be used to build a more diverse training set than CircuitNet provides if CircuitNet's layout variety is insufficient
- Goes down to RTL level (not a full physical layout, but enough for a representative floormap)

Relevant if we go with the circuit-diversity approach and find CircuitNet too narrow.

---

## Is This Project Worth Doing?

We asked Eda directly: *"Just bluntly — do you think this is the right path? Is there novelty here?"*

**Her answer: Yes, with caveats.**

There is genuine engineering need for fast, physics-consistent thermal predictors. The use case she cited: design-space exploration (e.g., simulated annealing) can require ~100,000 evaluation steps. Running a simulator at each step is prohibitive.

The IC prediction space is crowded — thermal, IR drop, and latency prediction all have substantial ML literature. Novelty requires a clear differentiator. Eda's view: **the differentiator is combining image-based input representation with physics-informed training**. Neither approach alone is new; together they are underexplored, especially for the out-of-distribution generalization problem.

Since the project is not about material design or circuit design per se, we don't need a high-fidelity commercial simulator. The claim we're making — faster, physics-consistent thermal prediction from layout images — is achievable with HotSpot or PACT as the ground-truth generator.

---

## 3D Extension (Future Direction, Not In Scope)

We asked: do any 2D problems become irrelevant once you move to 3D? Are there 3D-specific challenges that make 2D work obsolete?

**How the physics changes in 3D:**
- A square power source in 2D produces a roughly circular temperature gradient as heat spreads laterally through the chip plane
- In a 3D IC, heat spreads into a volume: more material → **higher valley temperatures** (the baseline everywhere is hotter) but **shallower gradients** (hotspot peaks are less severe, more diffuse, because heat can escape in more directions)
- Some thermal tolerances shift as a result — 3D chips run hotter on average but with less extreme local hotspots

**The new challenge specific to 3D — tier coupling:**
The key problem that has no 2D analog is cross-tier thermal coupling: a hotspot in one die affects temperature distributions in neighboring tiers above and below it. This cannot be captured by stacking independent 2D predictions for each die — the tiers interact. Eda's group specifically studies this coupling as a function of BEOL material choices.

**What a 3D-capable model would need:**
- Volumetric input (image with a third spatial dimension)
- Some inter-tier communication mechanism (e.g., cross-attention between tier feature maps, or 3D convolutions)
- All simulators Eda mentioned (HotSpot, PACT, COMSOL) are extendable to 3D

**Assessment:** A clean future-work section for the paper. The 2D model is a valid starting point; 3D is the natural extension with well-understood physics differences. Out of scope for the 2.5-week sprint.

---

## People to Contact

| Name | Affiliation | Why |
|------|------------|-----|
| **Eric Pop** | Stanford | Ran a large multi-scale thermal modeling project. Good source of physics background. |
| **Ayse Coskun** | Boston University | PI behind PACT and ML-PACT. Her group's work is the closest existing analog to what we're building. Worth emailing about datasets or guidance. |
| **Dennis Rich** | Mitra group (Stanford) | Has unsanitized thermal sweep results. Low confidence on availability — he didn't volunteer it in the thread. |

---

## Action Items

- [ ] Decide on the physics-informed loss approach — Eda's strong recommendation; this is the core novelty claim
- [ ] Lock the parameter sweep axis: circuit diversity vs. thermal/material parameters (or a narrow slice of both)
- [ ] Define specific parameter ranges for the sweep: die thickness, power distribution, die size
- [ ] Confirm random power distribution as an acceptable label-generation simplification (it is)
- [ ] Read ML-PACT paper (Ayse Coskun's group) — closest reference to our approach
- [ ] Email Ayse Coskun about datasets or collaboration
- [ ] Email Eric Pop about multi-scale thermal modeling background
- [ ] Follow up with Dennis Rich about sweep data (low priority)
- [ ] Look into Chipyard if CircuitNet layout diversity is insufficient
- [ ] Reframe project problem statement around out-of-distribution generalization, not just accuracy vs. HotSpot
- [ ] Confirm focus on steady-state (not transient) thermal prediction

---

## Revised Problem Framing (Post-Meeting)

**Original:** Train a CNN to replicate HotSpot thermal maps from floorplan + power inputs, faster than HotSpot.

**Problem with original:** Replicating HotSpot with a neural net has been done. The model would learn to approximate HotSpot's approximation with no mechanism to generalize beyond the training distribution. Limited novelty.

**Revised:** Train an image-based thermal predictor with a **physics-informed loss** (heat flux / temperature gradient residuals from the heat equation) that generalizes beyond the training parameter range — specifically to die thicknesses and HTCs not seen during training.

**Novelty claim:** Combining image-based input representation (2-channel floorplan + power map) with physics-backed loss constraints is underexplored. This approach:
1. Reduces dependence on large labeled datasets by encoding physical constraints directly in the loss
2. Improves out-of-distribution generalization — the model cannot produce thermally implausible predictions
3. Does not require a high-fidelity commercial simulator; HotSpot or PACT are sufficient for ground truth

**Timeline:** ~2–2.5 weeks remaining as of the meeting date.
