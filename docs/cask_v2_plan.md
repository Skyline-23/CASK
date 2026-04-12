# CASK V2 Plan

This document defines how CASK should move from the current v1-style package
to a v2 paper that can survive stronger top-tier review. The goal is not to
append more witness examples to the current narrative. The goal is to change
the paper from "CASK is somewhat better than TriAttention on selected regimes"
into "reasoning KV compression is primarily a structured consolidation problem,
and folding is the main behavior-preserving lever."

## 1. V2 Headline

The v2 paper should make one primary claim:

> Reasoning KV compression is not primarily a ranking problem. The main lever
> is preserving core states while folding redundant scratch into
> representative states.

This immediately implies three consequences:

1. Scorer refinement is a secondary lever.
2. Folding is not an implementation detail; it is the core mechanism.
3. Evaluation must explain when structured consolidation works, when it fails,
   and why.

## 2. What V1 Already Established

The current package is not useless and should not be discarded. It already
establishes:

1. A cleaner problem framing:
   ranking/eviction -> core protection + selective consolidation.
2. Same-budget reasoning-fidelity wins and some budget crossings on the H100
   replay gate.
3. Prompt-heavy witness separation into active regimes and
   `prefix_budget_exhausted` boundary regimes.
4. Useful diagnostics:
   lost representative mass, kappa-dispersion, cluster collapse.

These assets remain part of v2. They are not the destination.

## 3. Why V1 Is Not Enough

The current v1 story is still vulnerable in four ways:

1. It can be read as better policy engineering against one main baseline.
2. Folding appears as a downstream implementation choice instead of the main
   reason performance is preserved.
3. The scorer-side negative result is present but not yet weaponized.
4. The experimental package is still easier to read as witness-centric than as
   a general statement about the structure of reasoning KV compression.

Therefore, v2 must not be framed as "more examples." It must be framed as
"better explanation of the true primitive."

## 4. V2 Structural Changes

### 4.1 Promote Folding to the Main Object

In v2, folding must move from method detail to central object.

The paper should explicitly compare:

- discard
- preserve-only
- preserve + fold

The main question is:

> If the objective is to preserve future behavior under a fixed budget, why is
> folding redundant scratch better than simply discarding more states?

This is the real theoretical and empirical axis of v2.

### 4.2 Turn Phase 1 into a Strong Negative Result

The Phase 1 story is not just historical context. It is evidence that:

> Better scalar scoring did not materially reorganize the keep-set.

This should become an explicit negative result, not a footnote. The purpose is
to justify why the field should stop treating reasoning KV compression as
primarily a better-ranking problem.

### 4.3 Reframe Evaluation as Regime Decomposition

V2 experiments should answer:

- When does structured consolidation help?
- When does it fail?
- Which control knobs matter?

This means replacing a "selected win" mindset with a "regime map" mindset.

## 5. Required V2 Theory / Diagnostics

V2 does not need a grand theorem pretending the whole method is globally
optimal. It does need a cleaner principled story around folding.

### 5.1 Minimum theory target

The v2 draft should include a theory/diagnostic section centered on:

1. Representative mass preservation
2. Representative error under folded scratch
3. Collapse conditions when core states are merged or insufficiently protected

The target is not a heroic universal theorem. The target is a credible answer
to:

> Why should folding preserve future behavior better than discard-only
> compression when scratch is redundant?

### 5.2 Minimum diagnostic objects

The following should be first-class diagnostic objects in v2:

- `representative mass ratio`
- `lost representative mass`
- `kappa-dispersion`
- `cluster collapse`
- `keep-set churn` under scorer changes

The paper should visibly tie each object to one concrete failure mode.

## 6. Required V2 Experiment Blocks

V2 should be organized around experiment blocks, not ad hoc tables.

### Block A. Scorer Failure Study

Purpose:
show that better scalar scoring is not the main lever.

Minimum output:

1. Compare multiple scorer variants.
2. Measure keep-set churn / overlap / critical-token movement.
3. Show that score changes are much larger than actual set reorganization.

If this block lands cleanly, it strengthens the pivot away from ranking.

### Block B. Discard vs Fold Ablation

Purpose:
show that folding is the key behavior-preserving mechanism.

Required rows:

- TriAttention baseline
- core protection only
- scratch folding only or degraded fold variant
- full CASK

Key output:

1. fidelity comparison
2. saved-ratio comparison
3. representative-mass comparison

This block should become a centerpiece figure/table in v2.

### Block C. Structure Sensitivity

Purpose:
show that the method is understandable, not arbitrary.

Minimum sweeps:

- core ratio
- merge strength / merge threshold
- prefix reserve or stage-1 budget share

Do not run this across every dataset. Use one reasoning slice and one
prompt-heavy slice.

### Block D. Regime Map

Purpose:
explain where the method is active versus boundary-limited.

Minimum axes:

- prompt-heavy vs decode-heavy
- active two-stage regime vs `prefix_budget_exhausted`
- healthy folding vs collapse boundary

This should replace vague "sometimes works, sometimes not" language.

### Block E. Replay -> Actual Bridge

Purpose:
defend replay metrics as meaningful rather than convenient.

Minimum requirement:

- at least one reasoning-side bridge
- at least one prompt-heavy same-budget bridge
- at least one budget-crossing bridge

The current `multi_news`, `qasper`, and one reasoning task are acceptable as a
base, but v2 should present them as a deliberate bridge set, not scattered
supporting evidence.

## 7. H100 Priority Order

The H100 queue should not be spent uniformly. The order matters.

### Tier 1: Must-run

1. scorer failure study
2. discard vs fold ablation
3. structure sensitivity on one reasoning slice and one prompt-heavy slice

Without these, v2 is still just a broader v1.

### Tier 2: Strongly recommended

1. broader reasoning replay sweep
2. broader prompt-heavy regime sweep
3. one extra non-TriAttention baseline if implementation cost is reasonable

These improve defense, but they are not the conceptual core.

### Tier 3: Only if time remains

1. more witness extensions
2. more cosmetic bridge tasks
3. larger but structurally redundant sweeps

These help least relative to cost.

## 8. What Not To Do

To avoid burning H100 time, do not spend the next round on the following:

1. More scorer variants without keep-set churn analysis
2. More witness rows that do not isolate a new regime
3. Pure sample-count expansion without structure ablations
4. Cosmetic figure work before the v2 core blocks land

The current bottleneck is not presentation polish. It is causal closure.

## 9. Proposed V2 Paper Outline

The paper structure should shift accordingly.

### 9.1 Introduction

- Current methods over-emphasize ranking/eviction.
- Phase 1 showed scorer refinement has limited leverage.
- We instead treat reasoning KV compression as structured consolidation.

### 9.2 Failure of Ranking-Centric Compression

- scorer invariance
- limited keep-set churn
- why better scoring is not enough

### 9.3 CASK as Structured Consolidation

- core vs scratch
- folding / representative states
- two-stage handling for prompt-heavy regimes

### 9.4 Diagnostics and Behavior Preservation

- representative mass
- kappa-dispersion
- collapse conditions

### 9.5 Experiments

- Block A: scorer failure
- Block B: discard vs fold
- Block C: sensitivity / regime decomposition
- Block D: replay -> actual bridge

### 9.6 Discussion

- what CASK does solve
- what remains boundary-limited
- where discard-only methods still remain competitive

## 10. V2 Success Criteria

V2 should be considered ready only if the paper can defend all four questions:

1. Why is ranking not the main lever?
2. Why is folding the main lever?
3. When does the method work, and when does it fail?
4. Why should replay gains be taken seriously for real generation?

If any of these remain answered only by intuition, v2 is not finished.

## 11. Immediate Action Items

The next concrete actions should be:

1. Define exact H100 commands for Block A, Block B, and Block C.
2. Decide one reasoning slice and one prompt-heavy slice for sensitivity runs.
3. Promote Phase 1 keep-set churn analysis from background note to main
   artifact.
4. Draft the new v2 outline in the paper source before launching all runs.

## 12. Final Read

V1 is already submit-able as a credible paper package.
V2 should not be interpreted as "the same paper with more data."

V2 should be interpreted as:

> a paper that explicitly argues the field is focusing on the wrong primitive,
> and that behavior-preserving folding is the correct center of reasoning KV
> compression.
