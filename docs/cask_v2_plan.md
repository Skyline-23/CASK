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

## 7. V2 Evidence Requirements

V2 should not be justified by "more scale" in the abstract. Every new run
must strengthen one concrete evidence requirement that a stronger submission is
expected to satisfy.

| Evidence requirement | Why it matters | What must be added in v2 |
| --- | --- | --- |
| principled folding story | the paper must explain why folding is the main behavior-preserving primitive, not just a useful implementation choice | representative-mass recovery, representative-error diagnostics, and collapse-boundary analysis |
| clear method separation | the paper must show that CASK is more than a small merge-policy variant | discard vs preserve-only vs preserve+fold ablation, plus one external merge-style baseline |
| faithful evaluation bridge | replay fidelity must be connected to actual model behavior rather than left as an isolated proxy | explicit replay-to-actual bridge set with one reasoning bridge, one same-budget prompt-heavy bridge, and one budget-crossing bridge |
| breadth beyond selected witnesses | the package must read as a structured study rather than a handful of favorable cases | block-structured sweeps across reasoning and prompt-heavy regimes, plus one second-model stress point |
| interpretable policy controls | core/scratch decomposition and folding knobs must look stable and understandable rather than arbitrary | sensitivity map for core ratio, merge strength, and prefix reserve with narrow interpretable sweeps |

If a new experiment does not strengthen one of these rows, it is not v2-critical.

### 7.1 Translation Rule

This plan should encode defense requirements, not anticipated criticism in
reviewer language. The document should answer questions like:

1. what evidence makes folding look principled?
2. what evidence makes CASK look structurally distinct?
3. what evidence makes replay metrics defensible?
4. what evidence makes the package look broad rather than witness-picked?
5. what evidence makes the policy controls interpretable?

That is the right tone for a working paper plan. It keeps the document
submission-oriented without turning it into a list of negative talking points.

## 8. B200 Scale-Up Priority Order

The B200 budget should be spent to close review risks in order, not to inflate
row count uniformly.

### Tier 0: Protocol Lock Before Any Large Run

Before launching the scaled package, lock the following:

1. primary reasoning metric: teacher-forced replay fidelity vs full-KV
2. bridge metrics: official task metric, lexical overlap, semantic similarity
3. main baselines: `triattention`, `cask`, `snapkv`
4. main model: `Qwen3-8B`
5. supplementary model: one DeepSeek distilled model as a stress-point, not a
   full second matrix

If these are not frozen first, the B200 run will create more ambiguity than
evidence.

### Tier 1: Must-run

1. Block A: scorer failure study
2. Block B: discard vs fold ablation
3. Block C: structure sensitivity on one reasoning slice and one prompt-heavy
   slice

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

## 9. B200 Execution Package

The scaled package should be split into five run groups. The goal is not to run
everything at once. The goal is to finish the highest-value block first and
freeze it before moving on.

### Group 1. Scorer Failure Package

Purpose:
convert Phase 1 into an explicit negative result.

Required comparison:

- `triattention`
- `horizonkv`
- any retained scorer-side variant already in-tree

Required outputs:

1. keep-set overlap / churn
2. critical-token movement
3. score delta vs set delta

This group closes the "why not just improve the scorer?" objection.

### Group 2. Discard vs Fold Package

Purpose:
make folding the central mechanism rather than a hidden implementation detail.

Required rows:

- `triattention`
- preserve-only degraded variant
- fold-only or weakened-fold variant
- `cask`
- `snapkv`

Required outputs:

1. replay fidelity
2. representative mass recovery
3. saved ratio
4. collapse diagnostics on at least one failure-boundary task

This group is the centerpiece of v2.

### Group 3. Structure Sensitivity Package

Purpose:
show that the method is understandable and not arbitrary.

Minimum sweeps:

- core ratio
- merge threshold or merge strength
- stage-1 prefix reserve

Required tasks:

- one reasoning slice
- one prompt-heavy slice

Do not spread this over many tasks. The point is interpretability, not breadth.

### Group 4. Breadth Package

Purpose:
remove the "selected witness" criticism.

Minimum matrix:

- reasoning: `aime24`, `aime25`, `math500`
- prompt-heavy: `qasper`, `hotpotqa`, `multi_news`, `musique`, `2wikimqa`
- probes: `vcsum`, `qmsum`, `gov_report`
- model: `Qwen3-8B`
- baselines: `triattention`, `cask`, `snapkv`

This group should be read as breadth support, not as the main conceptual block.

### Group 5. Bridge Package

Purpose:
show that replay gains are not disconnected from actual generation.

Required bridges:

1. one reasoning-side bridge
2. one same-budget prompt-heavy bridge
3. one budget-crossing bridge

Current candidates:

- reasoning-side: strongest stable AIME witness from the scaled replay gate
- same-budget prompt-heavy: `multi_news`
- budget crossing: `qasper`

This group is necessary to defend replay, but it should remain smaller than the
replay package itself.

## 10. Minimum Matrix Needed To Upgrade The Submission

The submission should not be upgraded from the current v1 package to v2 unless
all of the following are true.

### 10.1 Ranking Critique Is Explicitly Closed

Required:

1. a scorer-failure block exists
2. the paper can show that score changes are larger than keep-set changes
3. this result is placed in the main paper, not buried in appendix history

### 10.2 Folding Is Shown To Be The Main Lever

Required:

1. discard vs preserve-only vs preserve+fold appears in one central table or
   figure
2. representative-mass recovery is reported alongside fidelity
3. `snapkv` is included in at least one central comparison block

### 10.3 Breadth Is No Longer Witness-Only

Required:

1. full reasoning replay block on `aime24`, `aime25`, `math500`
2. full prompt-heavy replay block on the current main witness set
3. one second-model stress point

### 10.4 Replay Is Defended By Bridge Evidence

Required:

1. one reasoning actual-generation bridge
2. one same-budget prompt-heavy bridge
3. one budget-crossing bridge

If any of these are missing, the paper is still closer to strengthened v1 than
to true v2.

## 11. Stop Rules For The B200 Queue

The queue should not be allowed to grow without decision checkpoints.

### Stop Rule A

If Group 1 fails to show a meaningful scorer-failure story, do not keep adding
scorer variants. Pivot all remaining time to Group 2 and Group 4.

### Stop Rule B

If Group 2 does not show a clear discard-vs-fold separation, v2 loses its
center and should not be pitched as a primitive shift.

### Stop Rule C

If Group 4 only reproduces current witness behavior without broadening the same
trend, the venue target should be reduced rather than hidden behind more plots.

### Stop Rule D

If Group 5 fails to produce at least one clean reasoning bridge and one clean
prompt-heavy bridge, replay must remain a defended internal metric rather than
a headline claim.

## 12. Script-Level Execution Path

The scaled package should use the current generic runners instead of ad hoc
host-specific queues.

Primary entry points:

1. `scripts/run_replay_fidelity_frontier.py` for replay blocks
2. `scripts/run_promptheavy_pack.py` for prompt-heavy reference + replay packs
3. `scripts/run_kv_benchmark_bundle.py` for broader benchmark bundles
4. `scripts/run_qwen3_frontier_preset.py` for conservative single-device
   frontier presets
5. `scripts/run_replay_queue.ps1` when a replay block needs queued JSON-driven
   execution on Windows
6. `scripts/run_v2_group1_score_failure.sh` for the scorer-failure package
7. `scripts/run_v2_group2_fold_ablation.sh` for the discard-vs-fold package
8. `scripts/run_v2_group3_sensitivity.sh` for the narrow sensitivity package

This matters because v2 should be reproducible as a package, not as a sequence
of one-off terminal sessions.

### 12.1 Shared Environment For The Scale-Up Run

All command templates below assume one shared environment block:

```bash
export MODEL_ALIAS=Qwen3-8B
export MODEL_PATH=experiments/models/Qwen3-8B
export STATS_PATH=cask/calibration/for_aime25_experiment/qwen3_8b.pt
export DTYPE=bfloat16
export ATTN_IMPL=sdpa
```

If the run is launched from another environment, the values should change, but
the command structure should not.

### 12.2 Group 1 Command Template: Scorer Failure Package

The goal of Group 1 is not to chase accuracy directly. It is to measure how
much scorer changes actually move the selected set.

Step 1: generate or reuse a compact full-KV reference slice.

```bash
python scripts/cli.py run-one \
  --model "$MODEL_ALIAS" \
  --dataset aime24 \
  --method fullkv \
  --run-tag v2_group1_ref_aime24 \
  --max-examples 6 \
  --num-samples 1 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE"
```

Step 2: run score-dump variants on the same slice.

```bash
python scripts/cli.py run-one \
  --model "$MODEL_ALIAS" \
  --dataset aime24 \
  --method triattention \
  --budget 384 \
  --stats-path "$STATS_PATH" \
  --run-tag v2_group1_tri384 \
  --max-examples 6 \
  --num-samples 1 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --score-dump-dir experiments/analysis/v2/group1/aime24_triattention \
  --score-dump-max-events 16

python scripts/cli.py run-one \
  --model "$MODEL_ALIAS" \
  --dataset aime24 \
  --method horizonkv \
  --budget 384 \
  --stats-path "$STATS_PATH" \
  --run-tag v2_group1_horizon384 \
  --max-examples 6 \
  --num-samples 1 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --triattention-horizon-mode adaptive \
  --triattention-norm-mode rms2 \
  --score-dump-dir experiments/analysis/v2/group1/aime24_horizonkv \
  --score-dump-max-events 16

python scripts/cli.py run-one \
  --model "$MODEL_ALIAS" \
  --dataset aime24 \
  --method cask \
  --budget 384 \
  --stats-path "$STATS_PATH" \
  --run-tag v2_group1_cask384 \
  --max-examples 6 \
  --num-samples 1 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --score-dump-dir experiments/analysis/v2/group1/aime24_cask \
  --score-dump-max-events 16
```

Step 3: summarize the resulting dumps.

```bash
python scripts/diff_score_dumps.py \
  --baseline-dir experiments/analysis/v2/group1/aime24_triattention \
  --candidate-dir experiments/analysis/v2/group1/aime24_horizonkv \
  --json-output experiments/analysis/v2/group1/aime24_tri_vs_horizonkv.json

python scripts/summarize_selection_dumps.py \
  --dump-dir experiments/analysis/v2/group1/aime24_cask \
  --json-output experiments/analysis/v2/group1/aime24_cask_selection_summary.json
```

The output of this block should be interpreted as:

1. score-family deltas
2. set-level overlap or churn
3. whether CASK selection changes are larger than scorer-only changes

### 12.3 Group 2 Command Template: Discard vs Fold Package

The goal of Group 2 is to show that folding, not just protected selection, is
the main behavior-preserving lever.

The current codebase already supports an executable approximation of the
required rows:

| Row name in the paper | Current implementation mapping |
| --- | --- |
| discard-only baseline | `--method triattention` |
| external merge baseline | `--method snapkv` |
| preserve-only degraded CASK | `--method cask --cask-decode-merge-enabled false` |
| fold-weakened CASK | `--method cask --cask-merge-operator mean --cask-representative-mode weighted_latest --cask-use-phase-markers false` |
| full CASK | `--method cask` with current defaults |

Important note:
the preserve-only row is an executable degraded in-code variant, not an oracle
policy. That is acceptable for v2 as long as the paper describes it honestly.

Reference generation:

```bash
python scripts/cli.py run-one \
  --model "$MODEL_ALIAS" \
  --dataset aime24 \
  --method fullkv \
  --run-tag v2_group2_ref_aime24 \
  --max-examples 6 \
  --num-samples 1 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE"
```

Replay fidelity rows:

```bash
python scripts/replay_reference_fidelity.py \
  --reference experiments/outputs/aime24/Qwen3-8B/sample1/fullkv/full_v2_group2_ref_aime24/merged/merged.jsonl \
  --model-path "$MODEL_PATH" \
  --method triattention \
  --budget 384 \
  --triattention-stats-file "$STATS_PATH" \
  --max-records 6 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --json-output experiments/analysis/v2/group2/aime24_triattention384.json \
  --csv-output experiments/analysis/v2/group2/aime24_triattention384.csv

python scripts/replay_reference_fidelity.py \
  --reference experiments/outputs/aime24/Qwen3-8B/sample1/fullkv/full_v2_group2_ref_aime24/merged/merged.jsonl \
  --model-path "$MODEL_PATH" \
  --method snapkv \
  --budget 384 \
  --triattention-stats-file "$STATS_PATH" \
  --max-records 6 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --json-output experiments/analysis/v2/group2/aime24_snapkv384.json \
  --csv-output experiments/analysis/v2/group2/aime24_snapkv384.csv

python scripts/replay_reference_fidelity.py \
  --reference experiments/outputs/aime24/Qwen3-8B/sample1/fullkv/full_v2_group2_ref_aime24/merged/merged.jsonl \
  --model-path "$MODEL_PATH" \
  --method cask \
  --budget 384 \
  --triattention-stats-file "$STATS_PATH" \
  --max-records 6 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --cask-decode-merge-enabled false \
  --json-output experiments/analysis/v2/group2/aime24_cask_preserve_only384.json \
  --csv-output experiments/analysis/v2/group2/aime24_cask_preserve_only384.csv

python scripts/replay_reference_fidelity.py \
  --reference experiments/outputs/aime24/Qwen3-8B/sample1/fullkv/full_v2_group2_ref_aime24/merged/merged.jsonl \
  --model-path "$MODEL_PATH" \
  --method cask \
  --budget 384 \
  --triattention-stats-file "$STATS_PATH" \
  --max-records 6 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --cask-merge-operator mean \
  --cask-representative-mode weighted_latest \
  --cask-use-phase-markers false \
  --json-output experiments/analysis/v2/group2/aime24_cask_fold_weakened384.json \
  --csv-output experiments/analysis/v2/group2/aime24_cask_fold_weakened384.csv

python scripts/replay_reference_fidelity.py \
  --reference experiments/outputs/aime24/Qwen3-8B/sample1/fullkv/full_v2_group2_ref_aime24/merged/merged.jsonl \
  --model-path "$MODEL_PATH" \
  --method cask \
  --budget 384 \
  --triattention-stats-file "$STATS_PATH" \
  --max-records 6 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --json-output experiments/analysis/v2/group2/aime24_cask_full384.json \
  --csv-output experiments/analysis/v2/group2/aime24_cask_full384.csv
```

This block should be read together with:

1. replay fidelity
2. terminal saved ratio
3. representative-mass diagnostics from selection dumps

### 12.4 Group 3 Command Template: Structure Sensitivity Package

The goal of Group 3 is not to maximize performance. The goal is to show that
the policy knobs behave in understandable ways.

Recommended reasoning slice:

- `aime24`

Recommended prompt-heavy slice:

- `multi_news`

Current executable sweep axes:

| Axis | CLI flag | Recommended values |
| --- | --- | --- |
| core ratio | `--cask-protected-core-ratio` | `0.35`, `0.50`, `0.65` |
| prefix reserve | `--cask-prefix-coverage-ratio` | `0.0`, `0.0625`, `0.125` |
| merge strictness | `--cask-similarity-threshold` | `0.975`, `0.985`, `0.992` |

Reasoning-side template:

```bash
python scripts/run_cask_frontier.py \
  --model "$MODEL_ALIAS" \
  --datasets aime24 \
  --methods cask \
  --budgets 384 \
  --frontier-tag v2_group3_aime24_core035 \
  --stats-path "$STATS_PATH" \
  --num-samples 1 \
  --max-examples 6 \
  --job-parallel 1 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --cask-protected-core-ratio 0.35
```

Prompt-heavy template:

```bash
python scripts/run_promptheavy_pack.py \
  --tag v2_group3_multi_news_cov0125 \
  --stage all \
  --main-tasks multi_news \
  --methods triattention cask snapkv \
  --budgets 384 \
  --max-examples 1 \
  --max-records 1 \
  --ref-parallel 1 \
  --replay-parallel 1 \
  --replay-inner-parallel 1
```

When the prompt-heavy axis is the object of study, use direct replay commands
after the reference step to inject the exact CASK knob being swept, for example:

```bash
python scripts/replay_reference_fidelity.py \
  --reference experiments/v2_group3_multi_news_cov0125_refs/longbench/Qwen3-8B/runs/multi_news/merged/merged.jsonl \
  --model-path "$MODEL_PATH" \
  --method cask \
  --budget 384 \
  --triattention-stats-file "$STATS_PATH" \
  --max-records 1 \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --count-prompt-tokens true \
  --slack-budget-trigger true \
  --allow-prefill-compression false \
  --cask-prefix-coverage-ratio 0.125 \
  --json-output experiments/analysis/v2/group3/multi_news_cask_cov0125.json \
  --csv-output experiments/analysis/v2/group3/multi_news_cask_cov0125.csv
```

The same pattern should be repeated for the selected values of each axis rather
than hidden inside one giant mixed sweep.

### 12.5 Packaging Rule

For B200 execution, every completed group should emit:

1. a manifest or command log
2. compact CSV/JSON summaries
3. one short readout note stating whether the group closed its intended
   evidence requirement

If a block finishes without that packaging step, it is not ready to inform the
paper.

## 13. What Not To Do

To avoid burning B200 time, do not spend the next round on the following:

1. more scorer variants without keep-set churn analysis
2. more witness rows that do not isolate a new regime
3. pure sample-count expansion without structure ablations
4. cosmetic figure work before the v2 core blocks land
5. full second-model duplication before the `Qwen3-8B` v2 center closes

The current bottleneck is not presentation polish. It is causal closure.

## 14. Proposed V2 Paper Outline

The paper structure should shift accordingly.

### 14.1 Introduction

- Current methods over-emphasize ranking/eviction.
- Phase 1 showed scorer refinement has limited leverage.
- We instead treat reasoning KV compression as structured consolidation.

### 14.2 Failure of Ranking-Centric Compression

- scorer invariance
- limited keep-set churn
- why better scoring is not enough

### 14.3 CASK as Structured Consolidation

- core vs scratch
- folding / representative states
- two-stage handling for prompt-heavy regimes

### 14.4 Diagnostics and Behavior Preservation

- representative mass
- kappa-dispersion
- collapse conditions

### 14.5 Experiments

- Block A: scorer failure
- Block B: discard vs fold
- Block C: sensitivity / regime decomposition
- Block D: replay -> actual bridge

### 14.6 Discussion

- what CASK does solve
- what remains boundary-limited
- where discard-only methods still remain competitive

## 15. V2 Success Criteria

V2 should be considered ready only if the paper can defend all four questions:

1. Why is ranking not the main lever?
2. Why is folding the main lever?
3. When does the method work, and when does it fail?
4. Why should replay gains be taken seriously for real generation?

If any of these remain answered only by intuition, v2 is not finished.

## 16. Immediate Action Items

The next concrete actions should be:

1. Define exact B200 commands for Group 1, Group 2, and Group 3.
2. Freeze one reasoning slice and one prompt-heavy slice for the sensitivity
   package.
3. Promote Phase 1 keep-set churn analysis from background note to a tracked
   main artifact.
4. Define the discard-only / preserve-only / fold-enabled ablation rows before
   launching the scale-up run.
5. Draft the new v2 outline in the paper source before launching all runs.

## 17. Final Read

V1 is already submit-able as a credible paper package.
V2 should not be interpreted as "the same paper with more data."

V2 should be interpreted as:

> a paper that explicitly argues the field is focusing on the wrong primitive,
> and that behavior-preserving folding is the correct center of reasoning KV
> compression.
