# H100 Gate Plan (2026-04-12)

## Goal

The current objective is not to finish the full frontier.
The objective is to preserve the GPU time already spent on the active `triattention@384` runs,
extract the first paper-relevant signals, and stop the queue before the long follow-on matrix burns another full day.

## Current Decision

Use option `1`.

1. Let the currently running `triattention@384` jobs finish.
2. Read the first merged outputs and fidelity-relevant artifacts from those jobs.
3. Stop the remaining follow-on frontier runs before they roll into the full `cask / snapkv / 512` cascade.
4. Relaunch only the smallest comparison package that is actually needed for the paper.

## Why This Changed

The earlier compact-frontier plan expanded too far.
Even with `max_examples=10`, a full frontier on this host turns into a long sequential queue because each AIME run is much slower than the original rough estimate.
The main risk is no longer "insufficient breadth"; it is wasting the night on a queue that does not tighten the paper quickly enough.

## Active Queues

### Main Qwen Gate

- model: `Qwen3-8B`
- datasets: `aime24`, `aime25`
- methods: `triattention`, `cask`, `snapkv`
- budgets: `384`, `512`
- purpose: main paper comparison package

Current tags:

- `h100_gate_qwen_aime25_6run_20260412`
- `h100_gate_qwen_aime24_6run_20260412`

Current logs to watch:

- `experiments/logs/aime25/Qwen3-8B/sample1/triattention/budget_384_h100_gate_qwen_aime25_6run_20260412/`
- `experiments/logs/aime24/Qwen3-8B/sample1/triattention/budget_384_h100_gate_qwen_aime24_6run_20260412/`

### DeepSeek Supplementary Gate

- model: `DeepSeek-R1-Distill-Llama-8B`
- datasets: `aime24`, `aime25`
- methods: `triattention`, `cask`, `snapkv`
- budgets: `384`, `512`
- purpose: second-model generalization check

Current live tag:

- `dsg`

Important note:

- The previous DeepSeek tag `h100_gate_dsllama8b_20260412` hit Windows `WinError 206` during eval output creation.
- The short tag `dsg` exists only to avoid the path-length failure.

Current logs to watch:

- `experiments/logs/aime24/DeepSeek-R1-Distill-Llama-8B/sample1/triattention/budget_384_dsg/`

## Explicitly Not Running

- `math500`
- `r1kv`

Reason:

- They widen the queue faster than they close the paper.
- They are not part of the minimal gate needed for the current decision.

## Primary Readout

Do not use the legacy `acc` line as the main paper metric.
It is only a side-effect of the existing dispatch evaluation path.

Primary readout for the paper:

1. same-budget comparison readiness
2. replay / fidelity artifacts
3. saved-ratio and bridge-style comparisons

Secondary readout:

- legacy eval `acc` if it happens to be produced cleanly

## Stop Condition

Stop the frontier expansion after the active `triattention@384` runs finish and their merged artifacts are available.

The first stop checkpoint is reached when these three runs have usable merged outputs:

1. `Qwen3-8B / aime25 / triattention / 384`
2. `Qwen3-8B / aime24 / triattention / 384`
3. `DeepSeek-R1-Distill-Llama-8B / aime24 / triattention / 384`

At that point, do not keep the frontier alive just because more runs are queued.
Read the outputs first, then decide the next minimal comparison batch.

This cutover is now automated by:

- `scripts/h100_cutover_watch.ps1`

The watcher waits for the three required `merged.jsonl` files, kills the long frontier queues, and relaunches the reduced comparison block automatically.

## Next Minimal Relaunch

If the first checkpoint looks healthy, the next relaunch should be the smallest same-budget comparison block, not a fresh full frontier.

Preferred order:

1. `Qwen3-8B / aime24,aime25 / cask,snapkv / 384`
2. add `Qwen3-8B / aime24,aime25 / cask,snapkv / 512` only if `384` shows a real ranking signal
3. keep `DeepSeek-R1-Distill-Llama-8B` as a single stress-point supplementary check, not a second full matrix

## What To Cut

Cut the queue by deleting work that does not add a new paper claim.

Keep:

1. current `Qwen3-8B / triattention / 384 / aime24,aime25`
2. current `DeepSeek-R1-Distill-Llama-8B / triattention / 384 / aime24`
3. the smallest follow-on same-budget comparison block needed to place `CASK` against `TriAttention` and `SnapKV`

Cut:

1. any second-model full frontier beyond the first stress-point check
2. `512` runs before `384` has shown whether the ranking is informative
3. any additional dataset breadth before the AIME gate closes

## Why This Closes The Paper Faster

Existing evidence already covers several other axes:

1. prompt-heavy witnesses already exist
2. output-level bridge already exists
3. same-budget fidelity story is already visible in prior local artifacts

What is still weak is the broad reasoning-side same-budget ranking under stress.

That means the highest-ROI missing axis is:

`Qwen3-8B + AIME24/AIME25 + 384-budget + TriAttention/CASK/SnapKV`

This axis does four jobs at once:

1. broadens beyond single witness cases
2. keeps the comparison at the stress budget where separation is most likely
3. preserves the older-baseline comparison through `SnapKV`
4. makes the final main table much easier to defend

By contrast, `512` and full `DeepSeek` breadth are lower-ROI until the `384` ranking signal is known.

## Practical Reminder

The right question tonight is:

`What is the smallest completed batch that makes the paper materially tighter?`

The wrong question is:

`Can the GPU be kept busy with more frontier entries?`
