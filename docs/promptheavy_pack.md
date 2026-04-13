# Prompt-Heavy Pack

This note fixes the pre-execution structure for the prompt-heavy package.
The goal is not to average everything into one headline. The goal is to run a
role-separated package with a stable replay metric and enough baseline coverage
to support a `v2` paper update.

## 1. Core Principle

Use a three-way replay read for every tracked prompt-heavy task:

- `fullkv`: reference continuation source
- `triattention`: eviction baseline
- `cask`: main method
- `snapkv`: merge-style external baseline

`fullkv` is not a budgeted baseline. It is the reference trace used by
teacher-forced replay. The budgeted comparison block is therefore
`triattention / cask / snapkv`.

## 2. Package Roles

Main witnesses:

- `qasper`: prefix-dominant crossing witness
- `hotpotqa`: strongest same-budget witness
- `multi_news`: decode-active witness with output bridge value
- `musique`: weaker-gain witness
- `2wikimqa`: retained boundary

Follow-up probes:

- `vcsum`: decode-active replay follow-up
- `qmsum`: prefix-only boundary
- `gov_report`: failure boundary

These roles should not be merged in interpretation. The package is only valid if
the witness taxonomy is preserved in the final readout.

## 3. Default Matrix

Replay main matrix:

- tasks: `qasper`, `hotpotqa`, `multi_news`, `musique`, `2wikimqa`
- methods: `triattention`, `cask`, `snapkv`
- budgets: `256`, `384`

Replay probe matrix:

- tasks: `vcsum`, `qmsum`, `gov_report`
- methods: `triattention`, `cask`, `snapkv`
- budgets: `256`, `384`

Reference jobs:

- one `fullkv` run per task

Unit counts:

- main references: `5`
- main replay frontier jobs: `5 tasks x 3 methods x 2 budgets = 30`
- probe references: `3`
- probe replay frontier jobs: `3 tasks x 3 methods x 2 budgets = 18`
- total replay package: `56` task-level units

## 4. Parallel Structure

The recommended structure is task-level parallelism with bounded inner
parallelism. Parallelism is optional and the default runner configuration keeps
everything single-process until you explicitly raise it.

Safe default:

- `ref_parallel = 1`
- `replay_parallel = 1`
- `replay_inner_parallel = 1`

Recommended scale-up starting point:

- `ref_parallel = 2`
- `replay_parallel = 3`
- `replay_inner_parallel = 1`

Reason:

- each replay frontier job loads the model again through the worker path
- raising both outer and inner parallelism at the same time makes scheduling
  harder and increases the chance of wasting memory on duplicate model loads
- task-level outer parallelism is easier to monitor and easier to resume

If the target machine still has memory headroom, the first knob to raise is:

- `replay_parallel: 3 -> 4`

Do not raise `replay_inner_parallel` before confirming that outer task
parallelism is already saturating the device effectively.

## 5. Output Layout

Reference roots:

- `experiments/<tag>_refs/longbench/Qwen3-8B/runs/<task>/merged/merged.jsonl`

Replay frontier roots:

- `experiments/frontier/Qwen3-8B/<tag>_<role>_<task>/`

Master plan:

- `experiments/reports/<tag>/master_plan.json`

This separation matters:

- refs are reusable across all budgeted methods
- each replay frontier remains task-local and resumable
- the master plan gives one place to inspect the entire replay package

## 6. Execution Script

Use:

```bash
python scripts/run_promptheavy_pack.py \
  --tag promptheavy_phaseA \
  --stage all \
  --main-tasks qasper hotpotqa multi_news \
  --probe-tasks vcsum qmsum gov_report \
  --methods triattention cask snapkv \
  --budgets 256 384 \
  --max-examples 1 \
  --max-records 1 \
  --ref-parallel 2 \
  --replay-parallel 3 \
  --replay-inner-parallel 1 \
  --skip-existing
```

For a full package:

```bash
python scripts/run_promptheavy_pack.py \
  --tag promptheavy_full \
  --stage all \
  --main-tasks qasper hotpotqa multi_news musique 2wikimqa \
  --probe-tasks vcsum qmsum gov_report \
  --methods triattention cask snapkv \
  --budgets 256 384 \
  --ref-parallel 2 \
  --replay-parallel 3 \
  --replay-inner-parallel 1 \
  --skip-existing
```

Planning only:

```bash
python scripts/run_promptheavy_pack.py \
  --tag promptheavy_plan \
  --stage plan
```

## 7. What This Does Not Yet Do

This script intentionally stops at:

- `fullkv` reference generation
- replay frontier generation for `triattention / cask / snapkv`

It does not yet package:

- prompt-heavy replay summary JSON/CSV/MD
- prompt-heavy figures
- actual-output bridge generation

Those should be a second layer, built only after the replay matrix is complete.

## 8. Next Layer After Execution

After the replay pack finishes, the next required step is a new summary
builder that reads the replay frontiers and writes:

- `artifacts/.../promptheavy_replay_fidelity_summary.json`
- `artifacts/.../promptheavy_replay_fidelity_summary.csv`
- `artifacts/.../promptheavy_replay_fidelity_readout.md`

The current `saved_ratio_audit` package should remain as a diagnostic, not the
main prompt-heavy paper package.
