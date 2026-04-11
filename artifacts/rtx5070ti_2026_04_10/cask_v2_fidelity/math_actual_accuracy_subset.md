# Math Actual-Accuracy Bridge Check

This note records the smallest math-side actual-generation check added after the
main fidelity package. The goal is not to claim a closed benchmark win; it is
to test whether the local fidelity advantage sometimes survives contact with
real answer extraction.

## Setup

- Dataset: 3 tracked `math500` witnesses
  - `test/prealgebra/1622.json` (`hexagon`)
  - `test/geometry/248.json`
  - `test/geometry/434.json`
- Budget: `104`
- Draws: `4`
- Methods: `triattention`, `cask`
- Generation:
  - `do_sample=true`
  - `stop_on_final_answer=true`
  - `max_new_tokens=512`
  - `count_prompt_tokens=true`
  - `slack_budget_trigger=true`

Primary runs:
- `experiments/outputs/math500_actual_witnesses/Qwen3-8B/triattention_b104_s4/`
- `experiments/outputs/math500_actual_witnesses/Qwen3-8B/cask_b104_s4/`

Prompt-control rerun for `geometry/434`:
- `experiments/outputs/math500_actual_witnesses/Qwen3-8B/triattention_geometry434_b104_s4_stable/`
- `experiments/outputs/math500_actual_witnesses/Qwen3-8B/cask_geometry434_b104_s4_stable/`

## Main Results

Per-problem exact match over 4 draws:

| Witness | TriAttention | CASK |
| --- | ---: | ---: |
| `hexagon` | `2/4` | `4/4` |
| `geometry/248` | `0/4` | `0/4` |
| `geometry/434` | `0/4` | `0/4` |

Aggregate:

| Metric | TriAttention | CASK |
| --- | ---: | ---: |
| draw-level exact match | `2/12` | `4/12` |
| problem-level `pass@4` | `1/3` | `1/3` |

## Prompt-Control Check

`geometry/434` originally came from an older witness prompt template, so it was
rerun with the current answer-stabilized math prompt.

| Witness | Prompt Style | TriAttention | CASK |
| --- | --- | ---: | ---: |
| `geometry/434` | current answer-stabilized prompt | `0/4` | `0/4` |

This means the `geometry/434` failure is not just a stale prompt-format
artifact.

## Interpretation

- This check does show a real math-side task gain, but it is concentrated in
  the strongest witness rather than spread across the whole subset.
- The clearest signal is `hexagon`, where `cask @ 104` is robust across all 4
  draws while `triattention @ 104` succeeds on only 2 of 4.
- At the subset level, CASK doubles draw-level exact match (`2/12 -> 4/12`)
  but does not yet improve problem-level `pass@4` (`1/3 -> 1/3`).
- This is best used as **bridge evidence** from fidelity to answer correctness,
  not as the paper's headline math benchmark result.
