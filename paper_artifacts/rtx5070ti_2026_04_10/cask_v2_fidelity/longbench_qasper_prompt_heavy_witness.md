# LongBench Prompt-Heavy Witness: `qasper` @ `budget=512`

This note tracks the first LongBench prompt-heavy witness added after the
local math-only sweep.

Setup:
- model: `Qwen3-8B`
- task: `qasper`
- examples: `1`
- reference: `fullkv`
- compared methods: `triattention`, `cask`
- budget: `512`
- prompt tokens: `4072`
- reference total tokens: `4200`

Observed fidelity:

| Method | Top-1 | Top-5 | Strict Prefix | Mean NLL | First Mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `triattention @ 512` | `0.6563` | `0.9063` | `0.0156` | `1.3839` | `2` |
| `cask @ 512` | `0.7578` | `0.9453` | `0.0469` | `1.1091` | `6` |

Interpretation:
- This is a **prompt-heavy same-budget fidelity witness**.
- `cask` is better than `triattention` on `top1`, `top5`, `strict_prefix`,
  `mean_nll`, and `first_mismatch`.
- In this witness, `cask` activates the **prefix stage only**:
  - `prefix_compression_events = 1`
  - `total_prefix_evicted_tokens = 3561`
  - `compression_events = 0`
- The witness therefore isolates the value of the **two-stage prefix-aware
  policy** rather than decode-stage scratch merging.

Savings note:
- `triattention` reports `terminal_saved_ratio = 0.8481`.
- `cask` reports `terminal_saved_ratio = 0.0` under the candidate-state metric
  because the prefix stage already collapses the candidate cardinality before
  replay finishes.
- For paper text, the correct same-budget reading is **reference-length /
  physical-budget aligned savings**, which are equal here at about `84.8%`.

Raw report provenance:
- `experiments/reports/qasper_teacher_forced_fidelity_vs_fullkv_tri512.json`
- `experiments/reports/qasper_teacher_forced_fidelity_vs_fullkv_cask512.json`
