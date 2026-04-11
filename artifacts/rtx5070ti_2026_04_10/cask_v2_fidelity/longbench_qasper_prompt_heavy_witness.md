# LongBench Prompt-Heavy Witness: `qasper` @ `budget=512`

This note tracks the first LongBench prompt-heavy witness added after the
local math-only sweep.

Setup:
- model: `Qwen3-8B`
- task: `qasper`
- examples: `1`
- reference: `fullkv`
- compared methods: `triattention`, `cask`
- budgets: `256`, `384`, `512`
- prompt tokens: `4072`
- reference total tokens: `4200`

Observed fidelity:

| Method | Top-1 | Top-5 | Strict Prefix | Mean NLL | First Mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cask @ 256` | `0.6953` | `0.9531` | `0.0313` | `1.2583` | `4` |
| `cask @ 384` | `0.7656` | `0.9531` | `0.0469` | `1.0135` | `6` |
| `triattention @ 384` | `0.6719` | `0.9063` | `0.0156` | `1.3659` | `2` |
| `triattention @ 512` | `0.6563` | `0.9063` | `0.0156` | `1.3839` | `2` |
| `cask @ 512` | `0.7578` | `0.9453` | `0.0469` | `1.1091` | `6` |

Interpretation:
- This is now a **prompt-heavy budget-crossing witness**.
- `cask @ 384` is better than `triattention @ 512` on every tracked fidelity
  metric while using `25%` less physical KV budget.
- `cask @ 256` is still better than `triattention @ 512` on `top1`, `top5`,
  `strict_prefix`, `mean_nll`, and `first_mismatch`, which gives a `50%`
  budget-crossing example on this witness.
- In this witness, `cask` activates the **prefix stage only**:
  - `prefix_compression_events = 1`
  - `total_prefix_evicted_tokens = 3561`
  - `compression_events = 0`
- The witness therefore isolates the value of the **two-stage prefix-aware
  policy** rather than decode-stage scratch merging.

Savings note:
- The replay script's candidate-state `terminal_saved_ratio` is not the right
  headline metric here because prefix-stage compaction changes the candidate
  cardinality before replay completes.
- The correct paper-facing reading is the physical budget crossing itself:
  `cask @ 384` versus `triattention @ 512`, and `cask @ 256` versus
  `triattention @ 512`.

Raw report provenance:
- `experiments/reports/qasper_teacher_forced_fidelity_vs_fullkv_tri512.json`
- `experiments/reports/qasper_teacher_forced_fidelity_vs_fullkv_cask512.json`
