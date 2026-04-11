# Prompt-Heavy Summary

Authoritative prompt-heavy teacher-forced replay results now come from the
H100 two-stage rerun:

- source: `experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411/overnight_status.md`
- key runtime settings:
  - `count_prompt_tokens=true`
  - `slack_budget_trigger=true`
  - `allow_prefill_compression=false`

## 1. H100 Prompt-Heavy Replay Matrix

| Task | Method | Budget | Top-1 | Top-5 | Mean NLL | First Mismatch | Saved Ratio | CASK Compression Events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qasper` | `TriAttention` | `256` | `67.2%` | `89.8%` | `1.315` | `2` | `90.9%` | `-` |
| `qasper` | `TriAttention` | `384` | `66.4%` | `90.6%` | `1.398` | `2` | `87.9%` | `-` |
| `qasper` | `CASK` | `256` | `71.1%` | `93.8%` | `1.247` | `4` | `90.9%` | `0` |
| `qasper` | `CASK` | `384` | `73.4%` | `92.2%` | `1.297` | `4` | `87.9%` | `0` |
| `multi_news` | `TriAttention` | `256` | `54.7%` | `82.0%` | `2.060` | `2` | `88.1%` | `-` |
| `multi_news` | `TriAttention` | `384` | `53.7%` | `81.6%` | `2.052` | `2` | `84.1%` | `-` |
| `multi_news` | `CASK` | `256` | `60.0%` | `87.9%` | `1.652` | `2` | `88.1%` | `3` |
| `multi_news` | `CASK` | `384` | `61.3%` | `89.5%` | `1.540` | `2` | `84.1%` | `3` |
| `hotpotqa` | `TriAttention` | `256` | `81.3%` | `90.6%` | `1.374` | `2` | `97.6%` | `-` |
| `hotpotqa` | `TriAttention` | `384` | `81.3%` | `90.6%` | `1.344` | `2` | `96.5%` | `-` |
| `hotpotqa` | `CASK` | `256` | `93.8%` | `100.0%` | `0.151` | `11` | `97.6%` | `0` |
| `hotpotqa` | `CASK` | `384` | `96.9%` | `100.0%` | `0.110` | `11` | `96.5%` | `0` |
| `musique` | `TriAttention` | `256` | `59.4%` | `71.9%` | `2.697` | `2` | `98.3%` | `-` |
| `musique` | `TriAttention` | `384` | `53.1%` | `75.0%` | `2.862` | `2` | `97.5%` | `-` |
| `musique` | `CASK` | `256` | `65.6%` | `75.0%` | `2.713` | `3` | `98.3%` | `0` |
| `musique` | `CASK` | `384` | `62.5%` | `75.0%` | `2.650` | `3` | `97.5%` | `0` |
| `2wikimqa` | `TriAttention` | `256` | `59.4%` | `68.8%` | `3.368` | `2` | `96.1%` | `-` |
| `2wikimqa` | `TriAttention` | `384` | `59.4%` | `68.8%` | `3.415` | `2` | `94.4%` | `-` |
| `2wikimqa` | `CASK` | `256` | `62.5%` | `81.3%` | `2.375` | `2` | `96.1%` | `0` |
| `2wikimqa` | `CASK` | `384` | `56.3%` | `81.3%` | `2.397` | `2` | `94.4%` | `0` |

## 2. Readout

- `qasper` is now a clean same-budget prompt-heavy win at both `256` and `384`.
- `multi_news` is the clean decode-active witness. It is the only H100
  prompt-heavy task where decode-stage consolidation clearly fires
  (`compression_events = 3`).
- `hotpotqa` is the strongest same-budget witness in the H100 matrix.
- `musique` shows smaller but consistent same-budget fidelity gains.
- `2wikimqa` remains the boundary case:
  - `cask @ 256` wins on `top1`, `top5`, and `mean_nll`
  - `cask @ 384` improves `top5` and `mean_nll` but does not flip `top1`

## 3. Stage Interpretation

- `multi_news`: genuinely two-stage active
- `qasper`, `hotpotqa`, `musique`, `2wikimqa`: prefix-dominant in this replay
  setting (`compression_events = 0`)

The prompt-heavy story should therefore be told as:

- one clean decode-active witness: `multi_news`
- one very strong same-budget witness: `hotpotqa`
- one stable same-budget prompt-heavy witness family: `qasper`
- one retained boundary case: `2wikimqa`

## 4. Output-Level Sanity

| Task | Method | Final Answer Match | Sequence Ratio | Prefix Token Ratio |
| --- | --- | ---: | ---: | ---: |
| `multi_news` | `TriAttention @ 384` | `0.0%` | `0.000` | `0.000` |
| `multi_news` | `CASK @ 384` | `0.0%` | `0.169` | `0.081` |
| `qasper` | `TriAttention @ 512` | `0.0%` | `0.042` | `0.018` |
| `qasper` | `CASK @ 512` | `0.0%` | `0.174` | `0.045` |
| `2wikimqa` | `TriAttention @ 512` | `0.0%` | `0.083` | `0.125` |
| `2wikimqa` | `CASK @ 384` | `0.0%` | `0.703` | `0.625` |

Readout:

- `multi_news` is the strongest same-budget prompt-heavy bridge from replay
  fidelity to task-visible output quality.
- `qasper` remains a useful prompt-heavy output-level sanity check.
- `2wikimqa` remains the boundary case.
