# Decode-Active Replay Probe

## `vcsum` same-budget replay

| Dataset | Method | Budget | Top-1 | Top-5 | Mean NLL | First Mismatch | Prefix Events | Decode Events | Saved Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vcsum` | `TriAttention` | `256` | `51.37%` | `77.15%` | `2.337` | `3` | `-` | `-` | `93.38%` |
| `vcsum` | `CASK` | `256` | `57.81%` | `82.23%` | `2.042` | `4` | `2` | `3` | `93.38%` |
| `vcsum` | `TriAttention` | `384` | `53.91%` | `78.71%` | `2.269` | `3` | `-` | `-` | `91.16%` |
| `vcsum` | `CASK` | `384` | `58.20%` | `82.03%` | `2.013` | `6` | `2` | `3` | `91.16%` |

## Readout

| Claim | Evidence |
| --- | --- |
| not prefix-only | `prefix_events = 2`, `decode_events = 3` |
| merge is active | `total_scratch_saved_tokens = 375`, `total_scratch_merged_groups = 6` |
| same-budget win holds | `256` and `384` both satisfy `CASK > TriAttention` on `top-1`, `top-5`, `mean NLL`, and `first mismatch` |
| output is long enough | `reference_output_tokens = 512` |

## Provenance

- `experiments/frontier/Qwen3-8B/h100_decode_probe_vcsum_20260411_vcsum_replay/cask_budget_256.json`
- `experiments/frontier/Qwen3-8B/h100_decode_probe_vcsum_20260411_vcsum_replay/cask_budget_384.json`
- `experiments/frontier/Qwen3-8B/h100_decode_probe_vcsum_20260411_vcsum_replay/triattention_budget_256.json`
- `experiments/frontier/Qwen3-8B/h100_decode_probe_vcsum_20260411_vcsum_replay/triattention_budget_384.json`
