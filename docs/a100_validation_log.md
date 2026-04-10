# A100 Validation Log

This note records A100 observations from 2026-04-09. Raw experiment outputs are ignored; this file preserves the paper-relevant conclusions and the exact scope of each check.

Tracked raw artifacts for the observations in this note are stored under `paper_artifacts/a100_2026_04_09/`.

## Paper-Safe Takeaway

- HorizonKV is wired as `adaptive + rms2` and its dump metadata confirms that identity in the A100 probe.
- KV compression is real on the HF path: in a decode probe with budget 512, both TriAttention and HorizonKV pruned resident per-head cache to 512 entries.
- Do not claim broad AIME25 non-regression at budget 512. The completed AIME25 stress run regressed vs FullKV by one draw on one question.
- The observed AIME25 regression was budget-sensitive: on the regressed question, HorizonKV with budget 1024 reproduced the FullKV prediction vector.

## Completed Reasoning Runs

### AIME24, sample8, max_new_tokens 512, budget 2048

| method | accuracy |
| --- | ---: |
| fullkv | 0.4 |
| triattention | 0.4 |
| horizonkv | 0.4 |

Scope note: this is a no-difference wiring/control observation. The absolute score is low and should not be presented as a standalone paper benchmark.

### AIME25, sample8, max_new_tokens 512

| method | budget | accuracy |
| --- | ---: | ---: |
| fullkv | none | 2.1 |
| triattention | 512 | 1.7 |
| horizonkv | 512 | 1.7 |

Comparison summary:

| comparison | acc delta | output tokens/s ratio | question-level result |
| --- | ---: | ---: | --- |
| horizonkv 512 vs triattention 512 | 0.0 | 0.996542 | 30 unchanged |
| horizonkv 512 vs fullkv | -0.4 | 0.898826 | 29 unchanged, 1 regressed |

The only differing AIME25 question in this run was `idx=13`.

| method | budget | idx=13 pass@1 | idx=13 predictions |
| --- | ---: | ---: | --- |
| fullkv | none | 62.5 | `['60', '0', '6.062', '60', '60', '60', '60', '0']` |
| triattention | 512 | 50.0 | `['60', '0', '6.062', '0', '0', '60', '60', '60']` |
| horizonkv | 512 | 50.0 | `['6', '0', '6.062', '60', '0', '60', '60', '60']` |
| horizonkv targeted control | 640 | 62.5 | `['60', '0', '6.062', '60', '60', '60', '60', '0']` |
| horizonkv targeted control | 1024 | 62.5 | `['60', '0', '6.062', '60', '60', '60', '60', '0']` |

Interpretation: the observed `idx=13` regression is consistent with an overly aggressive 512-token budget in this setting. The targeted 640- and 1024-budget controls recovered the FullKV prediction vector for the same problem and same draw seeds.

Scope note: with `slack_budget_trigger=true` and `divide_length=128`, the budget-640 targeted control has an expected trigger threshold of 768. The observed output length for this question was 697 total tokens, so the budget-640 control is a budget-stress/quality control, not a compression-event observation.

## KV Compression Probe

Probe setup:

- dataset/problem: AIME25 `idx=0`
- max_new_tokens: 700
- KV budget: 512
- score dump max events: 64
- This is an instrumentation probe, not a quality benchmark.

Observed score-dump events:

| method | score mode | event | absolute position | cache before event | retained per KV head | event-local reduction | retained vs full position |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| triattention | `fixed + tri` | 0 | 640 | 640 | 512 | 20.0% | 20.0% |
| triattention | `fixed + tri` | 1 | 768 | 640 | 512 | 20.0% | 33.3% |
| horizonkv | `adaptive + rms2` | 0 | 640 | 640 | 512 | 20.0% | 20.0% |
| horizonkv | `adaptive + rms2` | 1 | 768 | 640 | 512 | 20.0% | 33.3% |

Interpretation: the HF compression path pruned to the requested 512-entry per-head cache. At absolute position 768, retaining 512 entries corresponds to a 33.3% resident-cache reduction relative to uncompressed FullKV at that position.

## How To Use This In The Paper

Supported by these observations:

- HorizonKV can execute on A100 using v2 stats and the `adaptive + rms2` score path.
- The HF path applies actual KV compression events and retains the configured budget.
- In the AIME25 stress run, HorizonKV matched TriAttention quality but did not beat it.
- The budget-512 AIME25 regression is not evidence of a broken HorizonKV formula; a targeted budget-1024 control recovered the FullKV outputs on the single regressed question.

Do not claim from this log alone:

- Do not claim HorizonKV is broadly more accurate than TriAttention.
- Do not claim AIME25 budget-512 has no regression vs FullKV.
- Do not claim throughput speedup from the current HF scorer path. The completed AIME25 HF run shows HorizonKV slower than FullKV and essentially tied with TriAttention.
- Do not claim the compression probe's 33.3% resident-cache reduction as whole-benchmark average KV memory reduction.

## 2026-04-09 Follow-Up: Unsafe RMS2 Default

After this artifact bundle was archived, a targeted follow-up tested the old `horizonkv` default on AIME25 `idx=13`, draw 0, budget 544.

Observation:

- `triattention` / `fixed + tri`: predicted `60`
- old `horizonkv` / `adaptive + rms2`: predicted `6`
- `horizonkv` ablation / `adaptive + tri`: predicted `60`
- `horizonkv` ablation / `fixed + rms2`: predicted `60`
- `horizonkv` ablation / `adaptive + rms2`, `norm_lambda=0.25`: predicted `6`
- `horizonkv` ablation / `adaptive + rms2`, `norm_lambda=0.0`: predicted `60`

Code default correction:

- HF/CLI `horizonkv` now defaults to `adaptive + tri`.
- `rms2` remains available through `--triattention-norm-mode rms2`, but it should be treated as a risky ablation until it has a larger paired validation set.

Paper implication:

- Keep the paper claim centered on the adaptive horizon kernel.
- Do not position RMS2 as an established quality improvement.
