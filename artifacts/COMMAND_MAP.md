# Command Map

This file is the shortest path from a paper-facing table to the script and raw files that produced it.

## Runner Map

| Script | Role | Typical use |
| --- | --- | --- |
| `scripts/cli.py run-one` | one-off generation run | local sanity checks, single benchmark runs, direct `fullkv` / `triattention` / `cask` execution |
| `scripts/replay_reference_fidelity.py` | replay one candidate run against one full-KV reference | single replay comparison and per-run JSON/CSV generation |
| `scripts/run_replay_fidelity_frontier.py` | batch replay grid around one reference | `method x budget` frontier sweeps |
| `scripts/run_longbench_suite.py` | LongBench generation runner | full-KV references and actual-output bridge runs |
| `scripts/compare_kv_fidelity.py` | compare actual outputs against full-KV outputs | sequence ratio, prefix ratio, task-metric bridge |
| `scripts/build_promptheavy_saved_ratio_audit.py` | package prompt-heavy replay tables | turn replay JSONs into paper-facing prompt-heavy tables |
| `scripts/build_actual_bridge_artifacts.py` | package actual-output bridge tables | turn bridge JSONs and eval JSONs into paper-facing summaries |

## Package-to-Command Map

| Package | Main summary files | Producing scripts | Exact command source | Raw roots |
| --- | --- | --- | --- | --- |
| [`h100_2026_04_10/cask_h100_fidelity/`](h100_2026_04_10/cask_h100_fidelity/README.md) | [`aime24_ref6_h100_fidelity_summary.csv`](h100_2026_04_10/cask_h100_fidelity/aime24_ref6_h100_fidelity_summary.csv), [`aime25_ref6_h100_fidelity_summary.csv`](h100_2026_04_10/cask_h100_fidelity/aime25_ref6_h100_fidelity_summary.csv) | `scripts/replay_reference_fidelity.py` | [`experiments/frontier/Qwen3-8B/h100_aime24_fidelity_gate_20260410/fidelity_manifest.json`](../experiments/frontier/Qwen3-8B/h100_aime24_fidelity_gate_20260410/fidelity_manifest.json), [`experiments/frontier/Qwen3-8B/h100_aime25_fidelity_gate_20260410/fidelity_manifest.json`](../experiments/frontier/Qwen3-8B/h100_aime25_fidelity_gate_20260410/fidelity_manifest.json) | [`experiments/outputs/aime24/Qwen3-8B/sample1/fullkv/full_h100_aime24_ref6_fidelity_20260410/`](../experiments/outputs/aime24/Qwen3-8B/sample1/fullkv/full_h100_aime24_ref6_fidelity_20260410/), [`experiments/outputs/aime25/Qwen3-8B/sample1/fullkv/full_h100_aime25_ref6_fidelity_20260410/`](../experiments/outputs/aime25/Qwen3-8B/sample1/fullkv/full_h100_aime25_ref6_fidelity_20260410/), [`experiments/frontier/Qwen3-8B/h100_aime24_fidelity_gate_20260410/`](../experiments/frontier/Qwen3-8B/h100_aime24_fidelity_gate_20260410/), [`experiments/frontier/Qwen3-8B/h100_aime25_fidelity_gate_20260410/`](../experiments/frontier/Qwen3-8B/h100_aime25_fidelity_gate_20260410/) |
| [`h100_2026_04_11/`](h100_2026_04_11/README.md) | [`promptheavy_replay_readout.md`](h100_2026_04_11/promptheavy_saved_ratio_audit/promptheavy_replay_readout.md), [`decode_active_replay_probe.md`](h100_2026_04_11/decode_active_replay_probe.md), [`coverage_followup_probe.md`](h100_2026_04_11/coverage_followup_probe.md) | `scripts/run_longbench_suite.py`, `scripts/run_replay_fidelity_frontier.py`, `scripts/build_promptheavy_saved_ratio_audit.py` | [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411/overnight_manifest.json`](../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411/overnight_manifest.json), per-task [`fidelity_manifest.json`](../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_qasper_replay/fidelity_manifest.json) roots | [`experiments/longbench_h100_refs/`](../experiments/longbench_h100_refs/), [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411/`](../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411/), [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_qasper_replay/`](../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_qasper_replay/), [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_multi_news_replay/`](../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_multi_news_replay/), [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_hotpotqa_replay/`](../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_hotpotqa_replay/), [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_musique_replay/`](../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_musique_replay/), [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_2wikimqa_replay/`](../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_2wikimqa_replay/), [`experiments/frontier/Qwen3-8B/h100_decode_probe_vcsum_20260411_vcsum_replay/`](../experiments/frontier/Qwen3-8B/h100_decode_probe_vcsum_20260411_vcsum_replay/), [`experiments/frontier/Qwen3-8B/h100_decode_probe_qmsum_20260411_qmsum_replay/`](../experiments/frontier/Qwen3-8B/h100_decode_probe_qmsum_20260411_qmsum_replay/), [`experiments/frontier/Qwen3-8B/h100_decode_probe_gov_report_20260411_gov_report_replay/`](../experiments/frontier/Qwen3-8B/h100_decode_probe_gov_report_20260411_gov_report_replay/) |
| [`h100_2026_04_11/cask_h100_actual_bridge/`](h100_2026_04_11/cask_h100_actual_bridge/README.md) | [`actual_bridge_summary.md`](h100_2026_04_11/cask_h100_actual_bridge/actual_bridge_summary.md), [`stage_ablation_summary.md`](h100_2026_04_11/cask_h100_actual_bridge/stage_ablation_summary.md) | `scripts/run_longbench_suite.py`, `scripts/compare_kv_fidelity.py`, `scripts/build_actual_bridge_artifacts.py` | packaged [`actual_bridge_summary.json`](h100_2026_04_11/cask_h100_actual_bridge/actual_bridge_summary.json) and [`stage_ablation_summary.json`](h100_2026_04_11/cask_h100_actual_bridge/stage_ablation_summary.json) carry `source_json` / `source_eval_json` back-pointers | [`experiments/frontier/Qwen3-8B/h100_actual_bridge_metrics_20260411/`](../experiments/frontier/Qwen3-8B/h100_actual_bridge_metrics_20260411/), [`experiments/longbench_h100_actual_bridge_20260411/`](../experiments/longbench_h100_actual_bridge_20260411/), [`experiments/longbench_h100_stage_ablation/`](../experiments/longbench_h100_stage_ablation/), [`experiments/frontier/Qwen3-8B/h100_promptheavy_stage_ablation_20260411/`](../experiments/frontier/Qwen3-8B/h100_promptheavy_stage_ablation_20260411/) |

## Representative Commands

These are command patterns that match the tracked runs.

### H100 reasoning replay gate

```bash
python scripts/replay_reference_fidelity.py \
  --reference experiments/outputs/aime24/Qwen3-8B/sample1/fullkv/full_h100_aime24_ref6_fidelity_20260410/merged/merged.jsonl \
  --model-path experiments/models/Qwen3-8B \
  --method cask \
  --budget 384 \
  --triattention-stats-file cask/calibration/for_aime25_experiment/qwen3_8b.pt \
  --max-records 6 \
  --attn-implementation sdpa \
  --load-dtype bfloat16 \
  --json-output experiments/frontier/Qwen3-8B/h100_aime24_fidelity_gate_20260410/cask_budget_384.json \
  --csv-output experiments/frontier/Qwen3-8B/h100_aime24_fidelity_gate_20260410/cask_budget_384.csv
```

### H100 prompt-heavy replay matrix

```bash
python scripts/run_longbench_suite.py \
  --model-path experiments/models/Qwen3-8B \
  --output-root experiments/longbench_h100_refs \
  --tasks multi_news \
  --max-examples 1 \
  --method fullkv \
  --attn_implementation sdpa \
  --load_dtype bfloat16 \
  --max-length -1

python scripts/run_replay_fidelity_frontier.py \
  --reference experiments/longbench_h100_refs/longbench/Qwen3-8B/runs/multi_news/merged/merged.jsonl \
  --model-path experiments/models/Qwen3-8B \
  --methods triattention cask \
  --budgets 256 384 \
  --triattention-stats-file cask/calibration/for_aime25_experiment/qwen3_8b.pt \
  --frontier-tag h100_promptheavy_twostage_rerun_20260411_multi_news_replay \
  --job-parallel 2 \
  --max-records 1 \
  --attn-implementation sdpa \
  --load-dtype bfloat16 \
  --count-prompt-tokens true \
  --slack-budget-trigger true \
  --allow-prefill-compression false
```

### H100 actual-output bridge

```bash
python scripts/run_longbench_suite.py \
  --model-path experiments/models/Qwen3-8B \
  --output-root experiments/longbench_h100_actual_bridge_20260411/qasper_cask256 \
  --tasks qasper \
  --max-examples 1 \
  --method cask \
  --budget 256 \
  --stats-path cask/calibration/for_aime25_experiment/qwen3_8b.pt \
  --attn_implementation sdpa \
  --load_dtype bfloat16

python scripts/compare_kv_fidelity.py \
  --reference experiments/longbench_h100_refs/longbench/Qwen3-8B/predictions/qasper.jsonl \
  --candidate experiments/longbench_h100_actual_bridge_20260411/qasper_cask256/longbench/Qwen3-8B/predictions/qasper.jsonl \
  --json-output experiments/frontier/Qwen3-8B/h100_actual_bridge_metrics_20260411/qasper_cask256.json \
  --csv-output experiments/frontier/Qwen3-8B/h100_actual_bridge_metrics_20260411/qasper_cask256.csv
```

## Cleanup Rule

Tracked evidence should keep `manifest`, `json`, `csv`, `md`, and raw eval/reference trees. Temporary launcher logs, smoke-run directories, and ad-hoc status notes do not belong in the paper-facing artifact set.

