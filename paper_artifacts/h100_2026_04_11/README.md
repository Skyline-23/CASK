# H100 Prompt-Heavy Follow-Up Package

This directory is the prompt-heavy follow-up package. Open it when you want to understand the full prompt-heavy story: which tasks are decode-active wins, which ones are only prefix-stage wins, and which ones are retained boundaries.

## Read Order

1. [`promptheavy_saved_ratio_audit/promptheavy_replay_readout.md`](promptheavy_saved_ratio_audit/promptheavy_replay_readout.md)
2. [`decode_active_replay_probe.md`](decode_active_replay_probe.md)
3. [`coverage_followup_probe.md`](coverage_followup_probe.md)
4. [`cask_h100_actual_bridge/README.md`](cask_h100_actual_bridge/README.md) if you need the output-level bridge layer

## What Each File Is For

| File | What it answers | How to use it |
| --- | --- | --- |
| [`promptheavy_saved_ratio_audit/promptheavy_replay_readout.md`](promptheavy_saved_ratio_audit/promptheavy_replay_readout.md) | What are the main prompt-heavy replay numbers? | Use this for the wide replay table, weighted aggregates, and `384` scaling interpretation |
| [`decode_active_replay_probe.md`](decode_active_replay_probe.md) | Do we have another decode-active replay witness besides `multi_news`? | Use this to cite `vcsum` as the second replay-level decode-active witness |
| [`coverage_followup_probe.md`](coverage_followup_probe.md) | What happens when we broaden the prompt-heavy coverage set? | Use this to discuss `qmsum` / `gov_report` as `prefix_budget_exhausted` boundaries |
| [`cask_h100_actual_bridge/README.md`](cask_h100_actual_bridge/README.md) | Does replay advantage show up in actual generation? | Use this when you need output-level bridge evidence |

## Main Read

| Category | Current conclusion |
| --- | --- |
| Strong same-budget replay wins | `qasper`, `multi_news`, `hotpotqa`, `musique` |
| Replay-level decode-active witnesses | `multi_news`, `vcsum` |
| Output-level bridge read | `multi_news` is the clean decode-active bridge; `vcsum` is a lexical-semantic split boundary |
| Retained boundary | `2wikimqa` |
| Prefix-budget-exhausted boundaries | `qmsum`, `gov_report` |

## What To Cite For Which Claim

| Claim | Best file |
| --- | --- |
| "CASK has prompt-heavy same-budget replay wins" | [`promptheavy_saved_ratio_audit/promptheavy_replay_readout.md`](promptheavy_saved_ratio_audit/promptheavy_replay_readout.md) |
| "CASK is not just a prefix trick at replay level" | [`decode_active_replay_probe.md`](decode_active_replay_probe.md) |
| "Prompt-heavy evidence must be split into active and inactive regimes" | [`coverage_followup_probe.md`](coverage_followup_probe.md) |
| "The strongest decode-active output bridge is `multi_news`" | [`cask_h100_actual_bridge/README.md`](cask_h100_actual_bridge/README.md) |

## Command Provenance

| Item | File |
| --- | --- |
| Prompt-heavy batch manifest | [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411/overnight_manifest.json`](../../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411/overnight_manifest.json) |
| Package command map | [`paper_artifacts/COMMAND_MAP.md`](../COMMAND_MAP.md) |

## Raw Provenance

Use the tracked summaries first, then trace into these tracked raw roots:

- [`experiments/longbench_h100_refs/`](../../experiments/longbench_h100_refs/)
- [`experiments/frontier/Qwen3-8B/h100_actual_bridge_metrics_20260411/`](../../experiments/frontier/Qwen3-8B/h100_actual_bridge_metrics_20260411/)
- [`experiments/longbench_h100_actual_bridge_20260411/`](../../experiments/longbench_h100_actual_bridge_20260411/)
- [`experiments/longbench_h100_stage_ablation/`](../../experiments/longbench_h100_stage_ablation/)
- [`experiments/frontier/Qwen3-8B/h100_promptheavy_stage_ablation_20260411/`](../../experiments/frontier/Qwen3-8B/h100_promptheavy_stage_ablation_20260411/)
- [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411/`](../../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411/)
- [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_qasper_replay/`](../../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_qasper_replay/)
- [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_multi_news_replay/`](../../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_multi_news_replay/)
- [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_hotpotqa_replay/`](../../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_hotpotqa_replay/)
- [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_musique_replay/`](../../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_musique_replay/)
- [`experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_2wikimqa_replay/`](../../experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411_2wikimqa_replay/)
- [`experiments/frontier/Qwen3-8B/h100_decode_probe_vcsum_20260411_vcsum_replay/`](../../experiments/frontier/Qwen3-8B/h100_decode_probe_vcsum_20260411_vcsum_replay/)
- [`experiments/frontier/Qwen3-8B/h100_decode_probe_qmsum_20260411_qmsum_replay/`](../../experiments/frontier/Qwen3-8B/h100_decode_probe_qmsum_20260411_qmsum_replay/)
- [`experiments/frontier/Qwen3-8B/h100_decode_probe_gov_report_20260411_gov_report_replay/`](../../experiments/frontier/Qwen3-8B/h100_decode_probe_gov_report_20260411_gov_report_replay/)
