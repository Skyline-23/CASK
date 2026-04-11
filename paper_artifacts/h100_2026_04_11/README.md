# H100 2026-04-11 Assets

- `cask_h100_actual_bridge/`
  - output-level bridge package for `qasper`, `multi_news`, and `hotpotqa`
  - includes the `multi_news` stage-ablation boundary row
- `decode_active_replay_probe.md`
  - replay-only decode-active follow-up for `vcsum`
  - packages the second same-budget decode-active witness
- `coverage_followup_probe.md`
  - replay-only coverage expansion for `qmsum` and `gov_report`
  - records two additional prompt-heavy boundaries under the same-budget protocol
- `promptheavy_saved_ratio_audit/`
  - saved-ratio audit package for prompt-heavy H100 replay rows
  - use `promptheavy_replay_readout.md` first for paper-facing interpretation

Experiment roots:
- `experiments/frontier/Qwen3-8B/h100_actual_bridge_metrics_20260411/`
- `experiments/longbench_h100_actual_bridge_20260411/`
- `experiments/longbench_h100_stage_ablation/`
- `experiments/frontier/Qwen3-8B/h100_promptheavy_stage_ablation_20260411/`
