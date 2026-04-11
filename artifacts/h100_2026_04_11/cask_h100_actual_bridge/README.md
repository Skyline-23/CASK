# CASK H100 Actual-Bridge Assets

This directory packages the H100 output-level evidence that complements the replay-fidelity gates in `artifacts/h100_2026_04_10/cask_h100_fidelity/`.

Tracked here:
- actual-output bridge rows for `qasper`, `multi_news`, and `hotpotqa`
- the `multi_news` stage-ablation row needed to keep the stage-2 claim honest
- clean CSV / JSON / Markdown tables with direct provenance links
- output-level evaluation reported on official task metric, lexical overlap, and semantic/reference similarity

Primary files:
- `actual_bridge_summary.csv`
- `actual_bridge_summary.json`
- `actual_bridge_summary.md`
- `stage_ablation_summary.csv`
- `stage_ablation_summary.json`
- `stage_ablation_summary.md`

Artifact builders:
- `scripts/build_actual_bridge_artifacts.py`
- `scripts/build_promptheavy_saved_ratio_audit.py`

Experiment roots:
- replay metrics: `experiments/frontier/Qwen3-8B/h100_actual_bridge_metrics_20260411/`
- actual-generation runs: `experiments/longbench_h100_actual_bridge_20260411/`
- stage-ablation generation run: `experiments/longbench_h100_stage_ablation/`
- stage-ablation replay metrics: `experiments/frontier/Qwen3-8B/h100_promptheavy_stage_ablation_20260411/`

Headline read:
- `qasper`: `CASK @ 256` crosses above `TriAttention @ 512` on lexical overlap, semantic similarity, and task metric.
- `multi_news`: `CASK @ 384` beats `TriAttention @ 384` on lexical overlap, semantic similarity, and task metric at the same budget.
- `hotpotqa`: `CASK @ 256` is an output-parity non-regression witness rather than a win case.
- `multi_news` stage ablation stays in the package as a claim boundary: it does not justify a large standalone stage-2 output gain.

- bridge rows packaged: `6`
- stage-ablation rows packaged: `2`
