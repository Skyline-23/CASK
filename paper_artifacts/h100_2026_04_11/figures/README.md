# H100 Figure Pack

This directory stores the current paper-facing figure pack generated from the tracked H100 artifact summaries.

## Files

- [`fig1_reasoning_gate_frontier.png`](fig1_reasoning_gate_frontier.png) / [`fig1_reasoning_gate_frontier.pdf`](fig1_reasoning_gate_frontier.pdf): Figure 1. H100 reasoning replay gate frontier.
- [`fig2_promptheavy_aggregate.png`](fig2_promptheavy_aggregate.png) / [`fig2_promptheavy_aggregate.pdf`](fig2_promptheavy_aggregate.pdf): Figure 2. H100 prompt-heavy weighted aggregate.
- [`fig3_promptheavy_witness_map.png`](fig3_promptheavy_witness_map.png) / [`fig3_promptheavy_witness_map.pdf`](fig3_promptheavy_witness_map.pdf): Figure 3. Simplified prompt-heavy witness summary by task and budget.
- [`fig4_actual_output_bridge.png`](fig4_actual_output_bridge.png) / [`fig4_actual_output_bridge.pdf`](fig4_actual_output_bridge.pdf): Figure 4. Actual-output bridge across lexical, semantic, and task-level metrics.
- [`fig5_method_overview.png`](fig5_method_overview.png) / [`fig5_method_overview.pdf`](fig5_method_overview.pdf): Figure 5. Method overview for two-stage CASK.

## Source packages

- [`paper_artifacts/h100_2026_04_10/cask_h100_fidelity/`](../h100_2026_04_10/cask_h100_fidelity/)
- [`paper_artifacts/h100_2026_04_11/promptheavy_saved_ratio_audit/`](../promptheavy_saved_ratio_audit/)
- [`paper_artifacts/h100_2026_04_11/cask_h100_actual_bridge/`](../cask_h100_actual_bridge/)
- [`paper_artifacts/h100_2026_04_11/decode_active_replay_probe.md`](../decode_active_replay_probe.md)
- [`paper_artifacts/h100_2026_04_11/coverage_followup_probe.md`](../coverage_followup_probe.md)

## Regeneration

```bash
python scripts/build_paper_figures.py
```
