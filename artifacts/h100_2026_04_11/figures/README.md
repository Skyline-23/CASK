# H100 Figure Pack

Rendered figures are now stored under [`docs/assets/`](../../../docs/assets/).
This directory remains only as the provenance anchor for the H100 figure-producing summaries.

## Source packages

- [`artifacts/h100_2026_04_10/cask_h100_fidelity/`](../h100_2026_04_10/cask_h100_fidelity/)
- [`artifacts/h100_2026_04_11/promptheavy_saved_ratio_audit/`](../promptheavy_saved_ratio_audit/)
- [`artifacts/h100_2026_04_11/cask_h100_actual_bridge/`](../cask_h100_actual_bridge/)
- [`artifacts/h100_2026_04_11/decode_active_replay_probe.md`](../decode_active_replay_probe.md)
- [`artifacts/h100_2026_04_11/coverage_followup_probe.md`](../coverage_followup_probe.md)

## Regeneration

```bash
python scripts/build_paper_figures.py
```

