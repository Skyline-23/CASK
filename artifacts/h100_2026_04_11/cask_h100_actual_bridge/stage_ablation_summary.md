# Stage-Ablation Summary

| Task | Variant | Budget | Sequence Ratio | Prefix Ratio | Task Metric | Terminal Saved | Compression Events | Cumulative Saved Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multi_news` | `CASK @ 384 (full two-stage)` | `384` | `16.9%` | `8.1%` | `15.16` | `84.1%` | `3.0` | `375.0` |
| `multi_news` | `CASK @ 384 (stage 1 only)` | `384` | `16.9%` | `8.1%` | `16.25` | `84.1%` | `3.0` | `0.0` |

## Readout

- `multi_news` stage ablation is useful as a guardrail, not as a standalone stage-2 headline.
- On this single witness, `stage 1 only` and `full two-stage` have nearly identical output-level similarity; the current paper should claim the overall two-stage policy, not a large standalone decode-merge output gain from this one case.