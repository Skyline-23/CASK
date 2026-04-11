# Actual-Bridge Summary

| Task | Variant | Budget | Sequence Ratio | Prefix Ratio | Semantic Sim. | Output Ratio | Task Metric | Terminal Saved | Compression Events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qasper` | `TriAttention @ 512` | `512` | `17.3%` | `3.7%` | `0.678` | `100.0%` | `11.94` | `-` | `-` |
| `qasper` | `CASK @ 256` | `256` | `23.8%` | `5.5%` | `0.791` | `100.0%` | `12.77` | `90.9%` | `1.0` |
| `multi_news` | `TriAttention @ 384` | `384` | `0.0%` | `0.0%` | `0.452` | `100.0%` | `0.00` | `-` | `-` |
| `multi_news` | `CASK @ 384` | `384` | `16.9%` | `8.1%` | `0.952` | `100.0%` | `15.16` | `84.1%` | `3.0` |
| `hotpotqa` | `TriAttention @ 256` | `256` | `100.0%` | `100.0%` | `1.000` | `100.0%` | `27.27` | `-` | `-` |
| `hotpotqa` | `CASK @ 256` | `256` | `100.0%` | `100.0%` | `1.000` | `100.0%` | `27.27` | `97.6%` | `0.0` |

## Readout

- `qasper`: `CASK @ 256` beats `TriAttention @ 512` on `sequence_ratio`, official task metric, and semantic similarity, giving the cleanest actual-output budget crossing in the current H100 package.
- `multi_news`: `CASK @ 384` recovers a non-zero actual-output bridge where `TriAttention @ 384` collapses on both lexical overlap and task metric; semantic similarity shows the same direction even more strongly.
- `hotpotqa`: `CASK @ 256` matches `TriAttention @ 256` exactly at the output level on this witness while still ending with a `97.6%` terminal saved ratio.