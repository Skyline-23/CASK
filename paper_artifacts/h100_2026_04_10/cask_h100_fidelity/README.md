# CASK H100 Fidelity Assets

This directory tracks the H100 paper-facing fidelity assets that are stable
enough to cite directly in the manuscript.

Tracked here:
- compact summary tables derived from completed H100 replay runs
- crossing summaries that are small enough to review by hand
- provenance links back to ignored raw outputs under `experiments/frontier/`

Not tracked here:
- raw H100 launcher logs
- raw merged outputs and replay JSONs under `experiments/frontier/`
- prompt-heavy reference caches under `experiments/longbench_h100_refs/`
- local overnight orchestration glue such as `scripts/run_h100_fidelity_overnight.py`

Current scope:
- model: `Qwen3-8B`
- hardware: `H100 PCIe`
- dataset slices:
  - `AIME24` reference replay on `6` examples
  - `AIME25` reference replay on `6` examples
- methods: `triattention`, `cask`
- budgets: `256`, `384`, `512`

Companion package:
- output-level bridge evidence is packaged separately under
  `paper_artifacts/h100_2026_04_11/cask_h100_actual_bridge/`

Primary files:
- `aime24_ref6_h100_fidelity_summary.csv`
- `aime24_ref6_h100_fidelity_summary.json`
- `aime25_ref6_h100_fidelity_summary.csv`
- `aime25_ref6_h100_fidelity_summary.json`
- `MANIFEST.sha256`

Headline read from the packaged H100 gates:
- `AIME24`: same-budget `cask` beats `triattention` on `top1`, `top5`, and
  `mean_nll` at `256`, `384`, and `512`
- `AIME24`: `cask @ 256` is already ahead of `triattention @ 384`
- `AIME24`: `cask @ 384` is ahead of `triattention @ 512` while preserving
  much higher KV savings
- `AIME25`: same-budget `cask` again beats `triattention` on `top1`, `top5`,
  and `mean_nll` at `256`, `384`, and `512`
- `AIME25`: `cask @ 384` is ahead of `triattention @ 512`
- `AIME25`: `cask @ 256` does not beat `triattention @ 384`, so the crossing
  pattern is real but weaker than on `AIME24`

Important caveat:
- these are **fidelity** assets, not final benchmark-accuracy claims
- they are intended to support the paper's full-KV similarity story and to
  guide which larger H100 packages are worth running next

Raw provenance:
- all summary rows include `source_json`
- current raw source roots:
  - `experiments/frontier/Qwen3-8B/h100_aime24_fidelity_gate_20260410/`
  - `experiments/frontier/Qwen3-8B/h100_aime25_fidelity_gate_20260410/`
