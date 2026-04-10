# A100 Paper Artifacts, 2026-04-09

This directory contains the raw paper-support artifacts referenced by `docs/a100_validation_log.md`.

The artifacts are intentionally scoped. They include raw generations/evaluations for completed A100 checks and the raw score-dump tensors for the short KV-compression probe. They do not include model weights, calibration stats, cache directories, venvs, or full ignored experiment trees.

## Contents

| path | contents |
| --- | --- |
| `aime24_paper_v04/` | FullKV, TriAttention budget2048, and HorizonKV budget2048 raw merged generations, evaluator JSONL, and metric JSON. |
| `aime25_claim_core/` | FullKV, TriAttention budget512, and HorizonKV budget512 raw merged generations, evaluator JSONL, and metric JSON. |
| `idx13_budget640_control/` | Targeted AIME25 idx13 HorizonKV budget640 control: merged generations, evaluator JSONL, and metric JSON. Quality-control only; no score dump was captured for this run. |
| `idx13_budget1024_control/` | Targeted AIME25 idx13 HorizonKV budget1024 control: merged generations, evaluator JSONL, and metric JSON. |
| `kvprobe_512/kvprobe_score_dump_metadata.json` | JSON metadata/retention summary extracted from the score-dump tensors. |
| `kvprobe_512/raw_score_dumps/` | Raw `.pt` score-dump tensors for the AIME25 idx0 budget512 decode probe. |
| `summary/` | Completed compare JSON and claim snapshot from the A100 claim-core run. |
| `MANIFEST.sha256` | SHA256 checksums for artifact files. |

## Important Scope Notes

- AIME25 budget512 is a stress result, not a non-regression result versus FullKV.
- HorizonKV budget640 and budget1024 were run only as targeted controls on AIME25 `idx=13`.
- KV probe tensors are instrumentation artifacts. They support compression-event inspection, but are not a whole-benchmark memory average.
- Historical logs/configs remain ignored outside this artifact subset.
