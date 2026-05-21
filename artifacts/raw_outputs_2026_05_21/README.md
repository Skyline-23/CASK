# Raw Output Snapshot 2026-05-21

This package preserves compact raw outputs that were present in the local
workspace but ignored by the normal `experiments/outputs/` rule.

## Scope

Included snapshots:

- `experiments/outputs/longbench_2wikimqa`
- `experiments/outputs/longbench_multi_news`
- `experiments/outputs/longbench_qasper`
- `experiments/outputs/math500_actual_witnesses`

Excluded snapshots:

- `experiments/outputs/math500`

The excluded `math500` tree contains legacy exploratory runs from the pre-CASK
naming period. It is intentionally left out of this public artifact snapshot to
avoid mixing obsolete method names into the paper-facing evidence package.

## Why This Exists

The main paper should cite compact summaries and figures, not raw generation
logs. The raw logs still matter for provenance: they let a reader trace a
paper-facing table or claim back to the model outputs that produced it.

This directory is therefore an audit snapshot, not a new headline result.

## Integrity

`MANIFEST.sha256.csv` records SHA-256 hashes, byte sizes, and repository-relative
paths for every file in the snapshot.
