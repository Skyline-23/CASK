# Coverage Follow-up Probe

## `qmsum` same-budget replay

| Dataset | Method | Budget | Top-1 | Top-5 | Mean NLL | First Mismatch | Guard | Decode Events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `qmsum` | `TriAttention` | `256` | `82.62%` | `92.58%` | `0.845` | `12` | `-` | `-` |
| `qmsum` | `CASK` | `256` | `98.83%` | `100.00%` | `0.109` | `12` | `prefix_budget_exhausted` | `0` |
| `qmsum` | `TriAttention` | `384` | `95.31%` | `97.85%` | `0.309` | `12` | `-` | `-` |
| `qmsum` | `CASK` | `384` | `99.41%` | `100.00%` | `0.111` | `114` | `prefix_budget_exhausted` | `0` |

## `gov_report` same-budget replay

| Dataset | Method | Budget | Top-1 | Top-5 | Mean NLL | First Mismatch | Guard | Decode Events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `gov_report` | `TriAttention` | `256` | `89.65%` | `95.90%` | `0.595` | `353` | `-` | `-` |
| `gov_report` | `CASK` | `256` | `84.77%` | `92.38%` | `1.081` | `353` | `prefix_budget_exhausted` | `0` |
| `gov_report` | `TriAttention` | `384` | `98.24%` | `99.61%` | `0.240` | `481` | `-` | `-` |
| `gov_report` | `CASK` | `384` | `96.09%` | `98.44%` | `0.422` | `326` | `prefix_budget_exhausted` | `0` |

## Readout

| Probe | Takeaway |
| --- | --- |
| `qmsum` | broad coverage check succeeds as a same-budget replay win for CASK, but the win is prefix-driven because decode never activates |
| `gov_report` | broad coverage check becomes a failure boundary: CASK hits the same prefix budget guard and loses to TriAttention |
| shared message | these follow-ups expand benchmark coverage, but they do not add new decode-active evidence; they sharpen the regime split instead |

## Provenance

- `experiments/frontier/Qwen3-8B/h100_decode_probe_qmsum_20260411_qmsum_replay/`
- `experiments/frontier/Qwen3-8B/h100_decode_probe_gov_report_20260411_gov_report_replay/`
