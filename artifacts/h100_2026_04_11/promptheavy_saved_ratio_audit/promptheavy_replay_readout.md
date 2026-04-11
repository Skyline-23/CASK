# Prompt-Heavy Replay Readout

## Weighted aggregate (percent)

| Method | Budget | Weighted Top-1 | Weighted Top-5 |
| --- | ---: | ---: | ---: |
| `TriAttention` | `256` | `58.42%` | `82.74%` |
| `TriAttention` | `384` | `57.34%` | `82.74%` |
| `CASK` | `256` | `63.72%` | `88.59%` |
| `CASK` | `384` | `64.81%` | `89.40%` |

## Weighted aggregate (measured)

| Method | Budget | Top-1 Matches | Top-5 Matches | Total Replay Tokens | Weighted Mean NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `TriAttention` | `256` | `430` | `609` | `736` | `1.985` |
| `TriAttention` | `384` | `422` | `609` | `736` | `2.002` |
| `CASK` | `256` | `469` | `652` | `736` | `1.594` |
| `CASK` | `384` | `477` | `658` | `736` | `1.521` |

## Same-budget comparison (percent)

| Dataset | Budget | Tri Top-1 | CASK Top-1 | Tri Top-5 | CASK Top-5 | Tri Mean NLL | CASK Mean NLL | Saved Ratio | CASK Compression Events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qasper` | `256` | `67.19%` | `71.09%` | `89.84%` | `93.75%` | `1.315` | `1.247` | `90.90%` | `0` |
| `qasper` | `384` | `66.41%` | `73.44%` | `90.62%` | `92.19%` | `1.398` | `1.297` | `87.85%` | `0` |
| `multi_news` | `256` | `54.69%` | `59.96%` | `82.03%` | `87.89%` | `2.060` | `1.652` | `88.07%` | `3` |
| `multi_news` | `384` | `53.71%` | `61.33%` | `81.64%` | `89.45%` | `2.052` | `1.540` | `84.07%` | `3` |
| `hotpotqa` | `256` | `81.25%` | `93.75%` | `90.62%` | `100.00%` | `1.374` | `0.151` | `97.57%` | `0` |
| `hotpotqa` | `384` | `81.25%` | `96.88%` | `90.62%` | `100.00%` | `1.344` | `0.110` | `96.49%` | `0` |
| `musique` | `256` | `59.38%` | `65.62%` | `71.88%` | `75.00%` | `2.697` | `2.713` | `98.27%` | `0` |
| `musique` | `384` | `53.12%` | `62.50%` | `75.00%` | `75.00%` | `2.862` | `2.650` | `97.49%` | `0` |
| `2wikimqa` | `256` | `59.38%` | `62.50%` | `68.75%` | `81.25%` | `3.368` | `2.375` | `96.14%` | `0` |
| `2wikimqa` | `384` | `59.38%` | `56.25%` | `68.75%` | `81.25%` | `3.415` | `2.397` | `94.41%` | `0` |

## Same-budget comparison (measured)

| Dataset | Budget | Output Tokens | Tri Top-1 Matches | CASK Top-1 Matches | Tri Top-5 Matches | CASK Top-5 Matches | Tri First Mismatch | CASK First Mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qasper` | `256` | `128` | `86` | `91` | `115` | `120` | `2` | `4` |
| `qasper` | `384` | `128` | `85` | `94` | `116` | `118` | `2` | `4` |
| `multi_news` | `256` | `512` | `280` | `307` | `420` | `450` | `2` | `2` |
| `multi_news` | `384` | `512` | `275` | `314` | `418` | `458` | `2` | `2` |
| `hotpotqa` | `256` | `32` | `26` | `30` | `29` | `32` | `2` | `11` |
| `hotpotqa` | `384` | `32` | `26` | `31` | `29` | `32` | `2` | `11` |
| `musique` | `256` | `32` | `19` | `21` | `23` | `24` | `2` | `3` |
| `musique` | `384` | `32` | `17` | `20` | `24` | `24` | `2` | `3` |
| `2wikimqa` | `256` | `32` | `19` | `20` | `22` | `26` | `2` | `2` |
| `2wikimqa` | `384` | `32` | `19` | `18` | `22` | `26` | `2` | `2` |

## Budget scaling (percent)

| Dataset | Method | Top-1 Delta (`384 - 256`) | Top-5 Delta (`384 - 256`) | Mean NLL Delta | First Mismatch |
| --- | --- | ---: | ---: | ---: | --- |
| `qasper` | `TriAttention` | `-0.78%p` | `+0.78%p` | `+0.083` | `2 -> 2` |
| `qasper` | `CASK` | `+2.34%p` | `-1.56%p` | `+0.050` | `4 -> 4` |
| `multi_news` | `TriAttention` | `-0.98%p` | `-0.39%p` | `-0.008` | `2 -> 2` |
| `multi_news` | `CASK` | `+1.37%p` | `+1.56%p` | `-0.112` | `2 -> 2` |
| `hotpotqa` | `TriAttention` | `+0.00%p` | `+0.00%p` | `-0.031` | `2 -> 2` |
| `hotpotqa` | `CASK` | `+3.13%p` | `+0.00%p` | `-0.041` | `11 -> 11` |
| `musique` | `TriAttention` | `-6.25%p` | `+3.12%p` | `+0.165` | `2 -> 2` |
| `musique` | `CASK` | `-3.12%p` | `+0.00%p` | `-0.063` | `3 -> 3` |
| `2wikimqa` | `TriAttention` | `+0.00%p` | `+0.00%p` | `+0.047` | `2 -> 2` |
| `2wikimqa` | `CASK` | `-6.25%p` | `+0.00%p` | `+0.022` | `2 -> 2` |

## Budget scaling (measured)

| Dataset | Method | Output Tokens | Top-1 Match Count Delta | Top-5 Match Count Delta | Mean NLL Delta | First Mismatch |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `qasper` | `TriAttention` | `128` | `-1` | `+1` | `+0.083` | `2 -> 2` |
| `qasper` | `CASK` | `128` | `+3` | `-2` | `+0.050` | `4 -> 4` |
| `multi_news` | `TriAttention` | `512` | `-5` | `-2` | `-0.008` | `2 -> 2` |
| `multi_news` | `CASK` | `512` | `+7` | `+8` | `-0.112` | `2 -> 2` |
| `hotpotqa` | `TriAttention` | `32` | `0` | `0` | `-0.031` | `2 -> 2` |
| `hotpotqa` | `CASK` | `32` | `+1` | `0` | `-0.041` | `11 -> 11` |
| `musique` | `TriAttention` | `32` | `-2` | `+1` | `+0.165` | `2 -> 2` |
| `musique` | `CASK` | `32` | `-1` | `0` | `-0.063` | `3 -> 3` |
| `2wikimqa` | `TriAttention` | `32` | `0` | `0` | `+0.047` | `2 -> 2` |
| `2wikimqa` | `CASK` | `32` | `-2` | `0` | `+0.022` | `2 -> 2` |
