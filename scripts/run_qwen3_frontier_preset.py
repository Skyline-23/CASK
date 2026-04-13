#!/usr/bin/env python3
"""Qwen3-8B CASK frontier preset with conservative single-device defaults."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTIER_SCRIPT = REPO_ROOT / "scripts" / "run_cask_frontier.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["aime24", "aime25", "math500"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[384, 512, 768, 1024])
    parser.add_argument("--methods", nargs="+", default=["triattention", "cask"])
    parser.add_argument("--frontier-tag", default="qwen3_frontier")
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--core-selection-mode", choices=["vote", "score"], default="vote")
    parser.add_argument("--merge-operator", choices=["keepkv", "mean"], default="keepkv")
    parser.add_argument("--value-projection-threshold", type=float, default=None)
    parser.add_argument("--use-phase-markers", choices=["true", "false"], default="true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()
    cmd = [
        sys.executable,
        str(FRONTIER_SCRIPT),
        "--model",
        "Qwen3-8B",
        "--datasets",
        *args.datasets,
        "--methods",
        *args.methods,
        "--budgets",
        *[str(budget) for budget in args.budgets],
        "--frontier-tag",
        args.frontier_tag,
        "--gpus",
        args.gpus,
        "--num-shards",
        str(args.num_shards),
        "--num-samples",
        str(args.num_samples),
        "--max-examples",
        str(args.max_examples),
        "--attn-implementation",
        "sdpa",
        "--load-dtype",
        "float16",
        "--cask-core-selection-mode",
        args.core_selection_mode,
        "--cask-merge-operator",
        args.merge_operator,
        "--cask-use-phase-markers",
        args.use_phase_markers,
    ]
    if args.max_length is not None:
        cmd.extend(["--max-length", str(args.max_length)])
    if args.max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.stats_path is not None:
        cmd.extend(["--stats-path", str(args.stats_path)])
    if args.value_projection_threshold is not None:
        cmd.extend(
            [
                "--cask-value-projection-threshold",
                str(args.value_projection_threshold),
            ]
        )
    if args.summary_json is not None:
        cmd.extend(["--summary-json", str(args.summary_json)])
    if args.summary_csv is not None:
        cmd.extend(["--summary-csv", str(args.summary_csv)])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.extend(passthrough)

    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
