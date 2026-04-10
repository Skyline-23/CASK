#!/usr/bin/env python3
"""Run a historical Qwen3-8B FullKV vs TriAttention pair and compare the results."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = REPO_ROOT / "scripts" / "cli.py"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_experiment_runs.py"
OUTPUTS_ROOT = REPO_ROOT / "experiments" / "outputs"
COMPARISONS_ROOT = REPO_ROOT / "experiments" / "comparisons"
DATASETS = ("aime24", "aime25", "math500")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--budget", required=True, type=int, help="TriAttention KV budget for the historical baseline pair.")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument(
        "--pair-tag",
        default=None,
        help="Shared suffix used to derive full/triattention run tags. Defaults to a timestamped tag.",
    )
    parser.add_argument("--gpus", default=None, help="GPU ids override forwarded to cli.py run-one.")
    parser.add_argument("--num-shards", type=int, default=None, help="Shard count override forwarded to cli.py.")
    parser.add_argument("--num-samples", type=int, default=None, help="Generation draws override.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional dataset cap for smoke tests.")
    parser.add_argument("--max-length", type=int, default=None, help="Generation max_length override.")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "flash_attention_2", "sdpa"],
    )
    parser.add_argument("--load-dtype", default=None, choices=["bfloat16", "float16"])
    parser.add_argument(
        "--full-extra-config",
        action="append",
        default=[],
        help="Extra YAML config merged into the FullKV run (repeatable).",
    )
    parser.add_argument(
        "--tri-extra-config",
        action="append",
        default=[],
        help="Extra YAML config merged into the TriAttention run (repeatable).",
    )
    parser.add_argument("--skip-fullkv", action="store_true", help="Do not launch the FullKV run.")
    parser.add_argument("--skip-triattention", action="store_true", help="Do not launch the TriAttention run.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return parser.parse_args()


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not cleaned:
        return "paper"
    return cleaned[:64]


def build_base_command(args: argparse.Namespace, method: str, run_tag: str) -> List[str]:
    cmd = [
        sys.executable,
        str(CLI_SCRIPT),
        "run-one",
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--method",
        method,
        "--run-tag",
        run_tag,
    ]
    if method != "fullkv":
        cmd.extend(["--budget", str(args.budget)])
    if args.gpus is not None:
        cmd.extend(["--gpus", args.gpus])
    if args.num_shards is not None:
        cmd.extend(["--num-shards", str(args.num_shards)])
    if args.num_samples is not None:
        cmd.extend(["--num-samples", str(args.num_samples)])
    if args.max_examples is not None:
        cmd.extend(["--max-examples", str(args.max_examples)])
    if args.max_length is not None:
        cmd.extend(["--max-length", str(args.max_length)])
    if args.attn_implementation is not None:
        cmd.extend(["--attn-implementation", args.attn_implementation])
    if args.load_dtype is not None:
        cmd.extend(["--load-dtype", args.load_dtype])
    return cmd


def add_extra_configs(cmd: List[str], paths: List[str]) -> None:
    for path in paths:
        cmd.extend(["--extra-config", path])


def resolve_run_dir(dataset: str, model: str, method: str, run_tag: str) -> Path:
    root = OUTPUTS_ROOT / dataset / model
    matches = sorted(root.glob(f"sample*/{method}/*_{run_tag}"))
    if not matches:
        raise FileNotFoundError(f"No output directory found for {dataset}/{model}/{method} with run tag {run_tag}")
    if len(matches) > 1:
        raise ValueError(f"Multiple output directories found for run tag {run_tag}: {matches}")
    return matches[0]


def run_command(cmd: List[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    if args.skip_fullkv and args.skip_triattention:
        raise SystemExit("At least one of FullKV or TriAttention must run.")

    base_tag = sanitize_tag(
        args.pair_tag or f"{args.dataset}_budget{args.budget}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    full_run_tag = f"{base_tag}_full"
    tri_run_tag = f"{base_tag}_tri"

    full_cmd = build_base_command(args, "fullkv", full_run_tag)
    add_extra_configs(full_cmd, args.full_extra_config)

    tri_cmd = build_base_command(args, "triattention", tri_run_tag)
    add_extra_configs(tri_cmd, args.tri_extra_config)

    if not args.skip_fullkv:
        run_command(full_cmd, args.dry_run)
    if not args.skip_triattention:
        run_command(tri_cmd, args.dry_run)

    if args.dry_run:
        return

    full_run_dir = resolve_run_dir(args.dataset, args.model, "fullkv", full_run_tag)
    tri_run_dir = resolve_run_dir(args.dataset, args.model, "triattention", tri_run_tag)

    summary_output = args.summary_output
    if summary_output is None:
        summary_output = COMPARISONS_ROOT / args.dataset / args.model / f"{base_tag}.json"

    compare_cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--baseline",
        str(full_run_dir),
        "--candidate",
        str(tri_run_dir),
        "--json-output",
        str(summary_output),
    ]
    run_command(compare_cmd, dry_run=False)

    print(f"full_run_dir={full_run_dir}")
    print(f"triattention_run_dir={tri_run_dir}")
    print(f"comparison_summary={summary_output.resolve()}")


if __name__ == "__main__":
    main()
