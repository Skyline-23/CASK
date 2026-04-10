#!/usr/bin/env python3
"""Run the paper benchmark bundle for one KV-compression method configuration."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "cli.py"
DFS_RUNNER = REPO_ROOT / "scripts" / "run_dfs_suite.py"
LONGBENCH_RUNNER = REPO_ROOT / "scripts" / "run_longbench_suite.py"
RULER_RUNNER = REPO_ROOT / "scripts" / "run_ruler_suite.py"
DEFAULT_DFS_DATASET = (
    REPO_ROOT / "triattention" / "benchmarks" / "dfs" / "datasets" / "legacy" / "dfs_state_query_100.json"
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli-model", type=str, required=True, help="Model alias used by scripts/cli.py (e.g. Qwen3-8B).")
    parser.add_argument("--model-path", type=Path, required=True, help="Local HF model path used by suite runners.")
    parser.add_argument("--method", type=str, required=True, choices=["fullkv", "triattention", "horizonkv", "cask", "expectedattention", "r1kv", "snapkv"])
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "bundle")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--run-tag", type=str, default="bundle")
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="TriAttention-family stats file forwarded to both run-one and direct suite workers.",
    )
    parser.add_argument("--suite-kv-budget", type=int, default=None, help="Shared KV budget for DFS/LongBench/RULER when method is budgeted.")
    parser.add_argument("--aime24-budget", type=int, default=4096)
    parser.add_argument("--aime25-budget", type=int, default=3072)
    parser.add_argument("--math500-budget", type=int, default=1024)
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap for smoke runs across all suites.")
    parser.add_argument("--longbench-e", action="store_true", help="Use LongBench-E instead of full LongBench.")
    parser.add_argument("--longbench-tasks", nargs="*", default=None)
    parser.add_argument("--ruler-tasks", nargs="*", default=None)
    parser.add_argument("--ruler-seq-lengths", nargs="*", type=int, default=[4096, 8192, 16384, 32768, 65536, 131072])
    parser.add_argument("--dfs-dataset", type=Path, default=DEFAULT_DFS_DATASET)
    parser.add_argument("--dfs-max-new-tokens", type=int, default=1024)
    parser.add_argument("--skip-reasoning", action="store_true")
    parser.add_argument("--skip-dfs", action="store_true")
    parser.add_argument("--skip-longbench", action="store_true")
    parser.add_argument("--skip-ruler", action="store_true")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def run_command(command: list[str], manifest: list[dict]) -> None:
    manifest.append({"command": command})
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def strip_passthrough_args(args: list[str], blocked_flags: set[str]) -> list[str]:
    """Drop CLI-only passthrough flags before forwarding to direct suite workers."""
    filtered: list[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token in blocked_flags:
            skip_next = True
            continue
        filtered.append(token)
    return filtered


def main() -> None:
    args, passthrough = parse_args()
    if args.method != "fullkv" and args.suite_kv_budget is None:
        raise ValueError("--suite-kv-budget is required for non-fullkv methods.")
    if args.method in {"triattention", "horizonkv", "cask"} and args.stats_path is None:
        raise ValueError("--stats-path is required for triattention-family bundle runs.")

    bundle_root = args.output_root / args.cli_model / args.method / args.run_tag
    bundle_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    common_suite_args = strip_passthrough_args(list(passthrough), {"--gpus"})
    common_suite_args.extend(["--attn-implementation", "sdpa"])
    common_suite_args.extend(["--method", args.method])
    if args.stats_path is not None:
        common_suite_args.extend(["--stats-path", str(args.stats_path)])
    if args.method != "fullkv":
        common_suite_args.extend(["--kv-budget", str(args.suite_kv_budget)])

    if not args.skip_reasoning:
        reasoning_jobs = [
            ("aime24", args.aime24_budget),
            ("aime25", args.aime25_budget),
            ("math500", args.math500_budget),
        ]
        for dataset, budget in reasoning_jobs:
            cmd = [
                args.python,
                str(CLI_PATH),
                "run-one",
                "--model",
                args.cli_model,
                "--dataset",
                dataset,
                "--method",
                args.method,
                "--run-tag",
                f"{args.run_tag}_{dataset}",
            ]
            if args.method != "fullkv":
                cmd.extend(["--budget", str(budget)])
            if args.stats_path is not None:
                cmd.extend(["--stats-path", str(args.stats_path)])
            if args.max_examples is not None:
                cmd.extend(["--max-examples", str(args.max_examples)])
            cmd.extend(passthrough)
            run_command(cmd, manifest)

    if not args.skip_dfs:
        cmd = [
            args.python,
            str(DFS_RUNNER),
            "--model-path",
            str(args.model_path),
            "--output-root",
            str(bundle_root / "dfs"),
            "--dataset",
            str(args.dfs_dataset),
            "--max-new-tokens",
            str(args.dfs_max_new_tokens),
        ]
        if args.max_examples is not None:
            cmd.extend(["--max-examples", str(args.max_examples)])
        cmd.extend(common_suite_args)
        run_command(cmd, manifest)

    if not args.skip_longbench:
        cmd = [
            args.python,
            str(LONGBENCH_RUNNER),
            "--model-path",
            str(args.model_path),
            "--output-root",
            str(bundle_root / "longbench"),
        ]
        if args.longbench_e:
            cmd.append("--longbench-e")
        if args.longbench_tasks:
            cmd.extend(["--tasks", *args.longbench_tasks])
        if args.max_examples is not None:
            cmd.extend(["--max-examples", str(args.max_examples)])
        cmd.extend(common_suite_args)
        run_command(cmd, manifest)

    if not args.skip_ruler:
        cmd = [
            args.python,
            str(RULER_RUNNER),
            "--model-path",
            str(args.model_path),
            "--output-root",
            str(bundle_root / "ruler"),
            "--seq-lengths",
            *[str(length) for length in args.ruler_seq_lengths],
        ]
        if args.ruler_tasks:
            cmd.extend(["--tasks", *args.ruler_tasks])
        if args.max_examples is not None:
            cmd.extend(["--num-samples", str(args.max_examples)])
        cmd.extend(common_suite_args)
        run_command(cmd, manifest)

    manifest_path = bundle_root / "bundle_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Wrote bundle manifest to {manifest_path}")


if __name__ == "__main__":
    main()
