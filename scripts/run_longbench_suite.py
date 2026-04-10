#!/usr/bin/env python3
"""Prepare, run, merge, and evaluate LongBench v1 / LongBench-E through the HF worker path."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triattention.benchmarks.longbench.prepare import resolve_tasks, write_task_jsonl


WORKER_PATH = REPO_ROOT / "scripts" / "worker.py"
MERGE_PATH = REPO_ROOT / "scripts" / "merge_shards.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "longbench")
    parser.add_argument("--longbench-e", action="store_true")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="TriAttention/HorizonKV stats file forwarded to worker as --triattention_stats_file.",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()
    task_set = "longbench_e" if args.longbench_e else "longbench"
    tasks = resolve_tasks(args.longbench_e, args.tasks)
    suite_root = args.output_root / task_set / args.model_path.name
    prepared_root = suite_root / "prepared"
    runs_root = suite_root / "runs"
    merged_root = suite_root / "predictions"
    merged_root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        prepared_path = prepared_root / f"{task}.jsonl"
        if not args.eval_only:
            write_task_jsonl(task, prepared_path, use_e=args.longbench_e, max_examples=args.max_examples)
        if args.prepare_only:
            continue
        task_root = runs_root / task
        task_run_root = task_root / "raw"
        if not args.eval_only:
            worker_cmd = [
                args.python,
                str(WORKER_PATH),
                "--dataset-path",
                str(prepared_path),
                "--output-dir",
                str(task_run_root),
                "--model-path",
                str(args.model_path),
                "--shard-id",
                "0",
                "--num-shards",
                "1",
                "--num-samples",
                str(args.num_samples),
                "--do-sample",
                "false",
                "--max-length",
                "-1",
            ]
            if args.stats_path is not None and "--triattention_stats_file" not in passthrough:
                worker_cmd.extend(["--triattention_stats_file", str(args.stats_path)])
            worker_cmd.extend(passthrough)
            subprocess.run(worker_cmd, check=True, cwd=REPO_ROOT)
            subprocess.run(
                [
                    args.python,
                    str(MERGE_PATH),
                    "--method-output-dir",
                    str(task_run_root),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
        merged_path = task_root / "merged" / "merged.jsonl"
        if not merged_path.exists():
            raise FileNotFoundError(f"Merged output missing for task {task}: {merged_path}")
        shutil.copyfile(merged_path, merged_root / f"{task}.jsonl")

    subprocess.run(
        [
            args.python,
            "-m",
            "triattention.benchmarks.longbench.evaluate",
            "--pred-dir",
            str(merged_root),
            "--output",
            str(suite_root / "longbench_eval.json"),
            *(["--longbench-e"] if args.longbench_e else []),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
