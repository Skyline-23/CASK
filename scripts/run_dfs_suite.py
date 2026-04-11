#!/usr/bin/env python3
"""Prepare, run, merge, and evaluate the DFS state-query benchmark via the HF worker path."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cask.benchmarks.dfs.scripts.prompt_utils import build_prompt


WORKER_PATH = REPO_ROOT / "scripts" / "worker.py"
MERGE_PATH = REPO_ROOT / "scripts" / "merge_shards.py"
DFS_EVAL_PATH = REPO_ROOT / "cask" / "benchmarks" / "dfs" / "scripts" / "eval_dfs_state_query_raw.py"
DEFAULT_DFS_DATASET = (
    REPO_ROOT / "cask" / "benchmarks" / "dfs" / "datasets" / "legacy" / "dfs_state_query_100.json"
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DFS_DATASET)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "dfs")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="TriAttention/CASK stats file forwarded to worker as --triattention_stats_file.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def prepare_dataset(source_path: Path, target_path: Path, *, max_examples: int | None, max_new_tokens: int) -> Path:
    with source_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(dataset):
            if max_examples is not None and index >= max_examples:
                break
            record = dict(item)
            record["index"] = index
            record["question"] = f"DFS state query sample {item.get('id', index)}"
            record["prompt"] = build_prompt(item)
            record["max_new_tokens"] = max_new_tokens
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target_path


def main() -> None:
    args, passthrough = parse_args()
    suite_root = args.output_root / args.model_path.name
    prepared_path = suite_root / "prepared" / "dfs_state_query.jsonl"
    task_root = suite_root / "runs"
    task_run_root = task_root / "raw"

    if not args.eval_only:
        prepare_dataset(args.dataset, prepared_path, max_examples=args.max_examples, max_new_tokens=args.max_new_tokens)
    if not args.prepare_only:
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
                "1",
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
            raise FileNotFoundError(f"Merged DFS output missing: {merged_path}")
        raw_output = suite_root / "dfs_raw_eval.jsonl"
        summary_output = suite_root / "dfs_summary.json"
        subprocess.run(
            [
                args.python,
                str(DFS_EVAL_PATH),
                "--merged-path",
                str(merged_path),
                "--dataset",
                str(args.dataset),
                "--output",
                str(raw_output),
                "--summary-output",
                str(summary_output),
            ],
            check=True,
            cwd=REPO_ROOT,
        )


if __name__ == "__main__":
    main()


