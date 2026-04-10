#!/usr/bin/env python3
"""Run a quality-vs-budget frontier sweep for CASK comparisons."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = REPO_ROOT / "scripts" / "cli.py"
SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize_cask_frontier.py"
OUTPUTS_ROOT = REPO_ROOT / "experiments" / "outputs"
FRONTIER_ROOT = REPO_ROOT / "experiments" / "frontier"
DATASETS = ("aime24", "aime25", "math500")
METHODS = ("fullkv", "triattention", "cask", "expectedattention", "r1kv", "snapkv")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["triattention", "cask"],
        choices=METHODS,
    )
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument(
        "--frontier-tag",
        default=None,
        help="Shared suffix for all runs in this sweep. Defaults to a timestamped tag.",
    )
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--attn-implementation", default=None, choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--load-dtype", default=None, choices=["bfloat16", "float16"])
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--job-parallel", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--baseline-method", default="triattention", choices=METHODS)
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()
    if args.job_parallel < 1:
        raise SystemExit("--job-parallel must be >= 1")
    return args, passthrough


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not cleaned:
        return "frontier"
    return cleaned[:64]


def run_budget_tag(method: str, budget: int | None) -> str:
    if method == "fullkv":
        return "full"
    if budget is None:
        raise ValueError(f"budget is required for {method}")
    return f"budget_{int(budget)}"


def resolve_run_dir(model: str, dataset: str, method: str, budget: int | None, run_tag: str) -> Path | None:
    model_root = OUTPUTS_ROOT / dataset / model
    tag = f"{run_budget_tag(method, budget)}_{run_tag}"
    matches = sorted(model_root.glob(f"sample*/{method}/{tag}"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple output directories found for {dataset}/{model}/{method}/{tag}: {matches}")
    return matches[0]


def build_command(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    budget: int | None,
    frontier_tag: str,
    passthrough: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(CLI_SCRIPT),
        "run-one",
        "--dataset",
        dataset,
        "--model",
        args.model,
        "--method",
        method,
        "--run-tag",
        frontier_tag,
    ]
    if method != "fullkv":
        cmd.extend(["--budget", str(int(budget))])
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
    if args.stats_path is not None and method in {"triattention", "cask"}:
        cmd.extend(["--stats-path", str(args.stats_path)])
    cmd.extend(passthrough)
    return cmd


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def run_entries(entries: list[dict[str, Any]], *, dry_run: bool, job_parallel: int) -> None:
    if dry_run:
        for entry in entries:
            command = entry.get("command")
            if command:
                print(" ".join(command))
        return

    running: list[tuple[subprocess.Popen[Any], dict[str, Any]]] = []

    def drain_one() -> None:
        while True:
            for index, (proc, entry) in enumerate(running):
                ret = proc.poll()
                if ret is None:
                    continue
                running.pop(index)
                entry["return_code"] = ret
                entry["status"] = "completed" if ret == 0 else "failed"
                if ret != 0:
                    raise SystemExit(
                        f"[error] Frontier job failed: dataset={entry['dataset']} "
                        f"method={entry['method']} budget={entry['budget']} status={ret}"
                    )
                return
            time.sleep(1.0)

    for entry in entries:
        if entry.get("status") == "skipped_existing":
            continue
        command = entry.get("command")
        if not command:
            continue
        print(" ".join(command))
        proc = subprocess.Popen(command, cwd=str(REPO_ROOT))
        entry["status"] = "running"
        running.append((proc, entry))
        if len(running) >= job_parallel:
            drain_one()

    while running:
        drain_one()


def main() -> None:
    args, passthrough = parse_args()
    frontier_tag = sanitize_tag(args.frontier_tag or f"frontier_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    manifest_path = args.manifest_output
    if manifest_path is None:
        manifest_path = FRONTIER_ROOT / args.model / frontier_tag / "frontier_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries: List[dict[str, Any]] = []
    for dataset in args.datasets:
        for method in args.methods:
            budgets = [None] if method == "fullkv" else [int(budget) for budget in args.budgets]
            for budget in budgets:
                existing_run_dir = resolve_run_dir(args.model, dataset, method, budget, frontier_tag)
                entry: dict[str, Any] = {
                    "dataset": dataset,
                    "method": method,
                    "budget": budget,
                    "run_tag": frontier_tag,
                    "run_dir": str(existing_run_dir.resolve()) if existing_run_dir is not None else None,
                }
                if existing_run_dir is not None and args.skip_existing:
                    entry["status"] = "skipped_existing"
                    entries.append(entry)
                    continue

                cmd = build_command(args, dataset, method, budget, frontier_tag, passthrough)
                entry["command"] = cmd
                entry["status"] = "planned" if args.dry_run else "pending"
                entries.append(entry)

    payload = {
        "model": args.model,
        "frontier_tag": frontier_tag,
        "job_parallel": args.job_parallel,
        "datasets": list(args.datasets),
        "methods": list(args.methods),
        "budgets": [int(value) for value in args.budgets],
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"frontier_manifest={manifest_path.resolve()}")

    run_entries(entries, dry_run=args.dry_run, job_parallel=args.job_parallel)

    if not args.dry_run:
        for entry in entries:
            if entry.get("status") != "completed":
                continue
            resolved_run_dir = resolve_run_dir(
                args.model,
                entry["dataset"],
                entry["method"],
                entry["budget"],
                frontier_tag,
            )
            entry["run_dir"] = str(resolved_run_dir.resolve()) if resolved_run_dir is not None else None
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run or (args.summary_json is None and args.summary_csv is None):
        return

    summary_cmd = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        "--manifest",
        str(manifest_path),
        "--baseline-method",
        args.baseline_method,
    ]
    if args.summary_json is not None:
        summary_cmd.extend(["--json-output", str(args.summary_json)])
    if args.summary_csv is not None:
        summary_cmd.extend(["--csv-output", str(args.summary_csv)])
    run_command(summary_cmd, dry_run=False)


if __name__ == "__main__":
    main()
