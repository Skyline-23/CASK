#!/usr/bin/env python3
"""Plan or launch a prompt-heavy replay package with optional task-level parallelism."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_LONGBENCH = ROOT / "scripts" / "run_longbench_suite.py"
RUN_REPLAY_FRONTIER = ROOT / "scripts" / "run_replay_fidelity_frontier.py"

DEFAULT_MAIN_TASKS = ["qasper", "hotpotqa", "multi_news", "musique", "2wikimqa"]
DEFAULT_PROBE_TASKS = ["vcsum", "qmsum", "gov_report"]
DEFAULT_METHODS = ["triattention", "cask", "snapkv"]
DEFAULT_BUDGETS = [256, 384]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=f"promptheavy_pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--model-path", type=Path, default=ROOT / "experiments" / "models" / "Qwen3-8B")
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=ROOT / "cask" / "calibration" / "for_aime25_experiment" / "qwen3_8b.pt",
    )
    parser.add_argument("--main-tasks", nargs="*", default=DEFAULT_MAIN_TASKS)
    parser.add_argument("--probe-tasks", nargs="*", default=DEFAULT_PROBE_TASKS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=["triattention", "cask", "snapkv"])
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--ref-parallel", type=int, default=1)
    parser.add_argument("--replay-parallel", type=int, default=1)
    parser.add_argument("--replay-inner-parallel", type=int, default=1)
    parser.add_argument("--stage", choices=["plan", "refs", "replay", "all"], default="plan")
    parser.add_argument("--longbench-e", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def task_list(args: argparse.Namespace) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    for task in dedupe(args.main_tasks):
        tasks.append((task, "main"))
    for task in dedupe(args.probe_tasks):
        tasks.append((task, "probe"))
    return tasks


def task_set_name(args: argparse.Namespace) -> str:
    return "longbench_e" if args.longbench_e else "longbench"


def ref_output_root(args: argparse.Namespace) -> Path:
    return ROOT / "experiments" / f"{args.tag}_refs"


def ref_merged_path(args: argparse.Namespace, task: str) -> Path:
    return ref_output_root(args) / task_set_name(args) / args.model_path.name / "runs" / task / "merged" / "merged.jsonl"


def replay_frontier_tag(args: argparse.Namespace, task: str, role: str) -> str:
    return f"{args.tag}_{role}_{task}"


def replay_manifest_path(args: argparse.Namespace, task: str, role: str) -> Path:
    return ROOT / "experiments" / "frontier" / args.model_path.name / replay_frontier_tag(args, task, role) / "fidelity_manifest.json"


def build_ref_command(args: argparse.Namespace, task: str) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_LONGBENCH),
        "--model-path",
        str(args.model_path),
        "--output-root",
        str(ref_output_root(args)),
        "--tasks",
        task,
        "--stats-path",
        str(args.stats_path),
        "--method",
        "fullkv",
        "--attn-implementation",
        "sdpa",
        "--load-dtype",
        "bfloat16",
    ]
    if args.longbench_e:
        cmd.append("--longbench-e")
    if args.max_examples is not None:
        cmd.extend(["--max-examples", str(int(args.max_examples))])
    return cmd


def build_replay_command(args: argparse.Namespace, task: str, role: str) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_REPLAY_FRONTIER),
        "--reference",
        str(ref_merged_path(args, task)),
        "--model-path",
        str(args.model_path),
        "--methods",
        *args.methods,
        "--budgets",
        *[str(int(value)) for value in args.budgets],
        "--triattention-stats-file",
        str(args.stats_path),
        "--frontier-tag",
        replay_frontier_tag(args, task, role),
        "--job-parallel",
        str(int(args.replay_inner_parallel)),
        "--attn-implementation",
        "sdpa",
        "--load-dtype",
        "bfloat16",
        "--count-prompt-tokens",
        "true",
        "--slack-budget-trigger",
        "true",
        "--allow-prefill-compression",
        "false",
    ]
    if args.max_records is not None:
        cmd.extend(["--max-records", str(int(args.max_records))])
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    tasks = task_list(args)
    refs: list[dict[str, Any]] = []
    replays: list[dict[str, Any]] = []

    for task, role in tasks:
        refs.append(
            {
                "task": task,
                "role": role,
                "output_root": str(ref_output_root(args)),
                "reference_merged_jsonl": str(ref_merged_path(args, task)),
                "command": build_ref_command(args, task),
            }
        )
        replays.append(
            {
                "task": task,
                "role": role,
                "reference_merged_jsonl": str(ref_merged_path(args, task)),
                "frontier_tag": replay_frontier_tag(args, task, role),
                "manifest_path": str(replay_manifest_path(args, task, role)),
                "methods": list(args.methods),
                "budgets": [int(value) for value in args.budgets],
                "command": build_replay_command(args, task, role),
            }
        )

    return {
        "tag": args.tag,
        "model_path": str(args.model_path),
        "stats_path": str(args.stats_path),
        "task_set": task_set_name(args),
        "main_tasks": dedupe(args.main_tasks),
        "probe_tasks": dedupe(args.probe_tasks),
        "methods": list(args.methods),
        "budgets": [int(value) for value in args.budgets],
        "max_examples": args.max_examples,
        "max_records": args.max_records,
        "ref_parallel": int(args.ref_parallel),
        "replay_parallel": int(args.replay_parallel),
        "replay_inner_parallel": int(args.replay_inner_parallel),
        "refs": refs,
        "replays": replays,
    }


def run_queue(entries: list[dict[str, Any]], parallelism: int, dry_run: bool) -> None:
    if dry_run:
        for entry in entries:
            print("[dry-run]", " ".join(entry["command"]))
        return

    running: list[tuple[subprocess.Popen[Any], dict[str, Any]]] = []

    def drain_one() -> None:
        while True:
            for index, (proc, entry) in enumerate(running):
                ret = proc.poll()
                if ret is None:
                    continue
                running.pop(index)
                entry["return_code"] = int(ret)
                entry["status"] = "completed" if ret == 0 else "failed"
                if ret != 0:
                    raise SystemExit(f"Queue item failed: {entry.get('task', entry.get('frontier_tag'))} exit={ret}")
                return
            time.sleep(1.0)

    for entry in entries:
        proc = subprocess.Popen(entry["command"], cwd=str(ROOT))
        entry["status"] = "running"
        running.append((proc, entry))
        if len(running) >= parallelism:
            drain_one()

    while running:
        drain_one()


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    report_dir = ROOT / "experiments" / "reports" / args.tag
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "master_plan.json"
    manifest_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"master_plan={manifest_path}")

    if args.stage == "plan":
        return

    if args.stage in {"refs", "all"}:
        run_queue(plan["refs"], max(1, int(args.ref_parallel)), args.dry_run)
        manifest_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    if args.stage in {"replay", "all"}:
        for replay in plan["replays"]:
            if args.skip_existing and Path(replay["manifest_path"]).exists():
                replay["status"] = "skipped_existing"
                continue
            if not Path(replay["reference_merged_jsonl"]).exists():
                raise FileNotFoundError(
                    f"Missing reference merged.jsonl for task={replay['task']}: {replay['reference_merged_jsonl']}"
                )
        replay_entries = [entry for entry in plan["replays"] if entry.get("status") != "skipped_existing"]
        run_queue(replay_entries, max(1, int(args.replay_parallel)), args.dry_run)
        manifest_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
