#!/usr/bin/env python3
"""Run a replay-fidelity budget sweep with optional parallel job scheduling."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = REPO_ROOT / "scripts" / "replay_reference_fidelity.py"
FRONTIER_ROOT = REPO_ROOT / "experiments" / "frontier"
METHODS = ("fullkv", "triattention", "horizonkv", "cask", "snapkv", "r1kv", "expectedattention")


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned[:64] if cleaned else "fidelity-frontier"


def model_label(model_path: str) -> str:
    return Path(model_path).name or "model"


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Reference run root or merged.jsonl path.")
    parser.add_argument("--model-path", required=True, help="HF model path or model id.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["triattention", "cask"],
        choices=METHODS,
    )
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument("--triattention-stats-file", type=Path, default=None)
    parser.add_argument("--frontier-tag", default=None)
    parser.add_argument("--job-parallel", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-output", type=Path, default=None)
    args, passthrough = parser.parse_known_args()
    if args.job_parallel < 1:
        raise SystemExit("--job-parallel must be >= 1")
    return args, passthrough


def build_command(
    args: argparse.Namespace,
    method: str,
    budget: int | None,
    json_output: Path,
    csv_output: Path,
    passthrough: List[str],
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPLAY_SCRIPT),
        "--reference",
        str(args.reference),
        "--model-path",
        args.model_path,
        "--method",
        method,
        "--json-output",
        str(json_output),
        "--csv-output",
        str(csv_output),
    ]
    if method != "fullkv":
        if budget is None:
            raise ValueError(f"budget is required for {method}")
        cmd.extend(["--budget", str(int(budget))])
    if method in {"triattention", "horizonkv", "cask"}:
        if args.triattention_stats_file is None:
            raise SystemExit("--triattention-stats-file is required for triattention/horizonkv/cask.")
        cmd.extend(["--triattention-stats-file", str(args.triattention_stats_file)])
    cmd.extend(passthrough)
    return cmd


def run_commands(
    entries: List[Dict[str, Any]],
    *,
    dry_run: bool,
    job_parallel: int,
) -> None:
    if dry_run:
        print(f"[dry-run] job_parallel={job_parallel}")
        for entry in entries:
            print("[dry-run]", " ".join(entry["command"]))
        return

    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any]]] = []

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
                    raise SystemExit(f"[error] Fidelity job failed: {entry['label']} (status {ret})")
                return
            time.sleep(1.0)

    for entry in entries:
        if entry.get("status") == "skipped_existing":
            continue
        proc = subprocess.Popen(entry["command"], cwd=str(REPO_ROOT))
        running.append((proc, entry))
        entry["status"] = "running"
        if len(running) >= job_parallel:
            drain_one()

    while running:
        drain_one()


def main() -> None:
    args, passthrough = parse_args()
    frontier_tag = sanitize_tag(args.frontier_tag or f"fidelity_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root = FRONTIER_ROOT / model_label(args.model_path) / frontier_tag
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_output or (root / "fidelity_manifest.json")

    entries: List[Dict[str, Any]] = []
    for method in args.methods:
        budgets = [None] if method == "fullkv" else [int(value) for value in args.budgets]
        for budget in budgets:
            label = f"{method}_full" if budget is None else f"{method}_budget_{budget}"
            json_output = root / f"{label}.json"
            csv_output = root / f"{label}.csv"
            entry: Dict[str, Any] = {
                "label": label,
                "method": method,
                "budget": budget,
                "json_output": str(json_output),
                "csv_output": str(csv_output),
            }
            if args.skip_existing and json_output.exists() and csv_output.exists():
                entry["status"] = "skipped_existing"
                entries.append(entry)
                continue
            entry["command"] = build_command(args, method, budget, json_output, csv_output, passthrough)
            entry["status"] = "planned" if args.dry_run else "pending"
            entries.append(entry)

    payload = {
        "reference": str(args.reference),
        "model_path": args.model_path,
        "frontier_tag": frontier_tag,
        "job_parallel": args.job_parallel,
        "methods": list(args.methods),
        "budgets": [int(v) for v in args.budgets],
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"fidelity_manifest={manifest_path.resolve()}")

    run_commands(entries, dry_run=args.dry_run, job_parallel=args.job_parallel)

    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

