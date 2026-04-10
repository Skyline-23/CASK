#!/usr/bin/env python3
"""Prepare, run, merge, and evaluate RULER tasks through the HF worker path."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triattention.benchmarks.ruler.constants import DEFAULT_TASKS


WORKER_PATH = REPO_ROOT / "scripts" / "worker.py"
MERGE_PATH = REPO_ROOT / "scripts" / "merge_shards.py"
RULER_REPO_URL = "https://github.com/NVIDIA/RULER.git"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "ruler")
    parser.add_argument("--ruler-repo", type=Path, default=REPO_ROOT / ".cache" / "ruler_official")
    parser.add_argument("--ruler-branch", type=str, default="main")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--seq-lengths", nargs="*", type=int, default=[4096, 8192, 16384, 32768, 65536, 131072])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--subset", type=str, default="validation")
    parser.add_argument("--tokenizer-type", type=str, default="hf")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="TriAttention/HorizonKV stats file forwarded to worker as --triattention_stats_file.",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_ruler_repo(repo_path: Path, branch: str) -> None:
    env = dict(os.environ)
    env["GIT_CLONE_PROTECTION_ACTIVE"] = "false"
    if repo_path.exists():
        subprocess.run(["git", "-C", str(repo_path), "fetch", "origin", branch], check=True, env=env)
        subprocess.run(["git", "-C", str(repo_path), "checkout", branch], check=True, env=env)
        subprocess.run(["git", "-C", str(repo_path), "pull", "--ff-only", "origin", branch], check=True, env=env)
        return
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--branch", branch, "--single-branch", RULER_REPO_URL, str(repo_path)],
        check=True,
        env=env,
    )


def patch_ruler_prepare_script(repo_path: Path) -> None:
    prepare_path = repo_path / "scripts" / "data" / "prepare.py"
    text = prepare_path.read_text(encoding="utf-8")
    updated = text
    if "import sys\n" not in updated:
        updated = updated.replace("import subprocess\n", "import subprocess\nimport sys\n", 1)
    updated = updated.replace('command = f"""python {script} \\', 'command = f"""{sys.executable} {script} \\')
    if updated != text:
        prepare_path.write_text(updated, encoding="utf-8")


def load_task_configs(repo_path: Path) -> dict:
    constants_module = load_module_from_path(
        "ruler_synthetic_constants",
        repo_path / "scripts" / "data" / "synthetic" / "constants.py",
    )
    with (repo_path / "scripts" / "synthetic.yaml").open("r", encoding="utf-8") as handle:
        custom = yaml.safe_load(handle)
    configs = {}
    for task, config in custom.items():
        merged = dict(config)
        merged.update(constants_module.TASKS[config["task"]])
        configs[task] = merged
    return configs


def ensure_assets(repo_path: Path, tasks: list[str], task_configs: dict) -> None:
    json_root = repo_path / "scripts" / "data" / "synthetic" / "json"
    needs_essay = any(
        task_configs[task]["task"] == "niah" and task_configs[task]["args"].get("type_haystack") == "essay"
        for task in tasks
    )
    needs_qa = any(task_configs[task]["task"] == "qa" for task in tasks)
    if needs_essay and not (json_root / "PaulGrahamEssays.json").exists():
        subprocess.run(
            [sys.executable, str(json_root / "download_paulgraham_essay.py")],
            check=True,
            cwd=json_root,
        )
    if needs_qa and not ((json_root / "squad.json").exists() and (json_root / "hotpotqa.json").exists()):
        bash = shutil.which("bash")
        if bash is None:
            raise RuntimeError("RULER QA asset download requires bash to run download_qa_dataset.sh.")
        subprocess.run([bash, str(json_root / "download_qa_dataset.sh")], check=True, cwd=json_root)


def prepare_task(
    args: argparse.Namespace,
    repo_path: Path,
    task: str,
    seq_length: int,
    data_root: Path,
) -> Path:
    prepare_script = repo_path / "scripts" / "data" / "prepare.py"
    env = dict(os.environ)
    python_dir = str(Path(args.python).resolve().parent)
    env["PATH"] = f"{python_dir}:{env.get('PATH', '')}"
    subprocess.run(
        [
            args.python,
            str(prepare_script),
            "--save_dir",
            str(data_root),
            "--benchmark",
            "synthetic",
            "--task",
            task,
            "--subset",
            args.subset,
            "--tokenizer_path",
            str(args.model_path),
            "--tokenizer_type",
            args.tokenizer_type,
            "--max_seq_length",
            str(seq_length),
            "--num_samples",
            str(args.num_samples),
            "--model_template_type",
            "base",
        ],
        check=True,
        cwd=repo_path / "scripts" / "data",
        env=env,
    )
    return data_root / task / f"{args.subset}.jsonl"


def convert_prepared_dataset(source_path: Path, target_path: Path, *, task: str, max_new_tokens: int) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as src, target_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            answer_prefix = record.get("answer_prefix", "")
            output_record = {
                "index": record["index"],
                "id": record.get("others", {}).get("id", record["index"]),
                "task": task,
                "benchmark": "ruler",
                "input": record["input"],
                "question": record["input"],
                "prompt": record["input"] + answer_prefix,
                "outputs": record.get("outputs", []),
                "others": record.get("others", {}),
                "length": record.get("length", -1),
                "truncation": record.get("truncation", -1),
                "max_new_tokens": max_new_tokens,
            }
            dst.write(json.dumps(output_record, ensure_ascii=False) + "\n")
    return target_path


def main() -> None:
    args, passthrough = parse_args()
    tasks = list(args.tasks) if args.tasks else list(DEFAULT_TASKS)
    ensure_ruler_repo(args.ruler_repo, args.ruler_branch)
    patch_ruler_prepare_script(args.ruler_repo)
    task_configs = load_task_configs(args.ruler_repo)
    missing = sorted(set(tasks) - set(task_configs))
    if missing:
        raise ValueError(f"Unsupported RULER tasks: {missing}")
    ensure_assets(args.ruler_repo, tasks, task_configs)

    suite_root = args.output_root / args.model_path.name
    length_summaries: dict[str, dict] = {}

    for seq_length in args.seq_lengths:
        seq_root = suite_root / f"seq_{seq_length}"
        official_data_root = seq_root / "prepared_official"
        worker_data_root = seq_root / "prepared_worker"
        runs_root = seq_root / "runs"
        merged_root = seq_root / "predictions"
        merged_root.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            task_config = task_configs[task]
            official_path = official_data_root / task / f"{args.subset}.jsonl"
            worker_path = worker_data_root / f"{task}.jsonl"
            if not args.eval_only:
                official_path = prepare_task(args, args.ruler_repo, task, seq_length, official_data_root)
                convert_prepared_dataset(
                    official_path,
                    worker_path,
                    task=task,
                    max_new_tokens=int(task_config["tokens_to_generate"]),
                )
            if args.prepare_only:
                continue
            task_root = runs_root / task
            task_run_root = task_root / "raw"
            if not args.eval_only:
                worker_cmd = [
                    args.python,
                    str(WORKER_PATH),
                    "--dataset-path",
                    str(worker_path),
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
                raise FileNotFoundError(f"Merged output missing for task {task}: {merged_path}")
            shutil.copyfile(merged_path, merged_root / f"{task}.jsonl")

        eval_output = seq_root / "ruler_eval.json"
        subprocess.run(
            [
                args.python,
                "-m",
                "triattention.benchmarks.ruler.evaluate",
                "--pred-dir",
                str(merged_root),
                "--output",
                str(eval_output),
                "--tasks",
                *tasks,
            ],
            check=True,
            cwd=REPO_ROOT,
        )
        with eval_output.open("r", encoding="utf-8") as handle:
            length_summaries[str(seq_length)] = json.load(handle)

    if not args.prepare_only:
        summary_path = suite_root / "ruler_lengths_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(length_summaries, handle, ensure_ascii=False, indent=2)
        print(json.dumps(length_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
