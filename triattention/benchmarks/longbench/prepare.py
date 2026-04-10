"""Dataset preparation helpers for LongBench v1 and LongBench-E."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

from huggingface_hub import hf_hub_download

from triattention.benchmarks.longbench.constants import (
    DATASET_MAX_NEW_TOKENS,
    DATASET_PROMPTS,
    LONG_BENCH_DATASETS,
    LONG_BENCH_E_DATASETS,
)


def resolve_tasks(use_e: bool, tasks: Iterable[str] | None = None) -> list[str]:
    available = LONG_BENCH_E_DATASETS if use_e else LONG_BENCH_DATASETS
    if tasks is None:
        return list(available)
    requested = list(tasks)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        scope = "LongBench-E" if use_e else "LongBench"
        raise ValueError(f"Unsupported {scope} tasks: {unknown}")
    return requested


def hf_split_name(task: str, use_e: bool) -> str:
    return f"{task}_e" if use_e else task


def build_prompt(task: str, record: dict) -> str:
    return DATASET_PROMPTS[task].format(**record)


def write_task_jsonl(
    task: str,
    output_path: Path,
    *,
    use_e: bool,
    max_examples: int | None = None,
) -> Path:
    archive_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    member_name = f"data/{hf_split_name(task, use_e)}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive, archive.open(member_name) as src, output_path.open("w", encoding="utf-8") as handle:
        for index, line in enumerate(src):
            if max_examples is not None and index >= max_examples:
                break
            record = json.loads(line.decode("utf-8"))
            output_record = {
                "index": index,
                "id": record.get("_id", index),
                "dataset": task,
                "task_set": "longbench_e" if use_e else "longbench",
                "input": record.get("input", ""),
                "question": record.get("input", ""),
                "context": record.get("context", ""),
                "answers": record.get("answers", []),
                "all_classes": record.get("all_classes"),
                "length": record.get("length"),
                "language": record.get("language"),
                "prompt": build_prompt(task, record),
                "max_new_tokens": DATASET_MAX_NEW_TOKENS[task],
            }
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
    return output_path
