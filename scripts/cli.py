#!/usr/bin/env python3
"""CLI helpers for the TriAttention experiments wrapper (defaults-driven)."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to run experiment scripts.") from exc

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
EXP_ROOT = REPO_ROOT / "experiments"
CONFIG_ROOT = REPO_ROOT / "triattention" / "configs" / "shared"
DEFAULTS_PATH = CONFIG_ROOT / "defaults.yaml"
BUDGETS_PATH = CONFIG_ROOT / "budgets.yaml"
RUNNER_DEFAULTS_PATH = CONFIG_ROOT / "runner_defaults.yaml"
MODELS_DIR = EXP_ROOT / "models"
LOGS_DIR = EXP_ROOT / "logs"
OUTPUTS_DIR = EXP_ROOT / "outputs"
STATS_DIR = EXP_ROOT / "stats"
DATA_DIR = REPO_ROOT / "data"
PACKAGED_STATS_ROOT = REPO_ROOT / "triattention" / "calibration"

MODEL_SPECS: Dict[str, str] = {
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}

PACKAGED_STATS_FILENAMES: Dict[str, str] = {
    "DeepSeek-R1-Distill-Qwen-7B": "ds_qwen7b.pt",
    "DeepSeek-R1-Distill-Llama-8B": "ds_llama8b.pt",
    "Qwen3-8B": "qwen3_8b.pt",
}

HF_DATASET_SPECS: Dict[str, Dict[str, object]] = {
    "aime24": {
        "hf_path": "HuggingFaceH4/aime_2024",
        "field_map": {"problem": "question"},
    },
    "aime25": {
        "hf_path": "MathArena/aime_2025",
        "field_map": {"problem": "question"},
    },
    "math500": {
        "hf_path": "HuggingFaceH4/MATH-500",
        "field_map": {},
    },
}

DATASETS = ["aime24", "aime25", "math500"]
MODES = ["fullkv", "r1kv", "snapkv", "triattention", "horizonkv", "cask", "expectedattention"]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unable to interpret boolean value '{value}'")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_default_budget(model_name: str | None = None) -> int:
    data = load_yaml(DEFAULTS_PATH)
    if "default_budget" not in data:
        raise ValueError(f"default_budget missing in {DEFAULTS_PATH}")
    if model_name is not None:
        by_model = data.get("default_budget_by_model", {})
        if isinstance(by_model, dict) and model_name in by_model:
            return int(by_model[model_name])
    return int(data["default_budget"])


def load_budgets() -> List[int]:
    data = load_yaml(BUDGETS_PATH)
    budgets = data.get("budgets")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError(f"budgets missing in {BUDGETS_PATH}")
    return [int(value) for value in budgets]


def load_runner_defaults() -> dict:
    data = load_yaml(RUNNER_DEFAULTS_PATH)
    if "experiment" not in data or "runner_args" not in data:
        raise ValueError(f"experiment/runner_args missing in {RUNNER_DEFAULTS_PATH}")
    return data


def load_extra_config(extra_paths: List[Path] | None) -> dict:
    if not extra_paths:
        return {}
    merged = {"experiment": {}, "runner_args": {}, "run_tag": None}
    for path in extra_paths:
        data = load_yaml(path)
        if not data:
            continue
        if not isinstance(data, dict):
            raise ValueError(f"extra config must be a mapping: {path}")
        if "run_tag" in data:
            merged["run_tag"] = data.get("run_tag")
        if "experiment" in data or "runner_args" in data:
            experiment = data.get("experiment", {}) or {}
            runner_args = data.get("runner_args", {}) or {}
            if not isinstance(experiment, dict) or not isinstance(runner_args, dict):
                raise ValueError(f"extra config experiment/runner_args must be mappings: {path}")
            merged["experiment"].update(experiment)
            merged["runner_args"].update(runner_args)
        else:
            runner_args = dict(data)
            runner_args.pop("run_tag", None)
            merged["runner_args"].update(runner_args)
    if not merged["experiment"] and not merged["runner_args"] and merged["run_tag"] is None:
        return {}
    return merged


def dataset_max_length(dataset: str, defaults: dict) -> int:
    mapping = defaults.get("dataset_max_length", {})
    if dataset in mapping:
        return int(mapping[dataset])
    if dataset in {"aime24", "aime25"}:
        return 32768
    return 8192


def download_dataset_jsonl(dataset: str, target_path: Path) -> Path:
    spec = HF_DATASET_SPECS.get(dataset)
    if spec is None:
        raise FileNotFoundError(f"Unsupported dataset for auto-download: {dataset}")
    try:
        from datasets import load_dataset, load_dataset_builder
    except ImportError as exc:
        raise SystemExit(
            "datasets is required for benchmark auto-download. Install requirements.txt first."
        ) from exc

    target_path.parent.mkdir(parents=True, exist_ok=True)
    repo_id = str(spec["hf_path"])
    field_map = dict(spec.get("field_map", {}))
    print(f"[download] dataset {dataset} -> {target_path} from {repo_id}")
    builder = load_dataset_builder(repo_id)
    split_names = list(builder.info.splits.keys())
    split_name = next((name for name in ("test", "validation", "train") if name in split_names), None)
    if split_name is None:
        raise ValueError(f"Unable to determine split for {repo_id}. Available splits: {split_names}")
    dataset_split = load_dataset(repo_id, split=split_name)
    with target_path.open("w", encoding="utf-8") as handle:
        for row in dataset_split:
            output_row = dict(row)
            for old_key, new_key in field_map.items():
                if old_key in output_row:
                    output_row[new_key] = output_row.pop(old_key)
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    return target_path


def resolve_dataset_path(dataset: str) -> Path:
    candidates = [
        PROJECT_ROOT / f"{dataset}.jsonl",
        DATA_DIR / f"{dataset}.jsonl",
    ]
    if dataset == "math500":
        candidates.extend(
            [
                PROJECT_ROOT / "math500.jsonl",
                DATA_DIR / "math500.jsonl",
                PROJECT_ROOT / "math.jsonl",
                DATA_DIR / "math.jsonl",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            if candidate.name != f"{dataset}.jsonl":
                print(
                    f"[warn] dataset {dataset} resolved to {candidate}",
                    file=sys.stderr,
                )
            return candidate
    return download_dataset_jsonl(dataset, DATA_DIR / f"{dataset}.jsonl")


def resolve_model_path(model_name: str) -> Path:
    return MODELS_DIR / model_name


def stats_path_for(
    dataset: str,
    model_name: str,
    budget: int,
    *,
    announce: bool = True,
) -> Path:
    # Keep stats dataset distinct from evaluation dataset.
    stats_dataset = "aime25" if dataset == "aime24" else "aime24"
    if announce:
        sys.stderr.write(
            f"[info] triattention stats dataset: eval={dataset} stats={stats_dataset}\n"
        )
    generated_stats = STATS_DIR / stats_dataset / model_name / f"stats_budget_{budget}.pt"
    if generated_stats.exists():
        return generated_stats

    packaged_name = PACKAGED_STATS_FILENAMES.get(model_name)
    if packaged_name:
        packaged_stats = (
            PACKAGED_STATS_ROOT
            / f"for_{stats_dataset}_experiment"
            / packaged_name
        )
        if packaged_stats.exists():
            if announce:
                sys.stderr.write(
                    f"[info] triattention stats fallback: using packaged stats {packaged_stats}\n"
                )
            return packaged_stats
    return generated_stats


def budget_tag(mode: str, budget: int | None) -> str:
    if mode == "fullkv":
        return "full"
    if budget is None:
        raise ValueError("budget is required for non-fullkv modes")
    return f"budget_{budget}"


def resolve_num_samples(runner_args: dict, dataset: str | None = None) -> int:
    value = runner_args.get("num_samples", 64)
    per_dataset = runner_args.get("num_samples_by_dataset", {})
    if dataset and isinstance(per_dataset, dict) and dataset in per_dataset:
        value = per_dataset[dataset]
    try:
        return int(value)
    except (TypeError, ValueError):
        return 64


def sample_tag(num_samples: int) -> str:
    return f"sample{num_samples}"


def effective_runner_args(defaults: dict, extra_config: dict | None) -> dict:
    runner_args = dict(defaults.get("runner_args", {}))
    if extra_config:
        extra_runner_args = extra_config.get("runner_args", {}) or {}
        if not isinstance(extra_runner_args, dict):
            raise ValueError("extra config runner_args must be a mapping")
        if "num_samples" in extra_runner_args and "num_samples_by_dataset" not in extra_runner_args:
            runner_args.pop("num_samples_by_dataset", None)
        runner_args.update(extra_runner_args)
    return runner_args

def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not cleaned:
        return "custom"
    return cleaned[:64]

def tag_with_suffix(tag: str, suffix: str | None) -> str:
    if not suffix:
        return tag
    return f"{tag}_{sanitize_tag(suffix)}"


def resolve_stats_path(value: str) -> Path:
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in stats path: {value}")
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = (EXP_ROOT / path).resolve()
    return path


def resolve_stats_override(
    extra_config: dict | None,
    stats_path_arg: str | None,
) -> Path | None:
    if stats_path_arg:
        return resolve_stats_path(stats_path_arg)
    if not extra_config:
        return None
    runner_args = extra_config.get("runner_args", {})
    if not isinstance(runner_args, dict):
        return None
    value = runner_args.get("triattention_stats_file")
    if not value:
        return None
    return resolve_stats_path(str(value))


def stats_supports_rms2(stats_path: Path) -> bool:
    payload = torch.load(stats_path, map_location="cpu")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    stats_version = int(metadata.get("stats_version", 1))
    if stats_version >= 2:
        return True
    stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
    if not isinstance(stats, dict):
        return False
    return any(isinstance(entry, dict) and "q_sq_abs_mean" in entry for entry in stats.values())


def stats_supports_variational_horizon(stats_path: Path) -> bool:
    payload = torch.load(stats_path, map_location="cpu")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if bool(metadata.get("build_variational_horizon", False)):
        return True
    stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
    if not isinstance(stats, dict):
        return False
    return any(
        isinstance(entry, dict)
        and "oracle_horizon_mean_real" in entry
        and "oracle_horizon_mean_imag" in entry
        for entry in stats.values()
    )


def validate_stats_path_for_runner_args(
    *,
    mode: str,
    runner_args: dict,
    stats_path: Path | None,
) -> None:
    if mode not in {"triattention", "horizonkv", "cask"} or stats_path is None or not stats_path.exists():
        return
    norm_mode = str(runner_args.get("triattention_norm_mode", "tri")).strip().lower()
    if mode == "cask" and norm_mode != "tri":
        raise ValueError("cask only supports triattention_norm_mode='tri'.")
    if norm_mode == "rms2" and not stats_supports_rms2(stats_path):
        raise ValueError(
            f"Stats file {stats_path} does not contain q_sq_abs_mean and cannot be used with "
            f"norm_mode='rms2'. Regenerate stats with scripts/calibrate.py and pass the result via "
            f"--stats-path."
        )
    horizon_mode = str(runner_args.get("triattention_horizon_mode", "fixed")).strip().lower()
    if mode == "cask" and horizon_mode != "fixed":
        raise ValueError("cask only supports triattention_horizon_mode='fixed'.")
    if horizon_mode == "variational" and not stats_supports_variational_horizon(stats_path):
        raise ValueError(
            f"Stats file {stats_path} does not contain offline variational horizon tensors and cannot be used "
            "with horizon_mode='variational'. Regenerate stats with scripts/calibrate.py "
            "--build-variational-horizon and pass the result via --stats-path."
        )

def resolve_run_tag(extra_config: dict | None, run_tag_arg: str | None) -> str | None:
    if run_tag_arg:
        return sanitize_tag(str(run_tag_arg))
    if not extra_config:
        return None
    value = extra_config.get("run_tag")
    if value is None:
        return None
    return sanitize_tag(str(value))


def config_output_path(
    dataset: str,
    model_name: str,
    mode: str,
    budget: int | None,
    run_tag: str | None,
) -> Path:
    slug = model_name.lower().replace("/", "-").replace(" ", "-")
    tag = tag_with_suffix(budget_tag(mode, budget), run_tag)
    return EXP_ROOT / "configs" / "generated" / dataset / slug / f"{mode}_{tag}.yaml"


def apply_defaults(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    merged.update(overrides)
    return merged


def merge_extra_config(base: dict | None, overlay: dict | None) -> dict | None:
    if not base and not overlay:
        return None
    merged = {"experiment": {}, "runner_args": {}, "run_tag": None}
    for source in (base or {}, overlay or {}):
        if not source:
            continue
        experiment = source.get("experiment", {}) or {}
        runner_args = source.get("runner_args", {}) or {}
        if not isinstance(experiment, dict) or not isinstance(runner_args, dict):
            raise ValueError("extra config experiment/runner_args must be mappings")
        merged["experiment"].update(experiment)
        merged["runner_args"].update(runner_args)
        if source.get("run_tag") is not None:
            merged["run_tag"] = source.get("run_tag")
    if not merged["experiment"] and not merged["runner_args"] and merged["run_tag"] is None:
        return None
    return merged


def build_config(
    dataset: str,
    dataset_path: Path,
    model_name: str,
    model_path: Path,
    mode: str,
    budget: int | None,
    stats_path: Path | None,
    run_tag: str | None,
    defaults: dict,
    extra_config: dict | None,
) -> dict:
    tag = tag_with_suffix(budget_tag(mode, budget), run_tag)
    exp_defaults = defaults.get("experiment", {})
    runner_defaults = defaults.get("runner_args", {})
    merged_runner_args = effective_runner_args(defaults, extra_config)
    num_samples = resolve_num_samples(merged_runner_args, dataset)
    sample_dir = sample_tag(num_samples)
    log_dir = LOGS_DIR / dataset / model_name / sample_dir / mode / tag
    output_dir = OUTPUTS_DIR / dataset / model_name / sample_dir / mode / tag

    experiment = apply_defaults(
        exp_defaults,
        {
            "name": f"{dataset}_{model_name}_{mode}_{tag}",
            "log_dir": str(log_dir),
            "method_output_dir": str(output_dir),
        },
    )
    runner_args = apply_defaults(
        runner_defaults,
        {
            "output_dir": str(output_dir / "shards"),
            "dataset_path": str(dataset_path),
            "model_path": str(model_path),
            "max_length": dataset_max_length(dataset, defaults),
            "method": mode,
            "kv_budget": budget,
        },
    )
    runner_args["num_samples"] = num_samples
    runner_args.pop("num_samples_by_dataset", None)

    if extra_config:
        extra_experiment = extra_config.get("experiment", {})
        extra_runner_args = extra_config.get("runner_args", {})
        if not isinstance(extra_experiment, dict) or not isinstance(extra_runner_args, dict):
            raise ValueError("extra config experiment/runner_args must be mappings")
        experiment = apply_defaults(experiment, extra_experiment)
        runner_args = apply_defaults(runner_args, extra_runner_args)

    if mode == "fullkv":
        runner_args["kv_budget"] = None
    if mode in {"triattention", "horizonkv", "cask"}:
        if stats_path is None and "triattention_stats_file" not in runner_args:
            raise ValueError(f"stats_path is required for {mode} mode")
        runner_args.setdefault("triattention_stats_file", str(stats_path) if stats_path else None)
        if "per_head_pruning" not in runner_args and "per_layer_perhead_pruning" not in runner_args:
            runner_args["per_head_pruning"] = True
        runner_args.setdefault("count_prompt_tokens", True)
        runner_args.setdefault("attention_layer_compression", True)
        runner_args.setdefault("slack_budget_trigger", True)
        runner_args.setdefault("triattention_normalize_scores", True)
        runner_args.setdefault("divide_length", 128)
        runner_args.setdefault("window_size", 128)
        runner_args.setdefault("round_window", 32)
        runner_args.setdefault("triattention_frequency_window", 65536)
        runner_args.setdefault("triattention_score_aggregation", "mean")
        runner_args.setdefault("pruning_seed", 0)
        if mode == "horizonkv":
            runner_args.setdefault("triattention_horizon_mode", "adaptive")
            runner_args.setdefault("triattention_norm_mode", "tri")
        elif mode == "cask":
            runner_args.setdefault("triattention_horizon_mode", "fixed")
            runner_args.setdefault("triattention_norm_mode", "tri")
            runner_args.setdefault("cask_prefix_coverage_ratio", 0.0625)
            runner_args.setdefault("cask_protected_core_ratio", 0.5)
            runner_args.setdefault("cask_min_protected_core_tokens", 1)
            runner_args.setdefault("cask_core_selection_mode", "vote")
            runner_args.setdefault("cask_merge_operator", "keepkv")
            runner_args.setdefault("cask_merge_local_window", 32)
            runner_args.setdefault("cask_similarity_threshold", 0.985)
            runner_args.setdefault("cask_representative_mode", "score_max_source")
            runner_args.setdefault("cask_promotion_score_ratio", None)
            runner_args.setdefault("cask_merge_score_mass_ratio_threshold", None)
            runner_args.setdefault("cask_use_phase_markers", True)
        validate_stats_path_for_runner_args(
            mode=mode,
            runner_args=runner_args,
            stats_path=Path(str(runner_args["triattention_stats_file"])),
        )

    return {"experiment": {**experiment, "runner_args": runner_args}}


def write_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def ensure_run_log(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / "run.log"
    shard_logs = sorted(log_dir.glob("*.log"))
    if not shard_logs:
        return
    if len(shard_logs) == 1:
        shutil.copyfile(shard_logs[0], run_log)
        return
    with run_log.open("w", encoding="utf-8") as handle:
        for shard_log in shard_logs:
            handle.write(f"=== {shard_log.name} ===\n")
            handle.write(shard_log.read_text(encoding="utf-8"))
            handle.write("\n")


def dispatch_run(config_path: Path, dataset: str, log_dir: Path, dry_run: bool) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{pythonpath}" if pythonpath else str(REPO_ROOT)


    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dispatch.py"),
        "--config",
        str(config_path),
        "--dataset",
        dataset,
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)
    ensure_run_log(log_dir)


def validate_model_exists(model_name: str, dry_run: bool) -> Path:
    model_path = resolve_model_path(model_name)
    if dry_run:
        print(f"[dry-run] model path: {model_path}")
        return model_path
    if not model_path.exists() or not any(model_path.iterdir()):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python scripts/cli.py download-models --model {model_name}` first."
        )
    return model_path


def run_one(
    dataset: str,
    model_name: str,
    mode: str,
    budget: int | None,
    *,
    require_stats: bool,
    stats_path_arg: str | None,
    run_tag: str | None,
    defaults: dict,
    extra_config: dict | None,
    dry_run: bool,
) -> None:
    dataset_path = resolve_dataset_path(dataset)
    model_path = validate_model_exists(model_name, dry_run)
    tag = budget_tag(mode, budget)
    num_samples = resolve_num_samples(effective_runner_args(defaults, extra_config), dataset)
    sample_dir = sample_tag(num_samples)
    resolved_run_tag = resolve_run_tag(extra_config, run_tag)

    stats_path = None
    if mode in {"triattention", "horizonkv", "cask"}:
        if budget is None:
            raise ValueError(f"budget is required for {mode} runs")
        stats_override = resolve_stats_override(extra_config, stats_path_arg)
        if stats_override is None:
            stats_path = stats_path_for(dataset, model_name, budget)
        else:
            stats_path = stats_override
            sys.stderr.write(f"[info] {mode} stats override: {stats_path}\n")
        if require_stats and stats_path and not stats_path.exists():
            if dry_run:
                print(
                    f"[dry-run] missing stats for {dataset}/{model_name}/budget {budget}: {stats_path}",
                    file=sys.stderr,
                )
            else:
                raise FileNotFoundError(
                    f"Stats missing for {mode} on {dataset}/{model_name}/budget {budget}. "
                    f"Run scripts/experiments/build_all_stats.sh first."
                )

    tag = tag_with_suffix(tag, resolved_run_tag)
    log_dir = LOGS_DIR / dataset / model_name / sample_dir / mode / tag
    output_dir = OUTPUTS_DIR / dataset / model_name / sample_dir / mode / tag

    config_path = config_output_path(dataset, model_name, mode, budget, resolved_run_tag)
    config = build_config(
        dataset,
        dataset_path,
        model_name,
        model_path,
        mode,
        budget,
        stats_path,
        resolved_run_tag,
        defaults,
        extra_config,
    )
    write_config(config, config_path)
    dispatch_run(config_path, dataset, log_dir, dry_run)

    output_dir.mkdir(parents=True, exist_ok=True)


def run_defaults(dry_run: bool) -> None:
    defaults = load_runner_defaults()
    for dataset in DATASETS:
        for model_name in MODEL_SPECS.keys():
            default_budget = load_default_budget(model_name)
            run_one(
                dataset,
                model_name,
                "fullkv",
                None,
                require_stats=False,
                stats_path_arg=None,
                run_tag=None,
                defaults=defaults,
                extra_config=None,
                dry_run=dry_run,
            )
            run_one(
                dataset,
                model_name,
                "r1kv",
                default_budget,
                require_stats=False,
                stats_path_arg=None,
                run_tag=None,
                defaults=defaults,
                extra_config=None,
                dry_run=dry_run,
            )
            run_one(
                dataset,
                model_name,
                "triattention",
                default_budget,
                require_stats=True,
                stats_path_arg=None,
                run_tag=None,
                defaults=defaults,
                extra_config=None,
                dry_run=dry_run,
            )


def run_sweep(dry_run: bool) -> None:
    defaults = load_runner_defaults()
    budgets = load_budgets()
    for dataset in DATASETS:
        for model_name in MODEL_SPECS.keys():
            for budget in budgets:
                run_one(
                    dataset,
                    model_name,
                    "r1kv",
                    budget,
                    require_stats=False,
                    stats_path_arg=None,
                    run_tag=None,
                    defaults=defaults,
                    extra_config=None,
                    dry_run=dry_run,
                )
            for budget in budgets:
                run_one(
                    dataset,
                    model_name,
                    "triattention",
                    budget,
                    require_stats=True,
                    stats_path_arg=None,
                    run_tag=None,
                    defaults=defaults,
                    extra_config=None,
                    dry_run=dry_run,
                )


def has_trace_data(trace_root: Path) -> bool:
    merged = trace_root / "merged" / "merged.jsonl"
    if merged.exists():
        return True
    shards = trace_root / "shards"
    if shards.exists() and any(shards.glob("*.jsonl")):
        return True
    if any(trace_root.glob("*.jsonl")):
        return True
    return False


def normalize_selection(
    selected: List[str] | None, allowed: List[str], kind: str
) -> List[str]:
    if not selected:
        return list(allowed)
    allowed_set = set(allowed)
    ordered: List[str] = []
    seen: set[str] = set()
    for value in selected:
        if value not in allowed_set:
            raise ValueError(f"Unsupported {kind}: {value}")
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_stats(
    dry_run: bool,
    models: List[str] | None = None,
    input_file: str | None = None,
    output_dir: str | None = None,
    max_length: int = 32768,
    job_parallel: int = 1,
    build_variational_horizon: bool = False,
    variational_query_samples: int = 128,
    variational_key_chunk_size: int = 512,
    variational_offset_max_length: int = 65536,
) -> None:
    if job_parallel < 1:
        raise ValueError("job_parallel must be >= 1")

    if input_file is None:
        raise SystemExit(
            "build-stats requires --input pointing to a plain text calibration file.\n"
            "Example: python scripts/cli.py build-stats --input calibration_text.txt"
        )
    input_path = Path(input_file)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    out_dir = Path(output_dir) if output_dir else REPO_ROOT / "triattention" / "calibration"
    model_list = normalize_selection(models, list(MODEL_SPECS.keys()), "model")

    commands: List[Dict[str, object]] = []

    for model_name in model_list:
        model_path = validate_model_exists(model_name, dry_run)
        stats_path = out_dir / f"{model_name.lower().replace('-', '_')}_stats.pt"
        if stats_path.exists():
            print(f"[skip] Stats already exist: {stats_path}")
            continue
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "calibrate.py"),
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--output",
            str(stats_path),
            "--max-length",
            str(max_length),
            "--device",
            "cuda",
            "--attn-implementation",
            "flash_attention_2",
        ]
        if build_variational_horizon:
            cmd.extend(
                [
                    "--build-variational-horizon",
                    "--variational-query-samples",
                    str(variational_query_samples),
                    "--variational-key-chunk-size",
                    str(variational_key_chunk_size),
                    "--variational-offset-max-length",
                    str(variational_offset_max_length),
                ]
            )
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".strip(":")
        commands.append(
            {
                "cmd": cmd,
                "cwd": str(REPO_ROOT),
                "env": env,
                "label": model_name,
            }
        )

    if not commands:
        print("[info] No pending stats jobs for requested targets.")
        return

    if dry_run:
        print(f"[dry-run] job_parallel={job_parallel}")
        batch_id = 1
        for idx in range(0, len(commands), job_parallel):
            labels = ", ".join(
                info["label"]  # type: ignore[index]
                for info in commands[idx : idx + job_parallel]
            )
            print(f"[dry-run] batch {batch_id}: {labels}")
            batch_id += 1
        for info in commands:
            cmd_str = " ".join(info["cmd"])  # type: ignore[index]
            print(f"[dry-run] {cmd_str}")
        return

    running: List[Tuple[subprocess.Popen, str]] = []

    def wait_for_first() -> None:
        if not running:
            return
        proc, label = running.pop(0)
        ret = proc.wait()
        if ret != 0:
            raise SystemExit(f"[error] Stats job {label} failed with status {ret}")

    for info in commands:
        proc = subprocess.Popen(info["cmd"], cwd=info["cwd"], env=info["env"])  # type: ignore[arg-type]
        running.append((proc, info["label"]))  # type: ignore[index]
        if len(running) >= job_parallel:
            wait_for_first()

    while running:
        wait_for_first()


def download_models(models: List[str] | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to download models.") from exc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selected_models = normalize_selection(models, list(MODEL_SPECS.keys()), "model")
    for model_name in selected_models:
        repo_id = MODEL_SPECS[model_name]
        target_dir = MODELS_DIR / model_name
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"[skip] {model_name} already present at {target_dir}")
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] {model_name} -> {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )


def resolve_budget_for_mode(mode: str, budget: int | None, model_name: str | None = None) -> int | None:
    if mode == "fullkv":
        return None
    if budget is not None:
        return int(budget)
    return load_default_budget(model_name)


def build_run_one_cli_overrides(args: argparse.Namespace) -> dict | None:
    experiment: Dict[str, object] = {}
    runner_args: Dict[str, object] = {}

    if args.gpus is not None:
        experiment["gpus"] = args.gpus
    if args.num_shards is not None:
        experiment["num_shards"] = args.num_shards
    if args.no_eval:
        experiment["no_eval"] = True
    if args.eval_output_dir:
        experiment["eval_output_dir"] = args.eval_output_dir

    optional_runner_args = {
        "num_samples": args.num_samples,
        "max_examples": args.max_examples,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "stop_on_final_answer": args.stop_on_final_answer,
        "attn_implementation": args.attn_implementation,
        "load_dtype": args.load_dtype,
        "triattention_score_dump_dir": args.score_dump_dir,
        "triattention_score_dump_max_events": args.score_dump_max_events,
        "triattention_horizon_mode": args.triattention_horizon_mode,
        "triattention_norm_mode": args.triattention_norm_mode,
        "triattention_kernel_c_lambda": args.triattention_kernel_c_lambda,
        "triattention_kernel_s0": args.triattention_kernel_s0,
        "triattention_kernel_s1": args.triattention_kernel_s1,
        "triattention_norm_lambda": args.triattention_norm_lambda,
        "cask_prefix_coverage_ratio": args.cask_prefix_coverage_ratio,
        "cask_protected_core_ratio": args.cask_protected_core_ratio,
        "cask_min_protected_core_tokens": args.cask_min_protected_core_tokens,
        "cask_core_selection_mode": args.cask_core_selection_mode,
        "cask_merge_operator": args.cask_merge_operator,
        "cask_merge_local_window": args.cask_merge_local_window,
        "cask_similarity_threshold": args.cask_similarity_threshold,
        "cask_value_projection_threshold": args.cask_value_projection_threshold,
        "cask_representative_mode": args.cask_representative_mode,
        "cask_promotion_score_ratio": args.cask_promotion_score_ratio,
        "cask_merge_score_mass_ratio_threshold": args.cask_merge_score_mass_ratio_threshold,
        "cask_use_phase_markers": args.cask_use_phase_markers,
        "expectedattention_n_future_positions": args.expectedattention_n_future_positions,
        "expectedattention_n_sink": args.expectedattention_n_sink,
        "expectedattention_use_covariance": args.expectedattention_use_covariance,
        "expectedattention_use_vnorm": args.expectedattention_use_vnorm,
        "expectedattention_epsilon": args.expectedattention_epsilon,
    }
    for key, value in optional_runner_args.items():
        if value is not None:
            runner_args[key] = value

    if not experiment and not runner_args:
        return None
    return {"experiment": experiment, "runner_args": runner_args, "run_tag": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_models_parser = subparsers.add_parser("download-models", help="Download required models.")
    download_models_parser.add_argument(
        "--model",
        action="append",
        choices=list(MODEL_SPECS.keys()),
        help="Models to include (repeatable). Defaults to all.",
    )
    subparsers.add_parser("run-default", help="Run all default-budget experiments.")
    subparsers.add_parser("run-sweep", help="Run all budget sweep experiments.")
    build_stats_parser = subparsers.add_parser(
        "build-stats", help="Calibrate TriAttention stats from plain text input."
    )
    build_stats_parser.add_argument(
        "--input",
        required=True,
        help="Plain text file for calibration input.",
    )
    build_stats_parser.add_argument(
        "--model",
        action="append",
        choices=list(MODEL_SPECS.keys()),
        help="Models to include (repeatable). Defaults to all.",
    )
    build_stats_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for stats files (default: calibration/).",
    )
    build_stats_parser.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Maximum token length for calibration (default: 32768).",
    )
    build_stats_parser.add_argument(
        "--job-parallel",
        type=int,
        default=1,
        help="Maximum concurrent stats jobs.",
    )
    build_stats_parser.add_argument(
        "--build-variational-horizon",
        action="store_true",
        help="Also estimate oracle horizon targets and solve the offline variational QP during calibration.",
    )
    build_stats_parser.add_argument(
        "--variational-query-samples",
        type=int,
        default=128,
        help="Number of sampled query positions per head for variational horizon estimation.",
    )
    build_stats_parser.add_argument(
        "--variational-key-chunk-size",
        type=int,
        default=512,
        help="Chunk size used for streamed attention when building variational horizon targets.",
    )
    build_stats_parser.add_argument(
        "--variational-offset-max-length",
        type=int,
        default=65536,
        help="Maximum dyadic offset length used by the offline variational horizon solver.",
    )

    run_one_parser = subparsers.add_parser("run-one", help="Run a single dataset/model/method/budget.")
    run_one_parser.add_argument("--dataset", required=True, choices=DATASETS)
    run_one_parser.add_argument("--model", required=True, choices=MODEL_SPECS.keys())
    run_one_parser.add_argument("--method", required=True, choices=MODES)
    run_one_parser.add_argument("--budget", type=int, default=None)
    run_one_parser.add_argument(
        "--stats-path",
        default=None,
        help="Override TriAttention stats path (supports env vars).",
    )
    run_one_parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional suffix for output/log/config dirs to avoid collisions.",
    )
    run_one_parser.add_argument(
        "--extra-config",
        action="append",
        default=None,
        help="YAML config overrides to merge into runner_args or experiment.",
    )
    run_one_parser.add_argument("--gpus", default=None, help="GPU ids override for dispatch (for example: 0 or 0,1).")
    run_one_parser.add_argument("--num-shards", type=int, default=None, help="Override shard count for dispatch.")
    run_one_parser.add_argument("--num-samples", type=int, default=None, help="Override generation draws per question.")
    run_one_parser.add_argument("--max-examples", type=int, default=None, help="Cap dataset examples for smoke tests.")
    run_one_parser.add_argument("--max-length", type=int, default=None, help="Override generation max_length.")
    run_one_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Fallback generation cap when dataset records do not define max_new_tokens.",
    )
    run_one_parser.add_argument(
        "--stop-on-final-answer",
        type=str2bool,
        default=None,
        help="Forward a final-answer stopping criterion to the HF worker.",
    )
    run_one_parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "flash_attention_2", "sdpa"],
        help="Override HuggingFace attention backend.",
    )
    run_one_parser.add_argument(
        "--load-dtype",
        default=None,
        choices=["bfloat16", "float16"],
        help="Override model load dtype.",
    )
    run_one_parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after shard merge.")
    run_one_parser.add_argument("--eval-output-dir", default=None, help="Override evaluation output directory.")
    run_one_parser.add_argument(
        "--score-dump-dir",
        default=None,
        help="Directory to dump TriAttention score tensors for baseline/modified diffing.",
    )
    run_one_parser.add_argument(
        "--score-dump-max-events",
        type=int,
        default=None,
        help="Maximum TriAttention score dump events per generation.",
    )
    run_one_parser.add_argument(
        "--triattention-horizon-mode",
        default=None,
        choices=["fixed", "adaptive", "variational"],
        help="Override the TriAttention horizon kernel mode.",
    )
    run_one_parser.add_argument(
        "--triattention-norm-mode",
        default=None,
        choices=["tri", "rms2"],
        help="Override the TriAttention norm coefficient mode.",
    )
    run_one_parser.add_argument(
        "--triattention-kernel-c-lambda",
        type=float,
        default=None,
        help="Override the adaptive horizon kernel center multiplier.",
    )
    run_one_parser.add_argument(
        "--triattention-kernel-s0",
        type=float,
        default=None,
        help="Override the adaptive horizon kernel base width.",
    )
    run_one_parser.add_argument(
        "--triattention-kernel-s1",
        type=float,
        default=None,
        help="Override the adaptive horizon kernel concentration-dependent width term.",
    )
    run_one_parser.add_argument(
        "--triattention-norm-lambda",
        type=float,
        default=None,
        help="Override the scalar multiplier applied to the selected norm coefficient.",
    )
    run_one_parser.add_argument(
        "--cask-prefix-coverage-ratio",
        type=float,
        default=None,
        help="Override the fraction of prefix budget reserved for evenly spaced coverage anchors in stage-1 eviction.",
    )
    run_one_parser.add_argument(
        "--cask-protected-core-ratio",
        type=float,
        default=None,
        help="Override the protected-core slot ratio used by cask.",
    )
    run_one_parser.add_argument(
        "--cask-min-protected-core-tokens",
        type=int,
        default=None,
        help="Override the minimum protected-core tokens reserved in active merge plans.",
    )
    run_one_parser.add_argument(
        "--cask-core-selection-mode",
        choices=["vote", "score"],
        default=None,
        help="Override how cask chooses protected-core tokens.",
    )
    run_one_parser.add_argument(
        "--cask-merge-operator",
        choices=["keepkv", "mean"],
        default=None,
        help="Override the scratch merge operator used by cask.",
    )
    run_one_parser.add_argument(
        "--cask-merge-local-window",
        type=int,
        default=None,
        help="Override the maximum local merge gap used by cask.",
    )
    run_one_parser.add_argument(
        "--cask-similarity-threshold",
        type=float,
        default=None,
        help="Override the preferred pre-RoPE cosine threshold used by cask.",
    )
    run_one_parser.add_argument(
        "--cask-value-projection-threshold",
        type=float,
        default=None,
        help="Override the optional ||W_O v_i - W_O v_j|| gate used by cask.",
    )
    run_one_parser.add_argument(
        "--cask-representative-mode",
        choices=["weighted_latest", "score_max_source"],
        default=None,
        help="Override how merged scratch groups choose their representative key anchor.",
    )
    run_one_parser.add_argument(
        "--cask-promotion-score-ratio",
        type=float,
        default=None,
        help="Override the near-core scratch promotion ratio used by cask.",
    )
    run_one_parser.add_argument(
        "--cask-merge-score-mass-ratio-threshold",
        type=float,
        default=None,
        help="Override the combined score-mass veto used by cask merges.",
    )
    run_one_parser.add_argument(
        "--cask-use-phase-markers",
        type=str2bool,
        default=None,
        help="Override whether cask restricts merges to weak decode-phase segments.",
    )
    run_one_parser.add_argument(
        "--expectedattention-n-future-positions",
        type=int,
        default=None,
        help="Override the number of future positions averaged by Expected Attention.",
    )
    run_one_parser.add_argument(
        "--expectedattention-n-sink",
        type=int,
        default=None,
        help="Override the number of sink tokens preserved by Expected Attention.",
    )
    run_one_parser.add_argument(
        "--expectedattention-use-covariance",
        type=str2bool,
        default=None,
        help="Override whether Expected Attention includes the covariance correction.",
    )
    run_one_parser.add_argument(
        "--expectedattention-use-vnorm",
        type=str2bool,
        default=None,
        help="Override whether Expected Attention rescales by value norms.",
    )
    run_one_parser.add_argument(
        "--expectedattention-epsilon",
        type=float,
        default=None,
        help="Override the stability term applied before Expected Attention value-norm scaling.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download-models":
        download_models(args.model)
        return
    if args.command == "run-default":
        run_defaults(args.dry_run)
        return
    if args.command == "run-sweep":
        run_sweep(args.dry_run)
        return
    if args.command == "build-stats":
        build_stats(
            args.dry_run,
            models=args.model,
            input_file=args.input,
            output_dir=args.output_dir,
            max_length=args.max_length,
            job_parallel=args.job_parallel,
            build_variational_horizon=args.build_variational_horizon,
            variational_query_samples=args.variational_query_samples,
            variational_key_chunk_size=args.variational_key_chunk_size,
            variational_offset_max_length=args.variational_offset_max_length,
        )
        return
    if args.command == "run-one":
        defaults = load_runner_defaults()
        budget = resolve_budget_for_mode(args.method, args.budget, args.model)
        file_overrides = load_extra_config(
            [Path(path) for path in (args.extra_config or [])]
        )
        extra_config = merge_extra_config(file_overrides, build_run_one_cli_overrides(args))
        run_one(
            args.dataset,
            args.model,
            args.method,
            budget,
            require_stats=(args.method in {"triattention", "horizonkv", "cask"}),
            stats_path_arg=args.stats_path,
            run_tag=args.run_tag,
            defaults=defaults,
            extra_config=extra_config,
            dry_run=args.dry_run,
        )
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
