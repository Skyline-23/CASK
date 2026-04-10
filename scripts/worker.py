"""Shard-aware inference worker for HuggingFace backend."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from triattention.integration.monkeypatch import replace_llama, replace_qwen2, replace_qwen3
from triattention.common.prompt_utils import (
    DEFAULT_SYSTEM_PROMPT,
    PROMPT_TEMPLATE,
    build_prompt,
    extract_question_from_record,
)


# QK capture stubs (no-op)
def activate_capture(*args, **kwargs):
    return

def deactivate_capture():
    return

def capture_requested_for_sample(*args, **kwargs):
    return False

def patch_llama_attention_for_capture():
    return False

dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "aime25": ["question", "answer"],
    "math": ["problem", "answer"],
    "math500": ["problem", "answer"],
}

dataset2max_length = {
    "gsm8k": 8192,
    "aime24": 32768,
    "aime25": 32768,
    "math": 8192,
    "math500": 8192,
}

RUN_SEED_STRIDE = 1_000_000
FINAL_ANSWER_MARKERS = ("Final answer:", "The answer is:")
FINAL_ANSWER_MAX_SCAN_CHARS = 1024


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unable to interpret boolean value '{value}'")


def has_complete_boxed_answer(
    text: str,
    *,
    markers: Sequence[str] = FINAL_ANSWER_MARKERS,
) -> bool:
    marker_pos = max((text.rfind(marker) for marker in markers), default=-1)
    if marker_pos < 0:
        return False

    suffix = text[marker_pos:]
    boxed_pos = suffix.find("\\boxed{")
    if boxed_pos < 0:
        return False

    payload = suffix[boxed_pos + len("\\boxed{") :]
    depth = 1
    seen_content = False
    for ch in payload:
        if ch == "{":
            depth += 1
            seen_content = True
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return seen_content
        elif not ch.isspace():
            seen_content = True
    return False


class FinalAnswerStoppingCriteria(StoppingCriteria):
    """Stop generation once a complete final-answer boxed span is emitted."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        prompt_length: int,
        *,
        markers: Sequence[str] = FINAL_ANSWER_MARKERS,
        max_scan_chars: int = FINAL_ANSWER_MAX_SCAN_CHARS,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.markers = tuple(markers)
        self.max_scan_chars = int(max_scan_chars)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return False
        start = min(self.prompt_length, int(input_ids.shape[1]))
        text = self.tokenizer.decode(input_ids[0, start:], skip_special_tokens=True)
        if self.max_scan_chars > 0 and len(text) > self.max_scan_chars:
            text = text[-self.max_scan_chars :]
        return has_complete_boxed_answer(text, markers=self.markers)


def resolve_torch_dtype(name: str):
    normalized = name.lower()
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_under_rkv(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "R-KV":
        path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return (REPO_ROOT / path).resolve()


def compute_local_runs(num_samples: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    """Shard-over-draws: split draws across shards, each shard runs all questions."""
    base = num_samples // num_shards
    extra = num_samples % num_shards
    start = shard_id * base + min(shard_id, extra)
    count = base + (1 if shard_id < extra else 0)
    return start, count


def compute_local_questions(total_questions: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    """Shard-over-questions: split questions across shards, each shard runs all draws."""
    base = total_questions // num_shards
    extra = total_questions % num_shards
    start = shard_id * base + min(shard_id, extra)
    count = base + (1 if shard_id < extra else 0)
    return start, count


def use_question_sharding(num_samples: int, num_shards: int) -> bool:
    """Use question-level sharding when draws are fewer than shards."""
    return num_samples < num_shards


def shard_run_dir(base_dir: Path, shard_id: int) -> Path:
    return base_dir / f"shard{shard_id:02d}"


def run_artifacts(base_dir: Path, shard_id: int, run_id: int) -> dict[str, Path]:
    run_dir = shard_run_dir(base_dir, shard_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = run_dir / f"run{run_id:03d}"
    return {
        "run": stem.with_suffix(".jsonl"),
        "tmp": stem.with_suffix(".jsonl.tmp"),
        "meta": stem.with_suffix(".meta.json"),
        "meta_tmp": stem.with_suffix(".meta.json.tmp"),
    }


def run_is_complete(run_path: Path, meta_path: Path, expected_records: int) -> bool:
    if not run_path.exists() or run_path.stat().st_size == 0 or not meta_path.exists():
        return False
    try:
        with meta_path.open(encoding="utf-8") as fp:
            meta = json.load(fp)
    except Exception:
        return False
    if meta.get("status") != "complete":
        return False
    recorded = meta.get("records")
    if expected_records > 0 and isinstance(recorded, int) and recorded < expected_records:
        return False
    if expected_records <= 0:
        return True
    try:
        with run_path.open(encoding="utf-8") as fp:
            lines = sum(1 for _ in fp)
        return lines >= expected_records
    except Exception:
        return False


def _record_sample_idx(record: dict) -> int | None:
    value = record.get("sample_idx")
    if isinstance(value, int):
        return value
    value = record.get("index")
    if isinstance(value, int):
        return value
    return None


def load_existing_sample_indices(path: Path) -> set[int]:
    indices: set[int] = set()
    if not path.exists():
        return indices
    try:
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    break
                sample_idx = _record_sample_idx(record)
                if sample_idx is not None:
                    indices.add(sample_idx)
    except Exception:
        return set()
    return indices


def write_run_meta(meta_path: Path, run_id: int, shard_id: int, records: int) -> None:
    meta_tmp = meta_path.with_suffix(".meta.json.tmp")
    meta = {
        "status": "complete",
        "records": records,
        "run_id": run_id,
        "shard_id": shard_id,
    }
    with meta_tmp.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp)
    meta_tmp.replace(meta_path)


def load_dataset(
    path: Path,
    dataset_name: str,
    tokenizer: AutoTokenizer,
    *,
    use_chat_template: bool,
    system_prompt: str,
    max_examples: int | None = None,
) -> tuple[List[str], List[dict]]:
    prompts: List[str] = []
    test_data: List[dict] = []
    fallback_keys: List[str] = []
    if dataset_name in dataset2key and dataset2key[dataset_name]:
        fallback_keys.append(dataset2key[dataset_name][0])

    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            prompt = example.get("prompt")
            if isinstance(prompt, str) and prompt:
                question = example.get("question") or example.get("input") or ""
            else:
                question = extract_question_from_record(example, fallback_keys=fallback_keys)
                prompt = build_prompt(
                    tokenizer,
                    question,
                    use_chat_template=use_chat_template,
                    system_prompt=system_prompt,
                )
            example["question"] = question
            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)
            if max_examples and len(test_data) >= max_examples:
                break
    return prompts, test_data


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--dataset_path", "--dataset-path", dest="dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("--model_path", "--model-path", dest="model_path", type=str, required=True)
    parser.add_argument("--max_length", "--max-length", dest="max_length", type=int, default=-1)
    parser.add_argument(
        "--max_new_tokens",
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="Fallback generation cap when the dataset record does not define max_new_tokens.",
    )
    parser.add_argument(
        "--stop_on_final_answer",
        "--stop-on-final-answer",
        dest="stop_on_final_answer",
        type=str2bool,
        default=False,
        help="Stop generation once a complete `Final answer: \\boxed{...}` span appears.",
    )
    parser.add_argument(
        "--do_sample",
        "--do-sample",
        dest="do_sample",
        type=str2bool,
        default=True,
        help="Enable sampling during generation. Set false for greedy/benchmark eval.",
    )
    parser.add_argument("--eval_batch_size", "--eval-batch-size", dest="eval_batch_size", type=int, default=1)
    parser.add_argument("--load_dtype", "--load-dtype", dest="load_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--attn_implementation",
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["r1kv", "fullkv", "snapkv", "triattention", "horizonkv", "cask", "expectedattention"],
    )
    parser.add_argument("--kv_budget", "--kv-budget", dest="kv_budget", type=int, default=None)
    parser.add_argument("--window_size", "--window-size", dest="window_size", type=int, default=8)
    parser.add_argument("--first_tokens", "--first-tokens", dest="first_tokens", type=int, default=4)
    parser.add_argument("--mix_lambda", "--mix-lambda", dest="mix_lambda", type=float, default=0.1)
    parser.add_argument("--retain_ratio", "--retain-ratio", dest="retain_ratio", type=float, default=0.2)
    parser.add_argument("--update_kv", "--update-kv", dest="update_kv", type=str2bool, default=True)
    parser.add_argument("--fp32_topk", "--fp32-topk", dest="fp32_topk", type=str2bool, default=False)
    parser.add_argument(
        "--protect_prefill",
        "--protect-prefill",
        dest="protect_prefill",
        type=str2bool,
        default=False,
        help="Protect prefill tokens from compression (ablation for R-KV method). "
             "When True, prefill tokens are always preserved; when False, all tokens compete for budget (default R-KV behavior).",
    )
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )
    parser.add_argument(
        "--divide_method",
        "--divide-method",
        type=str,
        default="step_length",
        choices=["newline", "step_length"],
    )
    parser.add_argument("--divide_length", "--divide-length", dest="divide_length", type=int, default=128)
    parser.add_argument(
        "--compression_content",
        "--compression-content",
        type=str,
        default="all",
        choices=["think", "all"],
        help="whether to compress the whole model output or only the think part",
    )
    parser.add_argument("--shard_id", "--shard-id", dest="shard_id", type=int, required=True)
    parser.add_argument("--num_shards", "--num-shards", dest="num_shards", type=int, required=True)
    parser.add_argument("--num_samples", "--num-samples", dest="num_samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--top_k",
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="Sampling top-k. None keeps HF/default behavior (typically 50). "
             "Set <=0 to disable top-k (aligns with vLLM top_k=-1).",
    )
    parser.add_argument(
        "--triattention_stats_file",
        type=str,
        default=None,
        help="Stats file for TriAttention-family scoring (required for triattention/horizonkv/cask).",
    )
    parser.add_argument(
        "--round_window",
        type=int,
        default=None,
        help="Round window for sparse pruning (defaults to window_size when unset).",
    )
    parser.add_argument(
        "--triattention_frequency_window",
        type=int,
        default=65536,
        help="Maximum offset length for sparse pruning frequency scoring.",
    )
    parser.add_argument(
        "--triattention_score_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Aggregation strategy for sparse round pruning scores.",
    )
    parser.add_argument(
        "--triattention_normalize_scores",
        type=str2bool,
        default=True,
        help="Normalize per-head sparse scores before aggregation.",
    )
    parser.add_argument(
        "--pruning_seed",
        type=int,
        default=0,
        help="Seed used by sparse pruner for noise / head shuffling.",
    )
    parser.add_argument(
        "--triattention_score_dump_dir",
        type=str,
        default=None,
        help="Optional directory for dumping TriAttention score tensors per compression event.",
    )
    parser.add_argument(
        "--triattention_score_dump_max_events",
        type=int,
        default=None,
        help="Maximum number of TriAttention score dump events per generation.",
    )
    parser.add_argument(
        "--per_head_pruning",
        type=str2bool,
        default=True,
        help="Enable per-KV-head independent pruning (each head selects tokens independently). Default: True",
    )
    parser.add_argument(
        "--per_layer_perhead_pruning",
        type=str2bool,
        default=False,
        help="Enable per-layer-per-head independent pruning (each (layer, KV head) selects independently). Default: False",
    )
    parser.add_argument(
        "--layer_perhead_aggregation",
        type=str,
        choices=["max", "mean"],
        default="max",
        help="Aggregation method for per-layer-perhead pruning: max (default) or mean.",
    )
    parser.add_argument(
        "--use_chat_template",
        type=str2bool,
        default=False,
        help="Wrap prompts with tokenizer.apply_chat_template when using sparse pruning.",
    )
    parser.add_argument(
        "--chat_system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when --use_chat_template is enabled for sparse pruning.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on number of dataset examples for quick smoke tests.",
    )
    # Alignment args for fair R-KV comparison
    parser.add_argument(
        "--count_prompt_tokens",
        "--count-prompt-tokens",
        dest="count_prompt_tokens",
        type=str2bool,
        default=True,
        help="Include prefill tokens in budget calculation (aligns with R-KV behavior).",
    )
    parser.add_argument(
        "--attention_layer_compression",
        "--attention-layer-compression",
        dest="attention_layer_compression",
        type=str2bool,
        default=True,
        help="Use attention-layer compression instead of generate wrapper.",
    )
    parser.add_argument(
        "--slack_budget_trigger",
        "--slack-budget-trigger",
        dest="slack_budget_trigger",
        type=str2bool,
        default=True,
        help="Trigger pruning at budget + divide_length (like generate wrapper).",
    )
    parser.add_argument(
        "--allow_prefill_compression",
        "--allow-prefill-compression",
        dest="allow_prefill_compression",
        type=str2bool,
        default=False,
        help="Allow prefill tokens to be compressed. When False, prefill is always preserved.",
    )
    parser.add_argument(
        "--disable_mlr",
        type=str2bool,
        default=False,
        help="Disable MLR term in TriAttention extra computation (use q_abs_mean directly).",
    )
    parser.add_argument(
        "--disable_trig",
        type=str2bool,
        default=False,
        help="Disable position-dependent term in TriAttention scoring (use additive term only).",
    )
    parser.add_argument(
        "--triattention_horizon_mode",
        "--triattention-horizon-mode",
        dest="triattention_horizon_mode",
        type=str,
        default=None,
        choices=["fixed", "adaptive", "variational"],
        help="TriAttention horizon kernel mode.",
    )
    parser.add_argument(
        "--triattention_norm_mode",
        "--triattention-norm-mode",
        dest="triattention_norm_mode",
        type=str,
        default=None,
        choices=["tri", "rms2"],
        help="TriAttention norm coefficient mode.",
    )
    parser.add_argument(
        "--triattention_kernel_c_lambda",
        "--triattention-kernel-c-lambda",
        dest="triattention_kernel_c_lambda",
        type=float,
        default=1.0,
        help="Adaptive horizon kernel center multiplier.",
    )
    parser.add_argument(
        "--triattention_kernel_s0",
        "--triattention-kernel-s0",
        dest="triattention_kernel_s0",
        type=float,
        default=1.0,
        help="Adaptive horizon kernel base width.",
    )
    parser.add_argument(
        "--triattention_kernel_s1",
        "--triattention-kernel-s1",
        dest="triattention_kernel_s1",
        type=float,
        default=1.0,
        help="Adaptive horizon kernel concentration-dependent width term.",
    )
    parser.add_argument(
        "--triattention_norm_lambda",
        "--triattention-norm-lambda",
        dest="triattention_norm_lambda",
        type=float,
        default=1.0,
        help="Scalar multiplier applied to the selected TriAttention norm coefficient.",
    )
    parser.add_argument(
        "--cask_protected_core_ratio",
        "--cask-protected-core-ratio",
        dest="cask_protected_core_ratio",
        type=float,
        default=0.5,
        help="Fraction of candidate slots reserved for protected-core tokens before scratch merging.",
    )
    parser.add_argument(
        "--cask_min_protected_core_tokens",
        "--cask-min-protected-core-tokens",
        dest="cask_min_protected_core_tokens",
        type=int,
        default=1,
        help="Minimum protected-core tokens to reserve during non-collapse planning.",
    )
    parser.add_argument(
        "--cask_core_selection_mode",
        "--cask-core-selection-mode",
        dest="cask_core_selection_mode",
        choices=["vote", "score"],
        default="vote",
        help="How CASK picks protected-core tokens: sampled-head voting or aggregate score.",
    )
    parser.add_argument(
        "--cask_merge_operator",
        "--cask-merge-operator",
        dest="cask_merge_operator",
        choices=["keepkv", "mean"],
        default="keepkv",
        help="Scratch merge operator: KeepKV-style weighted merge or count-weighted mean fallback.",
    )
    parser.add_argument(
        "--cask_merge_local_window",
        "--cask-merge-local-window",
        dest="cask_merge_local_window",
        type=int,
        default=32,
        help="Maximum positional gap allowed for local scratch merges.",
    )
    parser.add_argument(
        "--cask_similarity_threshold",
        "--cask-similarity-threshold",
        dest="cask_similarity_threshold",
        type=float,
        default=0.985,
        help="Preferred cosine threshold for pre-RoPE adjacent scratch merges.",
    )
    parser.add_argument(
        "--cask_value_projection_threshold",
        "--cask-value-projection-threshold",
        dest="cask_value_projection_threshold",
        type=float,
        default=None,
        help="Optional ||W_O v_i - W_O v_j|| threshold. Pairs above this distance are not merged.",
    )
    parser.add_argument(
        "--cask_representative_mode",
        "--cask-representative-mode",
        dest="cask_representative_mode",
        choices=["weighted_latest", "score_max_source"],
        default="score_max_source",
        help="How merged scratch groups choose their surviving key anchor.",
    )
    parser.add_argument(
        "--cask_promotion_score_ratio",
        "--cask-promotion-score-ratio",
        dest="cask_promotion_score_ratio",
        type=float,
        default=None,
        help="Promote near-core scratch tokens whose scores stay above this fraction of the current core cutoff.",
    )
    parser.add_argument(
        "--cask_merge_score_mass_ratio_threshold",
        "--cask-merge-score-mass-ratio-threshold",
        dest="cask_merge_score_mass_ratio_threshold",
        type=float,
        default=None,
        help="Forbid merges whose combined score mass exceeds this multiple of the mean retained-slot score mass.",
    )
    parser.add_argument(
        "--cask_use_phase_markers",
        "--cask-use-phase-markers",
        dest="cask_use_phase_markers",
        type=str2bool,
        default=True,
        help="Restrict scratch merges to weak decode-phase segments inferred from marker tokens.",
    )
    parser.add_argument(
        "--expectedattention_n_future_positions",
        "--expectedattention-n-future-positions",
        dest="expectedattention_n_future_positions",
        type=int,
        default=512,
        help="Number of future positions to average for Expected Attention.",
    )
    parser.add_argument(
        "--expectedattention_n_sink",
        "--expectedattention-n-sink",
        dest="expectedattention_n_sink",
        type=int,
        default=4,
        help="Number of sink tokens to preserve for Expected Attention.",
    )
    parser.add_argument(
        "--expectedattention_use_covariance",
        "--expectedattention-use-covariance",
        dest="expectedattention_use_covariance",
        type=str2bool,
        default=True,
        help="Use covariance correction in Expected Attention scoring.",
    )
    parser.add_argument(
        "--expectedattention_use_vnorm",
        "--expectedattention-use-vnorm",
        dest="expectedattention_use_vnorm",
        type=str2bool,
        default=True,
        help="Rescale Expected Attention scores by value norms.",
    )
    parser.add_argument(
        "--expectedattention_epsilon",
        "--expectedattention-epsilon",
        dest="expectedattention_epsilon",
        type=float,
        default=0.0,
        help="Stability term used before value-norm scaling in Expected Attention.",
    )
    # Note: --divide_length is already defined above (line 238) for R-KV, reused for TriAttention alignment
    return parser.parse_args()


def configure_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_cask_phase_marker_token_ids(tokenizer: AutoTokenizer) -> tuple[tuple[int, ...], ...]:
    marker_texts = [
        "\n",
        "\n\n",
        "Therefore",
        " Therefore",
        "So",
        " So",
        "Hence",
        " Hence",
    ]
    markers: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for text in marker_texts:
        token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        if not token_ids or token_ids in seen:
            continue
        seen.add(token_ids)
        markers.append(token_ids)
    return tuple(markers)


def main(args: argparse.Namespace) -> None:
    args.dataset_name = Path(args.dataset_path).name.split(".")[0]
    if (not args.max_length) or args.max_length <= 0:
        if args.dataset_name in dataset2max_length:
            args.max_length = dataset2max_length[args.dataset_name]
    if args.eval_batch_size != 1:
        raise ValueError("eval_batch_size must be 1 for current TriAttention sharded runner.")

    total_samples = args.num_samples
    question_mode = use_question_sharding(total_samples, args.num_shards)

    if question_mode:
        start_draw = 0
        local_samples = total_samples
        run_ids = list(range(total_samples))
    else:
        start_draw, local_samples = compute_local_runs(total_samples, args.num_shards, args.shard_id)
        if local_samples == 0:
            return
        run_ids = list(range(start_draw, start_draw + local_samples))
    output_root = Path(args.output_dir)

    method_lower = args.method.lower() if args.method else ""
    patch_family = {"triattention", "horizonkv", "cask"}
    is_triattention_family = method_lower in patch_family
    if is_triattention_family:
        if args.kv_budget is None:
            raise ValueError(f"kv_budget must be provided for {method_lower}.")
        if bool(args.use_chat_template):
            import warnings
            warnings.warn(
                "TriAttention paper/baseline uses plain prompt (no chat template). "
                "Enabling chat template may affect reproducibility with published results.",
                UserWarning
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )
    tokenizer = configure_tokenizer(tokenizer)

    prompt_use_chat = args.use_chat_template
    prompts, test_data = load_dataset(
        Path(args.dataset_path),
        args.dataset_name,
        tokenizer,
        use_chat_template=prompt_use_chat,
        system_prompt=args.chat_system_prompt,
        max_examples=args.max_examples,
    )

    if question_mode:
        total_questions = len(test_data)
        start_q, local_q = compute_local_questions(total_questions, args.num_shards, args.shard_id)
        if local_q == 0:
            return
        prompts = prompts[start_q : start_q + local_q]
        test_data = test_data[start_q : start_q + local_q]
        sys.stderr.write(
            f"[shard-over-questions] shard={args.shard_id} questions={start_q}-{start_q + local_q - 1} "
            f"({local_q}/{total_questions}), draws={total_samples}\n"
        )
        sys.stderr.flush()

    expected_records = len(test_data)
    if expected_records == 0:
        return

    method_name = method_lower if method_lower else None
    triattention_method_config: Dict[str, object] = {}
    expectedattention_method_config: Dict[str, object] = {}
    if is_triattention_family:
        if args.kv_budget is None:
            raise ValueError(f"kv_budget must be provided for {method_lower}.")
        triattention_method_config = {
            "kv_budget": args.kv_budget,
            "window_size": args.window_size,
            "triattention_stats_file": args.triattention_stats_file,
            "round_window": args.round_window or args.window_size,
            "triattention_frequency_window": args.triattention_frequency_window,
            "triattention_score_aggregation": args.triattention_score_aggregation,
            "pruning_seed": args.pruning_seed,
            "triattention_normalize_scores": args.triattention_normalize_scores,
        }
    elif method_lower == "expectedattention":
        if args.kv_budget is None:
            raise ValueError("kv_budget must be provided for expectedattention.")
        expectedattention_method_config = {
            "budget": args.kv_budget,
            "window_size": args.window_size,
            "n_future_positions": args.expectedattention_n_future_positions,
            "n_sink": args.expectedattention_n_sink,
            "use_covariance": args.expectedattention_use_covariance,
            "use_vnorm": args.expectedattention_use_vnorm,
            "epsilon": args.expectedattention_epsilon,
            "protect_prefill": args.protect_prefill,
            "model_path": args.model_path,
        }

    method_config = {"budget": args.kv_budget, "window_size": args.window_size}
    if method_name in {"r1kv", "snapkv"}:
        method_config.update(
            {
                "mix_lambda": args.mix_lambda,
                "retain_ratio": args.retain_ratio,
                "retain_direction": args.retain_direction,
                "first_tokens": args.first_tokens,
                "fp32_topk": args.fp32_topk,
                "protect_prefill": args.protect_prefill,
            }
        )
    if is_triattention_family:
        method_config = triattention_method_config
    elif method_lower == "expectedattention":
        method_config = expectedattention_method_config

    compression_config = {
        "method": method_name,
        "method_config": method_config,
        "compression": None,
        "update_kv": args.update_kv,
    }
    model_config = {
        "divide_method": args.divide_method,
        "divide_length": args.divide_length,
        "compression_content": args.compression_content,
    }

    if method_name and method_name not in {"fullkv", "triattention", "horizonkv", "cask"}:
        if "llama" in args.model_path.lower():
            replace_llama(compression_config)
        elif "qwen3" in args.model_path.lower():
            replace_qwen3(compression_config)
        elif "qwen" in args.model_path.lower():
            replace_qwen2(compression_config)
        else:
            raise ValueError(f"Unsupported model: {args.model_path}")

    dtype = resolve_torch_dtype(args.load_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    model.config.update(model_config)

    capture_root = os.environ.get("RKV_QK_CAPTURE_DIR")
    capture_root_path = Path(capture_root).expanduser() if capture_root else None
    capture_model_info = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "kv_budget": args.kv_budget,
        "window_size": args.window_size,
        "method": method_name,
        "attn_implementation": args.attn_implementation,
        "load_dtype": args.load_dtype,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop_on_final_answer": args.stop_on_final_answer,
    }
    if capture_root_path:
        patched = patch_llama_attention_for_capture()
        if not patched:
            sys.stderr.write("[qk_capture] failed to patch LlamaAttention for capture; proceeding without QK dumps.\n")

    if method_name and method_name not in {"fullkv", "triattention", "horizonkv", "cask"}:
        model.newline_token_ids = [
            tokenizer.encode("\n")[-1],
            tokenizer.encode(".\n")[-1],
            tokenizer.encode(")\n")[-1],
            tokenizer.encode("\n\n")[-1],
            tokenizer.encode(".\n\n")[-1],
            tokenizer.encode(")\n\n")[-1],
        ]
        model.after_think_token_ids = [
            tokenizer.encode("</think>")[-1],
        ]
    elif is_triattention_family:
        if args.triattention_stats_file is None:
            raise ValueError(f"triattention_stats_file must be provided for {method_lower}.")
        stats_path = resolve_under_rkv(args.triattention_stats_file)
        if not stats_path.exists():
            raise FileNotFoundError(f"TriAttention stats file not found: {stats_path}")
        metadata_expectations = {}
        horizon_mode = args.triattention_horizon_mode or "fixed"
        norm_mode = args.triattention_norm_mode or "tri"
        if method_lower == "horizonkv":
            horizon_mode = args.triattention_horizon_mode or "adaptive"
            norm_mode = args.triattention_norm_mode or "tri"
        if method_lower == "cask":
            if horizon_mode != "fixed":
                raise ValueError("cask does not support adaptive/variational horizon modes.")
            if norm_mode != "tri":
                raise ValueError("cask does not support RMS2 or other norm overrides.")
            from triattention.methods.cask import apply_cask_patch
            phase_marker_token_ids = (
                build_cask_phase_marker_token_ids(tokenizer)
                if args.cask_use_phase_markers
                else ()
            )

            apply_cask_patch(
                model,
                stats_path=stats_path,
                model_path=Path(args.model_path),
                kv_budget=int(args.kv_budget),
                offset_max_length=args.triattention_frequency_window,
                score_aggregation=args.triattention_score_aggregation,
                pruning_seed=args.pruning_seed,
                metadata_expectations=metadata_expectations,
                normalize_scores=args.triattention_normalize_scores,
                count_prompt_tokens=args.count_prompt_tokens,
                allow_prefill_compression=args.allow_prefill_compression,
                divide_length=args.divide_length,
                use_slack_trigger=args.slack_budget_trigger,
                disable_mlr=args.disable_mlr,
                disable_trig=args.disable_trig,
                recent_window_size=args.window_size,
                protected_core_ratio=args.cask_protected_core_ratio,
                min_protected_core_tokens=args.cask_min_protected_core_tokens,
                core_selection_mode=args.cask_core_selection_mode,
                merge_operator=args.cask_merge_operator,
                merge_local_window=args.cask_merge_local_window,
                merge_similarity_threshold=args.cask_similarity_threshold,
                value_projection_threshold=args.cask_value_projection_threshold,
                representative_mode=args.cask_representative_mode,
                promotion_score_ratio=args.cask_promotion_score_ratio,
                merge_score_mass_ratio_threshold=args.cask_merge_score_mass_ratio_threshold,
                use_phase_markers=args.cask_use_phase_markers,
                phase_marker_token_ids=phase_marker_token_ids,
                score_dump_dir=Path(args.triattention_score_dump_dir) if args.triattention_score_dump_dir else None,
                score_dump_max_events=args.triattention_score_dump_max_events,
            )
        else:
            from triattention.methods.triattention import apply_triattention_patch

            apply_triattention_patch(
                model,
                stats_path=stats_path,
                model_path=Path(args.model_path),
                kv_budget=int(args.kv_budget),
                offset_max_length=args.triattention_frequency_window,
                score_aggregation=args.triattention_score_aggregation,
                pruning_seed=args.pruning_seed,
                metadata_expectations=metadata_expectations,
                normalize_scores=args.triattention_normalize_scores,
                count_prompt_tokens=args.count_prompt_tokens,
                allow_prefill_compression=args.allow_prefill_compression,
                divide_length=args.divide_length,
                use_slack_trigger=args.slack_budget_trigger,
                per_head_pruning=args.per_head_pruning,
                per_layer_perhead_pruning=args.per_layer_perhead_pruning,
                layer_perhead_aggregation=args.layer_perhead_aggregation,
                disable_mlr=args.disable_mlr,
                disable_trig=args.disable_trig,
                horizon_mode=horizon_mode,
                norm_mode=norm_mode,
                kernel_c_lambda=args.triattention_kernel_c_lambda,
                kernel_s0=args.triattention_kernel_s0,
                kernel_s1=args.triattention_kernel_s1,
                norm_lambda=args.triattention_norm_lambda,
                score_dump_dir=Path(args.triattention_score_dump_dir) if args.triattention_score_dump_dir else None,
                score_dump_max_events=args.triattention_score_dump_max_events,
            )

    for run_id in run_ids:
        artifacts = run_artifacts(output_root, args.shard_id, run_id)
        if run_is_complete(artifacts["run"], artifacts["meta"], expected_records):
            continue

        existing_path = artifacts["tmp"] if artifacts["tmp"].exists() else artifacts["run"] if artifacts["run"].exists() else None
        existing_indices = load_existing_sample_indices(existing_path) if existing_path else set()
        completed = len(existing_indices)

        if completed >= expected_records and expected_records > 0:
            if existing_path == artifacts["tmp"]:
                existing_path.replace(artifacts["run"])
            if not artifacts["meta"].exists():
                write_run_meta(artifacts["meta"], run_id, args.shard_id, expected_records)
            continue

        if artifacts["meta_tmp"].exists():
            artifacts["meta_tmp"].unlink()

        out_path = existing_path or artifacts["tmp"]
        if completed:
            sys.stderr.write(
                f"[resume] shard={args.shard_id} run={run_id} done={completed}/{expected_records}\n"
            )
            sys.stderr.flush()

        with out_path.open("a", encoding="utf-8") as fout:
            progress = completed
            for local_idx, prompt in enumerate(prompts):
                tokenized_prompts = tokenizer(
                    [prompt],
                    padding="longest",
                    return_tensors="pt",
                    add_special_tokens=True,
                ).to("cuda")
                prefill_length = int(tokenized_prompts["attention_mask"].sum().item())
                sample_idx = test_data[local_idx]["index"]
                if sample_idx in existing_indices:
                    continue
                record_id = test_data[local_idx].get("id", sample_idx)
                seed_value = args.seed + run_id * RUN_SEED_STRIDE + sample_idx
                set_seed(seed_value)
                active_compressor = getattr(model, "_triattention_compressor", None)
                if active_compressor is None:
                    active_compressor = getattr(model, "_cask_compressor", None)
                if active_compressor is not None:
                    active_compressor.set_dump_context(
                        shard_id=args.shard_id,
                        run_id=run_id,
                        sample_idx=sample_idx,
                        record_id=record_id,
                        prefill_length=prefill_length,
                    )

                if capture_root_path and capture_requested_for_sample(record_id):
                    activate_capture(
                        capture_root_path,
                        shard_id=args.shard_id,
                        run_id=run_id,
                        sample_id=record_id,
                        prefill_length=prefill_length,
                        model_info=capture_model_info,
                    )
                else:
                    deactivate_capture()

                progress += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sys.stderr.write(
                    f"[progress {timestamp}] shard={args.shard_id} run={run_id} "
                    f"problem={progress}/{expected_records} sample_idx={sample_idx}\n"
                )
                sys.stderr.flush()
                sample_max_new_tokens = test_data[local_idx].get("max_new_tokens")
                if sample_max_new_tokens is not None:
                    sample_max_new_tokens = int(sample_max_new_tokens)
                elif args.max_new_tokens is not None:
                    sample_max_new_tokens = int(args.max_new_tokens)
                generation_kwargs = {
                    "num_beams": 1,
                    "num_return_sequences": 1,
                    "do_sample": bool(args.do_sample),
                }
                if args.stop_on_final_answer:
                    generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                        [FinalAnswerStoppingCriteria(tokenizer, prefill_length)]
                    )
                if sample_max_new_tokens and sample_max_new_tokens > 0:
                    generation_kwargs["max_new_tokens"] = sample_max_new_tokens
                elif args.max_length and args.max_length > 0:
                    generation_kwargs["max_length"] = args.max_length
                else:
                    raise ValueError(
                        "Generation requires either args.max_length > 0 or a per-record max_new_tokens field."
                    )
                if args.do_sample:
                    generation_kwargs["temperature"] = args.temperature
                    generation_kwargs["top_p"] = args.top_p
                    if args.top_k is not None:
                        generation_kwargs["top_k"] = args.top_k if args.top_k > 0 else 0

                generation_started = time.perf_counter()
                try:
                    output = model.generate(
                        **tokenized_prompts,
                        **generation_kwargs,
                    )
                finally:
                    if active_compressor is not None:
                        active_compressor.clear_dump_context()
                generation_seconds = time.perf_counter() - generation_started

                total_tokens = int((output[0] != tokenizer.pad_token_id).sum().item())
                output_tokens = total_tokens - prefill_length
                decoded = tokenizer.decode(
                    output[0][prefill_length:], skip_special_tokens=True
                )

                record = dict(test_data[local_idx])
                record["prompt"] = prompt
                record["output"] = decoded
                record["prefill_tokens"] = prefill_length
                record["output_tokens"] = output_tokens
                record["total_tokens"] = total_tokens
                record["generation_seconds"] = generation_seconds
                record["sample_idx"] = sample_idx
                record["draw_idx"] = run_id
                if active_compressor is not None and hasattr(active_compressor, "get_runtime_summary"):
                    compression_summary = active_compressor.get_runtime_summary()
                    if compression_summary:
                        record["compression_summary"] = compression_summary

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                deactivate_capture()
            torch.cuda.empty_cache()

        write_run_meta(artifacts["meta"], run_id, args.shard_id, expected_records)
        if out_path == artifacts["tmp"]:
            artifacts["tmp"].replace(artifacts["run"])
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    main(args)
