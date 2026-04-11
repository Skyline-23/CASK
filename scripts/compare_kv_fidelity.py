#!/usr/bin/env python3
"""Compare candidate run fidelity against a FullKV or high-budget reference."""
from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from compare_experiment_runs import load_jsonl, resolve_eval_jsonl_path, resolve_merged_jsonl_path


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_ANSWER_RE = re.compile(r"Final answer:\s*(.+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Reference run root or merged jsonl path.")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate run root or merged jsonl path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional per-row CSV output path.")
    parser.add_argument(
        "--semantic-model",
        type=str,
        default=None,
        help="Optional HF encoder model for semantic/reference similarity (for example sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).",
    )
    parser.add_argument(
        "--semantic-max-length",
        type=int,
        default=512,
        help="Max token length used by the semantic encoder.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def normalize_answer_text(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = normalize_text(text)
    if not normalized:
        return None
    normalized = normalized.strip("$")
    normalized = normalized.rstrip(".")
    return normalized


def tokenize_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split(" ") if normalized else []


def common_prefix_length(left: list[str], right: list[str]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def common_prefix_chars(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def ratio_or_none(numerator: float | int, denominator: float | int) -> float | None:
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return None
    return float(numerator) / denominator_value


def extract_answer_from_output(text: str) -> str | None:
    if not text:
        return None
    boxed_matches = BOXED_RE.findall(text)
    if boxed_matches:
        return normalize_answer_text(boxed_matches[-1])
    final_matches = FINAL_ANSWER_RE.findall(text)
    if final_matches:
        return normalize_answer_text(final_matches[-1])
    return None


def extract_eval_prediction(item: dict[str, Any]) -> str | None:
    pred = item.get("pred")
    if isinstance(pred, list) and pred:
        return normalize_answer_text(str(pred[0]))
    if pred is not None:
        return normalize_answer_text(str(pred))
    return None


def load_eval_map(path: Path | None) -> Dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    records: Dict[str, dict[str, Any]] = {}
    for item in load_jsonl(path):
        idx = item.get("idx")
        if idx is None:
            continue
        records[str(idx)] = item
    return records


def load_merged_map(path: Path) -> Dict[str, dict[str, Any]]:
    records: Dict[str, dict[str, Any]] = {}
    for item in load_jsonl(path):
        idx = item.get("index")
        if idx is None:
            idx = item.get("idx")
        if idx is None:
            continue
        records[str(idx)] = item
    return records


class SemanticSimilarityScorer:
    def __init__(self, model_name: str, *, max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = int(max_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu()

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        left = self.encode([left_text for left_text, _ in pairs])
        right = self.encode([right_text for _, right_text in pairs])
        cosine = (left * right).sum(dim=1)
        return [float(value.item()) for value in cosine]


def summarize_pair(
    reference: Path,
    candidate: Path,
    *,
    semantic_model: str | None = None,
    semantic_max_length: int = 512,
) -> dict[str, Any]:
    reference_merged_path = resolve_merged_jsonl_path(reference)
    candidate_merged_path = resolve_merged_jsonl_path(candidate)
    if reference_merged_path is None:
        raise FileNotFoundError(f"Could not resolve merged.jsonl under reference path: {reference}")
    if candidate_merged_path is None:
        raise FileNotFoundError(f"Could not resolve merged.jsonl under candidate path: {candidate}")

    reference_eval_path = resolve_eval_jsonl_path(reference)
    candidate_eval_path = resolve_eval_jsonl_path(candidate)
    reference_eval = load_eval_map(reference_eval_path)
    candidate_eval = load_eval_map(candidate_eval_path)
    reference_merged = load_merged_map(reference_merged_path)
    candidate_merged = load_merged_map(candidate_merged_path)
    semantic_scorer = (
        SemanticSimilarityScorer(semantic_model, max_length=semantic_max_length)
        if semantic_model
        else None
    )

    shared_indices = sorted(set(reference_merged) & set(candidate_merged), key=lambda item: int(item))
    rows: list[dict[str, Any]] = []
    semantic_pairs: list[tuple[str, str]] = []
    for idx in shared_indices:
        reference_record = reference_merged[idx]
        candidate_record = candidate_merged[idx]
        reference_eval_record = reference_eval.get(idx, {})
        candidate_eval_record = candidate_eval.get(idx, {})

        reference_output = str(reference_record.get("output", "") or "")
        candidate_output = str(candidate_record.get("output", "") or "")
        reference_output_normalized = normalize_text(reference_output)
        candidate_output_normalized = normalize_text(candidate_output)
        reference_tokens = tokenize_text(reference_output)
        candidate_tokens = tokenize_text(candidate_output)

        reference_pred = extract_eval_prediction(reference_eval_record) or extract_answer_from_output(reference_output)
        candidate_pred = extract_eval_prediction(candidate_eval_record) or extract_answer_from_output(candidate_output)

        prefix_tokens = common_prefix_length(reference_tokens, candidate_tokens)
        prefix_chars = common_prefix_chars(reference_output_normalized, candidate_output_normalized)
        sequence_ratio = float(SequenceMatcher(a=reference_tokens, b=candidate_tokens).ratio())

        compression = candidate_record.get("compression_summary")
        terminal_saved_tokens = None
        terminal_saved_ratio = None
        cache_ratio = None
        cumulative_saved_tokens = None
        compression_events = None
        current_cache_tokens = None
        current_total_cardinality = None
        if isinstance(compression, dict):
            current_cache_tokens = int(compression.get("current_cache_tokens", 0) or 0)
            current_total_cardinality = int(compression.get("current_total_cardinality", 0) or 0)
            compression_events = int(compression.get("compression_events", 0) or 0)
            cumulative_saved_tokens = int(compression.get("total_scratch_saved_tokens", 0) or 0)
            terminal_saved_tokens = max(current_total_cardinality - current_cache_tokens, 0)
            terminal_saved_ratio = ratio_or_none(terminal_saved_tokens, current_total_cardinality)
            cache_ratio = ratio_or_none(current_cache_tokens, current_total_cardinality)

        row = {
            "idx": int(idx),
            "unique_id": reference_record.get("unique_id") or candidate_record.get("unique_id"),
            "question_match": (
                reference_record.get("question") == candidate_record.get("question")
                and reference_record.get("problem") == candidate_record.get("problem")
            ),
            "prompt_match": reference_record.get("prompt") == candidate_record.get("prompt"),
            "reference_output_tokens": int(reference_record.get("output_tokens", 0) or 0),
            "candidate_output_tokens": int(candidate_record.get("output_tokens", 0) or 0),
            "reference_total_tokens": int(reference_record.get("total_tokens", 0) or 0),
            "candidate_total_tokens": int(candidate_record.get("total_tokens", 0) or 0),
            "reference_pred": reference_pred,
            "candidate_pred": candidate_pred,
            "final_answer_match": reference_pred is not None and reference_pred == candidate_pred,
            "exact_output_match": reference_output == candidate_output,
            "normalized_output_match": reference_output_normalized == candidate_output_normalized,
            "sequence_ratio": sequence_ratio,
            "prefix_token_ratio": ratio_or_none(prefix_tokens, len(reference_tokens)),
            "prefix_char_ratio": ratio_or_none(prefix_chars, len(reference_output_normalized)),
            "output_token_ratio": ratio_or_none(
                int(candidate_record.get("output_tokens", 0) or 0),
                int(reference_record.get("output_tokens", 0) or 0),
            ),
            "output_token_delta": int(candidate_record.get("output_tokens", 0) or 0)
            - int(reference_record.get("output_tokens", 0) or 0),
            "semantic_similarity": None,
            "compression_events": compression_events,
            "terminal_saved_tokens": terminal_saved_tokens,
            "terminal_saved_ratio": terminal_saved_ratio,
            "terminal_cache_ratio": cache_ratio,
            "cumulative_saved_tokens": cumulative_saved_tokens,
            "current_cache_tokens": current_cache_tokens,
            "current_total_cardinality": current_total_cardinality,
        }
        rows.append(row)
        if semantic_scorer is not None:
            semantic_pairs.append((reference_output, candidate_output))

    if semantic_scorer is not None and rows:
        semantic_scores = semantic_scorer.score_pairs(semantic_pairs)
        for row, score in zip(rows, semantic_scores):
            row["semantic_similarity"] = score

    fidelity = {
        "records_compared": len(rows),
        "question_match_rate": mean_or_none([1.0 if row["question_match"] else 0.0 for row in rows]),
        "prompt_match_rate": mean_or_none([1.0 if row["prompt_match"] else 0.0 for row in rows]),
        "final_answer_match_rate": mean_or_none([1.0 if row["final_answer_match"] else 0.0 for row in rows]),
        "exact_output_match_rate": mean_or_none([1.0 if row["exact_output_match"] else 0.0 for row in rows]),
        "normalized_output_match_rate": mean_or_none(
            [1.0 if row["normalized_output_match"] else 0.0 for row in rows]
        ),
        "mean_sequence_ratio": mean_or_none(
            [float(row["sequence_ratio"]) for row in rows if row["sequence_ratio"] is not None]
        ),
        "mean_prefix_token_ratio": mean_or_none(
            [float(row["prefix_token_ratio"]) for row in rows if row["prefix_token_ratio"] is not None]
        ),
        "mean_prefix_char_ratio": mean_or_none(
            [float(row["prefix_char_ratio"]) for row in rows if row["prefix_char_ratio"] is not None]
        ),
        "mean_output_token_ratio": mean_or_none(
            [float(row["output_token_ratio"]) for row in rows if row["output_token_ratio"] is not None]
        ),
        "mean_output_token_delta": mean_or_none([float(row["output_token_delta"]) for row in rows]),
        "mean_semantic_similarity": mean_or_none(
            [float(row["semantic_similarity"]) for row in rows if row["semantic_similarity"] is not None]
        ),
    }
    savings = {
        "records_with_compression_summary": sum(
            1 for row in rows if row["terminal_saved_tokens"] is not None
        ),
        "records_with_positive_terminal_savings": sum(
            1 for row in rows if (row["terminal_saved_tokens"] or 0) > 0
        ),
        "mean_compression_events": mean_or_none(
            [float(row["compression_events"]) for row in rows if row["compression_events"] is not None]
        ),
        "mean_terminal_saved_tokens": mean_or_none(
            [float(row["terminal_saved_tokens"]) for row in rows if row["terminal_saved_tokens"] is not None]
        ),
        "mean_terminal_saved_ratio": mean_or_none(
            [float(row["terminal_saved_ratio"]) for row in rows if row["terminal_saved_ratio"] is not None]
        ),
        "mean_terminal_cache_ratio": mean_or_none(
            [float(row["terminal_cache_ratio"]) for row in rows if row["terminal_cache_ratio"] is not None]
        ),
        "mean_cumulative_saved_tokens": mean_or_none(
            [float(row["cumulative_saved_tokens"]) for row in rows if row["cumulative_saved_tokens"] is not None]
        ),
        "mean_current_cache_tokens": mean_or_none(
            [float(row["current_cache_tokens"]) for row in rows if row["current_cache_tokens"] is not None]
        ),
        "mean_current_total_cardinality": mean_or_none(
            [float(row["current_total_cardinality"]) for row in rows if row["current_total_cardinality"] is not None]
        ),
    }
    return {
        "reference_input": str(reference.resolve()),
        "candidate_input": str(candidate.resolve()),
        "reference_merged_jsonl": str(reference_merged_path.resolve()),
        "candidate_merged_jsonl": str(candidate_merged_path.resolve()),
        "reference_eval_jsonl": str(reference_eval_path.resolve()) if reference_eval_path is not None else None,
        "candidate_eval_jsonl": str(candidate_eval_path.resolve()) if candidate_eval_path is not None else None,
        "semantic_model": semantic_model,
        "shared_indices": [int(idx) for idx in shared_indices],
        "fidelity": fidelity,
        "savings": savings,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    summary = summarize_pair(
        args.reference,
        args.candidate,
        semantic_model=args.semantic_model,
        semantic_max_length=args.semantic_max_length,
    )

    fidelity = summary["fidelity"]
    savings = summary["savings"]
    print(f"records_compared={fidelity['records_compared']}")
    print(f"final_answer_match_rate={fidelity['final_answer_match_rate']}")
    print(f"mean_sequence_ratio={fidelity['mean_sequence_ratio']}")
    print(f"mean_prefix_token_ratio={fidelity['mean_prefix_token_ratio']}")
    print(f"mean_output_token_ratio={fidelity['mean_output_token_ratio']}")
    print(f"mean_semantic_similarity={fidelity['mean_semantic_similarity']}")
    print(f"mean_compression_events={savings['mean_compression_events']}")
    print(f"mean_terminal_saved_tokens={savings['mean_terminal_saved_tokens']}")
    print(f"mean_terminal_saved_ratio={savings['mean_terminal_saved_ratio']}")
    print(f"mean_terminal_cache_ratio={savings['mean_terminal_cache_ratio']}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "idx",
            "unique_id",
            "question_match",
            "prompt_match",
            "reference_output_tokens",
            "candidate_output_tokens",
            "reference_total_tokens",
            "candidate_total_tokens",
            "reference_pred",
            "candidate_pred",
            "final_answer_match",
            "exact_output_match",
            "normalized_output_match",
            "sequence_ratio",
            "prefix_token_ratio",
            "prefix_char_ratio",
            "output_token_ratio",
            "output_token_delta",
            "semantic_similarity",
            "compression_events",
            "terminal_saved_tokens",
            "terminal_saved_ratio",
            "terminal_cache_ratio",
            "cumulative_saved_tokens",
            "current_cache_tokens",
            "current_total_cardinality",
        ]
        with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary["rows"]:
                writer.writerow({key: row.get(key) for key in fieldnames})


if __name__ == "__main__":
    main()
