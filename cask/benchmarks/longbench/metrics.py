"""Metrics adapted from the official LongBench evaluation code."""
from __future__ import annotations

import re
import string
from collections import Counter

import jieba
from fuzzywuzzy import fuzz
from rouge import Rouge


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def normalize_zh_answer(text: str) -> str:
    def white_space_fix(value: str) -> str:
        return "".join(value.split())

    def remove_punc(value: str) -> str:
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in value if ch not in all_punctuation)

    return white_space_fix(remove_punc(text.lower()))


def f1_score(prediction: list[str], ground_truth: list[str], **_: object) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_: object) -> float:
    return f1_score(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def qa_f1_zh_score(prediction: str, ground_truth: str, **_: object) -> float:
    prediction_tokens = [normalize_zh_answer(token) for token in jieba.cut(prediction, cut_all=False)]
    ground_truth_tokens = [normalize_zh_answer(token) for token in jieba.cut(ground_truth, cut_all=False)]
    prediction_tokens = [token for token in prediction_tokens if token]
    ground_truth_tokens = [token for token in ground_truth_tokens if token]
    return f1_score(prediction_tokens, ground_truth_tokens)


def classification_score(prediction: str, ground_truth: str, **kwargs: object) -> float:
    all_classes = kwargs["all_classes"]
    matches = [class_name for class_name in all_classes if class_name in prediction]
    matches = [match for match in matches if not (match in ground_truth and match != ground_truth)]
    if ground_truth in matches:
        return 1.0 / len(matches)
    return 0.0


def retrieval_score(prediction: str, ground_truth: str, **_: object) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(right_num / len(numbers))


def retrieval_zh_score(prediction: str, ground_truth: str, **_: object) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(right_num / len(numbers))


def count_score(prediction: str, ground_truth: str, **_: object) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return float(right_num / len(numbers))


def code_sim_score(prediction: str, ground_truth: str, **_: object) -> float:
    all_lines = prediction.lstrip("\n").split("\n")
    cleaned = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            cleaned = line
            break
    return fuzz.ratio(cleaned, ground_truth) / 100.0


def rouge_score(prediction: str, ground_truth: str, **_: object) -> float:
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction: str, ground_truth: str, **_: object) -> float:
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


DATASET_TO_METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

