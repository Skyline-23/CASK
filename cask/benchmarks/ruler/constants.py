"""RULER task definitions used by the local bridge runner."""
from __future__ import annotations

from collections import OrderedDict

DEFAULT_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

PARTIAL_MATCH_TASKS = {"qa_1", "qa_2"}

CATEGORY_TO_TASKS = OrderedDict(
    [
        (
            "retrieval",
            [
                "niah_single_1",
                "niah_single_2",
                "niah_single_3",
                "niah_multikey_1",
                "niah_multikey_2",
                "niah_multikey_3",
                "niah_multivalue",
                "niah_multiquery",
            ],
        ),
        ("multi_hop_tracing", ["vt"]),
        ("aggregation", ["cwe", "fwe"]),
        ("question_answering", ["qa_1", "qa_2"]),
    ]
)

