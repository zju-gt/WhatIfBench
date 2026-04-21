from __future__ import annotations

import json
from pathlib import Path

from src.evaluator import load_benchmark_items


def test_load_benchmark_items_from_payload(tmp_path: Path):
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"task": "rubrics_generator"},
                "items": [
                    {"canonical_id": "q1", "question": "a", "answer_type": "open", "human_answers": []},
                    {"canonical_id": "q2", "question": "b", "answer_type": "open", "human_answers": []},
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    items = load_benchmark_items(str(path))
    assert len(items) == 2
    assert items[0]["canonical_id"] == "q1"


def test_load_benchmark_items_from_list(tmp_path: Path):
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(
            [
                {"canonical_id": "q1", "question": "a", "answer_type": "open", "human_answers": []},
            ],
            ensure_ascii=False,
            indent=2,
        )
    )

    items = load_benchmark_items(str(path))
    assert len(items) == 1
    assert items[0]["canonical_id"] == "q1"

