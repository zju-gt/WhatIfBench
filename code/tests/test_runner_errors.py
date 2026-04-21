from __future__ import annotations

import json
from pathlib import Path

from src.runner import run


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def chat_completion(self, **kwargs):
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def test_run_records_null_and_continues(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    out_dir = tmp_path / "result"
    items = [
        {
            "id": 1,
            "canonical_id": "q0001",
            "question": "question 1",
            "domain": "STEM",
            "answer_type": "open",
            "human_answers": ["gold 1"],
        },
        {
            "id": 2,
            "canonical_id": "q0002",
            "question": "question 2",
            "domain": "STEM",
            "answer_type": "open",
            "human_answers": ["gold 2"],
        },
    ]
    dataset.write_text(json.dumps(items, ensure_ascii=False, indent=2))

    client = FakeClient([RuntimeError("boom"), "A2"])
    out = run(client=client, dataset_path=str(dataset), model="test-model", output_dir=str(out_dir))

    payload = json.loads(out.read_text())
    assert len(payload["items"]) == 2
    assert payload["items"][0]["model_answer"] is None
    assert payload["items"][0]["model_answer_error"]
    assert payload["items"][1]["model_answer"] == "A2"
    assert payload["meta"]["status"] == "partial"

