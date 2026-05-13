from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from src.runner import run


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def chat_completion(self, **kwargs):
        self.calls += 1
        return self.outputs.pop(0)


class TrackingClient:
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def chat_completion(self, **kwargs):
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return "answer"


def _item(i: int) -> dict:
    return {
        "id": i,
        "canonical_id": f"q{i:04d}",
        "question": f"question {i}",
        "domain": "STEM",
        "answer_type": "open",
        "human_answers": [f"gold {i}"],
    }


def test_run_resumes_and_rewrites_same_file(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    out_dir = tmp_path / "result"
    items = [_item(1), _item(2), _item(3)]
    dataset.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    out_dir.mkdir()
    out_file = out_dir / "test-model.json"
    out_file.write_text(
        json.dumps(
            {
                "meta": {"status": "partial"},
                "items": [
                    {**items[0], "model_name": "test-model", "model_answer": "A1"},
                    {**items[1], "model_name": "test-model", "model_answer": None},
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    client = FakeClient(["A2", "A3"])
    result = run(
        client=client,
        dataset_path=str(dataset),
        model="test-model",
        output_dir=str(out_dir),
    )

    assert result == out_file
    payload = json.loads(out_file.read_text())
    assert payload["meta"]["status"] == "complete"
    assert payload["meta"]["count"] == 3
    assert len(payload["items"]) == 3
    assert payload["items"][0]["model_answer"] == "A1"
    assert payload["items"][1]["model_answer"] == "A2"
    assert payload["items"][2]["model_answer"] == "A3"
    assert client.calls == 2


def test_run_uses_concurrency(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    out_dir = tmp_path / "result"
    items = [_item(1), _item(2), _item(3), _item(4)]
    dataset.write_text(json.dumps(items, ensure_ascii=False, indent=2))

    client = TrackingClient()
    run(
        client=client,
        dataset_path=str(dataset),
        model="test-model",
        output_dir=str(out_dir),
        concurrency=2,
    )

    payload = json.loads((out_dir / "test-model.json").read_text())
    assert payload["meta"]["concurrency"] == 2
    assert client.calls == 4
    assert client.max_active > 1
