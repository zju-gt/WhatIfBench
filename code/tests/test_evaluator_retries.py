from __future__ import annotations

from src.evaluator import judge_json


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def test_judge_json_retries_and_falls_back():
    client = FakeClient([None, "bad", '{"score": 1, "rationale": "ok"}'])
    parsed = judge_json(client, "model", [{"role": "user", "content": "x"}], {"score": 0.0})
    assert parsed["score"] == 1


def test_judge_json_passes_custom_max_tokens():
    client = FakeClient(['{"score": 1, "rationale": "ok"}'])
    parsed = judge_json(
        client,
        "model",
        [{"role": "user", "content": "x"}],
        {"score": 0.0},
        max_tokens=512,
    )
    assert parsed["score"] == 1
    assert client.calls[0]["max_tokens"] == 512
