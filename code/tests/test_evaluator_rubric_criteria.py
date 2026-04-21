from __future__ import annotations

from src import evaluator
from src.evaluator import eval_rubric


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.call_kwargs = []

    def chat_completion(self, **kwargs):
        self.calls += 1
        self.call_kwargs.append(kwargs)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _criteria(name: str) -> dict:
    return {
        "name": name,
        "criteria_description": "desc",
        "1-2": "bad",
        "3-4": "weak",
        "5-6": "ok",
        "7-8": "good",
        "9-10": "excellent",
    }


def test_eval_rubric_query_dependent_criteria_mode():
    rubric = {
        "evaluation_mode": "query_dependent_criteria",
        "criteria": [
            _criteria("c1"),
            _criteria("c2"),
            _criteria("c3"),
            _criteria("c4"),
            _criteria("c5"),
        ],
    }
    client = FakeClient(
        [
            '{"score": 7, "reason": "ok"}',
            '{"score": 8, "reason": "ok"}',
            '{"score": 9, "reason": "ok"}',
            '{"score": 6, "reason": "ok"}',
            '{"score": 10, "reason": "ok"}',
        ]
    )
    parsed = eval_rubric(client, "judge", "q", rubric, "a")
    assert parsed["criterion_scores"] == [7, 8, 9, 6, 10]
    assert parsed["raw_score_10"] == 8.0
    assert parsed["score"] == 0.8
    assert client.calls == 5
    assert all(call["max_tokens"] == 4096 for call in client.call_kwargs)


def test_eval_rubric_legacy_mode_without_criteria():
    rubric = {"rubric_title": "legacy"}
    client = FakeClient(['{"score": 0.6, "rationale": "ok"}'])
    parsed = eval_rubric(client, "judge", "q", rubric, "a")
    assert parsed["score"] == 0.6
    assert client.calls == 1


def test_eval_rubric_logs_criterion_context_and_retry_reason(monkeypatch):
    rubric = {
        "evaluation_mode": "query_dependent_criteria",
        "criteria": [
            _criteria("causal fidelity"),
            _criteria("mechanism depth"),
        ],
    }
    client = FakeClient(
        [
            "not json",
            '{"score": 7, "reason": "ok"}',
            '{"score": 8, "reason": "ok"}',
        ]
    )
    logs: list[str] = []
    monkeypatch.setattr(evaluator, "log_write", logs.append)

    parsed = eval_rubric(client, "judge", "q0003", rubric, "a")

    assert parsed["criterion_scores"] == [7, 8]
    assert "[q0003][rm][criterion 1/2: causal fidelity] judge attempt 1/2" in logs
    assert any(
        "[q0003][rm][criterion 1/2: causal fidelity] retry because: No valid JSON object found" in line
        for line in logs
    )
    assert "[q0003][rm][criterion 2/2: mechanism depth] judge attempt 1/2" in logs
