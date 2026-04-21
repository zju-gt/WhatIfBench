from __future__ import annotations

from src import evaluator
from src.evaluator import judge_edge_messages, judge_yes_no


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


def test_judge_yes_no_parses_yes():
    client = FakeClient(["yes"])
    parsed = judge_yes_no(client, "m", [{"role": "user", "content": "x"}])
    assert parsed["score"] == 1.0


def test_judge_yes_no_parses_no():
    client = FakeClient(["no"])
    parsed = judge_yes_no(client, "m", [{"role": "user", "content": "x"}])
    assert parsed["score"] == 0.0


def test_judge_edge_messages_includes_relation_type():
    messages = judge_edge_messages("q", "a", "b", "CAUSE", "full answer")
    assert "Relation type:" in messages[1]["content"]
    assert "CAUSE" in messages[1]["content"]


def test_judge_yes_no_uses_larger_max_tokens():
    client = FakeClient(["yes"])
    judge_yes_no(client, "m", [{"role": "user", "content": "x"}])
    assert client.calls[0]["max_tokens"] == 1028


def test_judge_yes_no_logs_failure_reason(monkeypatch):
    logs: list[str] = []
    monkeypatch.setattr(evaluator, "log_write", logs.append)
    client = FakeClient(
        [
            RuntimeError(
                'OpenAI-compatible request failed: Empty model output (finish_reason=length, raw_choice={"message":{"content":null,"reasoning":"..."}})'
            )
        ]
    )

    parsed = judge_yes_no(
        client,
        "m",
        [{"role": "user", "content": "x"}],
        log_context="[q0001][pm][edge 1/7]",
    )

    assert parsed["score"] == 0.0
    assert any("failure because: OpenAI-compatible request failed: Empty model output" in line for line in logs)
    assert any("finish_reason=length" in line for line in logs)
