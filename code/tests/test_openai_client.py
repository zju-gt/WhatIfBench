from __future__ import annotations

import json
import urllib.request

import pytest

from src.openai_client import MODELROUTER_BASE_URL, OpenAICompatibleClient, from_env


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_chat_completion_reports_finish_reason_on_empty_content(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {"content": None},
                "finish_reason": "length",
            }
        ]
    }

    def fake_urlopen(request, timeout=None):
        return FakeHTTPResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(api_key="test", base_url="https://example.com", max_retries=0)

    with pytest.raises(RuntimeError, match="finish_reason=length"):
        client.chat_completion(
            model="judge-model",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_chat_completion_merges_extra_body_fields(monkeypatch):
    seen: dict[str, dict] = {}
    payload = {
        "choices": [
            {
                "message": {"content": "ok"},
                "finish_reason": "stop",
            }
        ]
    }

    def fake_urlopen(request, timeout=None):
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(api_key="test", base_url="https://example.com", max_retries=0)

    result = client.chat_completion(
        model="judge-model",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={
            "reasoning": {"effort": "none", "exclude": True},
            "thinking": {"type": "disabled"},
        },
    )

    assert result == "ok"
    assert seen["body"]["reasoning"] == {"effort": "none", "exclude": True}
    assert seen["body"]["thinking"] == {"type": "disabled"}


def test_from_env_uses_modelrouter_defaults(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("MODELROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("MODELROUTER_API_KEY", "mr-key")

    client = from_env(api_source="modelrouter")

    assert client.api_key == "mr-key"
    assert client.base_url == MODELROUTER_BASE_URL
