from __future__ import annotations

import json
import urllib.request

import pytest

from src.openai_client import OpenAICompatibleClient


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
