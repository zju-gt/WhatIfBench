from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

try:
    from .utils import extract_json_text
except ImportError:  # pragma: no cover
    from utils import extract_json_text


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELROUTER_BASE_URL = "https://routify.alibaba-inc.com/protocol/openai/v1"
API_SOURCE_BASE_URLS = {
    "openrouter": OPENROUTER_BASE_URL,
    "modelrouter": MODELROUTER_BASE_URL,
}


@dataclass
class OpenAICompatibleClient:
    api_key: str
    base_url: str = OPENROUTER_BASE_URL
    timeout: int = 120
    max_retries: int = 3

    @staticmethod
    def _summarize_choice(choice: Any) -> str:
        try:
            rendered = json.dumps(choice, ensure_ascii=False)
        except TypeError:
            rendered = repr(choice)
        rendered = rendered.replace("\n", "\\n")
        return rendered[:240]

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        url = self.base_url.rstrip("/") + "/chat/completions"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choice = data["choices"][0]
                message = choice.get("message") or {}
                content = message.get("content")
                if content is None or content == "":
                    finish_reason = choice.get("finish_reason")
                    raw_choice = self._summarize_choice(choice)
                    raise ValueError(
                        "Empty model output "
                        f"(finish_reason={finish_reason!s}, raw_choice={raw_choice})"
                    )
                return content
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_seconds = 2 ** attempt
                time.sleep(sleep_seconds)
            except ValueError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI-compatible request failed: {last_error}")


def from_env(
    api_key: str | None = None,
    base_url: str | None = None,
    api_source: str | None = None,
) -> OpenAICompatibleClient:
    source = (api_source or os.getenv("API_SOURCE") or "openrouter").strip().lower()
    if source not in API_SOURCE_BASE_URLS:
        raise ValueError(f"Unsupported api_source: {source}. Choose openrouter or modelrouter.")

    if api_key is None:
        if source == "modelrouter":
            api_key = (
                os.getenv("MODELROUTER_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("API_KEY")
            )
        else:
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        if source == "modelrouter":
            raise EnvironmentError("Set MODELROUTER_API_KEY, LLM_API_KEY, OPENAI_API_KEY, or API_KEY")
        raise EnvironmentError("Set OPENROUTER_API_KEY, OPENAI_API_KEY, or API_KEY")

    if base_url is None:
        if source == "modelrouter":
            base_url = os.getenv("MODELROUTER_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        else:
            base_url = os.getenv("OPENAI_BASE_URL")
    base_url = base_url or API_SOURCE_BASE_URLS[source]
    return OpenAICompatibleClient(api_key=api_key, base_url=base_url)
