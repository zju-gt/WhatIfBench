from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def sanitize_filename(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "model"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN check
        return default
    return number


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def first_existing_path(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_input_path(path_str: str) -> Path:
    raw = Path(path_str)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(
            [
                repo_root() / raw,
                code_root() / raw,
                repo_root() / "mvp" / raw,
                repo_root() / "mvp" / "data" / raw.name,
                code_root() / "data" / raw.name,
            ]
        )
    resolved = first_existing_path(candidates)
    if resolved is None:
        raise FileNotFoundError(f"Could not find input path: {path_str}")
    return resolved


def resolve_output_path(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw
    return code_root() / raw


def item_key(item: dict[str, Any]) -> str:
    canonical_id = item.get("canonical_id")
    if canonical_id:
        return str(canonical_id)
    if "id" in item:
        return str(item["id"])
    return json.dumps(item, sort_keys=True, ensure_ascii=False)


def extract_json_text(text: str) -> Any:
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    decoder = json.JSONDecoder()

    # First try direct decoding from likely JSON starts.
    candidates = [i for i, ch in enumerate(text) if ch in "{["]
    for start in candidates:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue

    # Fall back to balanced brace/bracket extraction.
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        while start != -1:
            depth = 0
            for end in range(start, len(text)):
                ch = text[end]
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : end + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)

    raise ValueError(f"No valid JSON object found in model output: {text[:200]!r}")


def safe_get_first_answer(item: dict[str, Any]) -> str | None:
    answers = item.get("human_answers") or []
    if answers:
        return answers[0]
    expert_answers = item.get("expert_answers") or []
    if expert_answers:
        first = expert_answers[0]
        if isinstance(first, dict):
            return first.get("text")
        if isinstance(first, str):
            return first
    return item.get("gt") or item.get("answer")
