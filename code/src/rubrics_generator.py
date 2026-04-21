from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .openai_client import OpenAICompatibleClient
    from .utils import (
        ensure_dir,
        extract_json_text,
        item_key,
        read_json,
        resolve_input_path,
        resolve_output_path,
        sanitize_filename,
        utc_timestamp,
        write_json,
    )
except ImportError:  # pragma: no cover
    from openai_client import OpenAICompatibleClient
    from utils import (
        ensure_dir,
        extract_json_text,
        item_key,
        read_json,
        resolve_input_path,
        resolve_output_path,
        sanitize_filename,
        utc_timestamp,
        write_json,
    )


SYSTEM_PROMPT = (
    "You generate query-dependent evaluation criteria for open-ended long-range counterfactual reasoning. "
    "Return JSON only."
)

MAX_ATTEMPTS = 3


def rubric_messages(question: dict[str, Any]) -> list[dict[str, str]]:
    user = {
        "instruction": (
            "Create query-dependent criteria adapted from WritingBench's criteria-generation step, "
            "but for counterfactual long-range reasoning rather than writing quality."
        ),
        "required_output": (
            "Return a JSON array with exactly 5 criteria objects."
        ),
        "required_schema_per_criterion": {
            "name": "string",
            "criteria_description": "string",
            "1-2": "string",
            "3-4": "string",
            "5-6": "string",
            "7-8": "string",
            "9-10": "string",
        },
        "item": {
            "question": question["question"],
            "domain": question.get("domain"),
            "answer_type": question.get("answer_type"),
            "gold_answers": question.get("human_answers", [])[:3],
        },
        "constraints": [
            "Each criterion must be specific to this exact question, not generic.",
            "Prioritize causal fidelity under the counterfactual premise, mechanism coverage, long-range chain coherence, uncertainty calibration, and contradiction control.",
            "Keep each level description concise and behaviorally distinguishable.",
            "For closed items, include endpoint correctness in at least one criterion.",
            "For open items, emphasize plausibility, mechanism depth, and coverage of major downstream effects.",
        ],
    }
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)}]


def _normalize_criterion(raw: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or raw.get("criterion_name") or "").strip()
    desc = str(raw.get("criteria_description") or raw.get("description") or "").strip()
    if not name or not desc:
        return None
    levels = {}
    for band in ("1-2", "3-4", "5-6", "7-8", "9-10"):
        value = raw.get(band)
        if value is None:
            return None
        levels[band] = str(value).strip()
    if not all(levels.values()):
        return None
    return {"name": name, "criteria_description": desc, **levels}


def _build_query_dependent_rubric(item: dict[str, Any], criteria: list[dict[str, str]]) -> dict[str, Any]:
    domain = str(item.get("domain") or "unknown")
    answer_type = str(item.get("answer_type") or "open")
    return {
        "rubric_title": f"{domain}_{answer_type}_query_dependent_counterfactual_rubric",
        "question_focus": item.get("question", ""),
        "task_type": "counterfactual_long_horizon_reasoning",
        "evaluation_mode": "query_dependent_criteria",
        "criteria": criteria,
        "aggregation": "mean_over_criteria_then_divide_by_10",
    }


def normalize_rubric(item: dict[str, Any], raw_rubric: Any) -> dict[str, Any]:
    if isinstance(raw_rubric, dict) and isinstance(raw_rubric.get("rubric_title"), str) and isinstance(raw_rubric.get("score_guidance"), dict):
        return raw_rubric

    criteria_raw: list[Any] = []
    if isinstance(raw_rubric, list):
        criteria_raw = raw_rubric
    elif isinstance(raw_rubric, dict) and isinstance(raw_rubric.get("criteria"), list):
        criteria_raw = raw_rubric["criteria"]
    else:
        raise ValueError("Rubric output must be a criteria array or a rubric object containing criteria")

    criteria = []
    for raw in criteria_raw:
        criterion = _normalize_criterion(raw)
        if criterion is not None:
            criteria.append(criterion)

    if not criteria:
        raise ValueError("No valid criteria found in rubric output")

    if len(criteria) > 5:
        criteria = criteria[:5]
    return _build_query_dependent_rubric(item, criteria)


def generate_single_rubric(
    client: OpenAICompatibleClient,
    model: str,
    item: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = client.chat_completion(
                model=model,
                messages=rubric_messages(item),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            print("===========raw=============")
            print(raw)
            if not raw:
                raise ValueError("Empty model output")
            rubric = extract_json_text(raw)
            return normalize_rubric(item, rubric)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(1)
    raise RuntimeError(f"Failed to generate rubric after {MAX_ATTEMPTS} attempts: {last_error}")


def generate(
    client: OpenAICompatibleClient,
    benchmark_path: str,
    model: str,
    output_path: str,
    temperature: float = 0.6,
    max_tokens: int = 8192,
    limit: int | None = None,
) -> Path:
    dataset = read_json(resolve_input_path(benchmark_path))
    items = dataset[:limit] if limit is not None else dataset
    out_file = resolve_output_path(output_path)
    existing_payload: dict[str, Any] | None = None
    if out_file.exists():
        try:
            loaded = read_json(out_file)
            if isinstance(loaded, dict) and isinstance(loaded.get("items"), list):
                existing_payload = loaded
        except Exception:
            existing_payload = None

    output_map: dict[str, dict[str, Any]] = {}
    if existing_payload:
        for existing_item in existing_payload.get("items", []):
            if not isinstance(existing_item, dict):
                continue
            key = item_key(existing_item)
            if isinstance(existing_item.get("rubrics"), dict):
                output_map[key] = existing_item

    timestamp = (
        existing_payload.get("meta", {}).get("timestamp")
        if existing_payload and isinstance(existing_payload.get("meta"), dict)
        else utc_timestamp()
    )

    for item in tqdm(items, desc=f"rubrics:{model}", unit="item"):
        key = item_key(item)
        if key not in output_map:
            rubric = generate_single_rubric(
                client=client,
                model=model,
                item=item,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output_map[key] = {**item, "rubrics": rubric}
            payload = {
                "meta": {
                    "task": "rubrics_generator",
                    "model": model,
                    "timestamp": timestamp,
                    "source_benchmark": str(resolve_input_path(benchmark_path)),
                    "count": len(output_map),
                },
                "items": [output_map[item_key(x)] for x in items if item_key(x) in output_map],
            }
            write_json(out_file, payload)

    payload = {
        "meta": {
            "task": "rubrics_generator",
            "model": model,
            "timestamp": timestamp,
            "source_benchmark": str(resolve_input_path(benchmark_path)),
            "count": len(output_map),
        },
        "items": [output_map[item_key(x)] for x in items if item_key(x) in output_map],
    }

    write_json(out_file, payload)
    return out_file
