from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
        item_key,
        read_json,
        resolve_input_path,
        resolve_output_path,
        sanitize_filename,
        utc_timestamp,
        write_json,
    )


SYSTEM_PROMPT = (
    "You answer counterfactual benchmark questions. "
    "Give a direct, coherent, non-bulleted answer unless bullets are clearly helpful."
)


def build_messages(question: str, domain: str, answer_type: str) -> list[dict[str, str]]:
    user = (
        f"Question domain: {domain}\n"
        f"Question type: {answer_type}\n\n"
        f"Question:\n{question}\n\n"
        "Answer the question carefully and directly."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def load_existing_results(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return None
    return payload


def is_answer_missing(item: dict[str, Any]) -> bool:
    answer = item.get("model_answer")
    return answer is None or answer == ""


def build_ordered_items(items: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [results[item_key(x)] for x in items if item_key(x) in results]


def all_answers_present(items: list[dict[str, Any]]) -> bool:
    return all(not is_answer_missing(item) for item in items)


def build_output_payload(
    items: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    model: str,
    timestamp: str,
    dataset_path: str,
    concurrency: int,
) -> dict[str, Any]:
    ordered_items = build_ordered_items(items, results)
    return {
        "meta": {
            "task": "runner",
            "model": model,
            "timestamp": timestamp,
            "source_dataset": str(resolve_input_path(dataset_path)),
            "count": len(ordered_items),
            "concurrency": concurrency,
            "status": "complete" if all_answers_present(ordered_items) and len(ordered_items) == len(items) else "partial",
        },
        "items": ordered_items,
    }


def answer_item(
    client: OpenAICompatibleClient,
    item: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    question = item["question"]
    messages = build_messages(question, item.get("domain", ""), item.get("answer_type", ""))
    try:
        answer = client.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            **item,
            "model_name": model,
            "model_answer": answer,
            "model_answer_error": None,
        }
    except Exception as exc:
        return {
            **item,
            "model_name": model,
            "model_answer": None,
            "model_answer_error": str(exc),
        }


def run(
    client: OpenAICompatibleClient,
    dataset_path: str,
    model: str,
    output_dir: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    limit: int | None = None,
    concurrency: int = 1,
) -> Path:
    concurrency = max(1, int(concurrency))
    dataset = read_json(resolve_input_path(dataset_path))
    items = dataset[:limit] if limit is not None else dataset
    out_dir = ensure_dir(resolve_output_path(output_dir))
    out_file = out_dir / f"{sanitize_filename(model)}.json"
    existing_payload = load_existing_results(out_file)

    timestamp = (
        existing_payload.get("meta", {}).get("timestamp")
        if existing_payload and isinstance(existing_payload.get("meta"), dict)
        else utc_timestamp()
    )

    results: dict[str, dict[str, Any]] = {}
    if existing_payload:
        for existing_item in existing_payload.get("items", []):
            if isinstance(existing_item, dict):
                results[item_key(existing_item)] = existing_item

    pending_items: list[dict[str, Any]] = []
    for item in items:
        key = item_key(item)
        existing_item = results.get(key)
        if existing_item and not is_answer_missing(existing_item):
            continue
        pending_items.append(item)

    if concurrency == 1:
        for item in tqdm(pending_items, desc=f"run:{model}", unit="item"):
            results[item_key(item)] = answer_item(
                client=client,
                item=item,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            write_json(out_file, build_output_payload(items, results, model, timestamp, dataset_path, concurrency))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(answer_item, client, item, model, temperature, max_tokens): item
                for item in pending_items
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"run:{model}", unit="item"):
                item = futures[future]
                results[item_key(item)] = future.result()
                write_json(out_file, build_output_payload(items, results, model, timestamp, dataset_path, concurrency))

    payload = build_output_payload(items, results, model, timestamp, dataset_path, concurrency)
    write_json(out_file, payload)
    return out_file
