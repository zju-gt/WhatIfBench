from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BUCKETS = ("STEM", "Hybrid", "HASS")


def _coerce_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _metric_value(item: dict[str, Any], metric_name: str) -> float | None:
    direct = _coerce_float(item.get(f"{metric_name}_score"))
    if direct is not None:
        return direct

    metrics = item.get("metrics")
    if isinstance(metrics, dict):
        metric = metrics.get(metric_name)
        if isinstance(metric, dict):
            return _coerce_float(metric.get("score"))
    return None


def _bucket_name(item: dict[str, Any]) -> str | None:
    domain = str(item.get("domain", "")).strip().upper()
    if item.get("is_hybrid") is True or domain == "HYBRID":
        return "Hybrid"
    if domain == "STEM":
        return "STEM"
    if domain == "HASS":
        return "HASS"
    return None


def load_items(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    raw_items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise TypeError(f"Unsupported JSON format: {path}")
    items = [item for item in raw_items if isinstance(item, dict)]
    return meta if isinstance(meta, dict) else {}, items


def compute_domain_metrics(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {
        bucket: {"n": 0, "pm": [], "rm": []}
        for bucket in BUCKETS
    }
    for item in items:
        bucket = _bucket_name(item)
        if bucket is None:
            continue
        grouped[bucket]["n"] += 1
        pm = _metric_value(item, "pm")
        rm = _metric_value(item, "rm")
        if pm is not None:
            grouped[bucket]["pm"].append(pm)
        if rm is not None:
            grouped[bucket]["rm"].append(rm)

    return {
        bucket: {
            "n": values["n"],
            "pm_n": len(values["pm"]),
            "pm_mean": _mean(values["pm"]),
            "rm_n": len(values["rm"]),
            "rm_mean": _mean(values["rm"]),
        }
        for bucket, values in grouped.items()
    }


def _format_number(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def print_domain_metrics(path: Path) -> None:
    meta, items = load_items(path)
    metrics = compute_domain_metrics(items)
    answer_model = meta.get("answer_model") or meta.get("model") or path.stem
    judge_model = meta.get("judge_model")

    print(f"file: {path}")
    print(f"answer_model: {answer_model}")
    if judge_model:
        print(f"judge_model: {judge_model}")
    if meta.get("status"):
        print(f"status: {meta['status']}")
    print()
    print(f"{'domain':<8} {'n':>5} {'pm_n':>5} {'pm_mean':>8} {'rm_n':>5} {'rm_mean':>8}")
    print("-" * 47)
    for bucket in BUCKETS:
        row = metrics[bucket]
        print(
            f"{bucket:<8} "
            f"{row['n']:>5} "
            f"{row['pm_n']:>5} "
            f"{_format_number(row['pm_mean']):>8} "
            f"{row['rm_n']:>5} "
            f"{_format_number(row['rm_mean']):>8}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print PM/RM means for STEM, Hybrid, and HASS in an evaluated metrics JSON file."
    )
    parser.add_argument("json_path", help="Path to a completed evaluated metrics JSON file.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print_domain_metrics(Path(args.json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
