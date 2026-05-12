from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from src.evaluator import evaluate, parse_graph_record

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from main import build_parser


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def chat_completion(self, **kwargs):
        self.calls += 1
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _item(i: int) -> dict:
    return {
        "id": i,
        "canonical_id": f"q{i:04d}",
        "question": f"question {i}",
        "domain": "STEM",
        "answer_type": "open",
        "human_answers": [f"gold {i}"],
        "rubrics": {"rubric_title": "r"},
    }


def test_parse_graph_record_falls_back_on_empty_output():
    client = FakeClient([RuntimeError("boom")])
    record = parse_graph_record(client, "model", "q", "answer")
    assert record["source"] == "heuristic"
    assert record["graph"]["nodes"]


def test_evaluate_resumes_from_existing_metrics_file(tmp_path: Path):
    benchmark = tmp_path / "benchmark.json"
    answers = tmp_path / "answers.json"
    result_dir = tmp_path / "result"
    result_dir.mkdir()

    items = [_item(1), _item(2)]
    benchmark.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2))
    answers.write_text(
        json.dumps(
            {
                "meta": {"model": "test-model"},
                "items": [
                    {**items[0], "model_answer": "a"},
                    {**items[1], "model_answer": "first. second."},
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    existing_file = result_dir / "test-model_metrics_test_model.json"
    existing_file.write_text(
        json.dumps(
            {
                "meta": {"timestamp": "20260101-000000", "status": "partial"},
                "items": [
                    {
                        **items[0],
                        "model_answer": "a",
                        "pm_score": 0.5,
                        "rm_score": 0.5,
                        "om_score": None,
                        "total_score": 0.5,
                        "parsed_dag": {"nodes": [], "edges": []},
                        "parsed_dag_source": "heuristic",
                        "parsed_dag_error": None,
                        "metrics": {
                            "pm": {"score": 0.5, "edges": [], "graph": {"nodes": [], "edges": []}},
                            "rm": {"score": 0.5},
                            "om": None,
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    client = FakeClient(
        [
            """EDUs:
EDU1: first.
EDU2: second.

RST ANALYSIS:
RELATION(EDU2, EDU1): BACKGROUND [SN]

TREE STRUCTURE:
ROOT[1-2]
  NUCLEUS: EDU1
  SATELLITE: EDU2
""",
            "yes",
            """{"score": 1, "rationale": "ok"}""",
        ]
    )

    out = evaluate(
        client=client,
        benchmark_path=str(benchmark),
        answer_path=str(answers),
        judge_model="test/model",
        parser_model="test/model",
        output_dir=str(result_dir),
    )

    assert out == existing_file
    payload = json.loads(existing_file.read_text())
    assert payload["meta"]["status"] == "complete"
    assert payload["meta"]["answer_model"] == "test-model"
    assert len(payload["items"]) == 2
    assert payload["items"][1]["parsed_dag_source"] == "parser"
    assert payload["items"][1]["parsed_rst_source"] == "parser"
    assert payload["items"][1]["parsed_dag"]["edges"][0]["relation_type"] == "BACKGROUND"
    assert client.calls == 3


def test_evaluate_skips_items_with_missing_answers(tmp_path: Path):
    benchmark = tmp_path / "benchmark.json"
    answers = tmp_path / "answers.json"
    result_dir = tmp_path / "result"
    result_dir.mkdir()

    items = [_item(1), _item(2)]
    benchmark.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2))
    answers.write_text(
        json.dumps(
            {
                "meta": {"model": "test-model"},
                "items": [
                    {**items[0], "model_answer": None, "model_answer_error": "empty"},
                    {**items[1], "model_answer": "first. second."},
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    client = FakeClient(
        [
            """EDUs:
EDU1: first.
EDU2: second.

RST ANALYSIS:
RELATION(EDU2, EDU1): BACKGROUND [SN]

TREE STRUCTURE:
ROOT[1-2]
  NUCLEUS: EDU1
  SATELLITE: EDU2
""",
            "yes",
            """{"score": 1, "rationale": "ok"}""",
        ]
    )

    out = evaluate(
        client=client,
        benchmark_path=str(benchmark),
        answer_path=str(answers),
        judge_model="test/model",
        parser_model="test/model",
        output_dir=str(result_dir),
    )

    payload = json.loads(out.read_text())
    assert payload["meta"]["status"] == "complete"
    assert payload["meta"]["count"] == 1
    assert payload["meta"]["failure_count"] == 1
    assert payload["state"] == {
        "total": 2,
        "processed": 2,
        "succeeded": 1,
        "failed": 1,
        "pending": 0,
        "mean_score": 1.0,
    }
    assert payload["summary"]["n_items"] == 1
    assert payload["failures"][0]["canonical_id"] == "q0001"
    assert payload["failures"][0]["evaluation_status"] == "failed"
    assert [item["canonical_id"] for item in payload["items"]] == ["q0002"]
    assert client.calls == 3


def test_evaluate_runs_items_with_configured_concurrency(tmp_path: Path):
    benchmark = tmp_path / "benchmark.json"
    answers = tmp_path / "answers.json"
    result_dir = tmp_path / "result"
    result_dir.mkdir()

    items = [_item(1), _item(2)]
    for item in items:
        item["rubrics"] = {"criteria": []}
    benchmark.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2))
    answers.write_text(
        json.dumps(
            {
                "meta": {"model": "test-model"},
                "items": [
                    {**items[0], "model_answer": "single sentence"},
                    {**items[1], "model_answer": "single sentence"},
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    class SlowClient:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.calls = 0
            self.lock = threading.Lock()

        def chat_completion(self, **kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls += 1
            try:
                time.sleep(0.05)
                if kwargs["messages"][0]["content"] == "Return JSON only.":
                    return '{"score": 1, "rationale": "ok"}'
                return """EDUs:
EDU1: single sentence

RST ANALYSIS:

TREE STRUCTURE:
ROOT[1]
"""
            finally:
                with self.lock:
                    self.active -= 1

    client = SlowClient()
    out = evaluate(
        client=client,
        benchmark_path=str(benchmark),
        answer_path=str(answers),
        judge_model="test/model",
        parser_model="test/model",
        output_dir=str(result_dir),
        concurrency=2,
    )

    payload = json.loads(out.read_text())
    assert client.calls == 4
    assert client.max_active >= 2
    assert payload["state"]["succeeded"] == 2
    assert payload["state"]["failed"] == 0


def test_evaluate_parser_accepts_concurrency():
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--answers",
            "answers.json",
            "--judge-model",
            "judge",
            "--concurrency",
            "32",
        ]
    )

    assert args.concurrency == 32
