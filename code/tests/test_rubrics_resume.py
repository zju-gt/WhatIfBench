from __future__ import annotations

import json
from pathlib import Path

from src.rubrics_generator import generate


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def chat_completion(self, **kwargs):
        self.calls += 1
        return self.outputs.pop(0)


def _make_item(i: int) -> dict:
    return {
        "id": i,
        "canonical_id": f"q{i:04d}",
        "question": f"question {i}",
        "domain": "STEM",
        "answer_type": "open",
        "human_answers": [f"gold {i}"],
    }


def test_generate_resumes_from_existing_output(tmp_path: Path):
    benchmark = tmp_path / "benchmark.json"
    output = tmp_path / "rubrics.json"
    items = [_make_item(1), _make_item(2), _make_item(3)]
    benchmark.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    output.write_text(
        json.dumps(
            {
                "meta": {"timestamp": "20260101-000000"},
                "items": [
                    {**items[0], "rubrics": {"rubric_title": "done"}},
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    client = FakeClient(
        [
            """[
              {"name":"criterion_a","criteria_description":"desc","1-2":"a","3-4":"b","5-6":"c","7-8":"d","9-10":"e"}
            ]""",
            """[
              {"name":"criterion_b","criteria_description":"desc","1-2":"a","3-4":"b","5-6":"c","7-8":"d","9-10":"e"}
            ]""",
        ]
    )

    result = generate(
        client=client,
        benchmark_path=str(benchmark),
        model="test-model",
        output_path=str(output),
    )

    assert result == output
    payload = json.loads(output.read_text())
    assert len(payload["items"]) == 3
    assert payload["items"][0]["rubrics"]["rubric_title"] == "done"
    assert payload["items"][1]["rubrics"]["criteria"][0]["name"] == "criterion_a"
    assert payload["items"][2]["rubrics"]["criteria"][0]["name"] == "criterion_b"
    assert client.calls == 2
