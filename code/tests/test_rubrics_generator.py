from __future__ import annotations

from src.rubrics_generator import generate_single_rubric


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


def test_generate_single_rubric_retries_until_valid():
    client = FakeClient(
        [
            None,
            "not json",
            """```json
            [
              {
                "name": "causal_fidelity",
                "criteria_description": "Assesses whether the response follows coherent causal logic under the counterfactual premise.",
                "1-2": "Critical causal breakdown.",
                "3-4": "Major causal gaps.",
                "5-6": "Basic but incomplete causality.",
                "7-8": "Strong causal chain with minor issues.",
                "9-10": "Excellent, rigorous causal reasoning."
              }
            ]
            ```""",
        ]
    )
    rubric = generate_single_rubric(client, "model", {"question": "q", "human_answers": []}, 0.0, 16)
    assert rubric["rubric_title"]
    assert rubric["evaluation_mode"] == "query_dependent_criteria"
    assert len(rubric["criteria"]) == 1
    assert rubric["criteria"][0]["name"] == "causal_fidelity"
    assert client.calls == 3
