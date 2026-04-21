from __future__ import annotations

from src.rst_parser import (
    build_parser_messages,
    parse_edus_block,
    parse_rst_record,
    parse_relations_block,
    project_rst_to_graph,
)


class FakeClient:
    def __init__(self, output: str):
        self.output = output
        self.calls: list[dict] = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def test_build_parser_messages_uses_rst_sections():
    messages = build_parser_messages("A. B.")
    content = messages[1]["content"]
    assert "EDUs:" in content
    assert "RST ANALYSIS:" in content
    assert "TREE STRUCTURE:" in content


def test_parse_edus_block_extracts_ordered_edus():
    raw = "EDUs:\nEDU1: First.\nEDU2: Second."
    edus = parse_edus_block(raw)
    assert [x["id"] for x in edus] == ["EDU1", "EDU2"]
    assert [x["text"] for x in edus] == ["First.", "Second."]


def test_parse_relations_block_extracts_relation_and_nuclearity():
    raw = "RST ANALYSIS:\nRELATION(EDU2, EDU1): BACKGROUND [SN]"
    relations = parse_relations_block(raw)
    assert relations == [
        {
            "source": "EDU2",
            "target": "EDU1",
            "type": "BACKGROUND",
            "nuclearity": "SN",
        }
    ]


def test_parse_relations_block_accepts_model_emitted_relation_lines_and_spans():
    raw = "RST ANALYSIS:\nCONDITION(EDU1, EDU2): CONDITION [SN]\nSEQUENCE(EDU4-5, EDU6-7): SEQUENCE [NN]"
    relations = parse_relations_block(raw)
    assert relations == [
        {
            "source": "EDU1",
            "target": "EDU2",
            "type": "CONDITION",
            "nuclearity": "SN",
        },
        {
            "source": "EDU4-5",
            "target": "EDU6-7",
            "type": "SEQUENCE",
            "nuclearity": "NN",
        },
    ]


def test_project_rst_to_graph_emits_relation_typed_edges():
    record = {
        "edus": [{"id": "EDU1", "text": "core"}, {"id": "EDU2", "text": "support"}],
        "tree": {
            "nodes": [
                {
                    "id": "node_1",
                    "span": [1, 2],
                    "relation": "BACKGROUND",
                    "nuclearity": "SN",
                    "left_child": "EDU1",
                    "right_child": "EDU2",
                }
            ]
        },
    }
    graph = project_rst_to_graph(record)
    assert graph["nodes"] == [{"id": "EDU1", "text": "core"}, {"id": "EDU2", "text": "support"}]
    assert graph["edges"] == [
        {
            "source": "EDU2",
            "target": "EDU1",
            "relation_type": "BACKGROUND",
            "reason": "RST projection: satellite supports nucleus via BACKGROUND",
        }
    ]


def test_project_rst_to_graph_expands_span_references():
    record = {
        "edus": [
            {"id": "EDU1", "text": "support"},
            {"id": "EDU2", "text": "core a"},
            {"id": "EDU3", "text": "core b"},
        ],
        "tree": {
            "nodes": [
                {
                    "id": "node_1",
                    "span": [1, 3],
                    "relation": "ELABORATION",
                    "nuclearity": "SN",
                    "left_child": "EDU1",
                    "right_child": "EDU2-3",
                    "satellite_child": "EDU1",
                    "nucleus_child": "EDU2-3",
                }
            ]
        },
    }
    graph = project_rst_to_graph(record)
    assert graph["edges"] == [
        {
            "source": "EDU1",
            "target": "EDU2",
            "relation_type": "ELABORATION",
            "reason": "RST projection: satellite supports nucleus via ELABORATION",
        },
        {
            "source": "EDU1",
            "target": "EDU3",
            "relation_type": "ELABORATION",
            "reason": "RST projection: satellite supports nucleus via ELABORATION",
        },
    ]


def test_parse_rst_record_disables_reasoning_and_preserves_raw_output_on_parse_failure():
    raw = "RST ANALYSIS:\nCONDITION(EDU1, EDU2): CONDITION [SN]"
    client = FakeClient(raw)

    record = parse_rst_record(client, "model", "answer")

    assert client.calls[0]["max_tokens"] == 4096
    assert client.calls[0]["extra_body"] == {
        "reasoning": {"effort": "none", "exclude": True},
        "thinking": {"type": "disabled"},
    }
    assert record["source"] == "heuristic"
    assert record["raw_output"] == raw
    assert record["error"] == "Could not extract EDUs from parser output"
