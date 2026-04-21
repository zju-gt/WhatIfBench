from __future__ import annotations

from src.rst_parser import (
    build_parser_messages,
    parse_edus_block,
    parse_relations_block,
    project_rst_to_graph,
)


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
