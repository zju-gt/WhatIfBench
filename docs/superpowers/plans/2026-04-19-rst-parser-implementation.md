# RST Parser Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current evaluator parser with a standalone RST parser that emits EDU/core-support/tree artifacts and projects them back into the existing PM graph.

**Architecture:** Add a new `src/rst_parser.py` module for prompt generation, raw-output normalization, tree recovery, and graph projection. Keep `src/evaluator.py` as the orchestrator that stores both RST artifacts and the backward-compatible projected graph.

**Tech Stack:** Python 3, pytest, OpenAI-compatible client, existing evaluator/result JSON schema

---

### Task 1: Add Failing Parser Tests

**Files:**
- Create: `code/tests/test_rst_parser.py`
- Modify: none
- Test: `code/tests/test_rst_parser.py`

- [ ] **Step 1: Write the failing test file**

```python
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


def test_parse_relations_block_extracts_relation_and_nuclearity():
    raw = "RST ANALYSIS:\nRELATION(EDU2, EDU1): BACKGROUND [SN]"
    relations = parse_relations_block(raw)
    assert relations[0]["type"] == "BACKGROUND"
    assert relations[0]["nuclearity"] == "SN"


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
    assert graph["edges"][0]["relation_type"] == "BACKGROUND"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest code/tests/test_rst_parser.py -v`
Expected: FAIL with import errors because `src.rst_parser` does not exist yet.

- [ ] **Step 3: Add integration-facing failing test**

```python
from src.evaluator import judge_edge_messages


def test_judge_edge_messages_includes_relation_type():
    messages = judge_edge_messages("q", "a", "b", "CAUSE", "full answer")
    assert "Relation type:" in messages[1]["content"]
    assert "CAUSE" in messages[1]["content"]
```

- [ ] **Step 4: Run focused evaluator test to verify it fails**

Run: `pytest code/tests/test_evaluator_yesno.py -v`
Expected: FAIL because the existing signature does not accept `relation_type`.

### Task 2: Implement Standalone RST Parser Module

**Files:**
- Create: `code/src/rst_parser.py`
- Modify: none
- Test: `code/tests/test_rst_parser.py`

- [ ] **Step 1: Write minimal parser module skeleton**

```python
from __future__ import annotations


def build_parser_messages(answer: str) -> list[dict[str, str]]:
    ...


def parse_edus_block(raw: str) -> list[dict[str, str]]:
    ...


def parse_relations_block(raw: str) -> list[dict[str, str]]:
    ...


def parse_rst_record(client, parser_model: str, answer: str) -> dict:
    ...


def project_rst_to_graph(rst_record: dict) -> dict:
    ...
```

- [ ] **Step 2: Implement prompt builder and text block parsers**

```python
def build_parser_messages(answer: str) -> list[dict[str, str]]:
    user = (
        "Segment the text into EDUs and produce an RST analysis.\n"
        "Return exactly these sections:\n"
        "EDUs:\n"
        "RST ANALYSIS:\n"
        "TREE STRUCTURE:\n\n"
        f"Text:\n{answer}"
    )
    return [{"role": "system", "content": "Return only the requested sections."}, {"role": "user", "content": user}]
```

- [ ] **Step 3: Implement normalization, fallback, and graph projection**

```python
def project_rst_to_graph(rst_record: dict) -> dict:
    nodes = [{"id": edu["id"], "text": edu["text"]} for edu in rst_record.get("edus", [])]
    edges = []
    for node in rst_record.get("tree", {}).get("nodes", []):
        ...
    return {"nodes": nodes, "edges": edges}
```

- [ ] **Step 4: Run parser tests**

Run: `pytest code/tests/test_rst_parser.py -v`
Expected: PASS

### Task 3: Integrate RST Parser Into Evaluator

**Files:**
- Modify: `code/src/evaluator.py`
- Test: `code/tests/test_evaluator_yesno.py`
- Test: `code/tests/test_evaluator_resume.py`

- [ ] **Step 1: Replace parser entrypoints**

```python
from .rst_parser import parse_rst_record, project_rst_to_graph
```

- [ ] **Step 2: Update edge judge prompt to include relation type**

```python
def judge_edge_messages(question: str, source: str, target: str, relation_type: str, answer: str) -> list[dict[str, str]]:
    ...
```

- [ ] **Step 3: Store RST artifacts and projected graph in evaluator payload**

```python
rst_record = parse_rst_record(...)
graph = project_rst_to_graph(rst_record)
...
"parsed_rst_raw": rst_record.get("raw_output"),
"parsed_rst": rst_record,
"parsed_dag": graph,
```

- [ ] **Step 4: Run focused evaluator tests**

Run: `pytest code/tests/test_evaluator_yesno.py code/tests/test_evaluator_resume.py -v`
Expected: PASS

### Task 4: Verify Full Regression Set

**Files:**
- Modify: `code/tests/test_evaluator_resume.py`
- Modify: `code/tests/test_evaluator_yesno.py`
- Test: `code/tests/test_rst_parser.py`
- Test: `code/tests/test_evaluator_resume.py`
- Test: `code/tests/test_evaluator_yesno.py`
- Test: `code/tests/test_evaluator_retries.py`

- [ ] **Step 1: Update tests for new stored fields and signatures**

```python
assert payload["items"][1]["parsed_rst_source"] == "parser"
assert payload["items"][1]["parsed_dag"]["edges"]
```

- [ ] **Step 2: Run targeted regression suite**

Run: `pytest code/tests/test_rst_parser.py code/tests/test_evaluator_resume.py code/tests/test_evaluator_yesno.py code/tests/test_evaluator_retries.py -v`
Expected: PASS

- [ ] **Step 3: Run broader suite**

Run: `pytest code/tests -v`
Expected: PASS

### Task 5: Final Quality Checks

**Files:**
- Modify: `code/src/rst_parser.py`
- Modify: `code/src/evaluator.py`

- [ ] **Step 1: Review naming and fallback behavior**

Check:
- RST record keys match the design spec
- fallback paths always set `source` and `error`
- `parsed_dag` remains backward compatible

- [ ] **Step 2: Run final verification commands**

Run: `pytest code/tests -v`
Expected: PASS

Run: `python3 code/src/main.py --help`
Expected: exit 0 and CLI help output

- [ ] **Step 3: Commit**

Current workspace is not a git repository, so no commit step is executable here.
