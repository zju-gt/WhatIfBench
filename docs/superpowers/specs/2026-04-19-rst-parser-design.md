# RST Parser Refactor Design

Date: 2026-04-19

## Goal

Replace the current paragraph-to-JSON causal graph parser in the MVP evaluator with a paper-faithful RST parsing stage inspired by Disco-RAG (arXiv:2601.04377), while preserving compatibility with the existing PM evaluation pipeline.

The new parser must:

- produce EDUs from model answers
- label discourse units with nucleus/core vs satellite/support structure
- recover an RST-style tree with relation labels
- project the RST structure into the existing PM graph format so downstream edge judging still works
- keep robust fallbacks so evaluation does not fail hard on parser output variation

## Scope

This design covers only the first core-function refactor:

- add a new standalone module: `code/src/rst_parser.py`
- refactor evaluator-side parser integration
- preserve the current rubric and outcome evaluation flow

This design does not cover:

- RM or OM redesign
- benchmark schema redesign
- retrieval or external evidence injection for PM
- full inter-chunk rhetorical graph from Disco-RAG

## Reference Work

Primary reference:

- Disco-RAG, arXiv:2601.04377

Relevant method elements to reuse:

- intra-chunk parsing with LLM-based EDU segmentation
- explicit nuclearity assignment: nucleus vs satellite
- discourse relation prediction over discourse units
- textual RST tree output rather than direct JSON graph output

We will reuse the paper's parser style for single-answer parsing, but we will not implement the full multi-chunk inter-chunk rhetorical graph in this phase.

## Current Problem

The current parser in `code/src/evaluator.py` asks the model to directly output a small causal DAG in JSON. This has three limitations:

- it skips the intermediate discourse structure the paper relies on
- it forces all structure into a causal graph even when the answer contains non-causal relations such as contrast or elaboration
- it mixes parser prompting, parser parsing, fallback behavior, and PM graph production into one file

## Target Architecture

Introduce a dedicated parser module:

- `code/src/rst_parser.py`

This module will own four layers:

1. `prompt layer`
   - builds the paper-style parser prompt
2. `raw parse layer`
   - calls the LLM and returns the raw RST-style textual output
3. `normalization layer`
   - extracts EDUs, relation statements, and tree lines into structured Python objects
4. `projection layer`
   - converts the normalized RST tree into the existing PM graph format

The evaluator will stop treating parser output as native graph output. Instead it will:

1. parse answer into RST
2. normalize and validate RST
3. project RST to PM graph
4. run the existing PM edge judge on the projected graph

## Module Interface

`code/src/rst_parser.py` will expose a narrow API:

```python
def parse_rst_record(
    client: OpenAICompatibleClient,
    parser_model: str,
    answer: str,
) -> dict[str, Any]:
    ...

def project_rst_to_graph(rst_record: dict[str, Any]) -> dict[str, Any]:
    ...
```

The `parse_rst_record` return value will be the canonical parser artifact. It will include both raw text and normalized structure.

## Canonical Parser Record

The normalized parser record will use this shape:

```json
{
  "raw_output": "...",
  "edus": [
    {"id": "EDU1", "text": "..."},
    {"id": "EDU2", "text": "..."}
  ],
  "relations": [
    {
      "source": "EDU2",
      "target": "EDU1",
      "type": "BACKGROUND",
      "nuclearity": "SN"
    }
  ],
  "tree": {
    "ascii_lines": ["ROOT[1-2]", "..."],
    "nodes": [
      {
        "id": "node_1",
        "span": [1, 2],
        "relation": "BACKGROUND",
        "nuclearity": "SN",
        "left_child": "EDU1",
        "right_child": "EDU2"
      }
    ]
  },
  "source": "parser",
  "error": null
}
```

Notes:

- `source` tracks whether the record came from parser output, partial recovery, or heuristic fallback
- `tree.ascii_lines` preserves the paper-style tree representation for debugging and paper-faithful inspection
- `tree.nodes` is our internal reconstruction and is not expected from the model directly

## Prompt Strategy

The parser prompt should follow the paper's intra-chunk RST parsing style rather than our current JSON-only graph prompt.

Prompt requirements:

- ask the model to segment the input text into EDUs
- ask the model to identify nucleus vs satellite roles
- ask the model to identify discourse relation types
- ask the model to output a complete tree structure
- ask for no extra explanation outside the specified sections

Required output sections:

- `EDUs:`
- `RST ANALYSIS:`
- `TREE STRUCTURE:`

We will not ask for JSON in the first-pass prompt. That would weaken fidelity to the paper and reduce parser observability.

## Supported Relation Inventory

The parser will accept the paper-style discourse relation inventory and normalize labels to uppercase snake-case.

Supported labels include:

- `ELABORATION`
- `EXPLANATION`
- `EVIDENCE`
- `EXAMPLE`
- `CONTRAST`
- `COMPARISON`
- `CONCESSION`
- `ANTITHESIS`
- `CAUSE`
- `RESULT`
- `CONSEQUENCE`
- `PURPOSE`
- `CONDITION`
- `TEMPORAL`
- `SEQUENCE`
- `BACKGROUND`
- `CIRCUMSTANCE`
- `SUMMARY`
- `RESTATEMENT`
- `EVALUATION`
- `INTERPRETATION`
- `ATTRIBUTION`
- `DEFINITION`
- `CLASSIFICATION`

Unknown labels will be retained as normalized strings and marked for debug visibility instead of discarded immediately.

## Tree Reconstruction

The model output will likely be semi-structured text rather than machine-safe syntax. Tree reconstruction therefore needs explicit robustness rules.

Reconstruction steps:

1. parse `EDUs:` block into ordered EDU records
2. parse `RST ANALYSIS:` block into pairwise relation entries
3. parse `TREE STRUCTURE:` block into line-based tree rows
4. infer spans from EDU references and indentation or explicit span markers
5. build internal tree nodes bottom-up

If the tree section is incomplete but EDUs and relation lines are available, construct a minimal binary or flat recovered tree.

## Projection Into PM Graph

The current PM evaluator expects a graph with:

- `nodes`
- `edges`

The new projection layer will convert the RST tree into a graph without pretending that all relations are pure causality.

Projection rules:

1. graph nodes are EDU-level units, not sentence-level units
2. for a nucleus-satellite relation, emit a directed edge from the satellite-promoted EDU to the nucleus-promoted EDU
3. for multinuclear relations, connect promoted EDUs in textual order
4. attach the discourse relation label to every emitted edge
5. preserve a short textual reason derived from the relation and nuclearity

Example projected edge:

```json
{
  "source": "EDU2",
  "target": "EDU1",
  "relation_type": "BACKGROUND",
  "reason": "RST projection: satellite supports nucleus via BACKGROUND"
}
```

This preserves compatibility with the current PM judge while making the graph semantically richer.

## Promotion Rules

Tree-to-graph projection will use promotion sets.

Rules:

- for `NS`, the promoted set is inherited from the nucleus side
- for `SN`, the promoted set is inherited from the nucleus side
- for `NN`, the promoted set is the union of both child promoted sets
- leaf EDU nodes promote themselves

These promotion rules are the bridge between rhetorical structure and graph-level dependencies.

## Evaluator Integration

`code/src/evaluator.py` will change in these ways:

- replace `parse_graph_record` with `parse_rst_record`
- replace `parser_messages` with a wrapper imported from `rst_parser.py`
- keep a projected graph object for PM
- store both original RST artifacts and the graph projection in the result payload

New item-level result fields:

- `parsed_rst_raw`
- `parsed_rst`
- `parsed_rst_source`
- `parsed_rst_error`
- `parsed_dag`
- `parsed_dag_source`
- `parsed_dag_error`

`parsed_dag` remains for backward compatibility and will now refer to the graph projected from RST.

## PM Judge Adjustment

The current edge judge only sees source text and target text. After this refactor it should also receive the relation type.

Updated judge input should include:

- source EDU text
- target EDU text
- relation type
- full answer text

The judge still returns `yes` or `no`, but now answers the narrower question:

- is this discourse relation between these two projected units supported by the answer text?

This avoids collapsing `CAUSE`, `BACKGROUND`, `CONTRAST`, and `ELABORATION` into one undifferentiated logical check.

## Fallback Strategy

Parser robustness is mandatory because MVP evaluation must continue even with imperfect LLM formatting.

Fallback order:

1. full RST parse succeeds
2. EDU extraction plus relation extraction succeeds, tree reconstructed heuristically
3. EDU extraction only succeeds, flat tree synthesized
4. fallback to current sentence-based heuristic graph

Every fallback path must set:

- `source`
- `error`

so downstream analysis can distinguish parser quality from model quality.

## Testing

Add focused tests for:

- prompt output section expectations
- EDU block parsing
- relation block parsing
- tree reconstruction from textual tree lines
- RST projection to PM graph
- fallback from partial RST to flat tree
- fallback from invalid RST to current heuristic graph
- evaluator integration preserving resume behavior

Existing resume tests must continue to pass.

## Rollout Plan

Implementation should happen in this order:

1. create `code/src/rst_parser.py` with prompt builder and parsing helpers
2. add unit tests for normalization and projection
3. integrate parser record into evaluator
4. update PM judge prompt to include relation type
5. verify backward-compatible result JSON fields

## Risks

Main risks:

- model output variability in textual tree formatting
- ambiguity when reconstructing internal tree nodes from loose text
- projected EDU graph may increase graph size and evaluation cost

Mitigations:

- preserve raw parser output
- keep layered fallback behavior
- enforce small-answer parsing limits if needed in a later optimization pass

## Open Decisions Resolved

The following decisions are fixed by this spec:

- use a standalone `rst_parser.py`
- keep paper-style textual RST output as the primary parser artifact
- do not force first-pass parser output into JSON
- project RST into the existing PM graph instead of replacing PM entirely
- keep backward-compatible `parsed_dag` fields in evaluator outputs

## Constraints

Current workspace state:

- this directory is not a git repository
- therefore the spec can be written to disk, but not committed here

If a git repo is introduced later, this spec can be committed then without changing its content.

## Sources

- Disco-RAG arXiv abstract: https://arxiv.org/abs/2601.04377
- Disco-RAG HTML paper: https://arxiv.org/html/2601.04377v4
