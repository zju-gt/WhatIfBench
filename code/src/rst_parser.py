from __future__ import annotations

import re
from typing import Any

try:
    from .openai_client import OpenAICompatibleClient
except ImportError:  # pragma: no cover
    from openai_client import OpenAICompatibleClient


RELATION_LABELS = [
    "ELABORATION",
    "EXPLANATION",
    "EVIDENCE",
    "EXAMPLE",
    "CONTRAST",
    "COMPARISON",
    "CONCESSION",
    "ANTITHESIS",
    "CAUSE",
    "RESULT",
    "CONSEQUENCE",
    "PURPOSE",
    "CONDITION",
    "TEMPORAL",
    "SEQUENCE",
    "BACKGROUND",
    "CIRCUMSTANCE",
    "SUMMARY",
    "RESTATEMENT",
    "EVALUATION",
    "INTERPRETATION",
    "ATTRIBUTION",
    "DEFINITION",
    "CLASSIFICATION",
]

SECTION_HEADERS = ("EDUs:", "RST ANALYSIS:", "TREE STRUCTURE:")
PARSER_EXTRA_BODY = {
    "reasoning": {"effort": "none", "exclude": True},
    "thinking": {"type": "disabled"},
}
PARSER_MAX_TOKENS = 4096


def build_parser_messages(answer: str) -> list[dict[str, str]]:
    relation_list = ", ".join(RELATION_LABELS)
    user = (
        "You are an RST discourse parser.\n"
        "Analyze the text by first segmenting it into meaningful elementary discourse units (EDUs), "
        "then marking core vs supporting structure, then assigning discourse relation labels, and finally "
        "writing a compact RST tree.\n\n"
        "Use only these relation labels when possible:\n"
        f"{relation_list}\n\n"
        "Output exactly these sections and no extra commentary:\n"
        "EDUs:\n"
        "RST ANALYSIS:\n"
        "TREE STRUCTURE:\n\n"
        "Formatting rules:\n"
        "- In EDUs, use one line per EDU in the form `EDU1: ...`.\n"
        "- In RST ANALYSIS, use one line per relation in the form "
        "`RELATION(EDU_source, EDU_target): RELATION_TYPE [NUCLEARITY]`.\n"
        "- For asymmetric relations, EDU_source must be the supporting/satellite unit and EDU_target must be the core/nucleus unit.\n"
        "- Use nuclearity tags `SN`, `NS`, or `NN`.\n"
        "- In TREE STRUCTURE, write a compact textual tree using ROOT / NUCLEUS / SATELLITE / MULTINUCLEAR labels.\n\n"
        f"Text:\n{answer}"
    )
    return [
        {"role": "system", "content": "Return only the requested sections."},
        {"role": "user", "content": user},
    ]


def _normalize_relation_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", label.strip().upper()).strip("_")


def _normalize_edu_ref(ref: str) -> str:
    stripped = ref.strip()
    match = re.fullmatch(r"EDU\s*(\d+)(?:\s*-\s*(\d+))?", stripped, flags=re.IGNORECASE)
    if not match:
        return stripped.upper()
    start = match.group(1)
    end = match.group(2)
    return f"EDU{start}" if end is None else f"EDU{start}-{end}"


def _edu_span_bounds(edu_ref: str) -> tuple[int, int]:
    match = re.fullmatch(r"EDU(\d+)(?:-(\d+))?", _normalize_edu_ref(edu_ref))
    if not match:
        return (0, 0)
    start = int(match.group(1))
    end = int(match.group(2) or match.group(1))
    if end < start:
        start, end = end, start
    return (start, end)


def _extract_section(raw: str, header: str) -> str:
    pattern = rf"{re.escape(header)}\s*(.*?)(?=^({'|'.join(re.escape(x) for x in SECTION_HEADERS if x != header)})\s*$|\Z)"
    match = re.search(pattern, raw, flags=re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_edus_block(raw: str) -> list[dict[str, str]]:
    block = _extract_section(raw, "EDUs:") or raw
    edus: list[dict[str, str]] = []
    current_id: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_lines
        if current_id is None:
            return
        text = " ".join(line.strip() for line in current_lines).strip()
        edus.append({"id": current_id, "text": text})
        current_id = None
        current_lines = []

    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^(EDU\s*(\d+)|EDU(\d+))\s*[:\-]\s*(.*)$", stripped, flags=re.IGNORECASE)
        if match:
            flush()
            edu_number = match.group(2) or match.group(3)
            current_id = f"EDU{edu_number}"
            current_lines = [match.group(4).strip()]
        elif current_id is not None:
            current_lines.append(stripped)
    flush()
    return [edu for edu in edus if edu["text"]]


def parse_relations_block(raw: str) -> list[dict[str, str]]:
    block = _extract_section(raw, "RST ANALYSIS:") or raw
    relations: list[dict[str, str]] = []
    edu_ref = r"EDU\s*\d+(?:\s*-\s*\d+)?"
    patterns = [
        re.compile(
            rf"RELATION\(\s*({edu_ref})\s*,\s*({edu_ref})\s*\)\s*:\s*([A-Za-z _/-]+?)(?:\s*\[\s*(SN|NS|NN)\s*\])?$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"(EDU\d+(?:-\d+)?)\s*->\s*(EDU\d+(?:-\d+)?)\s*:\s*([A-Za-z _/-]+?)(?:\s*\[\s*(SN|NS|NN)\s*\])?$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"([A-Za-z _/-]+)\(\s*({edu_ref})\s*,\s*({edu_ref})\s*\)\s*:\s*([A-Za-z _/-]+?)(?:\s*\[\s*(SN|NS|NN)\s*\])?$",
            flags=re.IGNORECASE,
        ),
    ]
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in patterns:
            match = pattern.match(stripped)
            if not match:
                continue
            if pattern is patterns[2]:
                source_ref = match.group(2)
                target_ref = match.group(3)
                relation_label = match.group(4) or match.group(1)
                nuclearity = match.group(5)
            else:
                source_ref = match.group(1)
                target_ref = match.group(2)
                relation_label = match.group(3)
                nuclearity = match.group(4)
            relations.append(
                {
                    "source": _normalize_edu_ref(source_ref),
                    "target": _normalize_edu_ref(target_ref),
                    "type": _normalize_relation_label(relation_label),
                    "nuclearity": (nuclearity or "").upper() or "SN",
                }
            )
            break
    return relations


def parse_tree_lines(raw: str) -> list[str]:
    block = _extract_section(raw, "TREE STRUCTURE:")
    return [line.rstrip() for line in block.splitlines() if line.strip()]


def _edu_index(edu_id: str) -> int:
    start, _ = _edu_span_bounds(edu_id)
    return start


def _sort_edu_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if _edu_index(a) <= _edu_index(b) else (b, a)


def _build_tree_nodes_from_relations(relations: list[dict[str, str]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for index, relation in enumerate(relations, start=1):
        left_child, right_child = _sort_edu_pair(relation["source"], relation["target"])
        left_start, left_end = _edu_span_bounds(left_child)
        right_start, right_end = _edu_span_bounds(right_child)
        span_start = min(left_start, right_start)
        span_end = max(left_end, right_end)
        node: dict[str, Any] = {
            "id": f"node_{index}",
            "span": [span_start, span_end],
            "span_ref": f"EDU{span_start}" if span_start == span_end else f"EDU{span_start}-{span_end}",
            "relation": relation["type"],
            "nuclearity": relation["nuclearity"],
            "left_child": left_child,
            "right_child": right_child,
        }
        if relation["nuclearity"] in {"SN", "NS"}:
            node["satellite_child"] = relation["source"]
            node["nucleus_child"] = relation["target"]
        nodes.append(node)
    return nodes


def _build_flat_tree_nodes(edus: list[dict[str, str]], relation_type: str = "SEQUENCE") -> list[dict[str, Any]]:
    relations = []
    for left, right in zip(edus, edus[1:]):
        relations.append(
            {
                "source": left["id"],
                "target": right["id"],
                "type": relation_type,
                "nuclearity": "NN",
            }
        )
    return _build_tree_nodes_from_relations(relations)


def _split_answer_into_edus(answer: str) -> list[dict[str, str]]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", answer) if part.strip()]
    return [{"id": f"EDU{i+1}", "text": part} for i, part in enumerate(parts[:8])]


def heuristic_rst_record(answer: str, error: str | None = None) -> dict[str, Any]:
    return heuristic_rst_record_with_raw(answer, error=error, raw_output="")


def heuristic_rst_record_with_raw(answer: str, error: str | None = None, raw_output: str = "") -> dict[str, Any]:
    edus = _split_answer_into_edus(answer)
    relations = []
    for left, right in zip(edus, edus[1:]):
        relations.append(
            {
                "source": left["id"],
                "target": right["id"],
                "type": "SEQUENCE",
                "nuclearity": "NN",
            }
        )
    return {
        "raw_output": raw_output,
        "edus": edus,
        "relations": relations,
        "tree": {
            "ascii_lines": [],
            "nodes": _build_tree_nodes_from_relations(relations),
        },
        "source": "heuristic",
        "error": error,
    }


def _recover_tree(edus: list[dict[str, str]], relations: list[dict[str, str]], tree_lines: list[str]) -> tuple[dict[str, Any], str, str | None]:
    if relations:
        return {"ascii_lines": tree_lines, "nodes": _build_tree_nodes_from_relations(relations)}, "parser", None
    if len(edus) > 1:
        return {"ascii_lines": tree_lines, "nodes": _build_flat_tree_nodes(edus)}, "partial", "Recovered flat tree from EDUs without explicit relations"
    return {"ascii_lines": tree_lines, "nodes": []}, "partial", "RST output contained a single EDU and no explicit relations"


def parse_rst_record(
    client: OpenAICompatibleClient,
    parser_model: str,
    answer: str,
) -> dict[str, Any]:
    raw = ""
    try:
        raw = client.chat_completion(
            model=parser_model,
            messages=build_parser_messages(answer),
            temperature=0.0,
            max_tokens=PARSER_MAX_TOKENS,
            extra_body=PARSER_EXTRA_BODY,
        )
        if not raw:
            raise ValueError("Empty parser output")
        edus = parse_edus_block(raw)
        if not edus:
            raise ValueError("Could not extract EDUs from parser output")
        relations = parse_relations_block(raw)
        tree_lines = parse_tree_lines(raw)
        tree, source, recovery_error = _recover_tree(edus, relations, tree_lines)
        return {
            "raw_output": raw,
            "edus": edus,
            "relations": relations,
            "tree": tree,
            "source": source,
            "error": recovery_error,
        }
    except Exception as exc:
        return heuristic_rst_record_with_raw(answer, error=str(exc), raw_output=raw)


def _resolve_promoted(ref: str, edu_ids: set[str], node_map: dict[str, dict[str, Any]], memo: dict[str, list[str]]) -> list[str]:
    if ref in memo:
        return memo[ref]
    normalized_ref = _normalize_edu_ref(ref)
    if normalized_ref in memo:
        return memo[normalized_ref]

    if normalized_ref in edu_ids:
        memo[normalized_ref] = [normalized_ref]
        return memo[normalized_ref]

    if ref in edu_ids:
        memo[ref] = [ref]
        return memo[ref]
    node = node_map.get(normalized_ref) or node_map.get(ref)
    if not node:
        start, end = _edu_span_bounds(normalized_ref)
        if start and end and end > start:
            expanded = [f"EDU{i}" for i in range(start, end + 1) if f"EDU{i}" in edu_ids]
            if expanded:
                memo[normalized_ref] = expanded
                return memo[normalized_ref]
        memo[ref] = []
        return memo[ref]

    nuclearity = node.get("nuclearity", "")
    if nuclearity in {"SN", "NS"}:
        nucleus_ref = node.get("nucleus_child")
        if nucleus_ref:
            memo[ref] = _resolve_promoted(nucleus_ref, edu_ids, node_map, memo)
            return memo[ref]

    left = _resolve_promoted(str(node.get("left_child", "")), edu_ids, node_map, memo)
    right = _resolve_promoted(str(node.get("right_child", "")), edu_ids, node_map, memo)
    if nuclearity == "NN":
        memo[ref] = left + [item for item in right if item not in left]
    else:
        memo[ref] = left or right
    return memo[ref]


def project_rst_to_graph(rst_record: dict[str, Any]) -> dict[str, Any]:
    edus = rst_record.get("edus", [])
    nodes = [{"id": edu["id"], "text": edu["text"]} for edu in edus]
    edu_ids = {node["id"] for node in nodes}
    tree_nodes = list(rst_record.get("tree", {}).get("nodes", []))
    if not tree_nodes and rst_record.get("relations"):
        tree_nodes = _build_tree_nodes_from_relations(list(rst_record["relations"]))

    node_map: dict[str, dict[str, Any]] = {}
    for node in tree_nodes:
        node_id = node.get("id")
        span_ref = node.get("span_ref")
        if isinstance(node_id, str):
            node_map[node_id] = node
        if isinstance(span_ref, str):
            node_map[span_ref] = node
    promoted_cache: dict[str, list[str]] = {}
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for node in tree_nodes:
        relation = _normalize_relation_label(str(node.get("relation", "ELABORATION"))) or "ELABORATION"
        nuclearity = str(node.get("nuclearity", "")).upper()
        if nuclearity in {"SN", "NS"}:
            satellite_ref = node.get("satellite_child") or node.get("source") or node.get("right_child")
            nucleus_ref = node.get("nucleus_child") or node.get("target") or node.get("left_child")
            source_refs = _resolve_promoted(str(satellite_ref), edu_ids, node_map, promoted_cache)
            target_refs = _resolve_promoted(str(nucleus_ref), edu_ids, node_map, promoted_cache)
            reason = f"RST projection: satellite supports nucleus via {relation}"
        else:
            left_ref = node.get("left_child") or node.get("source")
            right_ref = node.get("right_child") or node.get("target")
            source_refs = _resolve_promoted(str(left_ref), edu_ids, node_map, promoted_cache)
            target_refs = _resolve_promoted(str(right_ref), edu_ids, node_map, promoted_cache)
            reason = f"RST projection: multinuclear relation via {relation}"

        for source in source_refs:
            for target in target_refs:
                if not source or not target or source == target:
                    continue
                edge_key = (source, target, relation)
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "relation_type": relation,
                        "reason": reason,
                    }
                )

    return {"nodes": nodes, "edges": edges}
