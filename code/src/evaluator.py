from __future__ import annotations

import json
import time
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
    from .rst_parser import parse_rst_record, project_rst_to_graph
    from .utils import (
        clamp,
        coerce_float,
        ensure_dir,
        extract_json_text,
        first_existing_path,
        item_key,
        mean,
        read_json,
        resolve_input_path,
        resolve_output_path,
        sanitize_filename,
        safe_get_first_answer,
        utc_timestamp,
        write_json,
    )
except ImportError:  # pragma: no cover
    from openai_client import OpenAICompatibleClient
    from rst_parser import parse_rst_record, project_rst_to_graph
    from utils import (
        clamp,
        coerce_float,
        ensure_dir,
        extract_json_text,
        first_existing_path,
        item_key,
        mean,
        read_json,
        resolve_input_path,
        resolve_output_path,
        sanitize_filename,
        safe_get_first_answer,
        utc_timestamp,
        write_json,
    )


log_write = getattr(tqdm, "write", print)

EDGE_RETRIES = 1
RUBRIC_RETRIES = 2
OM_RETRIES = 2
PM_MAX_TOKENS = 1028
RM_MAX_TOKENS = 4096
DEFAULT_JSON_MAX_TOKENS = 256


def parse_graph_record(
    client: OpenAICompatibleClient,
    parser_model: str,
    question: str,
    answer: str,
) -> dict[str, Any]:
    rst_record = parse_rst_record(
        client=client,
        parser_model=parser_model,
        answer=answer,
    )
    graph = project_rst_to_graph(rst_record)
    return {
        "graph": graph,
        "source": rst_record.get("source", "heuristic"),
        "error": rst_record.get("error"),
        "rst_record": rst_record,
    }


def judge_edge_messages(question: str, source: str, target: str, relation_type: str, answer: str) -> list[dict[str, str]]:
    user = f"""
You are an extremely strict judge of one extracted discourse edge in a counterfactual answer.
Answer with only one token: yes or no.

Judge only whether the full answer explicitly supports this exact directed relation:
source claim -> target claim with relation type = {relation_type}.

Use this conservative standard:
- Say yes only if both claims are present or unambiguously paraphrased in the answer, and the answer clearly links the source to the target with the stated relation.
- Say no if the edge is merely plausible, world-knowledge-supported, implied by the question, or inferable only by filling gaps yourself.
- Say no if the answer mentions both claims but does not clearly connect them in the stated direction.
- Say no if the relation type is too strong, too specific, reversed, contradicted, or only weakly hinted at.
- Do not reward fluent prose, general relevance, or broad topical overlap.

Question:
{question}

Source claim:
{source}

Target claim:
{target}

Relation type:
{relation_type}

Full answer:
{answer}
"""
    return [{"role": "system", "content": "Reply with only yes or no."}, {"role": "user", "content": user.strip()}]


def rubric_messages(question: str, rubric: Any, answer: str) -> list[dict[str, str]]:
    user = f"""
You are an extremely strict rubric judge for counterfactual long-range reasoning.
Return JSON only with keys score and rationale.

Score from 0 to 1 and judge the answer against the rubric.

Use a conservative grading scale:
- 0.90-1.00: exceptional; satisfies nearly every rubric requirement with precise causal detail and no important errors.
- 0.75-0.89: strong; mostly satisfies the rubric but has minor omissions, shallow links, or limited uncertainty handling.
- 0.55-0.74: adequate; partially satisfies the rubric but misses important mechanisms, consequences, or constraints.
- 0.30-0.54: weak; generic, incomplete, or only loosely tied to the counterfactual premise.
- 0.00-0.29: poor; contradicts the premise, is mostly unsupported, or fails the rubric.

Rules:
- Grade evidence actually present in the answer, not what a knowledgeable reader could infer.
- Penalize generic alternate-history summaries, unsupported leaps, missing long-range consequences, and lack of mechanism detail.
- Penalize confident claims that ignore uncertainty or historical/domain constraints.
- Do not give a high score for fluency, length, or topical relevance alone.
- If the answer is correct only at a high level but shallow, the score should usually be at most 0.65.
- If it misses a major rubric dimension, the score should usually be at most 0.75.

Question:
{question}

Rubric:
{rubric}

Answer:
{answer}
"""
    return [{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": user.strip()}]


def criterion_score_messages(question: str, criterion: dict[str, Any], answer: str) -> list[dict[str, str]]:
    criterion_json = json.dumps(criterion, ensure_ascii=False, indent=2)
    user = f"""
You are an extremely strict evaluator of one query-dependent criterion for a counterfactual long-range reasoning task.
Score from 1 to 10 using the criterion's own band definitions.
Return JSON only with keys score and reason.

Question:
{question}

Criterion:
{criterion_json}

Answer:
{answer}

Rules:
- Select one integer score from 1 to 10.
- Follow the criterion's 1-2 / 3-4 / 5-6 / 7-8 / 9-10 descriptions.
- Grade only what is explicitly present or unambiguously paraphrased in the answer.
- Treat 5 as the default for a partially relevant but shallow answer; move upward only for concrete evidence.
- Scores 9-10 require outstanding, criterion-specific coverage with precise mechanisms, long-range implications, constraints, and no serious gaps.
- Scores 7-8 require strong criterion-specific reasoning, but still allow minor omissions or minor uncertainty issues.
- Scores 5-6 mean the answer addresses the criterion in a basic or uneven way but lacks depth, coverage, or grounding.
- Scores 3-4 mean the answer is mostly generic, underdeveloped, or weakly connected to the criterion.
- Scores 1-2 mean the answer ignores, contradicts, or seriously mishandles the criterion.
- Penalize unsupported causal leaps, missing major consequences, premise drift, contradictions, overconfidence, and vague lists of outcomes.
- Do not reward fluency, length, or broad topical relevance unless the criterion is actually satisfied.
- If the answer only states a plausible conclusion without explaining mechanisms, score at most 6.
- If the answer misses a central element named in the criterion, score at most 7 even if the rest is good.
"""
    return [{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": user.strip()}]


def om_messages(question: str, gold: str, answer: str) -> list[dict[str, str]]:
    user = f"""
You are a strict outcome judge.
Return JSON only with keys score and rationale.

Score 1 if the model answer is semantically consistent with the gold answer, otherwise 0.
Be tolerant to paraphrase and different wording.

Question:
{question}

Gold answer:
{gold}

Model answer:
{answer}
"""
    return [{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": user.strip()}]


def parse_graph(
    client: OpenAICompatibleClient,
    parser_model: str,
    question: str,
    answer: str,
) -> dict[str, Any]:
    return parse_graph_record(client, parser_model, question, answer)["graph"]


def judge_json(
    client: OpenAICompatibleClient,
    model: str,
    messages: list[dict[str, str]],
    fallback: dict[str, Any],
    retries: int = 3,
    log_context: str | None = None,
    max_tokens: int = DEFAULT_JSON_MAX_TOKENS,
) -> dict[str, Any]:
    last_error: Exception | None = None
    log_prefix = log_context or f"[{model}]"
    for attempt in range(retries):
        try:
            log_write(f"{log_prefix} judge attempt {attempt + 1}/{retries}")
            raw = client.chat_completion(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            if not raw:
                raise ValueError("Empty model output")
            parsed = extract_json_text(raw)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Judge output must be a JSON object")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                if log_context:
                    log_write(f"{log_prefix} retry because: {exc}")
                time.sleep(1)
    result = dict(fallback)
    result["error"] = str(last_error) if last_error else "unknown error"
    return result


def judge_yes_no(
    client: OpenAICompatibleClient,
    model: str,
    messages: list[dict[str, str]],
    retries: int = 1,
    log_context: str | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    log_prefix = log_context or f"[{model}]"
    for attempt in range(retries):
        try:
            log_write(f"{log_prefix} judge attempt {attempt + 1}/{retries}")
            raw = client.chat_completion(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=PM_MAX_TOKENS,
            )
            if not raw:
                raise ValueError("Empty model output")
            normalized = raw.strip().lower()
            if normalized.startswith("yes"):
                return {"score": 1.0, "raw": raw}
            if normalized.startswith("no"):
                return {"score": 0.0, "raw": raw}
            raise ValueError(f"Unexpected yes/no output: {raw!r}")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                if log_context:
                    log_write(f"{log_prefix} retry because: {exc}")
                time.sleep(1)
            elif log_context:
                log_write(f"{log_prefix} failure because: {exc}")
    return {"score": 0.0, "raw": "", "error": str(last_error) if last_error else "unknown error"}


def eval_edge(
    client: OpenAICompatibleClient,
    judge_model: str,
    question: str,
    source: str,
    target: str,
    relation_type: str,
    answer: str,
    log_context: str | None = None,
) -> dict[str, Any]:
    parsed = judge_yes_no(
        client=client,
        model=judge_model,
        messages=judge_edge_messages(question, source, target, relation_type, answer),
        retries=EDGE_RETRIES,
        log_context=log_context,
    )
    score = clamp(coerce_float(parsed.get("score", 0.0), 0.0), 0.0, 1.0)
    return {"score": score, "rationale": "", "raw": parsed}


def eval_rubric(
    client: OpenAICompatibleClient,
    judge_model: str,
    question: str,
    rubric: Any,
    answer: str,
) -> dict[str, Any]:
    if isinstance(rubric, dict) and isinstance(rubric.get("criteria"), list) and rubric.get("criteria"):
        criteria = [x for x in rubric.get("criteria", []) if isinstance(x, dict)]
        per_criterion: list[dict[str, Any]] = []
        criterion_scores: list[int] = []
        for idx, criterion in enumerate(criteria, start=1):
            criterion_name = str(criterion.get("name", f"criterion_{idx}"))
            parsed = judge_json(
                client=client,
                model=judge_model,
                messages=criterion_score_messages(question, criterion, answer),
                fallback={"score": 1, "reason": ""},
                retries=RUBRIC_RETRIES,
                log_context=f"[{question}][rm][criterion {idx}/{len(criteria)}: {criterion_name}]",
                max_tokens=RM_MAX_TOKENS,
            )
            score_int = int(round(clamp(coerce_float(parsed.get("score", 1.0), 1.0), 1.0, 10.0)))
            criterion_scores.append(score_int)
            per_criterion.append(
                {
                    "index": idx,
                    "name": criterion_name,
                    "score": score_int,
                    "reason": str(parsed.get("reason", "")),
                    "raw": parsed,
                }
            )
        raw_score_10 = mean(criterion_scores) if criterion_scores else 1.0
        score = clamp(raw_score_10 / 10.0, 0.0, 1.0)
        return {
            "score": score,
            "raw_score_10": raw_score_10,
            "criterion_scores": criterion_scores,
            "criterion_evaluations": per_criterion,
            "raw": {"mode": "query_dependent_criteria", "criteria_count": len(criteria)},
        }

    parsed = judge_json(
        client=client,
        model=judge_model,
        messages=rubric_messages(question, rubric, answer),
        fallback={"score": 0.0, "rationale": ""},
        retries=RUBRIC_RETRIES,
        max_tokens=RM_MAX_TOKENS,
    )
    score = clamp(coerce_float(parsed.get("score", 0.0), 0.0), 0.0, 1.0)
    return {"score": score, "rationale": parsed.get("rationale", ""), "raw": parsed}


def eval_outcome(
    client: OpenAICompatibleClient,
    judge_model: str,
    question: str,
    gold: str,
    answer: str,
) -> dict[str, Any]:
    parsed = judge_json(
        client=client,
        model=judge_model,
        messages=om_messages(question, gold, answer),
        fallback={"score": 0.0, "rationale": ""},
        retries=OM_RETRIES,
    )
    score = 1.0 if coerce_float(parsed.get("score", 0.0), 0.0) >= 0.5 else 0.0
    return {"score": score, "rationale": parsed.get("rationale", ""), "raw": parsed}


def load_answer_map(answer_path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(answer_path)
    items = payload.get("items", payload)
    mapping: dict[str, dict[str, Any]] = {}
    for item in items:
        mapping[item_key(item)] = item
    return mapping


def is_missing_answer_text(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def resolve_answer_text(answer_item: dict[str, Any]) -> str | None:
    model_answer = answer_item.get("model_answer")
    if not is_missing_answer_text(model_answer):
        return str(model_answer)

    legacy_answer = answer_item.get("answer")
    if not is_missing_answer_text(legacy_answer):
        return str(legacy_answer)

    return None


def resolve_answer_model(answer_path: Path) -> str:
    payload = read_json(answer_path)
    if isinstance(payload, dict):
        meta = payload.get("meta")
        if isinstance(meta, dict):
            model = meta.get("model")
            if model:
                return str(model)
        items = payload.get("items")
        if isinstance(items, list) and items:
            first = items[0]
            if isinstance(first, dict):
                model = first.get("model_name")
                if model:
                    return str(model)
    return answer_path.stem


def load_benchmark_items(benchmark_path: str) -> list[dict[str, Any]]:
    payload = read_json(resolve_input_path(benchmark_path))
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise TypeError(f"Unsupported benchmark format in {benchmark_path}")

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"Benchmark item must be an object, got {type(item).__name__}")
        normalized.append(item)
    return normalized


def find_resume_file(output_dir: Path, judge_model: str) -> Path | None:
    prefix = sanitize_filename(judge_model) + "_metrics_"
    candidates = sorted(output_dir.glob(f"{prefix}*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_existing_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return None
    return payload


def is_completed_item(item: dict[str, Any]) -> bool:
    return isinstance(item.get("metrics"), dict) and "total_score" in item


def build_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    pm_scores = [float(item.get("pm_score", 0.0)) for item in items]
    rm_scores = [float(item.get("rm_score", 0.0)) for item in items]
    om_scores = [float(item["om_score"]) for item in items if item.get("om_score") is not None]
    total_scores = [float(item.get("total_score", 0.0)) for item in items]
    return {
        "pm_mean": mean(pm_scores),
        "rm_mean": mean(rm_scores),
        "om_mean": mean(om_scores) if om_scores else None,
        "total_mean": mean(total_scores),
        "n_items": len(items),
        "n_open": sum(1 for item in items if item.get("answer_type") == "open"),
        "n_closed": sum(1 for item in items if item.get("answer_type") == "closed"),
    }


def build_ordered_items(benchmark: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [results[item_key(item)] for item in benchmark if item_key(item) in results]


def build_progress_state(
    total_items: int,
    completed_items: list[dict[str, Any]],
    failed_items: list[dict[str, Any]],
) -> dict[str, Any]:
    succeeded = len(completed_items)
    failed = len(failed_items)
    processed = succeeded + failed
    scores = [float(item.get("total_score", 0.0)) for item in completed_items]
    return {
        "total": total_items,
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "pending": max(total_items - processed, 0),
        "mean_score": mean(scores) if scores else None,
    }


def build_failure_item(item: dict[str, Any], error: Any) -> dict[str, Any]:
    return {
        **item,
        "evaluation_status": "failed",
        "evaluation_error": str(error),
    }


def is_failed_item(item: dict[str, Any]) -> bool:
    return item.get("evaluation_status") == "failed" and bool(item.get("evaluation_error"))


def build_checkpoint_payload(
    benchmark: list[dict[str, Any]],
    completed_items: dict[str, dict[str, Any]],
    failed_items: dict[str, dict[str, Any]],
    answer_model: str,
    judge_model: str,
    parser_model: str,
    timestamp: str,
    benchmark_path: str,
    answer_file: Path,
    concurrency: int,
) -> dict[str, Any]:
    ordered_items = build_ordered_items(benchmark, completed_items)
    ordered_failures = build_ordered_items(benchmark, failed_items)
    state = build_progress_state(len(benchmark), ordered_items, ordered_failures)
    return {
        "meta": {
            "task": "evaluator",
            "answer_model": answer_model,
            "judge_model": judge_model,
            "parser_model": parser_model,
            "timestamp": timestamp,
            "source_benchmark": str(resolve_input_path(benchmark_path)),
            "source_answers": str(answer_file),
            "count": len(ordered_items),
            "failure_count": len(ordered_failures),
            "concurrency": concurrency,
            "status": "complete" if state["processed"] == len(benchmark) else "partial",
        },
        "state": state,
        "summary": build_summary(ordered_items),
        "items": ordered_items,
        "failures": ordered_failures,
    }


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def update_progress(progress: Any, state: dict[str, Any]) -> None:
    set_postfix = getattr(progress, "set_postfix", None)
    if not set_postfix:
        return
    postfix: dict[str, Any] = {
        "ok": state["succeeded"],
        "fail": state["failed"],
    }
    if state["mean_score"] is not None:
        postfix["score"] = f"{state['mean_score']:.3f}"
    set_postfix(postfix)


def evaluate_item(
    client: OpenAICompatibleClient,
    item: dict[str, Any],
    answer_item: dict[str, Any] | None,
    judge_model: str,
    parser_model: str,
    index: int,
    total_items: int,
) -> dict[str, Any]:
    key = item_key(item)
    if not answer_item:
        raise KeyError(f"Missing answer for item {key}")
    model_answer = resolve_answer_text(answer_item)
    if model_answer is None:
        answer_error = answer_item.get("model_answer_error") or answer_item.get("answer_error") or "missing answer"
        raise ValueError(f"missing answer: {answer_error}")

    question = item["question"]
    answer_type = item.get("answer_type", "open")

    question_preview = question.replace("\n", " ")
    if len(question_preview) > 120:
        question_preview = question_preview[:117] + "..."
    log_write(f"[{index}/{total_items}] {key} | {answer_type} | {question_preview}")

    log_write(f"[{key}] parser")
    graph_record = parse_graph_record(client, parser_model, question, model_answer)
    graph = graph_record["graph"]
    rst_record = graph_record.get("rst_record", {})
    edge_scores = []
    edges = graph.get("edges", [])
    for edge_idx, edge in enumerate(edges, start=1):
        source_id = edge.get("source")
        target_id = edge.get("target")
        relation_type = str(edge.get("relation_type", edge.get("type", "")))
        nodes = {node["id"]: node["text"] for node in graph.get("nodes", []) if "id" in node and "text" in node}
        source_text = nodes.get(source_id, str(source_id))
        target_text = nodes.get(target_id, str(target_id))
        source_preview = source_text.replace("\n", " ")
        target_preview = target_text.replace("\n", " ")
        if len(source_preview) > 80:
            source_preview = source_preview[:77] + "..."
        if len(target_preview) > 80:
            target_preview = target_preview[:77] + "..."
        log_write(f"[{key}] edge {edge_idx}/{len(edges)}")
        edge_scores.append(
            eval_edge(
                client,
                judge_model,
                question,
                source_text,
                target_text,
                relation_type,
                model_answer,
                log_context=f"[{key}][pm][edge {edge_idx}/{len(edges)}]",
            )
        )

    log_write(f"[{key}] rubric")
    pm_score = mean(score["score"] for score in edge_scores) if edge_scores else 0.0
    rubric_score = eval_rubric(client, judge_model, question, item.get("rubrics", {}), model_answer)
    rm_score = rubric_score["score"]

    om_score: float | None = None
    if answer_type == "closed":
        log_write(f"[{key}] om")
        gold = safe_get_first_answer(item)
        if gold is None:
            raise ValueError(f"Closed item {key} has no gold answer")
        om_score = eval_outcome(client, judge_model, question, gold, model_answer)["score"]

    available = [pm_score, rm_score] + ([om_score] if om_score is not None else [])
    total_score = mean(available)
    return {
        **item,
        "model_answer": model_answer,
        "pm_score": pm_score,
        "rm_score": rm_score,
        "om_score": om_score,
        "total_score": total_score,
        "parsed_rst_raw": rst_record.get("raw_output"),
        "parsed_rst": rst_record,
        "parsed_rst_source": rst_record.get("source"),
        "parsed_rst_error": rst_record.get("error"),
        "parsed_dag": graph,
        "parsed_dag_source": graph_record["source"],
        "parsed_dag_error": graph_record["error"],
        "metrics": {
            "pm": {"score": pm_score, "edges": edge_scores, "graph": graph},
            "rm": rubric_score,
            "om": None if om_score is None else {"score": om_score},
        },
    }


def evaluate(
    client: OpenAICompatibleClient,
    benchmark_path: str,
    answer_path: str,
    judge_model: str,
    parser_model: str,
    output_dir: str,
    concurrency: int = 1,
) -> Path:
    concurrency = max(1, int(concurrency))
    benchmark = load_benchmark_items(benchmark_path)
    answer_file = resolve_input_path(answer_path)
    answer_map = load_answer_map(answer_file)
    answer_model = resolve_answer_model(answer_file)
    out_dir = ensure_dir(resolve_output_path(output_dir))
    out_file = out_dir / f"{sanitize_filename(answer_model)}_metrics_{sanitize_filename(judge_model)}.json"
    resume_file = out_file if out_file.exists() else find_resume_file(out_dir, judge_model)
    existing_payload = load_existing_metrics(resume_file) if resume_file else None
    timestamp = (
        existing_payload.get("meta", {}).get("timestamp")
        if existing_payload and isinstance(existing_payload.get("meta"), dict)
        else utc_timestamp()
    )

    completed_items: dict[str, dict[str, Any]] = {}
    failed_items: dict[str, dict[str, Any]] = {}
    if existing_payload:
        for existing_item in existing_payload.get("items", []):
            if isinstance(existing_item, dict) and is_completed_item(existing_item):
                completed_items[item_key(existing_item)] = existing_item
        for existing_item in existing_payload.get("failures", []):
            if isinstance(existing_item, dict) and is_failed_item(existing_item):
                key = item_key(existing_item)
                if key not in completed_items:
                    failed_items[key] = existing_item

    total_items = len(benchmark)
    pending_items: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(benchmark, start=1):
        key = item_key(item)
        if key in completed_items:
            log_write(f"[{index}/{total_items}] {key} | skip (resumed)")
            continue
        if key in failed_items:
            log_write(f"[{index}/{total_items}] {key} | skip (failed)")
            continue
        pending_items.append((index, item))
    if not pending_items:
        payload = build_checkpoint_payload(
            benchmark,
            completed_items,
            failed_items,
            answer_model,
            judge_model,
            parser_model,
            timestamp,
            benchmark_path,
            answer_file,
            concurrency,
        )
        if payload["meta"]["status"] == "complete":
            log_write(f"eval:{judge_model} | skip (already complete)")
        write_checkpoint(out_file, payload)
        return out_file

    def finish_item(index: int, item: dict[str, Any]) -> None:
        key = item_key(item)
        try:
            completed_items[key] = evaluate_item(
                client,
                item,
                answer_map.get(key),
                judge_model,
                parser_model,
                index,
                total_items,
            )
            failed_items.pop(key, None)
        except Exception as exc:
            failed_items[key] = build_failure_item(item, exc)
            log_write(f"[{index}/{total_items}] {key} | failed: {exc}")

    if concurrency == 1:
        progress = tqdm(pending_items, desc=f"eval:{judge_model}", unit="item")
        update_progress(
            progress,
            build_progress_state(
                total_items,
                build_ordered_items(benchmark, completed_items),
                build_ordered_items(benchmark, failed_items),
            ),
        )
        for index, item in progress:
            finish_item(index, item)
            payload = build_checkpoint_payload(
                benchmark,
                completed_items,
                failed_items,
                answer_model,
                judge_model,
                parser_model,
                timestamp,
                benchmark_path,
                answer_file,
                concurrency,
            )
            write_checkpoint(out_file, payload)
            update_progress(progress, payload["state"])
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    evaluate_item,
                    client,
                    item,
                    answer_map.get(item_key(item)),
                    judge_model,
                    parser_model,
                    index,
                    total_items,
                ): (index, item)
                for index, item in pending_items
            }
            progress = tqdm(as_completed(futures), total=len(futures), desc=f"eval:{judge_model}", unit="item")
            update_progress(
                progress,
                build_progress_state(
                    total_items,
                    build_ordered_items(benchmark, completed_items),
                    build_ordered_items(benchmark, failed_items),
                ),
            )
            for future in progress:
                index, item = futures[future]
                key = item_key(item)
                try:
                    completed_items[key] = future.result()
                    failed_items.pop(key, None)
                except Exception as exc:
                    failed_items[key] = build_failure_item(item, exc)
                    log_write(f"[{index}/{total_items}] {key} | failed: {exc}")
                payload = build_checkpoint_payload(
                    benchmark,
                    completed_items,
                    failed_items,
                    answer_model,
                    judge_model,
                    parser_model,
                    timestamp,
                    benchmark_path,
                    answer_file,
                    concurrency,
                )
                write_checkpoint(out_file, payload)
                update_progress(progress, payload["state"])

    payload = build_checkpoint_payload(
        benchmark,
        completed_items,
        failed_items,
        answer_model,
        judge_model,
        parser_model,
        timestamp,
        benchmark_path,
        answer_file,
        concurrency,
    )
    write_checkpoint(out_file, payload)
    return out_file
