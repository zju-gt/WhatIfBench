from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from evaluator import evaluate
from openai_client import from_env
from runner import run
from rubrics_generator import generate


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api-key", default=None)
    common.add_argument("--base-url", default=None)

    parser = argparse.ArgumentParser(prog="whatif-mvp")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Collect model answers for the benchmark", parents=[common])
    run_cmd.add_argument("--dataset", default="data/benchmark_candidates.json")
    run_cmd.add_argument("--model", required=True)
    run_cmd.add_argument("--output-dir", default="result")
    run_cmd.add_argument("--temperature", type=float, default=0.6)
    run_cmd.add_argument("--max-tokens", type=int, default=4096)
    run_cmd.add_argument("--limit", type=int, default=None)

    rub_cmd = sub.add_parser("rubrics", help="Generate rubrics for the benchmark", parents=[common])
    rub_cmd.add_argument("--benchmark", default="data/benchmark_candidates.json")
    rub_cmd.add_argument("--model", required=True)
    rub_cmd.add_argument("--output", default="data/benchmark_candidates_with_rubrics.json")
    rub_cmd.add_argument("--temperature", type=float, default=0.6)
    rub_cmd.add_argument("--max-tokens", type=int, default=4096)
    rub_cmd.add_argument("--limit", type=int, default=None)

    eval_cmd = sub.add_parser("evaluate", help="Evaluate answers with rubrics and gold answers", parents=[common])
    eval_cmd.add_argument("--benchmark", default="data/benchmark_candidates_with_rubrics.json")
    eval_cmd.add_argument("--answers", required=True)
    eval_cmd.add_argument("--judge-model", required=True)
    eval_cmd.add_argument("--parser-model", default=None)
    eval_cmd.add_argument("--output-dir", default="result")
    eval_cmd.add_argument("--concurrency", type=int, default=1)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    client = from_env(api_key=args.api_key, base_url=args.base_url)

    if args.command == "run":
        out = run(
            client=client,
            dataset_path=args.dataset,
            model=args.model,
            output_dir=args.output_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            limit=args.limit,
        )
        print(out)
        return 0

    if args.command == "rubrics":
        out = generate(
            client=client,
            benchmark_path=args.benchmark,
            model=args.model,
            output_path=args.output,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            limit=args.limit,
        )
        print(out)
        return 0

    if args.command == "evaluate":
        parser_model = args.parser_model or args.judge_model
        out = evaluate(
            client=client,
            benchmark_path=args.benchmark,
            answer_path=args.answers,
            judge_model=args.judge_model,
            parser_model=parser_model,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
        )
        print(out)
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
