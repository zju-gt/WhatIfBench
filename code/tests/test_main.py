from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from main import build_parser


def test_parser_accepts_api_source_modelrouter():
    args = build_parser().parse_args(["run", "--model", "judge", "--api-source", "modelrouter"])

    assert args.api_source == "modelrouter"


def test_parser_accepts_run_concurrency():
    args = build_parser().parse_args(["run", "--model", "judge", "--concurrency", "16"])

    assert args.concurrency == 16


def test_parser_accepts_rubrics_concurrency():
    args = build_parser().parse_args(["rubrics", "--model", "judge", "--concurrency", "16"])

    assert args.concurrency == 16
