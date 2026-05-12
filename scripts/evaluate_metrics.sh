#!/usr/bin/env bash
OPENROUTER_API_KEY="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"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_SOURCE="${API_SOURCE:-openrouter}"
if [[ "${API_SOURCE}" == "modelrouter" ]]; then
  API_KEY="${API_KEY:-${MODELROUTER_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}}"
  BASE_URL="${BASE_URL:-${MODELROUTER_BASE_URL:-${LLM_BASE_URL:-https://routify.alibaba-inc.com/protocol/openai/v1}}}"
else
  API_KEY="${API_KEY:-${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-}}}"
  BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
fi
ANSWERS_FILE="${1:-}"
JUDGE_MODEL="${2:-}"
PARSER_MODEL="${3:-${JUDGE_MODEL}}"
CONCURRENCY="${4:-${CONCURRENCY:-1}}"

if [[ -z "${API_KEY}" ]]; then
  if [[ "${API_SOURCE}" == "modelrouter" ]]; then
    echo "Set MODELROUTER_API_KEY, LLM_API_KEY, OPENAI_API_KEY, or API_KEY." >&2
  else
    echo "Set OPENROUTER_API_KEY, OPENAI_API_KEY, or API_KEY." >&2
  fi
  exit 1
fi

if [[ -z "${ANSWERS_FILE}" || -z "${JUDGE_MODEL}" ]]; then
  echo "Usage: scripts/evaluate_metrics.sh ANSWERS_FILE JUDGE_MODEL [PARSER_MODEL] [CONCURRENCY]" >&2
  exit 1
fi

python3 "${ROOT_DIR}/code/src/main.py" \
  evaluate \
  --api-key "${API_KEY}" \
  --base-url "${BASE_URL}" \
  --api-source "${API_SOURCE}" \
  --benchmark "data/benchmark_candidates_with_rubrics.json" \
  --answers "${ANSWERS_FILE}" \
  --judge-model "${JUDGE_MODEL}" \
  --parser-model "${PARSER_MODEL}" \
  --output-dir "result" \
  --concurrency "${CONCURRENCY}"
