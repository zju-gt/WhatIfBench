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
MODEL="${1:-}"
CONCURRENCY="${2:-${CONCURRENCY:-1}}"

if [[ -z "${API_KEY}" ]]; then
  if [[ "${API_SOURCE}" == "modelrouter" ]]; then
    echo "Set MODELROUTER_API_KEY, LLM_API_KEY, OPENAI_API_KEY, or API_KEY." >&2
  else
    echo "Set OPENROUTER_API_KEY, OPENAI_API_KEY, or API_KEY." >&2
  fi
  exit 1
fi

if [[ -z "${MODEL}" ]]; then
  echo "Usage: scripts/generate_rubrics.sh MODEL [CONCURRENCY]" >&2
  exit 1
fi

python3 "${ROOT_DIR}/code/src/main.py" \
  rubrics \
  --api-key "${API_KEY}" \
  --base-url "${BASE_URL}" \
  --api-source "${API_SOURCE}" \
  --benchmark "mvp/data/benchmark_candidates.json" \
  --model "${MODEL}" \
  --output "data/benchmark_candidates_with_rubrics.json" \
  --concurrency "${CONCURRENCY}"
