#!/usr/bin/env bash
set -euo pipefail
MODELROUTER_API_KEY="sk-9eba1adb38fa4cb1af5dca05f58f8472"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_SOURCE="${API_SOURCE:-modelrouter}"
if [[ "${API_SOURCE}" == "modelrouter" ]]; then
  API_KEY="${API_KEY:-${MODELROUTER_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}}"
  BASE_URL="${BASE_URL:-${MODELROUTER_BASE_URL:-${LLM_BASE_URL:-https://routify.alibaba-inc.com/protocol/openai/v1}}}"
else
  API_KEY="${API_KEY:-${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-}}}"
  BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
fi
MODEL="${1:-"gpt-5.2-chat-latest"}"
CONCURRENCY="${2:-${CONCURRENCY:-16}}"

if [[ -z "${API_KEY}" ]]; then
  if [[ "${API_SOURCE}" == "modelrouter" ]]; then
    echo "Set MODELROUTER_API_KEY, LLM_API_KEY, OPENAI_API_KEY, or API_KEY." >&2
  else
    echo "Set OPENROUTER_API_KEY, OPENAI_API_KEY, or API_KEY." >&2
  fi
  exit 1
fi

if [[ -z "${MODEL}" ]]; then
  echo "Usage: scripts/run_model.sh MODEL [CONCURRENCY]" >&2
  exit 1
fi

python3 "${ROOT_DIR}/code/src/main.py" \
  run \
  --api-key "${API_KEY}" \
  --base-url "${BASE_URL}" \
  --api-source "${API_SOURCE}" \
  --dataset "/data/oss_bucket_0/wyc/EMNLP/benchmark_candidates_v1_0422_split.json" \
  --model "${MODEL}" \
  --output-dir "result" \
  --concurrency "${CONCURRENCY}"
