#!/usr/bin/env bash
# Runtime golden evaluation (post-ingest).
# Usage:
#   ./scripts/run_golden_eval.sh              # router, full 160
#   ./scripts/run_golden_eval.sh agent        # agent without LLM
#   ./scripts/run_golden_eval.sh agent-llm    # agent + Haiku (costs API)
#   ./scripts/run_golden_eval.sh multi_hop    # category filter
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

MODE="${1:-router}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="data/eval/reports"
mkdir -p "$OUT_DIR"

case "$MODE" in
  router)
    python -m govgrant.rag.cli eval --golden \
      --out "${OUT_DIR}/golden_router_${STAMP}.json"
    cp "${OUT_DIR}/golden_router_${STAMP}.json" "${OUT_DIR}/golden_router_latest.json"
    ;;
  agent)
    python -m govgrant.rag.cli eval --golden --backend agent \
      --out "${OUT_DIR}/golden_agent_${STAMP}.json"
    cp "${OUT_DIR}/golden_agent_${STAMP}.json" "${OUT_DIR}/golden_agent_latest.json"
    ;;
  agent-llm)
    python -m govgrant.rag.cli eval --golden --backend agent --llm \
      --out "${OUT_DIR}/golden_agent_llm_${STAMP}.json"
    cp "${OUT_DIR}/golden_agent_llm_${STAMP}.json" "${OUT_DIR}/golden_agent_llm_latest.json"
    ;;
  multi_hop|fact|boolean|list|comparison|scenario|not_found)
    python -m govgrant.rag.cli eval --golden --category "$MODE" \
      --out "${OUT_DIR}/golden_${MODE}_${STAMP}.json"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Use: router | agent | agent-llm | multi_hop | fact | boolean | list | comparison | scenario | not_found" >&2
    exit 2
    ;;
esac
