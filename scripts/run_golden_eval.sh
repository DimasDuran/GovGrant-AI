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
    # Prefer threshold gate when available
    python -m govgrant.rag.cli gate router \
      --out "${OUT_DIR}/gate_router_${STAMP}.json"
    cp "${OUT_DIR}/gate_router_${STAMP}.json" "${OUT_DIR}/gate_router_latest.json"
    ;;
  agent)
    python -m govgrant.rag.cli eval --golden --backend agent \
      --out "${OUT_DIR}/golden_agent_${STAMP}.json"
    cp "${OUT_DIR}/golden_agent_${STAMP}.json" "${OUT_DIR}/golden_agent_latest.json"
    ;;
  agent-llm|hard_llm)
    python -m govgrant.rag.cli gate hard_llm \
      --out "${OUT_DIR}/gate_hard_llm_${STAMP}.json"
    cp "${OUT_DIR}/gate_hard_llm_${STAMP}.json" "${OUT_DIR}/gate_hard_llm_latest.json"
    ;;
  multi_hop|fact|boolean|list|comparison|scenario|not_found)
    python -m govgrant.rag.cli eval --golden --category "$MODE" \
      --out "${OUT_DIR}/golden_${MODE}_${STAMP}.json"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Use: router | agent | agent-llm | hard_llm | multi_hop | fact | ..." >&2
    exit 2
    ;;
esac
