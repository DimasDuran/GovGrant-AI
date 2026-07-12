#!/usr/bin/env bash
# Quality gates (ordered). Usage:
#   ./scripts/run_gates.sh              # unit + router (if stack up)
#   ./scripts/run_gates.sh unit         # pytest only
#   ./scripts/run_gates.sh router       # golden router thresholds
#   ./scripts/run_gates.sh hard_llm     # agent+llm multi_hop/not_found
#   ./scripts/run_gates.sh checklist_corpus
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

MODE="${1:-all}"

run_unit() {
  echo "== gate: unit =="
  pytest -q tests/rag -k "not integration" --tb=short
}

run_named() {
  local gate="$1"
  echo "== gate: ${gate} =="
  python -m govgrant.rag.cli gate "$gate"
}

case "$MODE" in
  unit)
    run_unit
    ;;
  router|hard_llm|checklist_corpus)
    run_named "$MODE"
    ;;
  all)
    run_unit
    # Router requires local stack; fail clearly if unavailable
    if ! run_named router; then
      echo "router gate failed (is Qdrant+Ollama+ingest up?)" >&2
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [all|unit|router|hard_llm|checklist_corpus]" >&2
    exit 2
    ;;
esac

echo "OK: gate(s) passed ($MODE)"
