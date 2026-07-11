#!/usr/bin/env bash
# Quick health check: Qdrant + Ollama nomic-embed-text
set -euo pipefail

QDRANT_URL="${
:-http://localhost:6333}"
echo "== Qdrant =="
curl -sf "${QDRANT_URL}/collections" | head -c 500
echo
echo

echo "== Ollama models =="
curl -sf http://localhost:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print([m['name'] for m in d.get('models',[])])"
echo

echo "== nomic embed smoke =="
curl -sf http://localhost:11434/api/embeddings -d '{"model":"nomic-embed-text","prompt":"SBIR Phase I eligibility"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); e=d.get('embedding',[]); print('dim', len(e), 'first3', e[:3])"
echo
echo "OK"
