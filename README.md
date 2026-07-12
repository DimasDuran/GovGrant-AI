# GovGrant AI

Multimodal hybrid RAG + agent for **U.S. SBIR/STTR** grant compliance (DoD/DARPA, SBA, NIH SF424).

| Layer | Stack |
|-------|--------|
| Retrieval | LlamaIndex hybrid (Qdrant vectors + BM25 + RRF), local Ollama `nomic-embed-text` |
| Agent | LangGraph · Claude Haiku (grounded answers) |
| Sources | Agency PDFs, tables/figures, SBIR.gov topics |
| Eval | Runtime golden set (fact recall / precision) |
| Product | Gradio UI · compliance checklist · proposal PDF draft scoring |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set ANTHROPIC_API_KEY, LLAMAPARSE_API_KEY, Ollama/Qdrant URLs

# Ingest fixture PDFs (Qdrant + Ollama must be up)
python -m govgrant.rag.cli ingest

# Chat UI
python -m govgrant.ui.app
# → http://127.0.0.1:7860

# Golden eval (post-ingest)
python -m govgrant.rag.cli eval --golden

# Compliance checklist (corpus + optional draft PDF)
python -m govgrant.rag.cli checklist --package darpa --ot
python -m govgrant.rag.cli checklist --draft-pdf ./proposal.pdf --llm-draft --package darpa --ot
```

Local-only notes (not in git): `docs/r1-quickstart.md`, `About.md`, `Infra.md`, architecture plans — kept for development context.

## Layout

```
src/govgrant/
  rag/          # ingest, hybrid index, router, eval, CLI
  agent/        # LangGraph + Haiku
  compliance/   # multi-agency checklist + proposal PDF + draft LLM judge
  ui/           # Gradio console
data/eval/      # golden cases (01–10) + SCHEMA.json
tests/rag/      # unit tests
```

## Quality gates

Thresholds live in `data/eval/THRESHOLDS.json` (versioned). Reports stay under `data/eval/reports/` (gitignored).

```bash
./scripts/run_gates.sh unit              # pytest
./scripts/run_gates.sh router            # golden + thresholds (needs stack)
./scripts/run_gates.sh hard_llm          # agent+Haiku multi_hop/not_found
./scripts/run_gates.sh checklist_corpus  # DARPA critical corpus
python -m govgrant.rag.cli gate --list
```

| Gate | Command | Target |
|------|---------|--------|
| Unit | `pytest -q tests/rag` | pass |
| Router | `python -m govgrant.rag.cli gate router` | see THRESHOLDS.json |
| Hard LLM | `python -m govgrant.rag.cli gate hard_llm` | pre-release |

## Dev auth / multi-tenant

Default is open local mode (`AUTH_ENABLED=false`, tenant `local-dev`).

```bash
# Enable API-key → tenant binding (see data/auth/tenants.example.json)
export AUTH_ENABLED=true
# optional: cp data/auth/tenants.example.json data/auth/tenants.local.json

python -m govgrant.rag.cli agent "What is SBIR work-share?" --api-key dev-local-key
python -m govgrant.rag.cli query "cost volume" --api-key demo-beta-key --doc-id darpa-sbir-sttr-phase-II-instructions
```

- **Public agency docs** are listed in `public_doc_ids` and readable by all tenants.
- **User proposals** are registered under the caller's `tenant_id` (UI tab **My proposals**).
- `allowed_doc_ids: []` on a tenant restricts non-public docs (cross-tenant isolation).

```bash
# Programmatic upload (no Gradio)
python - <<'PY'
from govgrant.auth import resolve_request_auth
from govgrant.proposals import ProposalService
auth = resolve_request_auth(api_key="dev-local-key")  # or AUTH_ENABLED=false
svc = ProposalService()
print(svc.upload(auth, "path/to/proposal.pdf", index=True).to_dict())
print([r.doc_id for r in svc.list_proposals(auth)])
svc.delete(auth, "user-proposal-…")  # also purges Qdrant + BM25 + tables
PY
```

Deleting a proposal removes registry + file **and** index vectors (Qdrant filter on `tenant_id`+`gg_doc_id`, BM25 leaves, tabular rows).


## Branching

- `main` — stable baseline  
- `develop` — active feature integration  

## License

See [LICENSE](LICENSE).
