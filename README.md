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

| Gate | Command | Target |
|------|---------|--------|
| Unit tests | `pytest -q tests/rag` | pass |
| Router golden | `python -m govgrant.rag.cli eval --golden` | high pass rate post-ingest |
| Checklist | `python -m govgrant.rag.cli checklist --package darpa --ot` | critical controls pass |

## Branching

- `main` — stable baseline  
- `develop` — active feature integration  

## License

See [LICENSE](LICENSE).
