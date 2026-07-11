# R1 Quickstart — Hybrid RAG local

## Stack verificado

| Componente | Endpoint / modelo |
|------------|-------------------|
| Qdrant | `http://localhost:6333` |
| Embeddings | Ollama `nomic-embed-text` (768-d), local |
| Parse PDFs | LlamaParse (`LLAMAPARSE_API_KEY`) + fallback `pypdf` |
| Retrieve | Qdrant vectors + BM25 + RRF |

**Nota nomic:** se sanitizan TOC con “dot leaders” (`.....`) porque rompen algunos builds de Ollama+nomic con HTTP 500.

### Estado R1 (fixtures actuales)

| PDF | Leaf nodes (aprox.) |
|-----|---------------------|
| SBA SBIR/STTR Policy Directive | 172 |
| SF424 Application Guide | 563 |
| DARPA Phase II instructions | 31 |
| **Total Qdrant `user_docs`** | **766** |

## 1. PDFs

Copia tus 3 PDFs:

```bash
cp /ruta/a/*.pdf data/fixtures/pdfs/
ls data/fixtures/pdfs/
```

## 2. Entorno

```bash
cd /Users/usuario/Documents/GovGrant-AI
source .venv/bin/activate
bash scripts/check_stack.sh
```

## 3. Ingestar

```bash
python -m govgrant.rag.cli ingest
# o un archivo:
# python -m govgrant.rag.cli ingest data/fixtures/pdfs/mi-doc.pdf
# sin LlamaParse (solo texto pypdf):
# python -m govgrant.rag.cli ingest --no-llamaparse
```

## 4. Consultar (hybrid)

```bash
python -m govgrant.rag.cli query "What eligibility requirements are mentioned?"
python -m govgrant.rag.cli query "budget table indirect costs" --top-k 5
python -m govgrant.rag.cli query "SF-424" --tenant local-dev
# Filtro estable por documento (gg_doc_id)
python -m govgrant.rag.cli query "cost volume maximum" --doc-id darpa-sbir-sttr-phase-II-instructions
# Solo nodos de tablas (R2)
python -m govgrant.rag.cli query "indirect cost" --modality table
```

Cada hit incluye `score`, `gg_doc_id`, `modality`, `page` y `citation_uri`.

## 5. Tablas dual (R2)

```bash
python -m govgrant.rag.cli tables stats --tenant local-dev
python -m govgrant.rag.cli tables list --tenant local-dev
python -m govgrant.rag.cli tables search "Phase II" --limit 10
python -m govgrant.rag.cli tables get "SF424 SBIR_STTR Application Guide::p17::t2"
```

- **RAG path:** tablas como nodes `modality=table` en Qdrant/BM25  
- **Structured path:** SQLite en `data/indexes/tabular/tables.sqlite`

## 6. SBIR Topics (R3)

La API pública de SBIR a menudo devuelve **403 / maintenance**. El connector:

1. Intenta API (`SBIR_API_KEY` si aplica)
2. Si falla y `SBIR_USE_FIXTURES_ON_FAIL=true` → carga `data/fixtures/sbir/open_solicitations.sample.json`
3. Marca `stale=true` y adjunta **disclaimer** obligatorio

```bash
# Sync (API o fixtures)
python -m govgrant.rag.cli sbir sync
python -m govgrant.rag.cli sbir sync --fixtures          # forzar fixtures
python -m govgrant.rag.cli sbir sync --agency DOD

# Search hybrid + disclaimer
python -m govgrant.rag.cli sbir search "thermal batteries missile defense"
python -m govgrant.rag.cli sbir search "quantum imaging" --agency DOD --top-k 3

# Structured
python -m govgrant.rag.cli sbir list --agency HHS
python -m govgrant.rag.cli sbir get 12799
```

Colección Qdrant: `sbir_topics` (separada de `user_docs`).

## 7. Figuras / charts (R4)

Durante `ingest`, además de prose y tables:

1. Captions / `![alt](...)` del markdown LlamaParse  
2. Imágenes embebidas del PDF (PyMuPDF) → `data/indexes/figures/<doc_id>/`  
3. (Opcional) caption con visión local: `OLLAMA_VISION_MODEL=llava` (u otro)

```bash
# Ingest con figuras (default)
python -m govgrant.rag.cli ingest
python -m govgrant.rag.cli ingest --no-figures   # skip R4
python -m govgrant.rag.cli ingest --no-vision    # sin Ollama vision

# Query solo figuras/charts
python -m govgrant.rag.cli query "funding timeline chart" --modality figure
python -m govgrant.rag.cli query "performance graph" --modality chart
```

Sin modelo de visión, los nodes de figura usan caption/contexto textual (confidence más baja).

## 8. Multi-source ask (R5)

```bash
# Clasifica intent y enruta a docs / tables / figures / SBIR / cross-check
python -m govgrant.rag.cli ask "What foreign ownership disclosures are required?"
python -m govgrant.rag.cli ask "open topics thermal batteries missile defense"
python -m govgrant.rag.cli ask "indirect cost table budget" --intent table
python -m govgrant.rag.cli ask "Does my proposal align with open MDA topics?" --intent cross_check
python -m govgrant.rag.cli ask "diagram on page 1" --intent figure
python -m govgrant.rag.cli ask "..." --json   # intent + meta + text
```

## 9. Regression + golden eval (R6)

Runtime evaluation is **post-ingest** (needs Qdrant + BM25 indexes loaded).

### Smoke regression (multi-source router)

```bash
python -m govgrant.rag.cli eval
# Dataset: data/eval/regression_min.json
# (SBA foreign ownership, SF424, tables, SBIR topics, figures, cross-check, DARPA cost)
```

### Golden dataset (160 DARPA Phase II cases)

Unified schema under `data/eval/01_*.json` … `08_*.json` — see `data/eval/SCHEMA.json`.

```bash
# Full golden — retrieval/router scoring (fast, no Haiku cost)
python -m govgrant.rag.cli eval --golden \
  --out data/eval/reports/golden_router_latest.json

# By category
python -m govgrant.rag.cli eval --golden --category multi_hop
python -m govgrant.rag.cli eval --golden --category fact --category scenario

# Single cases
python -m govgrant.rag.cli eval --golden --id MH001 --id S018

# Agent path (LangGraph); add --llm to score Haiku answers + refusals
python -m govgrant.rag.cli eval --golden --backend agent --limit 10
python -m govgrant.rag.cli eval --golden --backend agent --llm \
  --category not_found --limit 5 \
  --out data/eval/reports/golden_agent_llm_sample.json
```

| Flag | Meaning |
|------|---------|
| `--golden` | Load all 01–08 golden files |
| `--backend router\|agent` | QueryRouter only vs full LangGraph |
| `--llm` | With agent: format answers with Haiku |
| `--category` / `--id` | Filters (repeatable) |
| `--limit N` | First N after filters |
| `--out PATH` | Save full per-case JSON report |
| `--full` | Print full report to stdout |

### Fact-based scoring (not full-string match)

| Field | Metric | Effect |
|-------|--------|--------|
| `facts_required` | **recall** | ≥60% must hit (override with `required_recall_threshold`) |
| `facts_optional` | optional_recall | Bonus only — never fails |
| `facts_forbidden` / `must_not_include` | **precision** | Fail if present in **answer** (agent+LLM); skipped on raw retrieval |
| `expected_answer` | — | Human reference only |

Reports include `avg_recall`, `avg_precision`, `avg_optional_recall`.  
Example multi-part Spanish case: `--id MH015`.

### Release quality gate (recommended)

```bash
# Fast: always (CI / pre-push)
python -m govgrant.rag.cli eval --golden \
  --out data/eval/reports/golden_router_latest.json
# Gate: pass_rate == 100, avg_recall >= 0.95

# Hard path: multi-hop + not_found with Haiku (pre-release)
python -m govgrant.rag.cli eval --golden --backend agent --llm \
  --category multi_hop --category not_found \
  --out data/eval/reports/golden_agent_llm_hard.json
# Gate: pass_rate >= 90, avg_recall >= 0.95, avg_precision >= 0.90
# Canonical UI multi-part: --id MH015
```

Haiku answers are scoped for **precision** (no unsolicited Volume 5/CCR digressions).

### Compliance checklist (DARPA · SBA · SF424 + draft)

Maps control points to retrieved instruction facts (corpus mode) and optionally scores a
**pasted proposal draft** for keyword signals (draft mode).

```bash
# All packages
python -m govgrant.rag.cli checklist --program sbir --ot

# One agency
python -m govgrant.rag.cli checklist --package darpa --ot
python -m govgrant.rag.cli checklist --package sba
python -m govgrant.rag.cli checklist --package sf424

# Draft scoring (text or PDF)
python -m govgrant.rag.cli checklist --package darpa --ot --draft-file ./my_sow.md
python -m govgrant.rag.cli checklist --package darpa --ot --draft-pdf ./my_proposal.pdf
# Extract + index proposal for chat Q&A
python -m govgrant.rag.cli checklist --draft-pdf ./my_proposal.pdf --index-proposal --package darpa --ot
python -m govgrant.rag.cli checklist --program sttr --ot --json
```

Also available in the Gradio UI tab **Compliance checklist** (packages + PDF upload + optional index).

Golden extras: `data/eval/09_sba_policy.json`, `data/eval/10_sf424_guide.json` (included in `--golden`).

Re-rank: lexical overlap boost on hybrid hits (no external re-ranker API required).

## 10. LangGraph agent (R7) + Claude Haiku

Requiere en `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
CHAT_MODEL=claude-haiku-4-5-20251001
CHAT_ENABLED=true
```

```bash
# Chat grounded en evidencia RAG (Haiku)
python -m govgrant.rag.cli agent "What foreign ownership disclosures are required?"
python -m govgrant.rag.cli agent "What open topics cover thermal batteries for missile defense?"
python -m govgrant.rag.cli agent "Does my proposal align with open MDA thermal topics?" --json

# Solo retrieve (sin LLM)
python -m govgrant.rag.cli agent "..." --no-llm
```

Grafo: `classify (heuristic) → retrieve → validate_evidence → format_answer (Haiku)`.

## 11. Gradio UI (local test)

```bash
source .venv/bin/activate
# Qdrant + Ollama nomic must be running
python -m govgrant.ui.app
# open http://127.0.0.1:7860
```

Tabs:
- **Chat agent** — LangGraph + Haiku (examples included)
- **Retrieve only** — raw hybrid hits
- **SBIR topics** — topic search + disclaimer
- **About** — stack status

Options: filter by document, agency, force intent, toggle Haiku, debug panel.

## Layout del código (R0/R1)

```text
src/govgrant/rag/
  contracts.py          # metadata canónica
  config.py             # env
  parsers/prose.py      # LlamaParse + pypdf fallback
  index/embeddings.py   # nomic via Ollama
  index/qdrant_store.py
  index/hybrid.py       # ingest + hybrid retrieve
  cli.py
```

## Siguiente después de R1

1. Validar 10 preguntas sobre tus 3 PDFs.
2. **R2** — lane tablas dual (SQL + RAG).
3. **R3** — SBIR API connector.
4. **R4** — figures/charts lane dedicado.
