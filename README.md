# GovGrant AI

![U.S. SBIR/STTR](https://res.cloudinary.com/diyzwz1mx/image/upload/v1783962148/U.S._SBIR_STTR_seexwv.png)

**A compliance assistant for U.S. federal SBIR/STTR grant proposals.**

GovGrant AI answers questions about agency regulations (DoD/DARPA, SBA, NIH SF424) using multimodal RAG over text, tables, and figures, orchestrated by a LangGraph agent with Claude Haiku. It also runs compliance checklists against the source corpus and scores proposal drafts against each agency's requirements.

[Architecture](#architecture) · [Quick start](#quick-start) · [Project layout](#project-layout) · [Quality gates](#quality-gates) · [Auth & multi-tenancy](#dev-auth--multi-tenant)

---

## Architecture

```mermaid
graph TB
    %% Entry points
    UI["Gradio UI"]
    CLI["CLI"]

    %% Auth
    AUTH["Auth<br/>tenant · roles · permissions"]

    %% Agent (LangGraph) — full pipeline
    subgraph AGENT["Agent (LangGraph + Claude Haiku)"]
        C["classify<br/><i>LLM tool‑use routing</i><br/>or heuristic fallback"]
        R["retrieve<br/><i>Qdrant dense + sparse</i>"]
        V["validate<br/><i>LLM judge</i><br/>mark_sufficient<br/>request_more_evidence"]
        F["format_answer<br/><i>LLM synthesis</i>"]
        S["self_check<br/><i>LLM verifies answer</i><br/>answer_complete<br/>answer_incomplete"]
        CK["checklist<br/><i>LLM‑planned compliance audit</i>"]

        C -->|doc_qa / table / topic| R
        C -->|checklist| CK
        C -->|chat| F
        R --> V
        V -->|sufficient| F
        V -->|retry + reformulated query| R
        F --> S
        S -->|incomplete + critique| F
        S -->|complete| E
        CK --> F
    end

    E(("END"))

    %% Router + RAG
    ROUTER["QueryRouter<br/>heuristic intent → source<br/><i>(also used by CLI)</i>"]
    HR["HybridRAGService<br/>Qdrant dense + sparse vectors<br/>+ RRF fusion"]
    SBIR["SBIRTopicService<br/>Qdrant dense + sparse + API"]
    TABULAR["TabularStore<br/>SQLite + FTS5"]

    %% External / infra
    QD[(Qdrant<br/>dense + sparse vectors)]
    OLL(("Ollama<br/>nomic-embed-text"))
    LLAMA("LlamaParse")
    ANTH("Anthropic<br/>Claude Haiku")
    SBIRAPI("SBIR.gov API")

    %% Flows
    UI --> AUTH
    CLI --> AUTH
    AUTH --> AGENT
    ROUTER -.->|CLI query path| HR
    ROUTER -.->|CLI query path| SBIR
    ROUTER -.->|CLI query path| TABULAR
    HR -->|dense + sparse| QD
    HR --> OLL
    SBIR -->|dense + sparse| QD
    SBIR --> SBIRAPI
    LLAMA -.-> HR
    ANTH -.-> AGENT
```

### Why hybrid RAG (dense + sparse + multimodal)

SBIR/STTR compliance documents mix two search modes that no single retriever handles well:

| Mode | Example query | Solved by |
|---|---|---|
| Semantic | "What does the work-share policy say?" | Dense vectors — meaning, synonyms, paraphrase |
| Lexical | `2 CFR 200`, `SF-424`, `40%`, `5500.7` | Sparse vectors (native Qdrant) — guaranteed recall on codes and clauses |
| Multimodal | Tables, figures, flowcharts | Modality-specific parser and indexing path |

### Why an agent (LangGraph)

Plain RAG can't orchestrate multi-step decisions. The agent structures every response with LLM-driven tool-calling at each stage:

1. **classify** — Claude Haiku selects a retrieval tool via Anthropic function-calling (`search_documents`, `search_tables`, `search_sbir_topics`, `cross_check`, `compliance_checklist`). Heuristic fallback when the LLM is unavailable.
2. **retrieve** — hybrid RAG (Qdrant dense + sparse vectors) + neighbor page expansion + force-include pages with exact phrase matches.
3. **validate** — LLM judge using `mark_sufficient` / `request_more_evidence(reason, suggested_query)` tools. If evidence is insufficient, the LLM reformulates the query and retries (up to 2 times) instead of using hardcoded string-matching heuristics.
4. **format_answer** — Claude Haiku synthesizes an answer grounded in the retrieved evidence.
5. **self_check** — LLM verifies the answer covers every sub-question using `answer_complete` / `answer_incomplete(critique)`. If incomplete, loops back to `format_answer` with specific revision guidance before returning to the user.
6. **checklist** — When the user asks for a compliance audit, the LLM selects which agency packages (DARPA, SBA, SF424) to check and dynamically runs the compliance checklist, then interprets the results.

Every decision point uses real tool-calling (not regex or keyword heuristics) — routing source selection, evidence sufficiency, query reformulation, answer quality, and checklist planning.

### Why LlamaIndex + LangGraph

- **LlamaIndex** handles data orchestration out of the box: hierarchical chunking (`HierarchicalNodeParser`), `QdrantVectorStore` with native hybrid mode, and ingestion into Qdrant in a few lines — replacing what would otherwise be ~400 lines of boilerplate.
- **LangGraph** models the flow as an explicit `StateGraph` with typed shared state rather than a linear chain. Each node (`classify`, `retrieve`, `validate`, `format_answer`, `self_check`, `checklist`) is independently testable and debuggable. `StateGraph.add_conditional_edges` enables runtime retry loops and branching without bespoke orchestration code. The alternative (LangChain Expression Language) is more verbose for the same graph.

## Data sources

The `QueryRouter` doesn't send every query to the vector store — it picks between three distinct sources depending on intent:

| Source | Backing store | What it covers |
|---|---|---|
| `HybridRAGService` | Qdrant (dense + sparse vectors) | Agency policy/regulation text, tables, and figures — the indexed document corpus |
| `TabularStore` | SQLite + FTS5 | Structured tabular data with exact lexical search, outside the vector index |
| `SBIRTopicService` | Qdrant (dense + sparse) **+ live SBIR.gov API** | SBIR/STTR topics and open funding opportunities, combining its own index with real-time external calls |

In short: two of the three paths sit outside the main vector DB, and one of those (`SBIRTopicService`) also reaches out to a live external API rather than relying solely on indexed data.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set ANTHROPIC_API_KEY, LLAMAPARSE_API_KEY, Ollama/Qdrant URLs

# Ingest fixture PDFs (Qdrant + Ollama must be running)
python -m govgrant.rag.cli ingest

# Chat UI (session API key shared across tabs)
python -m govgrant.ui.app
# → http://127.0.0.1:7860
# Set the session key once — Chat / My proposals / Checklist all share the same tenant

# Golden eval (run after ingest)
python -m govgrant.rag.cli eval --golden

# Agent (LLM‑routed: classify → retrieve → validate → format → self_check)
python -m govgrant.rag.cli agent "What is the DARPA Phase II work-share requirement?"
python -m govgrant.rag.cli agent "Run a compliance checklist for my DARPA proposal"
python -m govgrant.rag.cli agent "Cross‑check my abstract with open SBIR topics"

# Compliance checklist (direct, or via agent)
python -m govgrant.rag.cli checklist --package darpa --ot
python -m govgrant.rag.cli checklist --draft-pdf ./proposal.pdf --llm-draft --package darpa --ot
python -m govgrant.rag.cli checklist --package darpa --ot --export   # writes md+json to data/eval/reports/ (gitignored)
```

## Project layout

```
src/govgrant/
  rag/          # ingest, hybrid index, router, eval, CLI
  agent/        # LangGraph pipeline: classify → retrieve → validate → format → self_check (Claude Haiku)
  compliance/   # multi-agency checklist (DARPA, SBA, SF424) + proposal PDF + draft LLM judge
  ui/           # Gradio console
data/eval/      # golden cases (01–10) + THRESHOLDS.json
tests/          # unit tests (rag, agent, compliance)
```

## Quality gates

Thresholds are versioned in `data/eval/THRESHOLDS.json`. Reports are written to `data/eval/reports/` (gitignored).

| Gate | Command | Target |
|---|---|---|
| Unit | `pytest -q tests/rag` | pass |
| Router | `python -m govgrant.rag.cli gate router` | see `THRESHOLDS.json` |
| Hard LLM | `python -m govgrant.rag.cli gate hard_llm` | pre-release |
| Checklist corpus | `./scripts/run_gates.sh checklist_corpus` | DARPA critical corpus |

```bash
./scripts/run_gates.sh unit              # pytest
./scripts/run_gates.sh router            # golden + thresholds (needs stack)
./scripts/run_gates.sh hard_llm          # agent + Haiku, multi_hop/not_found
./scripts/run_gates.sh checklist_corpus  # DARPA critical corpus
python -m govgrant.rag.cli gate --list
```

## Dev auth / multi-tenant

Default mode is open and local (`AUTH_ENABLED=false`, tenant `local-dev`).

```bash
# Enable API-key → tenant binding (see data/auth/tenants.example.json)
export AUTH_ENABLED=true
# optional: cp data/auth/tenants.example.json data/auth/tenants.local.json

python -m govgrant.rag.cli agent "What is SBIR work-share?" --api-key dev-local-key
python -m govgrant.rag.cli query "cost volume" --api-key demo-beta-key --doc-id darpa-sbir-sttr-phase-II-instructions
```

- **Public agency docs** are listed in `public_doc_ids` and readable by all tenants.
- **User proposals** are registered under the caller's `tenant_id` (UI tab **My proposals**).
- `allowed_doc_ids: []` on a tenant restricts access to non-public docs (cross-tenant isolation).

```bash
# Programmatic upload (no Gradio)
python - <<'PY'
from govgrant.auth import resolve_request_auth
from govgrant.proposals import ProposalService

auth = resolve_request_auth(api_key="dev-local-key")  # or AUTH_ENABLED=false
svc = ProposalService()
print(svc.upload(auth, "path/to/proposal.pdf", index=True).to_dict())
print([r.doc_id for r in svc.list_proposals(auth)])
svc.delete(auth, "user-proposal-…")  # also purges Qdrant + page index + tables
PY

# CLI equivalents
python -m govgrant.rag.cli proposals whoami
python -m govgrant.rag.cli proposals list
python -m govgrant.rag.cli proposals upload ./proposal.pdf
python -m govgrant.rag.cli proposals get user-proposal-my-file
python -m govgrant.rag.cli proposals delete user-proposal-my-file
python -m govgrant.rag.cli proposals audit --limit 20
```

- **Capabilities** (session banner / `whoami`): `upload_proposals`, `delete_proposals` (admin, when `AUTH_ENABLED`), `run_checklist`.
- **Audit log**: upload / delete / delete_denied events per tenant (`proposals audit`).

Deleting a proposal removes the registry entry, the file, and all index data — Qdrant vectors filtered by `tenant_id` + `gg_doc_id`, plus the page index and tabular rows.

## Branching

- `main` — stable baseline
- `develop` — active feature integration

## License

See [LICENSE](LICENSE).