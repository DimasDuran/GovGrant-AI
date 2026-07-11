"""
Gradio UI for testing GovGrant AI (agent + RAG + SBIR).

Run:
  source .venv/bin/activate
  python -m govgrant.ui.app
  # open http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import time
from functools import lru_cache

import gradio as gr

from govgrant.agent.graph import run_agent
from govgrant.agent.llm import ChatLLM
from govgrant.compliance.checklist import run_checklist
from govgrant.rag.config import get_settings
from govgrant.rag.index.hybrid import HybridRAGService
from govgrant.rag.router.query_router import QueryRouter, RouteIntent
from govgrant.rag.sbir.service import SBIRTopicService

EXAMPLES = [
    "What foreign ownership disclosures are required for SBIR applicants?",
    "What is the maximum DARPA Phase II cost volume amount?",
    "What open topics cover thermal batteries for missile defense?",
    "What does the SF-424 say about indirect costs in the budget table?",
    "Does my proposal align with open MDA thermal battery topics?",
    "What are the Research Strategy page limits in the SF424 application guide?",
    (
        "Estoy preparando una propuesta DARPA Fase II. Si uso universidad y FFRDC, "
        "¿qué work-share aplica en SBIR vs STTR? Si envío una propuesta similar a otra "
        "agencia, ¿qué debo revelar y cuándo? Si pido OT, ¿qué debe incluir el plan de "
        "hitos y la estrategia de comercialización?"
    ),
]

DOC_CHOICES = [
    "(auto)",
    "SBA SBIR_STTR_POLICY_DIRECTIVE_May2023",
    "SF424 SBIR_STTR Application Guide",
    "darpa-sbir-sttr-phase-II-instructions",
]

INTENT_CHOICES = [
    "(auto)",
    "doc_qa",
    "table",
    "figure",
    "topic_search",
    "cross_check",
    "mixed",
]

AGENCY_CHOICES = ["(any)", "DOD", "HHS", "NASA", "NSF", "DOE", "USDA", "EPA"]


@lru_cache(maxsize=1)
def _services() -> tuple[HybridRAGService, SBIRTopicService, QueryRouter]:
    settings = get_settings()
    docs = HybridRAGService(settings)
    sbir = SBIRTopicService(settings)
    router = QueryRouter(settings, docs=docs, sbir=sbir)
    return docs, sbir, router


def _status_md() -> str:
    s = get_settings()
    llm = ChatLLM()
    docs, sbir, _ = _services()
    try:
        n_bm25 = len(docs._leaf_nodes)
    except Exception:  # noqa: BLE001
        n_bm25 = "?"
    try:
        n_sbir = sbir.store.count()
    except Exception:  # noqa: BLE001
        n_sbir = "?"
    return (
        f"**Chat model:** `{s.chat_model}` · "
        f"**LLM ready:** {'yes' if llm.available else 'no'} · "
        f"**BM25 nodes:** {n_bm25} · "
        f"**SBIR topics:** {n_sbir} · "
        f"**Qdrant:** `{s.qdrant_url}`"
    )


def _norm_choice(value: str | None, empty_labels: set[str]) -> str | None:
    if not value or value in empty_labels:
        return None
    return value


def chat_ask(
    message: str,
    history: list[dict[str, str]] | None,
    doc_id: str,
    agency: str,
    intent: str,
    use_llm: bool,
    show_debug: bool,
) -> tuple[list[dict[str, str]], str]:
    history = list(history or [])
    message = (message or "").strip()
    if not message:
        return history, _status_md()

    history.append({"role": "user", "content": message})
    t0 = time.time()
    try:
        result = run_agent(
            message,
            doc_id=_norm_choice(doc_id, {"(auto)"}),
            agency=_norm_choice(agency, {"(any)"}),
            use_llm=use_llm,
        )
        forced = _norm_choice(intent, {"(auto)"})
        if forced and result.get("intent") != forced:
            _, _, router = _services()
            routed = router.ask(
                message,
                doc_id=_norm_choice(doc_id, {"(auto)"}),
                agency=_norm_choice(agency, {"(any)"}),
                intent=RouteIntent(forced),
            )
            if use_llm and ChatLLM().available:
                answer = ChatLLM().answer_from_evidence(
                    query=message,
                    evidence=routed.text,
                    intent=routed.intent.value,
                    sources=routed.sources_used,
                )
                result = {
                    "intent": routed.intent.value,
                    "sources_used": routed.sources_used,
                    "answer": answer,
                    "evidence": routed.text,
                    "used_llm": True,
                    "insufficient": False,
                    "meta": routed.meta,
                }
            else:
                result = {
                    "intent": routed.intent.value,
                    "sources_used": routed.sources_used,
                    "answer": routed.text,
                    "evidence": routed.text,
                    "used_llm": False,
                    "insufficient": "[insufficient evidence]" in routed.text,
                    "meta": routed.meta,
                }

        dt = time.time() - t0
        answer = result.get("answer") or result.get("evidence") or "(empty)"
        meta_line = (
            f"\n\n---\n"
            f"*intent=`{result.get('intent')}` · "
            f"sources=`{result.get('sources_used')}` · "
            f"llm=`{result.get('used_llm')}` · "
            f"{dt:.1f}s*"
        )
        if show_debug:
            debug = {
                "intent": result.get("intent"),
                "sources_used": result.get("sources_used"),
                "used_llm": result.get("used_llm"),
                "insufficient": result.get("insufficient"),
                "meta": result.get("meta"),
                "seconds": round(dt, 2),
                "evidence_preview": (result.get("evidence") or "")[:1200],
            }
            meta_line += "\n\n```json\n"
            meta_line += json.dumps(debug, indent=2, ensure_ascii=False)[:4000]
            meta_line += "\n```"
        history.append({"role": "assistant", "content": answer + meta_line})
    except Exception as exc:  # noqa: BLE001
        history.append(
            {
                "role": "assistant",
                "content": f"**Error:** `{type(exc).__name__}: {exc}`",
            }
        )
    return history, _status_md()


def retrieve_only(
    query: str,
    doc_id: str,
    modality: str,
    top_k: float | int,
) -> str:
    query = (query or "").strip()
    if not query:
        return "(empty query)"
    docs, _, _ = _services()
    hits = docs.retrieve(
        query,
        doc_id=_norm_choice(doc_id, {"(auto)"}),
        modality=_norm_choice(modality, {"(any)"}),
        top_k=int(top_k or 5),
    )
    return docs.format_hits(hits)


def sbir_search(query: str, agency: str, top_k: float | int) -> str:
    query = (query or "").strip()
    if not query:
        return "(empty query)"
    _, sbir, _ = _services()
    result = sbir.search(
        query,
        agency=_norm_choice(agency, {"(any)"}),
        top_k=int(top_k or 5),
        include_disclaimer=True,
    )
    return result["text"]


def run_compliance_checklist(
    program: str,
    use_ot: bool,
    pkg_darpa: bool,
    pkg_sba: bool,
    pkg_sf424: bool,
    draft_text: str,
) -> tuple[str, str]:
    """Return (markdown report, json summary)."""
    docs, _, _ = _services()
    packages = []
    if pkg_darpa:
        packages.append("darpa")
    if pkg_sba:
        packages.append("sba")
    if pkg_sf424:
        packages.append("sf424")
    if not packages:
        packages = ["darpa", "sba", "sf424"]
    t0 = time.time()
    run = run_checklist(
        program=(program or "SBIR").lower(),
        use_ot=bool(use_ot),
        packages=packages,
        draft_text=(draft_text or "").strip() or None,
        docs=docs,
    )
    dt = time.time() - t0
    md = run.to_markdown()
    md += f"\n\n---\n*checklist finished in {dt:.1f}s*"
    payload = {
        "corpus": run.summary,
        "draft": run.draft_summary,
        "packages": run.packages,
        "items": len(run.items),
    }
    summary = json.dumps(payload, indent=2)
    return md, summary


def build_ui() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(title="GovGrant AI") as demo:
        gr.Markdown(
            """
# GovGrant AI — test console
Hybrid RAG (Qdrant + nomic) · SBIR topics · LangGraph agent · Claude Haiku
            """
        )
        status = gr.Markdown(_status_md())

        with gr.Tabs():
            with gr.Tab("Chat agent"):
                chatbot = gr.Chatbot(
                    height=480,
                    label="Conversation",
                    buttons=["copy", "copy_all"],
                    layout="bubble",
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder=(
                            "Ask about SBIR policy, SF-424, DARPA Phase II, open topics…"
                        ),
                        scale=5,
                        show_label=False,
                        lines=2,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)

                with gr.Accordion("Options", open=False):
                    with gr.Row():
                        doc_id = gr.Dropdown(
                            DOC_CHOICES, value="(auto)", label="Document filter"
                        )
                        agency = gr.Dropdown(
                            AGENCY_CHOICES, value="(any)", label="SBIR agency"
                        )
                        intent = gr.Dropdown(
                            INTENT_CHOICES, value="(auto)", label="Force intent"
                        )
                    with gr.Row():
                        use_llm = gr.Checkbox(
                            value=True,
                            label="Use Claude Haiku for answer",
                        )
                        show_debug = gr.Checkbox(
                            value=False,
                            label="Show debug (evidence meta)",
                        )

                gr.Examples(examples=EXAMPLES, inputs=msg, label="Examples")
                clear = gr.Button("Clear chat", size="sm")

                def _clear():
                    return [], _status_md()

                send.click(
                    chat_ask,
                    inputs=[
                        msg,
                        chatbot,
                        doc_id,
                        agency,
                        intent,
                        use_llm,
                        show_debug,
                    ],
                    outputs=[chatbot, status],
                ).then(lambda: "", outputs=msg)
                msg.submit(
                    chat_ask,
                    inputs=[
                        msg,
                        chatbot,
                        doc_id,
                        agency,
                        intent,
                        use_llm,
                        show_debug,
                    ],
                    outputs=[chatbot, status],
                ).then(lambda: "", outputs=msg)
                clear.click(_clear, outputs=[chatbot, status])

            with gr.Tab("Retrieve only"):
                gr.Markdown(
                    "Raw hybrid hits (vector + BM25 + RRF + re-rank), no chat LLM."
                )
                rq = gr.Textbox(label="Query", lines=2)
                with gr.Row():
                    rdoc = gr.Dropdown(DOC_CHOICES, value="(auto)", label="doc_id")
                    rmod = gr.Dropdown(
                        ["(any)", "prose", "table", "figure", "chart"],
                        value="(any)",
                        label="modality",
                    )
                    rtop = gr.Slider(1, 12, value=5, step=1, label="top_k")
                rbtn = gr.Button("Retrieve", variant="primary")
                rout = gr.Textbox(label="Hits", lines=20)
                rbtn.click(
                    retrieve_only, inputs=[rq, rdoc, rmod, rtop], outputs=rout
                )

            with gr.Tab("SBIR topics"):
                gr.Markdown(
                    "Search open SBIR topics (fixture cache if API is down). "
                    "Includes mandatory disclaimer."
                )
                sq = gr.Textbox(label="Query", lines=2)
                with gr.Row():
                    sag = gr.Dropdown(
                        AGENCY_CHOICES, value="(any)", label="agency"
                    )
                    stop = gr.Slider(1, 10, value=5, step=1, label="top_k")
                sbtn = gr.Button("Search topics", variant="primary")
                sout = gr.Textbox(label="Results", lines=20)
                sbtn.click(sbir_search, inputs=[sq, sag, stop], outputs=sout)

            with gr.Tab("Compliance checklist"):
                gr.Markdown(
                    """
### Multi-agency control points (DARPA · SBA · SF424)
1. **Corpus mode** — retrieves each rule from indexed instructions (is the rule grounded?).
2. **Draft mode** — optional: paste proposal text; we flag controls your draft appears
   to address (`draft_ok`) vs missing signals (`draft_gap`).

Draft scoring is keyword/signal based — **not** a legal determination.
                    """
                )
                with gr.Row():
                    c_prog = gr.Radio(
                        ["SBIR", "STTR"], value="SBIR", label="Program"
                    )
                    c_ot = gr.Checkbox(
                        value=True, label="Include Other Transaction (OT) items"
                    )
                with gr.Row():
                    c_darpa = gr.Checkbox(value=True, label="DARPA Phase II")
                    c_sba = gr.Checkbox(value=True, label="SBA Policy Directive")
                    c_sf424 = gr.Checkbox(value=True, label="SF424 Application Guide")
                c_draft = gr.Textbox(
                    label="Optional: paste proposal draft (work plan / SOW / strategy)",
                    lines=8,
                    placeholder=(
                        "Paste excerpt of your proposal to check draft signals "
                        "(work-share %, FFRDC, commercialization, Specific Aims…)"
                    ),
                )
                c_btn = gr.Button("Run checklist", variant="primary")
                c_sum = gr.Code(label="Summary counts", language="json", lines=8)
                c_out = gr.Markdown(label="Report")
                c_btn.click(
                    run_compliance_checklist,
                    inputs=[c_prog, c_ot, c_darpa, c_sba, c_sf424, c_draft],
                    outputs=[c_out, c_sum],
                )

            with gr.Tab("About"):
                gr.Markdown(
                    f"""
### Stack
- **Embeddings:** Ollama `{settings.embedding_model}` (local)
- **Vector store:** Qdrant `{settings.qdrant_collection}` + `sbir_topics`
- **Chat:** Anthropic `{settings.chat_model}`
- **Orchestration:** LangGraph (`classify → retrieve → guardrail → answer`)

### Tips
1. Keep Qdrant + Ollama running.
2. If answers look empty, re-ingest: `python -m govgrant.rag.cli ingest`
3. Force document with the dropdown for DARPA / SF424 / SBA.
4. Uncheck Haiku to inspect raw retrieve packs.
5. Use **Compliance checklist** for DARPA Phase II control points (work-share, OT, etc.).
6. Eval: `python -m govgrant.rag.cli eval --golden`
                    """
                )

        return demo


def main() -> None:
    try:
        _services()
        print("[ui] services warmed")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] service warm-up failed: {exc}")

    demo = build_ui()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=".gradio-container { max-width: 1100px !important; }",
    )


if __name__ == "__main__":
    main()
