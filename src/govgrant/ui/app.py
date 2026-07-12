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
from govgrant.auth import AuthError, resolve_request_auth
from govgrant.compliance.checklist import run_checklist
from govgrant.compliance.proposal import extract_proposal_text, proposal_doc_id
from govgrant.compliance.report import export_checklist_run, export_history_markdown
from govgrant.proposals import ProposalService
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


@lru_cache(maxsize=1)
def _proposal_service() -> ProposalService:
    docs, _, _ = _services()
    return ProposalService(docs=docs)


def _status_md(api_key: str | None = None) -> str:
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
    tenant = s.default_tenant_id
    n_props = "?"
    try:
        auth = resolve_request_auth(api_key=(api_key or "").strip() or None)
        tenant = auth.tenant_id
        n_props = str(len(_proposal_service().list_proposals(auth)))
    except AuthError:
        tenant = "(auth error)"
    return (
        f"**Chat model:** `{s.chat_model}` · "
        f"**LLM ready:** {'yes' if llm.available else 'no'} · "
        f"**tenant:** `{tenant}` · "
        f"**my proposals:** {n_props} · "
        f"**BM25 nodes:** {n_bm25} · "
        f"**SBIR topics:** {n_sbir} · "
        f"**Qdrant:** `{s.qdrant_url}`"
    )


def _doc_choices_for_key(api_key: str | None) -> list[str]:
    """Public agency docs + this tenant's registered proposals."""
    choices = list(DOC_CHOICES)
    try:
        auth = resolve_request_auth(api_key=(api_key or "").strip() or None)
        for rec in _proposal_service().list_proposals(auth):
            if rec.doc_id not in choices:
                choices.append(rec.doc_id)
    except AuthError:
        pass
    return choices


def _proposals_table_md(api_key: str) -> str:
    try:
        auth = resolve_request_auth(api_key=(api_key or "").strip() or None)
    except AuthError as exc:
        return f"**Auth error:** {exc}"
    rows = _proposal_service().list_proposals(auth)
    caps = auth.capabilities()
    cap_bits = ", ".join(k for k, v in caps.items() if v) or "(none)"
    header = (
        f"**Tenant:** `{auth.tenant_id}` · **roles:** `{', '.join(auth.roles)}` · "
        f"**capabilities:** {cap_bits}"
    )
    if not rows:
        return (
            f"{header}\n\n_No proposals for tenant `{auth.tenant_id}` yet. "
            "Upload a PDF below._"
        )
    lines = [
        f"{header} · **count:** {len(rows)}",
        "",
        "| doc_id | file | pages | indexed | created |",
        "|--------|------|------:|:-------:|---------|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.doc_id}` | {r.file_name} | {r.pages} | "
            f"{'yes' if r.indexed else 'no'} | {r.created_at} |"
        )
    return "\n".join(lines)


def _audit_table_md(api_key: str, *, limit: int = 15) -> str:
    try:
        auth = resolve_request_auth(api_key=(api_key or "").strip() or None)
    except AuthError as exc:
        return f"**Auth error:** {exc}"
    events = _proposal_service().list_events(auth, limit=limit)
    if not events:
        return f"_No audit events for tenant `{auth.tenant_id}` yet._"
    lines = [
        f"**Recent activity** (tenant `{auth.tenant_id}`)",
        "",
        "| when | action | doc_id | roles |",
        "|------|--------|--------|-------|",
    ]
    for e in events:
        lines.append(
            f"| {e.created_at} | `{e.action}` | `{e.doc_id}` | {e.actor_roles or '—'} |"
        )
    return "\n".join(lines)


def _delete_btn_update(api_key: str) -> gr.Button:
    """Enable delete only when session may delete proposals."""
    try:
        auth = resolve_request_auth(api_key=(api_key or "").strip() or None)
        can = auth.can_delete_proposals()
        label = (
            "Delete proposal"
            if can
            else "Delete (admin required when AUTH_ENABLED)"
        )
        return gr.update(interactive=can, value=label)
    except AuthError:
        return gr.update(interactive=False, value="Delete (auth error)")


def upload_proposal_ui(
    pdf_file,
    index: bool,
    api_key: str,
) -> tuple[str, str, str, gr.Dropdown]:
    """Upload → registry (+ optional index). Returns status, table, audit, doc dropdown."""
    if pdf_file is None:
        return (
            "Upload a PDF first.",
            _proposals_table_md(api_key),
            _audit_table_md(api_key),
            gr.update(),
        )
    path = getattr(pdf_file, "name", None) or pdf_file
    try:
        auth = resolve_request_auth(api_key=(api_key or "").strip() or None)
        result = _proposal_service().upload(auth, path, index=bool(index))
        rec = result.record
        msg = (
            f"**Registered** `{rec.doc_id}` for tenant `{rec.tenant_id}` · "
            f"pages={rec.pages} · chars={rec.chars:,} · "
            f"indexed={'yes' if rec.indexed else 'no'} · parser=`{result.extract_parser}`"
        )
        choices = _doc_choices_for_key(api_key)
        return (
            msg,
            _proposals_table_md(api_key),
            _audit_table_md(api_key),
            gr.update(choices=choices),
        )
    except AuthError as exc:
        return (
            f"**Auth error:** {exc}",
            _proposals_table_md(api_key),
            _audit_table_md(api_key),
            gr.update(),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            f"**Error:** `{type(exc).__name__}: {exc}`",
            _proposals_table_md(api_key),
            _audit_table_md(api_key),
            gr.update(),
        )


def refresh_proposals_ui(api_key: str) -> tuple[str, str, gr.Dropdown]:
    choices = _doc_choices_for_key(api_key)
    return (
        _proposals_table_md(api_key),
        _audit_table_md(api_key),
        gr.update(choices=choices),
    )


def delete_proposal_ui(
    doc_id: str, api_key: str
) -> tuple[str, str, str, gr.Dropdown]:
    doc_id = (doc_id or "").strip()
    if not doc_id or doc_id in DOC_CHOICES:
        return (
            "Select a user-proposal doc_id to delete.",
            _proposals_table_md(api_key),
            _audit_table_md(api_key),
            gr.update(),
        )
    try:
        auth = resolve_request_auth(api_key=(api_key or "").strip() or None)
        svc = _proposal_service()
        ok = svc.delete(auth, doc_id, purge_index=True, remove_file=True)
        if ok:
            purge = getattr(svc, "_last_delete_index", None) or {}
            msg = (
                f"Deleted `{doc_id}` (registry + file). "
                f"Index purge: bm25={purge.get('bm25_removed', '?')}, "
                f"qdrant≈{purge.get('qdrant_deleted_estimate', '?')}, "
                f"tables={purge.get('tabular_cleared', '?')}."
            )
        else:
            msg = f"Not found: `{doc_id}`."
        choices = _doc_choices_for_key(api_key)
        return (
            msg,
            _proposals_table_md(api_key),
            _audit_table_md(api_key),
            gr.update(choices=choices),
        )
    except AuthError as exc:
        return (
            f"**Auth error:** {exc}",
            _proposals_table_md(api_key),
            _audit_table_md(api_key),
            gr.update(),
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
    api_key: str,
) -> tuple[list[dict[str, str]], str]:
    history = list(history or [])
    message = (message or "").strip()
    if not message:
        return history, _status_md()

    history.append({"role": "user", "content": message})
    t0 = time.time()
    try:
        auth = resolve_request_auth(api_key=(api_key or "").strip() or None)
        doc = auth.filter_doc_id(_norm_choice(doc_id, {"(auto)"}))
        result = run_agent(
            message,
            tenant_id=auth.tenant_id,
            doc_id=doc,
            agency=_norm_choice(agency, {"(any)"}),
            use_llm=use_llm,
        )
        forced = _norm_choice(intent, {"(auto)"})
        if forced and result.get("intent") != forced:
            _, _, router = _services()
            routed = router.ask(
                message,
                tenant_id=auth.tenant_id,
                doc_id=doc,
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
    except AuthError as exc:
        history.append(
            {
                "role": "assistant",
                "content": f"**Auth error:** {exc}",
            }
        )
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
    draft_pdf,  # gradio File path or None
    index_proposal: bool,
    use_llm_draft: bool,
    api_key: str,
    selected_proposal: str,
    export_report: bool,
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

    draft = (draft_text or "").strip() or None
    extract_note = ""
    proposal_id = None
    t0 = time.time()

    try:
        auth = resolve_request_auth(api_key=(api_key or "").strip() or None)
    except AuthError as exc:
        return f"**Auth error:** {exc}", "{}"

    # Prefer registered proposal doc_id for draft text
    sel = (selected_proposal or "").strip()
    if sel and sel not in {"(none)", ""} and sel.startswith("user-proposal-"):
        try:
            draft = _proposal_service().read_draft_text(auth, sel)
            proposal_id = sel
            extract_note = f"**Draft from registry:** `{sel}` (tenant `{auth.tenant_id}`)"
        except Exception as exc:  # noqa: BLE001
            extract_note = f"**Registry read error:** `{exc}`"

    if draft_pdf is not None and not proposal_id:
        pdf_path = getattr(draft_pdf, "name", None) or draft_pdf
        if pdf_path:
            try:
                result = _proposal_service().upload(
                    auth, pdf_path, index=bool(index_proposal)
                )
                draft = extract_proposal_text(result.record.stored_path).text
                proposal_id = result.record.doc_id
                extract_note = (
                    f"**Registered** `{proposal_id}` · pages={result.record.pages} · "
                    f"indexed={'yes' if result.record.indexed else 'no'}"
                )
            except Exception as exc:  # noqa: BLE001
                extract_note = f"**PDF error:** `{type(exc).__name__}: {exc}`"

    run = run_checklist(
        program=(program or "SBIR").lower(),
        use_ot=bool(use_ot),
        packages=packages,
        draft_text=draft,
        docs=docs,
        use_llm_draft=bool(use_llm_draft),
        tenant_id=auth.tenant_id,
    )
    export_paths: dict = {}
    if export_report:
        export_paths = export_checklist_run(
            run,
            extra={
                "tenant_id": auth.tenant_id,
                "proposal_doc_id": proposal_id,
                "packages": packages,
            },
        )
    dt = time.time() - t0
    md = ""
    if extract_note:
        md += extract_note + "\n\n"
    md += run.to_markdown()
    if export_paths:
        md += f"\n\n**Exported:** `{export_paths.get('md', '')}`"
    md += f"\n\n---\n*checklist finished in {dt:.1f}s · tenant=`{auth.tenant_id}`*"
    payload = {
        "corpus": run.summary,
        "draft": run.draft_summary,
        "packages": run.packages,
        "items": len(run.items),
        "proposal_doc_id": proposal_id,
        "tenant_id": auth.tenant_id,
        "export_paths": export_paths or None,
    }
    summary = json.dumps(payload, indent=2)
    return md, summary


def _apply_session(api_key: str) -> tuple[str, str]:
    """Update status line + session banner from shared API key."""
    key = (api_key or "").strip()
    try:
        auth = resolve_request_auth(api_key=key or None)
        caps = auth.capabilities()
        cap_s = ", ".join(k for k, v in caps.items() if v) or "(none)"
        banner = (
            f"**Session:** tenant=`{auth.tenant_id}` · "
            f"roles=`{', '.join(auth.roles)}` · "
            f"auth_enabled=`{auth.auth_enabled}` · "
            f"source=`{auth.source}` · "
            f"capabilities=`{cap_s}`"
        )
    except AuthError as exc:
        banner = f"**Session auth error:** {exc}"
    return _status_md(key), banner


def build_ui() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(title="GovGrant AI") as demo:
        gr.Markdown(
            """
# GovGrant AI — test console
Hybrid RAG (Qdrant + nomic) · SBIR topics · LangGraph agent · Claude Haiku
            """
        )
        # Shared session (one API key for all tabs)
        with gr.Group():
            gr.Markdown("### Session")
            with gr.Row():
                session_key = gr.Textbox(
                    label="API key (shared across tabs)",
                    type="password",
                    placeholder="empty if AUTH_ENABLED=false · e.g. dev-local-key",
                    scale=4,
                )
                apply_session = gr.Button("Apply session", variant="secondary", scale=1)
            session_banner = gr.Markdown(_apply_session("")[1])
        status = gr.Markdown(_status_md())
        apply_session.click(
            _apply_session, inputs=[session_key], outputs=[status, session_banner]
        )
        session_key.submit(
            _apply_session, inputs=[session_key], outputs=[status, session_banner]
        )

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
                            DOC_CHOICES,
                            value="(auto)",
                            label="Document filter",
                            allow_custom_value=True,
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
                    refresh_docs = gr.Button("Refresh docs (my proposals)", size="sm")
                    refresh_docs.click(
                        lambda k: gr.update(choices=_doc_choices_for_key(k)),
                        inputs=[session_key],
                        outputs=[doc_id],
                    )

                gr.Examples(examples=EXAMPLES, inputs=msg, label="Examples")
                clear = gr.Button("Clear chat", size="sm")

                def _clear(key: str):
                    return [], _status_md(key)

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
                        session_key,
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
                        session_key,
                    ],
                    outputs=[chatbot, status],
                ).then(lambda: "", outputs=msg)
                clear.click(_clear, inputs=[session_key], outputs=[chatbot, status])

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

            with gr.Tab("My proposals"):
                gr.Markdown(
                    """
### Tenant-scoped proposals
Uses the **Session** API key above. Indexed docs get `doc_id=user-proposal-…`
for Chat filtering. **Admin** required to delete when `AUTH_ENABLED=true`.
Upload / delete actions are recorded in the tenant audit log.
                    """
                )
                p_table = gr.Markdown(_proposals_table_md(""))
                with gr.Row():
                    p_file = gr.File(
                        label="Upload proposal PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    p_index = gr.Checkbox(value=True, label="Index into hybrid RAG")
                with gr.Row():
                    p_up = gr.Button("Register / upload", variant="primary")
                    p_ref = gr.Button("Refresh list")
                p_status = gr.Markdown()
                p_del_id = gr.Textbox(
                    label="Delete doc_id",
                    placeholder="user-proposal-my-file",
                )
                p_del = gr.Button(
                    "Delete proposal",
                    size="sm",
                )
                p_audit = gr.Markdown(_audit_table_md(""))
                p_up.click(
                    upload_proposal_ui,
                    inputs=[p_file, p_index, session_key],
                    outputs=[p_status, p_table, p_audit, doc_id],
                )
                p_ref.click(
                    refresh_proposals_ui,
                    inputs=[session_key],
                    outputs=[p_table, p_audit, doc_id],
                )
                p_del.click(
                    delete_proposal_ui,
                    inputs=[p_del_id, session_key],
                    outputs=[p_status, p_table, p_audit, doc_id],
                )
                # Refresh table + delete affordance when session is applied
                apply_session.click(
                    lambda k: (
                        _proposals_table_md(k),
                        _audit_table_md(k),
                        _delete_btn_update(k),
                    ),
                    inputs=[session_key],
                    outputs=[p_table, p_audit, p_del],
                )
                session_key.submit(
                    lambda k: (
                        _proposals_table_md(k),
                        _audit_table_md(k),
                        _delete_btn_update(k),
                    ),
                    inputs=[session_key],
                    outputs=[p_table, p_audit, p_del],
                )

            with gr.Tab("Compliance checklist"):
                gr.Markdown(
                    """
### Multi-agency control points (DARPA · SBA · SF424)
1. **Corpus mode** — rule grounded in indexed instructions?
2. **Draft mode** — paste text, upload PDF, or pick a registered proposal.
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
                c_sel = gr.Dropdown(
                    choices=["(none)"],
                    value="(none)",
                    label="Registered proposal (optional)",
                    allow_custom_value=True,
                )
                c_pdf = gr.File(
                    label="Or upload proposal PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                c_index = gr.Checkbox(
                    value=True,
                    label="Register + index uploaded PDF under my tenant",
                )
                c_llm = gr.Checkbox(
                    value=False,
                    label="LLM draft judge (Haiku; falls back to keywords)",
                )
                c_export = gr.Checkbox(
                    value=True,
                    label="Export report to data/eval/reports/ (md + json)",
                )
                c_draft = gr.Textbox(
                    label="Or paste draft text",
                    lines=5,
                    placeholder="Paste SOW / strategy excerpt…",
                )
                c_btn = gr.Button("Run checklist", variant="primary")
                c_sum = gr.Code(label="Summary counts", language="json", lines=8)
                c_out = gr.Markdown(label="Report")
                c_hist = gr.Markdown(export_history_markdown())
                c_hist_btn = gr.Button("Refresh export history", size="sm")

                def _refresh_c_sel(key: str):
                    try:
                        auth = resolve_request_auth(
                            api_key=(key or "").strip() or None
                        )
                        ids = ["(none)"] + [
                            r.doc_id for r in _proposal_service().list_proposals(auth)
                        ]
                        return gr.update(choices=ids)
                    except AuthError:
                        return gr.update(choices=["(none)"])

                apply_session.click(
                    _refresh_c_sel, inputs=[session_key], outputs=[c_sel]
                )
                c_hist_btn.click(
                    lambda: export_history_markdown(), outputs=[c_hist]
                )
                c_btn.click(
                    run_compliance_checklist,
                    inputs=[
                        c_prog,
                        c_ot,
                        c_darpa,
                        c_sba,
                        c_sf424,
                        c_draft,
                        c_pdf,
                        c_index,
                        c_llm,
                        session_key,
                        c_sel,
                        c_export,
                    ],
                    outputs=[c_out, c_sum],
                ).then(
                    lambda: export_history_markdown(),
                    outputs=[c_hist],
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
