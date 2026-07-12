"""
LangGraph orchestration (R7 + Haiku chat).

Pipeline:
  classify (heuristic) → retrieve (QueryRouter) → validate_evidence → format_answer (Haiku)

Routing stays heuristic (more reliable for SBIR domain).
Haiku is used only to write the final grounded answer from retrieved evidence.
This is a document Q&A engine — not a conversational chatbot.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from govgrant.agent.llm import ChatLLM
from govgrant.agent.tools import RagToolBelt

# Soft doc targeting when user names a known fixture corpus
_DOC_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bDARPA\b", re.I), "darpa-sbir-sttr-phase-II-instructions"),
    (re.compile(r"\bSF-?424\b|\bapplication guide\b", re.I), "SF424 SBIR_STTR Application Guide"),
    (re.compile(r"\bpolicy directive\b|\bSBA\b", re.I), "SBA SBIR_STTR_POLICY_DIRECTIVE_May2023"),
]

# Greetings / meta-chat — do not retrieve or invent a chatbot intro
_NON_QUESTION_RE = re.compile(
    r"""^
    (
        hola|hello|hi|hey|buenas(?:\s+(tardes|noches|días|dias))?|
        buenos\s+días|buenos\s+dias|
        good\s+(morning|afternoon|evening)|
        (qué|que)\s+tal|(cómo|como)\s+estás|(cómo|como)\s+estas|
        who\s+are\s+you|(quién|quien)\s+eres|
        help|ayuda|start|comenzar|thanks|gracias|ok|okay
    )
    [\s.!?¿?]*$
    """,
    re.I | re.X,
)

_NON_QUESTION_REPLY = (
    "Este panel **consulta el corpus indexado** (DARPA / SBA / SF-424 / tus proposals) "
    "— no es un chatbot conversacional.\n\n"
    "Escribe una **pregunta concreta**, por ejemplo:\n"
    "• ¿Cuál es el máximo del Cost Volume en DARPA Phase II?\n"
    "• ¿Qué work-share aplica en SBIR Phase II?\n"
    "• ¿Qué dice el SF-424 sobre indirect costs?"
)


def is_non_substantive_query(query: str) -> bool:
    """True for greetings / empty / pure meta chat (no compliance question)."""
    q = (query or "").strip()
    if not q:
        return True
    if _NON_QUESTION_RE.match(q):
        return True
    # Very short with no domain token
    low = q.lower()
    domain = re.search(
        r"sbir|sttr|darpa|sba|sf-?424|phase|cost|budget|work[- ]?share|"
        r"eligib|proposal|volume|ot\b|milestone|topic|foa|nih|dod",
        low,
    )
    if len(q) < 12 and not domain:
        return True
    return False


class AgentState(TypedDict, total=False):
    query: str
    tenant_id: str
    doc_id: str | None
    agency: str | None
    intent: str
    sources_used: list[str]
    evidence: str
    answer: str
    insufficient: bool
    meta: dict[str, Any]
    used_llm: bool


def infer_doc_id(query: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for pattern, doc_id in _DOC_HINTS:
        if pattern.search(query):
            return doc_id
    return None


def re_search_hits(evidence: str) -> bool:
    return "score=" in (evidence or "") or "topic_id=" in (evidence or "")


def build_agent_graph(
    tools: RagToolBelt | None = None,
    llm: ChatLLM | None = None,
    *,
    use_llm: bool = True,
):
    tools = tools or RagToolBelt()
    llm = llm if llm is not None else ChatLLM()
    llm_on = bool(use_llm and llm.available)

    def classify(state: AgentState) -> AgentState:
        # Domain heuristics are more reliable than LLM routing for this stack
        q = state.get("query") or ""
        if is_non_substantive_query(q):
            return {
                **state,
                "intent": "doc_qa",
                "doc_id": state.get("doc_id"),
                "used_llm": False,
                "insufficient": True,
                "evidence": "",
                "sources_used": [],
                "answer": _NON_QUESTION_REPLY,
                "meta": {"skip_reason": "non_substantive_query"},
            }
        intent = tools.classify(q)
        doc_id = infer_doc_id(q, state.get("doc_id"))
        return {**state, "intent": intent, "doc_id": doc_id, "used_llm": False}

    def retrieve(state: AgentState) -> AgentState:
        # Already answered (greeting / empty)
        if state.get("answer"):
            return state
        # Multi-part compliance questions need broader evidence packs
        q = state["query"]
        multi = bool(
            re.search(
                r"\b(also|finally|additionally|a few questions|first|second|third|"
                r"además|por último|finalmente|asimismo)\b",
                q,
                re.I,
            )
        )
        result = tools.ask(
            state["query"],
            tenant_id=state.get("tenant_id"),
            doc_id=state.get("doc_id"),
            agency=state.get("agency"),
            intent=state.get("intent"),
            top_k=12 if multi else 8,
        )
        evidence = result.get("text") or ""
        has_hits = bool(re_search_hits(evidence))
        insufficient = (not has_hits) or (
            "[insufficient evidence]" in evidence and not has_hits
        )
        return {
            **state,
            "intent": result.get("intent") or state.get("intent"),
            "sources_used": result.get("sources_used") or [],
            "evidence": evidence,
            "insufficient": insufficient,
            "meta": {
                **(result.get("meta") or {}),
                "doc_id": state.get("doc_id"),
            },
        }

    def validate_evidence(state: AgentState) -> AgentState:
        # Keep pre-filled answers (e.g. greeting short-circuit)
        if state.get("answer"):
            return state
        if state.get("insufficient") or not (state.get("evidence") or "").strip():
            return {
                **state,
                "answer": (
                    "I don't have enough retrieved evidence to answer reliably. "
                    "Please refine the question, specify --doc-id, or ingest more sources."
                ),
                "insufficient": True,
            }
        return state

    def format_answer(state: AgentState) -> AgentState:
        if state.get("answer"):
            return state
        if llm_on and not state.get("insufficient"):
            try:
                answer = llm.answer_from_evidence(
                    query=state["query"],
                    evidence=state.get("evidence") or "",
                    intent=state.get("intent") or "doc_qa",
                    sources=list(state.get("sources_used") or []),
                )
                return {**state, "answer": answer, "used_llm": True}
            except Exception as exc:  # noqa: BLE001
                fallback = (
                    f"(LLM format failed: {exc})\n\n"
                    f"intent={state.get('intent')} | sources={state.get('sources_used')}\n\n"
                    f"{state.get('evidence', '')}"
                )
                return {**state, "answer": fallback, "used_llm": False}

        header = (
            f"intent={state.get('intent')} | "
            f"sources={state.get('sources_used')} | "
            f"insufficient={state.get('insufficient', False)}"
        )
        answer = f"{header}\n\n{state.get('evidence', '')}".strip()
        return {**state, "answer": answer, "used_llm": False}

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("validate_evidence", validate_evidence)
    graph.add_node("format_answer", format_answer)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "validate_evidence")
    graph.add_edge("validate_evidence", "format_answer")
    graph.add_edge("format_answer", END)

    return graph.compile()


def run_agent(
    query: str,
    *,
    tenant_id: str | None = None,
    doc_id: str | None = None,
    agency: str | None = None,
    tools: RagToolBelt | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    app = build_agent_graph(tools, use_llm=use_llm)
    from govgrant.rag.config import get_settings

    settings = get_settings()
    final = app.invoke(
        {
            "query": query,
            "tenant_id": tenant_id or settings.default_tenant_id,
            "doc_id": doc_id,
            "agency": agency,
        }
    )
    return dict(final)
