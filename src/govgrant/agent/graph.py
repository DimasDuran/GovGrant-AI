"""
LangGraph orchestration (R7 + Haiku).

Pipeline:
  classify → retrieve (QueryRouter, when needed) → validate_evidence → format_answer (Haiku)

Behaves like a normal AI assistant (Claude/GPT style), specialized on SBIR/STTR
compliance. Grounded answers use retrieved evidence; greetings/meta use a short
conversational turn without forcing a docs-only lecture.
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

# Greetings / small-talk → conversational path (no RAG required)
_CONVERSATIONAL_RE = re.compile(
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

def _normalize_chat_query(query: str) -> str:
    """Strip punctuation/emoji so '¡Hola! 👋' still counts as a greeting."""
    q = (query or "").strip().lower()
    q = re.sub(r"[^\w\sáéíóúüñ]", " ", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip()


def is_conversational_turn(query: str) -> bool:
    """True for greetings / empty / pure small-talk (no retrieval needed)."""
    q = (query or "").strip()
    if not q:
        return True
    norm = _normalize_chat_query(q)
    if not norm:
        return True
    if _CONVERSATIONAL_RE.match(norm):
        return True
    return False


def conversational_reply(query: str) -> str:
    """
    Short, natural replies for greetings/meta — no capability brochure.

    Fixed templates (no LLM): Haiku tends to dump topic menus on greetings.
    """
    q = (query or "").strip()
    low = _normalize_chat_query(q)
    spanish = bool(
        re.search(
            r"[áéíóúñ¿¡]|hola|buenas|quién|quien|gracias|ayuda|qué tal|como estas|cómo estás",
            q,
            re.I,
        )
    ) or low in {"hola", "buenas", ""}

    if not q or low in {"hola", "hello", "hi", "hey", "buenas", "ok", "okay"}:
        return (
            "¡Hola! Soy GovGrant AI, especializado en cumplimiento SBIR/STTR. ¿En qué te ayudo?"
            if spanish or low in {"hola", "buenas", ""}
            else "Hi — I'm GovGrant AI, specialized in SBIR/STTR compliance. How can I help?"
        )
    if re.search(r"buenos?\s+d[ií]as|good morning", low):
        return (
            "¡Buenos días! Soy GovGrant AI (SBIR/STTR). ¿En qué te ayudo?"
            if spanish
            else "Good morning — GovGrant AI here (SBIR/STTR). How can I help?"
        )
    if re.search(r"buenas?\s+(tardes|noches)|good (afternoon|evening)", low):
        return (
            "¡Hola! Soy GovGrant AI (SBIR/STTR). ¿En qué te ayudo?"
            if spanish
            else "Hi — GovGrant AI (SBIR/STTR). How can I help?"
        )
    if re.search(r"quién eres|quien eres|who are you", low):
        return (
            "Soy **GovGrant AI**: un asistente de IA enfocado en cumplimiento de "
            "propuestas y políticas SBIR/STTR (instrucciones de agencia, SBA, SF-424, "
            "topics y tus proposals). Pregúntame lo que necesites en ese ámbito."
            if spanish
            else "I'm **GovGrant AI** — an AI assistant focused on SBIR/STTR compliance "
            "(agency instructions, SBA, SF-424, open topics, and your proposals). "
            "Ask me anything in that domain."
        )
    if re.search(r"gracias|thanks|thank you", low):
        return "¡De nada! Si surge otra duda de SBIR/STTR, aquí estoy." if spanish else "You're welcome!"
    if re.search(r"help|ayuda|start|comenzar", low):
        return (
            "Claro. Dime tu duda de cumplimiento SBIR/STTR "
            "(p. ej. work-share, cost volume, elegibilidad, SF-424)."
            if spanish
            else "Sure — ask your SBIR/STTR compliance question "
            "(e.g. work-share, cost volume, eligibility, SF-424)."
        )
    # Default short open
    return (
        "Hola — soy GovGrant AI (SBIR/STTR). ¿En qué te ayudo?"
        if spanish
        else "Hi — I'm GovGrant AI (SBIR/STTR). How can I help?"
    )


# Back-compat alias for older imports/tests
def is_non_substantive_query(query: str) -> bool:
    return is_conversational_turn(query)


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
        if is_conversational_turn(q):
            return {
                **state,
                "intent": "chat",
                "doc_id": state.get("doc_id"),
                "used_llm": False,
                "insufficient": False,
                "evidence": "",
                "sources_used": [],
                "meta": {"mode": "conversation"},
            }
        intent = tools.classify(q)
        doc_id = infer_doc_id(q, state.get("doc_id"))
        return {
            **state,
            "intent": intent,
            "doc_id": doc_id,
            "used_llm": False,
            "meta": {"mode": "grounded"},
        }

    def retrieve(state: AgentState) -> AgentState:
        # Conversational turns (greetings) skip retrieval
        if (state.get("meta") or {}).get("mode") == "conversation":
            return state
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
        if state.get("answer"):
            return state
        # Conversational path does not need evidence
        if (state.get("meta") or {}).get("mode") == "conversation":
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

        # Greetings / meta: fixed short reply (LLM tends to dump capability menus)
        if (state.get("meta") or {}).get("mode") == "conversation":
            return {
                **state,
                "answer": conversational_reply(state.get("query") or ""),
                "used_llm": False,
            }

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
