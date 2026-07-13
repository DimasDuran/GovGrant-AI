"""
LangGraph orchestration (R7 + Haiku) with LLM-driven routing + retry + self-check.

Pipeline:
  classify → retrieve → validate_evidence → (retry retrieve | format_answer) → self_check → (revise | END)

classify  — LLM tool-use routing (or regex fallback).
validate_evidence — LLM judge (mark_sufficient / request_more_evidence) decides retry.
self_check — LLM verifies answer covers the question (answer_complete / answer_incomplete).
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from govgrant.agent.llm import ChatLLM
from govgrant.agent.tools import RagToolBelt
from govgrant.compliance.checklist import run_checklist

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

_TOOL_TO_INTENT = {
    "search_documents": "doc_qa",
    "search_tables": "table",
    "search_sbir_topics": "topic_search",
    "cross_check": "cross_check",
    "compliance_checklist": "checklist",
}

_MAX_RETRIES = 2
_MAX_SELF_CHECK_RETRIES = 1


def _normalize_chat_query(query: str) -> str:
    q = (query or "").strip().lower()
    q = re.sub(r"[^\w\sáéíóúüñ]", " ", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip()


def is_conversational_turn(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return True
    norm = _normalize_chat_query(q)
    if not norm:
        return True
    if _CONVERSATIONAL_RE.match(norm):
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
    # Agentic routing + retry
    next_action: str
    retries: int
    reformulated_query: str | None
    self_check_critique: str | None
    self_check_retries: int
    checklist_packages: list[str]
    checklist_program: str


def infer_doc_id(query: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for pattern, doc_id in _DOC_HINTS:
        if pattern.search(query):
            return doc_id
    return None


def build_agent_graph(
    tools: RagToolBelt | None = None,
    llm: ChatLLM | None = None,
    *,
    use_llm: bool = True,
    max_retries: int = _MAX_RETRIES,
):
    tools = tools or RagToolBelt()
    llm = llm if llm is not None else ChatLLM()
    llm_on = bool(use_llm and llm.available)

    def classify(state: AgentState) -> AgentState:
        q = state.get("query") or ""

        # 1. Greeting → bypass retrieval entirely
        if is_conversational_turn(q):
            return {
                **state,
                "next_action": "format_answer",
                "intent": "chat",
                "doc_id": state.get("doc_id"),
                "used_llm": False,
                "insufficient": False,
                "evidence": "",
                "sources_used": [],
                "meta": {"mode": "conversation"},
            }

        # 2. LLM tool-use routing (when available)
        if llm_on:
            result = llm.classify_with_tools(q)
            if result and result.get("tool") in _TOOL_TO_INTENT:
                intent = _TOOL_TO_INTENT[result["tool"]]
                args = result.get("input") or {}
                if intent == "checklist":
                    packages = args.get("packages") or ["darpa", "sba", "sf424"]
                    program = args.get("program") or "sbir"
                    return {
                        **state,
                        "next_action": "checklist",
                        "intent": intent,
                        "used_llm": True,
                        "meta": {"mode": "grounded"},
                        "checklist_packages": packages,
                        "checklist_program": program,
                    }
                doc_id = args.get("doc_id") or infer_doc_id(q, state.get("doc_id"))
                agency = args.get("agency") or state.get("agency")
                return {
                    **state,
                    "next_action": "retrieve",
                    "intent": intent,
                    "doc_id": doc_id,
                    "agency": agency,
                    "used_llm": True,
                    "meta": {"mode": "grounded"},
                }

        # 3. Fallback: heuristic classifier
        intent = tools.classify(q)
        doc_id = infer_doc_id(q, state.get("doc_id"))
        if intent == "checklist":
            return {
                **state,
                "next_action": "checklist",
                "intent": intent,
                "used_llm": False,
                "meta": {"mode": "grounded"},
                "checklist_packages": ["darpa", "sba", "sf424"],
                "checklist_program": "sbir",
            }
        return {
            **state,
            "next_action": "retrieve",
            "intent": intent,
            "doc_id": doc_id,
            "used_llm": False,
            "meta": {"mode": "grounded"},
        }

    def retrieve(state: AgentState) -> AgentState:
        mode = (state.get("meta") or {}).get("mode")
        if mode == "conversation" or state.get("answer"):
            return state

        q = state.get("reformulated_query") or state["query"]
        is_retry = bool(state.get("reformulated_query"))
        result = tools.ask(
            q,
            tenant_id=state.get("tenant_id"),
            doc_id=state.get("doc_id"),
            agency=state.get("agency"),
            intent=state.get("intent"),
            top_k=14 if is_retry else 8,
        )
        evidence = result.get("text") or ""
        return {
            **state,
            "reformulated_query": None,
            "intent": result.get("intent") or state.get("intent"),
            "sources_used": result.get("sources_used") or [],
            "evidence": evidence,
            "meta": {
                **(result.get("meta") or {}),
                "doc_id": state.get("doc_id"),
            },
        }

    def validate_evidence(state: AgentState) -> AgentState:
        if state.get("answer"):
            return {**state, "next_action": "format_answer"}
        if (state.get("meta") or {}).get("mode") == "conversation":
            return {**state, "next_action": "format_answer"}

        retries = state.get("retries", 0)
        evidence = state.get("evidence") or ""
        has_evidence = bool(evidence.strip())

        # 1. LLM judge (when available)
        if llm_on and has_evidence:
            judgement = llm.judge_evidence(
                query=state["query"],
                evidence=evidence,
                retry_count=retries,
            )
            if judgement:
                if judgement.get("action") == "sufficient":
                    reason = judgement.get("reason", "")
                    meta = dict(state.get("meta") or {})
                    meta["judge_reason"] = reason
                    return {
                        **state,
                        "next_action": "format_answer",
                        "insufficient": False,
                        "meta": meta,
                    }
                if judgement.get("action") == "retry":
                    if retries < max_retries:
                        return {
                            **state,
                            "next_action": "retrieve",
                            "retries": retries + 1,
                            "reformulated_query": judgement.get("suggested_query", state["query"]),
                        }
                    # LLM says insufficient but retries exhausted → hard stop
                    reason = judgement.get("reason", "")
                    meta = dict(state.get("meta") or {})
                    meta["judge_reason"] = f"insufficient after {retries + 1} attempts: {reason}"
                    return {
                        **state,
                        "next_action": "format_answer",
                        "answer": (
                            "I don't have enough retrieved evidence to answer reliably. "
                            "Please refine the question, specify --doc-id, or ingest more sources."
                        ),
                        "insufficient": True,
                        "meta": meta,
                    }

        # 2. Fallback when LLM unavailable / error
        if has_evidence:
            return {**state, "next_action": "format_answer", "insufficient": False}

        if retries < max_retries:
            return {
                **state,
                "next_action": "retrieve",
                "retries": retries + 1,
                "reformulated_query": state["query"],
            }

        return {
            **state,
            "next_action": "format_answer",
            "answer": (
                "I don't have enough retrieved evidence to answer reliably. "
                "Please refine the question, specify --doc-id, or ingest more sources."
            ),
            "insufficient": True,
        }

    def checklist(state: AgentState) -> AgentState:
        packages = state.get("checklist_packages") or ["darpa", "sba", "sf424"]
        program = state.get("checklist_program") or "sbir"
        try:
            run = run_checklist(
                program=program,
                use_ot=True,
                packages=packages,
                tenant_id=state.get("tenant_id"),
            )
            evidence = run.to_markdown()
            return {
                **state,
                "evidence": evidence,
                "sources_used": [],
                "insufficient": False,
                "next_action": "format_answer",
                "meta": {
                    **(state.get("meta") or {}),
                    "checklist_summary": run.summary.to_dict(),
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {
                **state,
                "evidence": f"(Checklist failed: {exc})",
                "insufficient": True,
                "next_action": "format_answer",
            }

    def format_answer(state: AgentState) -> AgentState:
        if state.get("answer") and not state.get("self_check_critique"):
            return state

        critique = state.get("self_check_critique")

        if llm_on and not state.get("insufficient"):
            try:
                query = state["query"]
                if critique:
                    query = (
                        f"{state['query']}\n\n"
                        f"[Revision requested by quality check: {critique}]"
                    )
                answer = llm.answer_from_evidence(
                    query=query,
                    evidence=state.get("evidence") or "",
                    intent=state.get("intent") or "doc_qa",
                    sources=list(state.get("sources_used") or []),
                )
                return {**state, "answer": answer, "used_llm": True, "self_check_critique": None}
            except Exception as exc:  # noqa: BLE001
                if (state.get("meta") or {}).get("mode") == "conversation":
                    return {**state, "answer": "(LLM unavailable)", "used_llm": False}
                fallback = (
                    f"(LLM format failed: {exc})\n\n"
                    f"intent={state.get('intent')} | sources={state.get('sources_used')}\n\n"
                    f"{state.get('evidence', '')}"
                )
                return {**state, "answer": fallback, "used_llm": False}

        if (state.get("meta") or {}).get("mode") == "conversation":
            return {**state, "answer": "(LLM not configured)", "used_llm": False}

        header = (
            f"intent={state.get('intent')} | "
            f"sources={state.get('sources_used')} | "
            f"insufficient={state.get('insufficient', False)}"
        )
        answer = f"{header}\n\n{state.get('evidence', '')}".strip()
        return {**state, "answer": answer, "used_llm": False}

    def self_check(state: AgentState) -> AgentState:
        if state.get("answer") and llm_on and not state.get("insufficient"):
            sc_retries = state.get("self_check_retries", 0)
            if sc_retries < _MAX_SELF_CHECK_RETRIES:
                check = llm.self_check_answer(
                    query=state["query"],
                    answer=state["answer"],
                )
                if check and check.get("action") == "incomplete":
                    return {
                        **state,
                        "next_action": "format_answer",
                        "self_check_critique": check.get("critique", ""),
                        "self_check_retries": sc_retries + 1,
                    }
        return {**state, "next_action": "end"}

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("validate_evidence", validate_evidence)
    graph.add_node("checklist", checklist)
    graph.add_node("format_answer", format_answer)
    graph.add_node("self_check", self_check)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        lambda s: s.get("next_action", "retrieve"),
        {"retrieve": "retrieve", "checklist": "checklist", "format_answer": "format_answer"},
    )
    graph.add_edge("retrieve", "validate_evidence")
    graph.add_edge("checklist", "format_answer")
    graph.add_conditional_edges(
        "validate_evidence",
        lambda s: s.get("next_action", "format_answer"),
        {"retrieve": "retrieve", "format_answer": "format_answer"},
    )
    graph.add_edge("format_answer", "self_check")
    graph.add_conditional_edges(
        "self_check",
        lambda s: s.get("next_action", "end"),
        {"format_answer": "format_answer", "end": END},
    )

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
