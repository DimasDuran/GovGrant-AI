"""Anthropic Claude chat client (Haiku by default) for agent answer generation."""

from __future__ import annotations

import re

from langsmith import traceable

from govgrant.agent.tools import (
    ANSWER_SYSTEM_BASE,
    ANSWER_SYSTEM_FOOTER,
    ANSWER_SYSTEM_INTENT_RULES,
    JUDGE_SYSTEM_TEMPLATE,
    JUDGE_TOOLS,
    ROUTING_SYSTEM,
    ROUTING_TOOLS,
    SELF_CHECK_SYSTEM,
    SELF_CHECK_TOOLS,
)
from govgrant.rag.config import Settings, get_settings


class ChatLLM:
    """Thin wrapper around Anthropic Messages API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.anthropic_api_key
        self.model = self.settings.chat_model
        self.max_tokens = self.settings.chat_max_tokens
        self._client = None
        if self.api_key:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return bool(self.api_key and self._client and self.settings.chat_enabled)

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        if not self.available or self._client is None:
            raise RuntimeError("Chat LLM unavailable. Set ANTHROPIC_API_KEY and CHAT_ENABLED=true.")
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: list[str] = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    @traceable(run_type="llm")
    def classify_with_tools(self, query: str) -> dict | None:
        """Use Anthropic tool-use to select retrieval route.

        Returns {"tool": str, "arguments": dict} or None if LLM unavailable / no tool chosen.
        """
        if not self.available or self._client is None:
            return None
        tools = ROUTING_TOOLS
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=200,
                system=ROUTING_SYSTEM,
                messages=[{"role": "user", "content": query}],
                tools=tools,
                tool_choice={"type": "any"},
            )
            for block in msg.content:
                if getattr(block, "type", None) == "tool_use":
                    return {"tool": block.name, "input": block.input}
        except Exception:  # noqa: BLE001
            pass
        return None

    @traceable(run_type="llm")
    def judge_evidence(
        self,
        *,
        query: str,
        evidence: str,
        retry_count: int,
        max_tokens: int = 300,
    ) -> dict | None:
        """LLM tool-calling judge: is the evidence sufficient to answer?

        Returns {"action": "sufficient", "reason": "..."}
        or {"action": "retry", "reason": "...", "suggested_query": "..."}
        or None if LLM unavailable / error.
        """
        if not self.available or self._client is None:
            return None
        tools = JUDGE_TOOLS
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=JUDGE_SYSTEM_TEMPLATE.format(retry=retry_count + 1),
                messages=[
                    {
                        "role": "user",
                        "content": f"User question: {query}\n\nRetrieved evidence:\n{evidence[:12000]}",
                    }
                ],
                tools=tools,
                tool_choice={"type": "any"},
            )
            for block in msg.content:
                if getattr(block, "type", None) == "tool_use":
                    if block.name == "mark_sufficient":
                        return {"action": "sufficient", "reason": block.input.get("reason", "")}
                    if block.name == "request_more_evidence":
                        return {
                            "action": "retry",
                            "reason": block.input.get("reason", ""),
                            "suggested_query": block.input.get("suggested_query", query),
                        }
        except Exception:  # noqa: BLE001
            pass
        return None

    @traceable(run_type="llm")
    def self_check_answer(
        self,
        *,
        query: str,
        answer: str,
        max_tokens: int = 200,
    ) -> dict | None:
        """LLM verifies answer covers the user's question before returning.

        Returns {"action": "complete", "reason": "..."}
        or {"action": "incomplete", "reason": "...", "critique": "..."}
        or None if LLM unavailable / error.
        """
        if not self.available or self._client is None:
            return None
        tools = SELF_CHECK_TOOLS
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=SELF_CHECK_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": f"User question:\n{query}\n\nDraft answer:\n{answer}",
                    }
                ],
                tools=tools,
                tool_choice={"type": "any"},
            )
            for block in msg.content:
                if getattr(block, "type", None) == "tool_use":
                    if block.name == "answer_complete":
                        return {"action": "complete", "reason": block.input.get("reason", "")}
                    if block.name == "answer_incomplete":
                        return {
                            "action": "incomplete",
                            "reason": block.input.get("reason", ""),
                            "critique": block.input.get("critique", ""),
                        }
        except Exception:  # noqa: BLE001
            pass
        return None

    @traceable(run_type="llm")
    def answer_from_evidence(
        self,
        *,
        query: str,
        evidence: str,
        intent: str,
        sources: list[str],
    ) -> str:
        system = ANSWER_SYSTEM_BASE
        intent_rule = ANSWER_SYSTEM_INTENT_RULES.get(intent)
        if intent_rule:
            system += intent_rule
        system += ANSWER_SYSTEM_FOOTER
        user = (
            f"Intent: {intent}\n"
            f"Sources used: {', '.join(sources) or 'n/a'}\n\n"
            f"User question:\n{query}\n\n"
            f"Retrieved evidence:\n{evidence[:48000]}\n\n"
            "Write the final answer for the user.\n"
            "- Cover every sub-question the user asked.\n"
            "- Stay within the asked scope (no extra volumes/programs).\n"
            "- If a later passage contains the answer, use it—do not stop at early pages.\n"
            "- Cite pages/files from the evidence when possible."
        )
        answer = self.complete(
            system=system,
            user=user,
            temperature=0.15,
            max_tokens=max(self.max_tokens, 1800),
        )
        return strip_unsolicited_digressions(answer, query=query)


def strip_unsolicited_digressions(answer: str, *, query: str) -> str:
    """
    Post-filter: drop trailing sections that digress into volumes/programs
    the user did not ask about (improves precision on multi-hop eval).
    """
    if not answer or not answer.strip():
        return answer
    q = (query or "").lower()
    # Topics the user would have to mention for these digressions to be kept
    allow = {
        "volume 5": any(
            k in q
            for k in (
                "volume 5",
                "supporting document",
                "supporting docs",
                "documentos de apoyo",
                "subcontract pricing",
                "data rights assertion",
            )
        ),
        "volume 4": any(k in q for k in ("volume 4", "ccr", "commercialization report")),
        "cost volume": any(
            k in q
            for k in (
                "cost volume",
                "budget",
                "1,800,000",
                "1800000",
                "volumen de costos",
                "plantilla de costos",
            )
        ),
        "volume 6": "volume 6" in q or "fraud" in q,
        "volume 7": "volume 7" in q or "foreign affiliation" in q,
    }

    # Split on markdown headings; drop unsolicited volume digression sections
    parts = re.split(r"(?=^#{1,3}\s+)", answer, flags=re.M)
    if len(parts) <= 1:
        return _strip_inline_volume5_footer(answer, allow=allow)

    kept: list[str] = []
    digression = re.compile(
        r"volume\s*5|supporting documents?|company commercialization report|\bCCR\b|"
        r"cost volume template|volumen\s*5|documentos de apoyo",
        re.I,
    )
    for part in parts:
        low = part.lower()
        is_heading_digression = bool(
            re.match(r"^#{1,3}\s+.*", part.strip()) and digression.search(part[:200])
        )
        if (
            is_heading_digression
            and ("volume 5" in low or "supporting document" in low)
            and not allow["volume 5"]
        ):
            continue
        if (
            is_heading_digression
            and ("volume 4" in low or "ccr" in low or "commercialization report" in low)
            and not allow["volume 4"]
            and "commercialization strategy" not in low
        ):
            continue
        if "cost volume" in low and not allow["cost volume"]:
            continue
        kept.append(part)

    cleaned = "".join(kept).strip() or answer
    return _strip_inline_volume5_footer(cleaned, allow=allow)


def _strip_inline_volume5_footer(answer: str, *, allow: dict[str, bool]) -> str:
    """Remove Volume 5 / CCR digressions when the user did not ask about them."""
    out = answer
    # Drop last section if it's only a Volume 5/CCR summary
    patterns = [
        r"\n##+\s*[^\n]*(?:Volume\s*5|Supporting Documents|CCR|Company Commercialization)[^\n]*\n[\s\S]*$",
        r"\n\*\*?(?:Resumen de documentación en Volume 5|Volume 5 \(Supporting)[^\n]*\n[\s\S]*$",
    ]
    if not allow.get("volume 5"):
        for pat in patterns:
            out = re.sub(pat, "", out, flags=re.I)
        # Drop bullet/paragraph lines that only push Volume 5 packaging
        out = re.sub(
            r"(?m)^[ \t]*(?:[-*]|\d+\.)[ \t]*[^\n]*(?:Volume\s*5|Supporting Documents Volume)[^\n]*\n?",
            "",
            out,
            flags=re.I,
        )
        out = re.sub(
            r"(?m)^[ \t]*[^\n]{0,40}\bVolume\s*5\b[^\n]*(?:Supporting|upload|include|puedes|también)[^\n]*\n?",
            "",
            out,
            flags=re.I,
        )
    if not allow.get("volume 4"):
        out = re.sub(
            r"(?m)^[ \t]*(?:[-*]|\d+\.)[ \t]*[^\n]*(?:\bCCR\b|Company Commercialization Report)[^\n]*\n?",
            "",
            out,
            flags=re.I,
        )
    if not allow.get("cost volume"):
        out = re.sub(
            r"(?m)^[ \t]*(?:[-*]|\d+\.)[ \t]*[^\n]*Cost Volume template[^\n]*\n?",
            "",
            out,
            flags=re.I,
        )
    # collapse excess blank lines
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.rstrip() + ("\n" if answer.endswith("\n") else "")
