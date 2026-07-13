"""Anthropic Claude chat client (Haiku by default) for agent answer generation."""

from __future__ import annotations

import re

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
            raise RuntimeError(
                "Chat LLM unavailable. Set ANTHROPIC_API_KEY and CHAT_ENABLED=true."
            )
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

    def classify_with_tools(self, query: str) -> dict | None:
        """Use Anthropic tool-use to select retrieval route.

        Returns {"tool": str, "arguments": dict} or None if LLM unavailable / no tool chosen.
        """
        if not self.available or self._client is None:
            return None
        tools = [
            {
                "name": "search_documents",
                "description": (
                    "Search SBIR/STTR agency documents (DARPA Phase II instructions, "
                    "SBA Policy Directive, SF424 Application Guide) for compliance rules, "
                    "eligibility, proposal instructions, work-share, milestone plans, "
                    "commercialization strategy, page limits, funding restrictions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Search terms — keep the user's own words",
                        }
                    },
                    "required": ["search_query"],
                },
            },
            {
                "name": "search_tables",
                "description": (
                    "Search structured table data extracted from PDFs: budget tables, "
                    "proposal forms, data rights assertion matrices, row/column data."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Search terms",
                        }
                    },
                    "required": ["search_query"],
                },
            },
            {
                "name": "search_sbir_topics",
                "description": (
                    "Search open SBIR/STTR funding topics and solicitations from "
                    "SBIR.gov — topic descriptions, agency, phase, deadlines."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Search terms — technology area or topic keywords",
                        },
                        "agency": {
                            "type": "string",
                            "description": "Agency code: DOD, NIH, NASA, NSF, DARPA, etc.",
                        },
                    },
                    "required": ["search_query"],
                },
            },
            {
                "name": "cross_check",
                "description": (
                    "Cross-reference user proposal or draft content with open SBIR topics "
                    "to check alignment, eligibility fit, and topic matching."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The proposal description or technology keywords",
                        },
                        "agency": {
                            "type": "string",
                            "description": "Target agency code if known",
                        },
                    },
                    "required": ["search_query"],
                },
            },
            {
                "name": "compliance_checklist",
                "description": (
                    "Run the SBIR/STTR compliance checklist against agency documents. "
                    "Use this when the user asks to run a compliance review, checklist, "
                    "or audit of their proposal against DARPA Phase II instructions, "
                    "SBA Policy Directive, or SF424 Application Guide requirements."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "packages": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["darpa", "sba", "sf424"],
                            },
                            "description": (
                                "Which compliance packages to check. "
                                "darpa = DARPA Phase II Proposal Instructions, "
                                "sba = SBA SBIR/STTR Policy Directive, "
                                "sf424 = NIH SF424 Application Guide. "
                                "Default to all three unless the user specifies an agency."
                            ),
                        },
                        "program": {
                            "type": "string",
                            "enum": ["sbir", "sttr"],
                            "description": "SBIR or STTR program (default: sbir)",
                        },
                    },
                    "required": ["packages"],
                },
            },
        ]
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=200,
                system=(
                    "You are a routing classifier for a SBIR/STTR compliance assistant. "
                    "Select the most appropriate retrieval tool for the user's question. "
                    "Always choose one tool — do not answer directly."
                ),
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
        tools = [
            {
                "name": "mark_sufficient",
                "description": (
                    "The retrieved evidence is sufficient to answer the user's question. "
                    "Call this when evidence directly addresses the query with relevant "
                    "details — even if partial or incomplete."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Brief explanation of why the evidence suffices",
                        }
                    },
                    "required": ["reason"],
                },
            },
            {
                "name": "request_more_evidence",
                "description": (
                    "The retrieved evidence does NOT sufficiently answer the user's "
                    "question. Call this to request a new retrieval with a reformulated "
                    "query. Use when evidence is off-topic, too generic, or missing "
                    "specific details the user asked about."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why the current evidence is insufficient",
                        },
                        "suggested_query": {
                            "type": "string",
                            "description": (
                                "A reformulated search query that should retrieve "
                                "better evidence. Focus on keywords from the user's "
                                "question that were missing in the results."
                            ),
                        },
                    },
                    "required": ["reason", "suggested_query"],
                },
            },
        ]
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=(
                    "You are an evidence judge for a SBIR/STTR compliance assistant. "
                    "Review the retrieved evidence and decide whether it is sufficient "
                    "to answer the user's question. "
                    f"Retry attempt {retry_count + 1} of up to 3.\n\n"
                    "Rules:\n"
                    "- Choose mark_sufficient if evidence addresses the question, "
                    "even if partially.\n"
                    "- Choose request_more_evidence if evidence is off-topic, empty, "
                    "or missing key information.\n"
                    "- Always pick one tool — do not answer directly."
                ),
                messages=[{"role": "user", "content": f"User question: {query}\n\nRetrieved evidence:\n{evidence[:12000]}"}],
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
        tools = [
            {
                "name": "answer_complete",
                "description": "The answer fully addresses the user's question. Call this when the answer covers every aspect the user asked about.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Brief confirmation"},
                    },
                    "required": ["reason"],
                },
            },
            {
                "name": "answer_incomplete",
                "description": "The answer does NOT fully address the user's question. Call this when the answer missed sub-questions, went off-topic, or is too vague.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why the answer is incomplete"},
                        "critique": {
                            "type": "string",
                            "description": "Specific guidance on what to add or change in the revised answer",
                        },
                    },
                    "required": ["reason", "critique"],
                },
            },
        ]
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=(
                    "You are a quality checker for a SBIR/STTR compliance assistant. "
                    "Review the answer against the user's question. Verify: "
                    "(1) every sub-question is addressed, "
                    "(2) the answer stays within the asked scope, "
                    "(3) no critical detail from the question is ignored.\n"
                    "Choose answer_complete if satisfactory, answer_incomplete otherwise."
                ),
                messages=[
                    {"role": "user", "content": f"User question:\n{query}\n\nDraft answer:\n{answer}"}
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

    def answer_from_evidence(
        self,
        *,
        query: str,
        evidence: str,
        intent: str,
        sources: list[str],
    ) -> str:
        system = (
            "You are GovGrant AI, a specialized AI assistant for U.S. SBIR/STTR "
            "grant compliance. Behave like Claude or ChatGPT: clear, natural, and direct — "
            "but you are a **vertical** product: only SBIR/STTR / federal small-business "
            "innovation funding compliance, proposal instructions, and related agency docs.\n"
            "Rules:\n"
            "1. For this turn you MUST ground the answer in the provided evidence. "
            "If evidence is weak or off-topic, say so briefly and ask a clarifying question.\n"
            "2. Be precise and concise (short paragraphs + bullets when useful).\n"
            "3. Cite sources inline using file names, page numbers, or "
            "https://www.sbir.gov/topics/{id} when present in evidence.\n"
            "4. Never invent award amounts, deadlines, eligibility, or proposal content.\n"
            "5. Prefer the highest-relevance evidence that directly answers the question; "
            "ignore table-of-contents noise when better pages exist.\n"
            "6. If evidence includes an SBIR disclaimer, keep a short disclaimer.\n"
            "7. Write in the same language as the user question.\n"
            "8. Do not write long capability menus. Answer the question; if out of domain, "
            "say you only cover SBIR/STTR compliance.\n"
            "\n"
            "PRECISION / SCOPE (critical):\n"
            "A. Answer ONLY what the user asked. Do not volunteer extra proposal volumes, "
            "sections, or programs that were not requested.\n"
            "B. Do NOT append digressions such as Volume 5 / Supporting Documents, Volume 4 / CCR, "
            "Cost Volume template details, Fraud/Waste training, or full proposal volume lists "
            "unless the user explicitly asked about those topics. "
            "Even if evidence mentions Volume 5, omit it unless the question is about supporting "
            "documents, data-rights packaging, or subcontract pricing documentation.\n"
            "C. Do NOT add unstated assumptions (e.g. 'Research Institution is typically a "
            "university') unless the evidence says so.\n"
            "D. Optional extras that are directly named in the evidence AND clearly related "
            "to the asked topic (e.g. optional Advocacy Letters when asked about "
            "commercialization strategy) are allowed; unrelated neighboring sections are not.\n"
            "E. Prefer grounded facts over exhaustive document tours.\n"
        )
        if intent == "cross_check":
            system += (
                "F. CROSS-CHECK MODE: Do NOT claim the user's proposal aligns with a topic "
                "unless the evidence contains the user's proposal/abstract text. "
                "If only official topics are present, list matching open topics and say "
                "you cannot judge alignment without the proposal content.\n"
            )
        if intent == "table":
            system += (
                "F. TABLE MODE: Prefer table rows, headers, and structured cell evidence "
                "over general narrative.\n"
            )

        system += (
            "9. If the user asked multiple questions, answer EACH one separately "
            "with clear headings. Only say evidence is missing after checking all "
            "retrieved passages carefully for that sub-question.\n"
            "10. Do not claim the evidence lacks a topic if a later passage covers it.\n"
            "11. When the user asks about optional documents/supporting materials, "
            "list EVERY optional item named in the evidence (e.g. Advocacy Letters AND "
            "Letters of Intent/Commitment). Do not stop after the first example.\n"
            "12. For Other Transaction / milestone questions, list EVERY required "
            "milestone field present in evidence (description, exit criteria, due date, "
            "payment schedule, government data rights).\n"
            "13. Distinguish carefully: 'Transition and Commercialization Strategy' "
            "in Technical Volume (proposal content, 5 pages) is NOT the same as the "
            "Transition and Commercialization Support Program (TCSP) agency program. "
            "Prefer evidence that mentions Technical Volume / Volume 2 / 5 pages for "
            "proposal strategy questions.\n"
            "14. When evidence includes 'THE FOLLOWING PERTAINS TO SBIR ONLY' and "
            "'THE FOLLOWING PERTAINS TO STTR ONLY', report BOTH sections fully "
            "(work-share %, FFRDC rules, funding flow, prohibitions).\n"
             "15. If the answer is not in the evidence, say so briefly—do not pad with "
             "other volumes or general SBIR background.\n"
             "16. When the user greets you (e.g. hola, hello, buenos días, good morning), "
             "respond naturally and briefly. Say a simple greeting back and ask how you "
             "can help — do NOT list capabilities, topics, or things you can do. "
             "Be warm and concise (1-2 sentences max). Never enumerate features. "
             "Do not use emojis.\n"
         )
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
        "volume 4": any(
            k in q for k in ("volume 4", "ccr", "commercialization report")
        ),
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
            re.match(r"^#{1,3}\s+.*", part.strip())
            and digression.search(part[:200])
        )
        if is_heading_digression:
            if "volume 5" in low or "supporting document" in low:
                if not allow["volume 5"]:
                    continue
            if "volume 4" in low or "ccr" in low or "commercialization report" in low:
                if not allow["volume 4"] and "commercialization strategy" not in low:
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
