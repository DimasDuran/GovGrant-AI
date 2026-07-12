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
