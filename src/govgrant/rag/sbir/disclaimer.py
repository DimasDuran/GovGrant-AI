"""Mandatory disclaimer for SBIR.gov-sourced answers."""

SBIR_DISCLAIMER = (
    "Disclaimer: Topics listed on SBIR.gov may be copies of agency solicitations "
    "and are not necessarily the latest official version. Always verify the "
    "solicitation on the awarding agency's official site and confirm dates, "
    "eligibility, and submission instructions before applying. "
    "Topic pages: https://www.sbir.gov/topics/{topic_id}"
)


def with_disclaimer(text: str, *, topic_ids: list[str] | None = None) -> str:
    """Append disclaimer, optionally listing citation URLs for topic ids."""
    lines = [text.rstrip(), "", "---", SBIR_DISCLAIMER]
    if topic_ids:
        lines.append("Citations:")
        for tid in topic_ids:
            lines.append(f"  - https://www.sbir.gov/topics/{tid}")
    return "\n".join(lines)
