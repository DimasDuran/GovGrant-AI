"""Canonical SBIR topic documents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TopicDocument(BaseModel):
    """Normalized topic used for structured store + RAG index."""

    topic_id: str
    topic_title: str
    topic_description: str = ""
    topic_code: str | None = None
    solicitation_title: str | None = None
    solicitation_number: str | None = None
    program: str | None = None
    phase: str | None = None
    agency: str | None = None
    branch: str | None = None
    solicitation_year: str | None = None
    release_date: str | None = None
    open_date: str | None = None
    close_date: str | None = None
    application_due_dates: list[str] = Field(default_factory=list)
    status: str = "open"
    solicitation_agency_url: str | None = None
    citation_uri: str = ""
    stale: bool = False
    source: str = "api"  # api | fixture | cache
    content_hash: str = ""

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        if not self.citation_uri and self.topic_id:
            self.citation_uri = f"https://www.sbir.gov/topics/{self.topic_id}"

    def to_rag_text(self) -> str:
        parts = [
            f"Topic: {self.topic_title}",
            f"Topic ID: {self.topic_id}",
        ]
        if self.topic_code:
            parts.append(f"Topic code: {self.topic_code}")
        if self.agency:
            parts.append(f"Agency: {self.agency}")
        if self.branch:
            parts.append(f"Branch: {self.branch}")
        if self.program:
            parts.append(f"Program: {self.program}")
        if self.phase:
            parts.append(f"Phase: {self.phase}")
        if self.status:
            parts.append(f"Status: {self.status}")
        if self.open_date or self.close_date:
            parts.append(
                f"Open: {self.open_date or '?'} | Close: {self.close_date or '?'}"
            )
        if self.solicitation_title:
            parts.append(f"Solicitation: {self.solicitation_title}")
        if self.solicitation_number:
            parts.append(f"Solicitation number: {self.solicitation_number}")
        if self.topic_description:
            parts.append("")
            parts.append(self.topic_description)
        return "\n".join(parts)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_type": "sbir_topic",
            "modality": "prose",
            "lane": "sbir_topic",
            "topic_id": self.topic_id,
            "topic_code": self.topic_code or "",
            "topic_title": self.topic_title,
            "agency": (self.agency or "").upper(),
            "branch": (self.branch or "").upper(),
            "program": (self.program or "").upper(),
            "phase": (self.phase or "").upper(),
            "status": (self.status or "").lower(),
            "solicitation_year": self.solicitation_year or "",
            "open_date": self.open_date or "",
            "close_date": self.close_date or "",
            "solicitation_number": self.solicitation_number or "",
            "citation_uri": self.citation_uri,
            "stale": bool(self.stale),
            "source": self.source,
            "content_hash": self.content_hash,
            "file_name": f"sbir-topic-{self.topic_id}",
            # public corpus — no tenant isolation
            "tenant_id": "public-sbir",
            "gg_doc_id": f"sbir-topic-{self.topic_id}",
            "doc_id": f"sbir-topic-{self.topic_id}",
        }
