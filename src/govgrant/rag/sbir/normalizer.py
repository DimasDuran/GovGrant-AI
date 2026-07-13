"""Normalize raw SBIR API / fixture payloads into TopicDocument list."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from govgrant.rag.sbir.models import TopicDocument


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _hash_text(*parts: str) -> str:
    blob = "||".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def normalize_solicitations(
    raw_items: list[dict[str, Any]],
    *,
    source: str = "api",
    stale: bool = False,
) -> list[TopicDocument]:
    """
    Expand solicitations into one TopicDocument per nested topic.

    If a solicitation has no nested topics, emit one synthetic topic from the
    solicitation title/description fields.
    """
    out: list[TopicDocument] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        topics = _as_list(
            item.get("solicitation_topics") or item.get("topics") or item.get("topic")
        )
        if not topics:
            topics = [
                {
                    "topic_title": item.get("solicitation_title")
                    or item.get("title")
                    or "Untitled topic",
                    "topic_number": item.get("topic_number")
                    or item.get("solicitation_number")
                    or item.get("id"),
                    "topic_description": item.get("description")
                    or item.get("topic_description")
                    or "",
                    "topic_code": item.get("topic_code"),
                }
            ]

        for t in topics:
            if not isinstance(t, dict):
                # sometimes API returns plain strings
                t = {"topic_title": str(t), "topic_description": ""}

            topic_id = _str(
                t.get("topic_number")
                or t.get("topic_id")
                or t.get("id")
                or item.get("topic_number")
            )
            title = _str(t.get("topic_title") or t.get("title") or item.get("solicitation_title"))
            if not topic_id or not title:
                # deterministic fallback id
                seed = f"{item.get('solicitation_number')}|{title}"
                topic_id = topic_id or hashlib.sha1(seed.encode()).hexdigest()[:10]
                title = title or "Untitled topic"

            desc = _str(t.get("topic_description") or t.get("description") or "") or ""
            status = (
                _str(item.get("current_status") or item.get("status") or t.get("status") or "open")
                or "open"
            )

            due = _as_list(item.get("application_due_date") or item.get("application_due_dates"))
            due_dates = [str(d) for d in due if d is not None]

            doc = TopicDocument(
                topic_id=str(topic_id),
                topic_title=title,
                topic_description=desc,
                topic_code=_str(t.get("topic_code") or t.get("code")),
                solicitation_title=_str(item.get("solicitation_title") or item.get("title")),
                solicitation_number=_str(item.get("solicitation_number")),
                program=_str(item.get("program") or t.get("program")),
                phase=_str(item.get("phase") or t.get("phase")),
                agency=_str(item.get("agency") or t.get("agency")),
                branch=_str(item.get("branch") or t.get("branch")),
                solicitation_year=_str(item.get("solicitation_year") or item.get("year")),
                release_date=_str(item.get("release_date")),
                open_date=_str(item.get("open_date")),
                close_date=_str(item.get("close_date")),
                application_due_dates=due_dates,
                status=status.lower(),
                solicitation_agency_url=_str(
                    item.get("solicitation_agency_url") or item.get("agency_url")
                ),
                stale=stale,
                source=source,
                content_hash=_hash_text(title, desc, status, str(item.get("close_date"))),
            )
            out.append(doc)
    return out


def load_fixture_json(path: str | Any) -> list[dict[str, Any]]:
    from pathlib import Path

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"Fixture must be a list, got {type(data)}")
    return data
