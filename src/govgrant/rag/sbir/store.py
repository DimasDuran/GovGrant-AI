"""SQLite structured store for SBIR topics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from govgrant.rag.sbir.models import TopicDocument


class SBIRStructuredStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sbir_topics (
                    topic_id TEXT PRIMARY KEY,
                    topic_title TEXT NOT NULL,
                    topic_description TEXT,
                    topic_code TEXT,
                    solicitation_title TEXT,
                    solicitation_number TEXT,
                    program TEXT,
                    phase TEXT,
                    agency TEXT,
                    branch TEXT,
                    solicitation_year TEXT,
                    release_date TEXT,
                    open_date TEXT,
                    close_date TEXT,
                    application_due_dates_json TEXT,
                    status TEXT,
                    solicitation_agency_url TEXT,
                    citation_uri TEXT,
                    stale INTEGER DEFAULT 0,
                    source TEXT,
                    content_hash TEXT,
                    payload_json TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_sbir_agency ON sbir_topics(agency);
                CREATE INDEX IF NOT EXISTS idx_sbir_status ON sbir_topics(status);
                CREATE INDEX IF NOT EXISTS idx_sbir_program ON sbir_topics(program);
                CREATE TABLE IF NOT EXISTS sbir_sync_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )

    def upsert_topics(self, topics: list[TopicDocument]) -> int:
        if not topics:
            return 0
        with self._connect() as conn:
            for t in topics:
                conn.execute(
                    """
                    INSERT INTO sbir_topics (
                        topic_id, topic_title, topic_description, topic_code,
                        solicitation_title, solicitation_number, program, phase,
                        agency, branch, solicitation_year, release_date, open_date,
                        close_date, application_due_dates_json, status,
                        solicitation_agency_url, citation_uri, stale, source,
                        content_hash, payload_json, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                    ON CONFLICT(topic_id) DO UPDATE SET
                        topic_title=excluded.topic_title,
                        topic_description=excluded.topic_description,
                        topic_code=excluded.topic_code,
                        solicitation_title=excluded.solicitation_title,
                        solicitation_number=excluded.solicitation_number,
                        program=excluded.program,
                        phase=excluded.phase,
                        agency=excluded.agency,
                        branch=excluded.branch,
                        solicitation_year=excluded.solicitation_year,
                        release_date=excluded.release_date,
                        open_date=excluded.open_date,
                        close_date=excluded.close_date,
                        application_due_dates_json=excluded.application_due_dates_json,
                        status=excluded.status,
                        solicitation_agency_url=excluded.solicitation_agency_url,
                        citation_uri=excluded.citation_uri,
                        stale=excluded.stale,
                        source=excluded.source,
                        content_hash=excluded.content_hash,
                        payload_json=excluded.payload_json,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        t.topic_id,
                        t.topic_title,
                        t.topic_description,
                        t.topic_code,
                        t.solicitation_title,
                        t.solicitation_number,
                        t.program,
                        t.phase,
                        t.agency,
                        t.branch,
                        t.solicitation_year,
                        t.release_date,
                        t.open_date,
                        t.close_date,
                        json.dumps(t.application_due_dates),
                        t.status,
                        t.solicitation_agency_url,
                        t.citation_uri,
                        1 if t.stale else 0,
                        t.source,
                        t.content_hash,
                        t.model_dump_json(),
                    ),
                )
        return len(topics)

    def get(self, topic_id: str) -> TopicDocument | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM sbir_topics WHERE topic_id=?",
                (topic_id,),
            ).fetchone()
        if not row:
            return None
        return TopicDocument.model_validate_json(row["payload_json"])

    def list_topics(
        self,
        *,
        status: str | None = "open",
        agency: str | None = None,
        program: str | None = None,
        limit: int = 100,
    ) -> list[TopicDocument]:
        sql = "SELECT payload_json FROM sbir_topics WHERE 1=1"
        params: list[Any] = []
        if status:
            sql += " AND lower(status)=?"
            params.append(status.lower())
        if agency:
            sql += " AND upper(agency)=?"
            params.append(agency.upper())
        if program:
            sql += " AND upper(program)=?"
            params.append(program.upper())
        sql += " ORDER BY agency, topic_title LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [TopicDocument.model_validate_json(r["payload_json"]) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return int(
                conn.execute("SELECT COUNT(*) AS c FROM sbir_topics").fetchone()["c"]
            )

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sbir_sync_meta(key, value) VALUES(?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM sbir_sync_meta WHERE key=?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def mark_all_stale(self) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE sbir_topics SET stale=1")
