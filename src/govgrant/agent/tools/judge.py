"""Evidence-judge tools — the LLM decides whether retrieved evidence suffices."""

from __future__ import annotations

JUDGE_TOOLS: list[dict] = [
    {
        "name": "mark_sufficient",
        "description": (
            "The retrieved evidence is sufficient to answer the user's question. "
            "Call this when evidence directly addresses the query with relevant "
            "details \u2014 even if partial or incomplete."
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
