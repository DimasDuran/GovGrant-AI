"""Self-check tools — the LLM verifies its own answer before returning it."""

from __future__ import annotations

SELF_CHECK_TOOLS: list[dict] = [
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
