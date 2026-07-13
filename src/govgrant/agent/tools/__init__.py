"""GovGrant agent tooling.

Exports:
  - Anthropic tool definitions (ROUTING_TOOLS, JUDGE_TOOLS, SELF_CHECK_TOOLS)
  - System prompts (ROUTING_SYSTEM, JUDGE_SYSTEM_TEMPLATE, SELF_CHECK_SYSTEM,
    ANSWER_SYSTEM_BASE, ANSWER_SYSTEM_FOOTER)
  - RagToolBelt — façade over the LlamaIndex RAG stack for LangGraph nodes
"""

from __future__ import annotations

from govgrant.agent.tools.belt import RagToolBelt
from govgrant.agent.tools.judge import JUDGE_TOOLS
from govgrant.agent.tools.prompts import (
    ANSWER_SYSTEM_BASE,
    ANSWER_SYSTEM_FOOTER,
    ANSWER_SYSTEM_INTENT_RULES,
    JUDGE_SYSTEM_TEMPLATE,
    ROUTING_SYSTEM,
    SELF_CHECK_SYSTEM,
)
from govgrant.agent.tools.routing import ROUTING_TOOLS
from govgrant.agent.tools.self_check import SELF_CHECK_TOOLS

__all__ = [
    "ANSWER_SYSTEM_BASE",
    "ANSWER_SYSTEM_FOOTER",
    "ANSWER_SYSTEM_INTENT_RULES",
    "JUDGE_SYSTEM_TEMPLATE",
    "JUDGE_TOOLS",
    "ROUTING_SYSTEM",
    "ROUTING_TOOLS",
    "SELF_CHECK_SYSTEM",
    "SELF_CHECK_TOOLS",
    "RagToolBelt",
]
