"""System prompts for the GovGrant AI ChatLLM.

Each module in this package exports prompt constants consumed by
ChatLLM methods.
"""

from __future__ import annotations

from govgrant.agent.tools.prompts.answer import (
    ANSWER_SYSTEM_BASE,
    ANSWER_SYSTEM_FOOTER,
    ANSWER_SYSTEM_INTENT_RULES,
)
from govgrant.agent.tools.prompts.judge import JUDGE_SYSTEM_TEMPLATE
from govgrant.agent.tools.prompts.routing import ROUTING_SYSTEM
from govgrant.agent.tools.prompts.self_check import SELF_CHECK_SYSTEM

__all__ = [
    "ANSWER_SYSTEM_BASE",
    "ANSWER_SYSTEM_FOOTER",
    "ANSWER_SYSTEM_INTENT_RULES",
    "JUDGE_SYSTEM_TEMPLATE",
    "ROUTING_SYSTEM",
    "SELF_CHECK_SYSTEM",
]
