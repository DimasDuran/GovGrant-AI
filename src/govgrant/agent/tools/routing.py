"""Routing tools — the LLM selects one to decide which source to query."""

from __future__ import annotations

ROUTING_TOOLS: list[dict] = [
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
                    "description": "Search terms \u2014 keep the user\u2019s own words",
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
            "SBIR.gov \u2014 topic descriptions, agency, phase, deadlines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Search terms \u2014 technology area or topic keywords",
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
