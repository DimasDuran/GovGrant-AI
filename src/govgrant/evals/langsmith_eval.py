"""LangSmith evaluation harness for GovGrant-AI.

Usage:
    python -m govgrant.evals.langsmith_eval sync   # sync dataset
    python -m govgrant.evals.langsmith_eval run     # run eval on all cases
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from govgrant.agent.graph import run_agent
from govgrant.core.telemetry import langsmith_client


def _golden_cases() -> list[dict]:
    """Load golden eval cases from ``data/eval/``."""
    cases: list[dict] = []
    eval_dir = Path(__file__).resolve().parents[3] / "data" / "eval"
    for f in sorted(eval_dir.glob("case_*.json")):
        cases.append(json.loads(f.read_text()))
    return cases


def sync_dataset(project: str = "govgrant") -> None:
    """Sync golden eval cases as a LangSmith dataset."""
    client = langsmith_client()
    if client is None:
        print("[error] LangSmith not configured — set LANGSMITH_API_KEY", file=sys.stderr)
        sys.exit(1)

    cases = _golden_cases()
    dataset = client.create_dataset(
        dataset_name=f"{project}-golden-eval",
        description="Golden eval cases for GovGrant-AI: intent routing + answer quality",
    )
    for case in cases:
        client.create_example(
            dataset_id=dataset.id,
            inputs={"query": case["query"]},
            outputs={
                "expected_intent": case["expected_intent"],
                "expected_sources": case.get("expected_sources", []),
            },
        )
    print(f"Synced {len(cases)} examples to LangSmith dataset '{dataset.name}'")


def run_eval(project: str = "govgrant") -> None:
    """Run agent on golden cases and log results to LangSmith."""
    client = langsmith_client()
    if client is None:
        print("[error] LangSmith not configured", file=sys.stderr)
        sys.exit(1)

    cases = _golden_cases()
    for case in cases:
        result = run_agent(
            case["query"],
            tenant_id="local-dev",
            doc_id=case.get("doc_id"),
            agency=case.get("agency"),
        )

        run_id = result.get("meta", {}).get("langsmith_run_id")
        if run_id:
            client.create_feedback(
                run_id=run_id,
                key="intent_match",
                score=1.0 if result.get("intent") == case["expected_intent"] else 0.0,
            )
            client.create_feedback(
                run_id=run_id,
                key="insufficient",
                score=1.0 if not result.get("insufficient") else 0.0,
            )

        status = "✓" if result.get("intent") == case["expected_intent"] else "✗"
        print(f"  {status} {case['query'][:60]:60s} → {result.get('intent')}")


def main() -> None:
    from govgrant.core.telemetry import setup_telemetry

    setup_telemetry()

    if len(sys.argv) < 2 or sys.argv[1] not in ("sync", "run"):
        print("Usage: python -m govgrant.evals.langsmith_eval sync|run", file=sys.stderr)
        sys.exit(2)

    if sys.argv[1] == "sync":
        sync_dataset()
    else:
        run_eval()


if __name__ == "__main__":
    main()
