"""Unit tests for quality gate evaluation (no Qdrant)."""

from __future__ import annotations

from govgrant.rag.eval.gates import (
    evaluate_checklist_report,
    evaluate_eval_report,
    load_thresholds,
)


def test_load_thresholds_has_router():
    cfg = load_thresholds()
    assert "router" in cfg["gates"]
    assert cfg["gates"]["router"]["min_pass_rate"] >= 90


def test_evaluate_eval_report_pass():
    report = {
        "total": 180,
        "passed": 180,
        "failed": 0,
        "metrics": {
            "pass_rate": 100.0,
            "avg_recall": 0.99,
            "avg_precision": 1.0,
        },
        "backend": "router",
    }
    gate_cfg = {
        "min_pass_rate": 98.0,
        "min_avg_recall": 0.95,
        "min_avg_precision": 0.95,
        "min_total_cases": 100,
    }
    result = evaluate_eval_report(report, gate_id="router", gate_cfg=gate_cfg)
    assert result.passed
    assert all(c.ok for c in result.checks)


def test_evaluate_eval_report_fail_recall():
    report = {
        "total": 180,
        "passed": 170,
        "failed": 10,
        "metrics": {
            "pass_rate": 94.4,
            "avg_recall": 0.90,
            "avg_precision": 1.0,
        },
    }
    gate_cfg = {
        "min_pass_rate": 98.0,
        "min_avg_recall": 0.95,
        "min_avg_precision": 0.95,
        "min_total_cases": 100,
    }
    result = evaluate_eval_report(report, gate_id="router", gate_cfg=gate_cfg)
    assert not result.passed
    failed_names = {c.name for c in result.checks if not c.ok}
    assert "pass_rate" in failed_names or "avg_recall" in failed_names


def test_checklist_critical_fail_gate():
    run = {
        "items": [
            {"status": "pass", "severity": "critical"},
            {"status": "fail", "severity": "critical"},
            {"status": "fail", "severity": "low"},
        ],
        "summary": {"pass": 1, "fail": 2},
    }
    result = evaluate_checklist_report(
        run, gate_id="checklist_corpus", gate_cfg={"max_critical_fail": 0}
    )
    assert not result.passed
    result_ok = evaluate_checklist_report(
        run, gate_id="checklist_corpus", gate_cfg={"max_critical_fail": 1}
    )
    assert result_ok.passed
