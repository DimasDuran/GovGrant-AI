"""
Quality gates over golden eval reports.

Pure, testable checks: load THRESHOLDS.json → evaluate metrics → pass/fail.
Does not run the stack; callers pass a report dict from run_regression.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from govgrant.rag.config import REPO_ROOT

DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "data" / "eval" / "THRESHOLDS.json"


@dataclass(frozen=True)
class GateCheck:
    name: str
    ok: bool
    actual: float | int | str | None
    threshold: float | int | str | None
    message: str


@dataclass
class GateResult:
    gate_id: str
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    report_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
            "report_summary": self.report_summary,
        }


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_THRESHOLDS_PATH
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "gates" not in data:
        raise ValueError(f"Invalid thresholds file (missing gates): {path}")
    return data


def evaluate_eval_report(
    report: dict[str, Any],
    *,
    gate_id: str,
    gate_cfg: dict[str, Any],
) -> GateResult:
    """
    Apply numeric thresholds to a run_regression report.

    Expected report keys: total, passed, failed, metrics{pass_rate, avg_recall, avg_precision}
    """
    metrics = report.get("metrics") or {}
    total = int(report.get("total") or 0)
    passed_n = int(report.get("passed") or 0)
    pass_rate = float(
        metrics.get("pass_rate")
        if metrics.get("pass_rate") is not None
        else (100.0 * passed_n / total if total else 0.0)
    )
    avg_recall = float(metrics.get("avg_recall") or 0.0)
    avg_precision = float(metrics.get("avg_precision") or 1.0)

    checks: list[GateCheck] = []

    def _min(name: str, actual: float, key: str) -> None:
        if key not in gate_cfg or gate_cfg[key] is None:
            return
        thr = float(gate_cfg[key])
        ok = actual + 1e-9 >= thr
        checks.append(
            GateCheck(
                name=name,
                ok=ok,
                actual=round(actual, 4),
                threshold=thr,
                message=f"{name}: {actual:.4g} {'>=' if ok else '<'} {thr}",
            )
        )

    _min("pass_rate", pass_rate, "min_pass_rate")
    _min("avg_recall", avg_recall, "min_avg_recall")
    _min("avg_precision", avg_precision, "min_avg_precision")

    if "min_total_cases" in gate_cfg and gate_cfg["min_total_cases"] is not None:
        thr = int(gate_cfg["min_total_cases"])
        ok = total >= thr
        checks.append(
            GateCheck(
                name="min_total_cases",
                ok=ok,
                actual=total,
                threshold=thr,
                message=f"total cases: {total} {'>=' if ok else '<'} {thr}",
            )
        )

    if not checks:
        checks.append(
            GateCheck(
                name="empty_gate",
                ok=False,
                actual=None,
                threshold=None,
                message="Gate has no numeric thresholds configured",
            )
        )

    summary = {
        "total": total,
        "passed": passed_n,
        "failed": report.get("failed"),
        "pass_rate": pass_rate,
        "avg_recall": avg_recall,
        "avg_precision": avg_precision,
        "backend": report.get("backend"),
        "use_llm": report.get("use_llm"),
    }
    return GateResult(
        gate_id=gate_id,
        passed=all(c.ok for c in checks),
        checks=checks,
        report_summary=summary,
    )


def evaluate_checklist_report(
    run_dict: dict[str, Any],
    *,
    gate_id: str,
    gate_cfg: dict[str, Any],
) -> GateResult:
    """Fail if too many critical corpus failures (status=fail + severity=critical)."""
    items = run_dict.get("items") or []
    critical_fail = 0
    for it in items:
        if it.get("status") == "fail" and it.get("severity") == "critical":
            critical_fail += 1
    max_fail = int(gate_cfg.get("max_critical_fail", 0))
    ok = critical_fail <= max_fail
    checks = [
        GateCheck(
            name="max_critical_fail",
            ok=ok,
            actual=critical_fail,
            threshold=max_fail,
            message=(
                f"critical corpus fails: {critical_fail} "
                f"{'<=' if ok else '>'} {max_fail}"
            ),
        )
    ]
    return GateResult(
        gate_id=gate_id,
        passed=ok,
        checks=checks,
        report_summary={
            "items": len(items),
            "critical_fail": critical_fail,
            "summary": run_dict.get("summary"),
        },
    )


def run_configured_gate(
    gate_id: str,
    *,
    thresholds_path: Path | None = None,
    out_path: Path | None = None,
) -> GateResult:
    """
    Execute a named gate from THRESHOLDS.json (may invoke eval/checklist).

    Supported: router, hard_llm, checklist_corpus.
    """
    cfg_root = load_thresholds(thresholds_path)
    gates = cfg_root["gates"]
    if gate_id not in gates:
        known = ", ".join(sorted(gates))
        raise KeyError(f"Unknown gate {gate_id!r}. Known: {known}")
    gate_cfg = gates[gate_id]

    if gate_id == "unit":
        # Unit is shell/pytest — not run here; marker only
        return GateResult(
            gate_id=gate_id,
            passed=True,
            checks=[
                GateCheck(
                    name="unit_marker",
                    ok=True,
                    actual="use pytest",
                    threshold=gate_cfg.get("command"),
                    message="Run unit tests via pytest (see thresholds.command)",
                )
            ],
        )

    if gate_id == "checklist_corpus":
        from govgrant.compliance.checklist import run_checklist

        run = run_checklist(
            program=str(gate_cfg.get("program") or "sbir"),
            use_ot=bool(gate_cfg.get("use_ot", True)),
            packages=list(gate_cfg.get("packages") or ["darpa"]),
        )
        result = evaluate_checklist_report(
            run.to_dict(), gate_id=gate_id, gate_cfg=gate_cfg
        )
    else:
        from govgrant.rag.eval.runner import run_regression

        categories = gate_cfg.get("categories")
        cat_set = set(categories) if categories else None
        report = run_regression(
            golden=bool(gate_cfg.get("golden", True)),
            backend=gate_cfg.get("backend") or "router",
            use_llm=bool(gate_cfg.get("use_llm", False)),
            categories=cat_set,
        )
        result = evaluate_eval_report(report, gate_id=gate_id, gate_cfg=gate_cfg)
        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {**report, "gate": result.to_dict()}
            out_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    if out_path and gate_id == "checklist_corpus":
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return result
