"""Deterministic regression gate for the offline SciFact fixture benchmark.

The gate executes the checked-in fixture end to end and compares the resulting
metrics against a checked-in baseline. A metric drop fails the gate; deliberate
improvements pass and are reported. Dataset, cache, prompt, seed, and sample
changes are reported as baseline drift instead of being silently scored, and an
invalid paired run never passes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

BASELINE_SCHEMA_VERSION = 1

#: Metrics pinned per architecture, as paths into the run's metrics section.
#: Claim metrics apply to every architecture; evidence/abstention/efficiency
#: metrics are pinned only when present in the baseline entry.
METRIC_PATHS: tuple[str, ...] = (
    "claim.accuracy",
    "claim.macro_f1",
    "evidence.evidence_f1",
    "evidence.document_recall_at_k",
    "evidence.mrr",
    "abstention.coverage",
    "efficiency.avg_llm_calls",
)

#: Manifest fields that must match the baseline for the comparison to count.
DRIFT_MANIFEST_FIELDS: tuple[str, ...] = (
    "dataset_revision",
    "dataset_hashes",
    "cache_version",
    "prompt_versions",
    "seed",
    "split",
    "sample_size",
)


class GateStatus(StrEnum):
    PASS = "PASS"
    REGRESSION = "REGRESSION"
    BASELINE_DRIFT = "BASELINE_DRIFT"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class _MetricRead:
    value: Any
    found: bool


def _read_path(node: dict[str, Any], dotted: str) -> _MetricRead:
    current: Any = node
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MetricRead(value=None, found=False)
        current = current[part]
    return _MetricRead(value=current, found=True)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def baseline_drift_reasons(manifest: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in DRIFT_MANIFEST_FIELDS:
        baseline_value = _read_path(baseline, f"environment.{field}")
        current_value = _read_path(manifest, field)
        if not baseline_value.found or not current_value.found:
            reasons.append(f"{field}: missing from baseline or run manifest")
        elif baseline_value.value != current_value.value:
            reasons.append(
                f"{field}: baseline={baseline_value.value!r} run={current_value.value!r}"
            )
    return reasons


def build_baseline_from_result(result: dict[str, Any], baseline_id: str) -> dict[str, Any]:
    manifest = result["manifest"]
    metrics = result["metrics"]
    architectures: dict[str, dict[str, float]] = {}
    for architecture, entry in metrics.items():
        pinned: dict[str, float] = {}
        for dotted in METRIC_PATHS:
            read = _read_path(entry, dotted)
            if read.found and _is_number(read.value):
                pinned[dotted] = float(read.value)
        if pinned:
            architectures[architecture] = pinned
    environment = {field: manifest[field] for field in DRIFT_MANIFEST_FIELDS if field in manifest}
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "description": (
            "Deterministic regression baseline for the checked-in offline SciFact "
            "fixture benchmark; regenerate only with scripts/eval_regression_gate.py "
            "--update-baseline and record the reason in the commit message."
        ),
        "environment": environment,
        "architectures": architectures,
    }


def load_baseline(path: Path) -> dict[str, Any]:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError(f"unsupported baseline schema_version: {baseline.get('schema_version')!r}")
    return baseline


def compare_to_baseline(
    result: dict[str, Any],
    baseline: dict[str, Any],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    manifest = result["manifest"]
    metrics = result["metrics"]

    invalid_run: list[str] = []
    if manifest.get("valid_paired_run") is not True:
        invalid_run.append("valid_paired_run is not true")
    if manifest.get("model_substitutions"):
        invalid_run.append(f"model_substitutions={manifest.get('model_substitutions')}")
    for architecture in baseline["architectures"]:
        current = metrics.get(architecture)
        if current is None:
            invalid_run.append(f"{architecture}: missing prediction metrics")
        elif current.get("complete") is not True or current.get("valid_for_comparison") is not True:
            invalid_run.append(f"{architecture}: incomplete or invalid for comparison")
    if invalid_run:
        return {
            "status": GateStatus.INVALID_RUN,
            "failures": [
                {"architecture": "run", "metric": "validity", "detail": detail}
                for detail in invalid_run
            ],
            "improvements": [],
            "drift_reasons": [],
        }

    drift_reasons = baseline_drift_reasons(manifest, baseline)
    if drift_reasons:
        return {
            "status": GateStatus.BASELINE_DRIFT,
            "failures": [],
            "improvements": [],
            "drift_reasons": drift_reasons,
        }

    failures: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    for architecture, baseline_metrics in baseline["architectures"].items():
        current_entry = metrics[architecture]
        for dotted, baseline_value in baseline_metrics.items():
            read = _read_path(current_entry, dotted)
            if not read.found or not _is_number(read.value):
                failures.append(
                    {
                        "architecture": architecture,
                        "metric": dotted,
                        "baseline": baseline_value,
                        "current": None,
                        "detail": "metric missing from run results",
                    }
                )
                continue
            current_value = float(read.value)
            if current_value + tolerance < baseline_value:
                failures.append(
                    {
                        "architecture": architecture,
                        "metric": dotted,
                        "baseline": baseline_value,
                        "current": current_value,
                        "detail": (f"metric dropped by {baseline_value - current_value:.6f}"),
                    }
                )
            elif current_value > baseline_value + tolerance:
                improvements.append(
                    {
                        "architecture": architecture,
                        "metric": dotted,
                        "baseline": baseline_value,
                        "current": current_value,
                    }
                )

    status = GateStatus.REGRESSION if failures else GateStatus.PASS
    return {
        "status": status,
        "failures": failures,
        "improvements": improvements,
        "drift_reasons": [],
    }


def describe_report(report: dict[str, Any]) -> str:
    lines = [f"REGRESSION_GATE={report['status'].value}"]
    for failure in report["failures"]:
        detail = failure.get("detail", "")
        lines.append(
            f"FAILURE architecture={failure['architecture']} metric={failure['metric']} "
            f"baseline={failure.get('baseline')} current={failure.get('current')} {detail}".rstrip()
        )
    for improvement in report["improvements"]:
        lines.append(
            f"IMPROVEMENT architecture={improvement['architecture']} "
            f"metric={improvement['metric']} baseline={improvement['baseline']} "
            f"current={improvement['current']}"
        )
    for reason in report["drift_reasons"]:
        lines.append(f"DRIFT {reason}")
    return "\n".join(lines)
