from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from evals.scifact.dataset import build_manifest, load_scifact
from evals.scifact.regression import (
    BASELINE_SCHEMA_VERSION,
    GateStatus,
    baseline_drift_reasons,
    build_baseline_from_result,
    compare_to_baseline,
    load_baseline,
)
from evals.scifact.retrieval import RetrievalConfig
from evals.scifact.runner import (
    ARCHITECTURES,
    OfflineBenchmarkProvider,
    load_and_validate_dataset,
    run_benchmark,
)

FIXTURE_DATA_DIR = Path("evals/fixtures/scifact/data")
FIXTURE_MANIFEST = Path("evals/fixtures/scifact/manifest.json")
CHECKED_IN_BASELINE = Path("evals/baselines/scifact-offline-fixture-smoke.json")


def _run_offline_fixture(tmp_path: Path) -> dict:
    corpus, manifest = load_and_validate_dataset(FIXTURE_DATA_DIR, FIXTURE_MANIFEST, "dev")
    stage_providers = {
        "single": "gemini",
        "judge": "gemini",
        "classifier": "groq",
        "auditor": "gemini",
        "critic": "groq",
    }
    return run_benchmark(
        corpus=corpus,
        manifest=manifest,
        profile="smoke",
        seed=42,
        architectures=ARCHITECTURES,
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers={"groq": OfflineBenchmarkProvider(), "gemini": OfflineBenchmarkProvider()},
        model_profile={
            "mode": "OFFLINE_FIXTURE",
            "groq": "deterministic-eval-fixture-v1",
            "gemini": "deterministic-eval-fixture-v1",
            "stage_providers": stage_providers,
        },
        stage_providers=stage_providers,
        retrieval_config=RetrievalConfig(),
    )


def _sample_result_metrics() -> dict:
    return {
        "manifest": {
            "valid_paired_run": True,
            "sample_size": 2,
            "seed": 42,
            "split": "dev",
            "dataset_revision": "fixture-v1",
            "dataset_hashes": {"corpus.jsonl": "aaa", "claims_dev.jsonl": "bbb"},
            "cache_version": "scifact-structured-cache-v2",
            "prompt_versions": {"judge": "judge_v1"},
            "model_substitutions": 0,
        },
        "metrics": {
            "A_SINGLE_LLM": {
                "complete": True,
                "valid_for_comparison": True,
                "claim": {"accuracy": 0.5, "macro_f1": 0.4},
            },
            "B_RETRIEVAL_JUDGE": {
                "complete": True,
                "valid_for_comparison": True,
                "claim": {"accuracy": 0.5, "macro_f1": 0.4},
                "evidence": {"evidence_f1": 0.75, "mrr": 0.9},
                "abstention": {"coverage": 1.0},
                "efficiency": {"avg_llm_calls": 1.0},
            },
        },
    }


def _sample_baseline() -> dict:
    result = _sample_result_metrics()
    return build_baseline_from_result(result, baseline_id="test-baseline")


def test_baseline_schema_version_is_explicit() -> None:
    assert _sample_baseline()["schema_version"] == BASELINE_SCHEMA_VERSION


def test_identical_metrics_pass() -> None:
    report = compare_to_baseline(_sample_result_metrics(), _sample_baseline())
    assert report["status"] == GateStatus.PASS
    assert report["failures"] == []


def test_metric_drop_is_a_regression() -> None:
    result = _sample_result_metrics()
    result["metrics"]["B_RETRIEVAL_JUDGE"]["claim"]["macro_f1"] = 0.2
    report = compare_to_baseline(result, _sample_baseline())
    assert report["status"] == GateStatus.REGRESSION
    assert any(
        failure["architecture"] == "B_RETRIEVAL_JUDGE" and failure["metric"] == "claim.macro_f1"
        for failure in report["failures"]
    )


def test_metric_improvement_passes_and_is_reported() -> None:
    result = _sample_result_metrics()
    result["metrics"]["B_RETRIEVAL_JUDGE"]["evidence"]["evidence_f1"] = 0.9
    report = compare_to_baseline(result, _sample_baseline())
    assert report["status"] == GateStatus.PASS
    assert any(
        improvement["architecture"] == "B_RETRIEVAL_JUDGE"
        and improvement["metric"] == "evidence.evidence_f1"
        for improvement in report["improvements"]
    )


def test_tolerated_drop_passes_and_beyond_tolerance_fails() -> None:
    baseline = _sample_baseline()
    within = _sample_result_metrics()
    within["metrics"]["B_RETRIEVAL_JUDGE"]["claim"]["macro_f1"] = 0.4 - 1e-9
    assert compare_to_baseline(within, baseline)["status"] == GateStatus.PASS

    beyond = _sample_result_metrics()
    beyond["metrics"]["B_RETRIEVAL_JUDGE"]["claim"]["macro_f1"] = 0.35
    assert compare_to_baseline(beyond, baseline, tolerance=0.01)["status"] == GateStatus.REGRESSION


def test_missing_baseline_metric_cannot_pass_silently() -> None:
    result = _sample_result_metrics()
    del result["metrics"]["B_RETRIEVAL_JUDGE"]["evidence"]
    report = compare_to_baseline(result, _sample_baseline())
    assert report["status"] == GateStatus.REGRESSION
    assert any(failure["metric"] == "evidence.evidence_f1" for failure in report["failures"])


def test_missing_architecture_never_passes() -> None:
    result = _sample_result_metrics()
    del result["metrics"]["A_SINGLE_LLM"]
    report = compare_to_baseline(result, _sample_baseline())
    assert report["status"] == GateStatus.INVALID_RUN
    assert any("A_SINGLE_LLM" in failure["detail"] for failure in report["failures"])


def test_invalid_run_never_passes() -> None:
    result = _sample_result_metrics()
    result["manifest"]["valid_paired_run"] = False
    report = compare_to_baseline(result, _sample_baseline())
    assert report["status"] == GateStatus.INVALID_RUN

    substituted = _sample_result_metrics()
    substituted["manifest"]["model_substitutions"] = 1
    assert compare_to_baseline(substituted, _sample_baseline())["status"] == GateStatus.INVALID_RUN

    incomplete = _sample_result_metrics()
    incomplete["metrics"]["B_RETRIEVAL_JUDGE"]["complete"] = False
    assert compare_to_baseline(incomplete, _sample_baseline())["status"] == GateStatus.INVALID_RUN


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_revision", "fixture-v2"),
        ("cache_version", "scifact-structured-cache-v3"),
        ("seed", 43),
        ("sample_size", 5),
        ("split", "train"),
    ],
)
def test_environment_drift_is_reported_not_scored(field: str, value: object) -> None:
    result = _sample_result_metrics()
    result["manifest"][field] = value
    report = compare_to_baseline(result, _sample_baseline())
    assert report["status"] == GateStatus.BASELINE_DRIFT
    assert report["failures"] == []
    assert any(field in reason for reason in report["drift_reasons"])


def test_prompt_version_and_dataset_hash_drift_is_reported() -> None:
    result = _sample_result_metrics()
    result["manifest"]["prompt_versions"]["judge"] = "judge_v2"
    assert compare_to_baseline(result, _sample_baseline())["status"] == GateStatus.BASELINE_DRIFT

    result = _sample_result_metrics()
    result["manifest"]["dataset_hashes"]["corpus.jsonl"] = "changed"
    assert baseline_drift_reasons(result["manifest"], _sample_baseline()) != []


def test_baseline_round_trips_through_json(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(_sample_baseline()), encoding="utf-8")
    loaded = load_baseline(path)
    assert compare_to_baseline(_sample_result_metrics(), loaded)["status"] == GateStatus.PASS


def test_gate_passes_end_to_end_on_checked_in_fixture_and_baseline(tmp_path: Path) -> None:
    result = _run_offline_fixture(tmp_path)
    baseline = load_baseline(CHECKED_IN_BASELINE)
    report = compare_to_baseline(result, baseline)
    assert report["status"] == GateStatus.PASS, report


def test_baseline_builder_records_run_context(tmp_path: Path) -> None:
    corpus = load_scifact(FIXTURE_DATA_DIR, "dev")
    assert corpus.label_distribution == {
        "CONTRADICT": 1,
        "NOT_ENOUGH_INFO": 1,
        "SUPPORT": 1,
    }
    manifest = build_manifest(
        data_dir=FIXTURE_DATA_DIR,
        archive_path=None,
        source="local-test-fixture",
        revision="fixture-v1",
        downloaded_at="2026-01-01T00:00:00+00:00",
    )
    assert manifest["split_rows"]["dev"]["rows"] == 3
    baseline = _sample_baseline()
    assert baseline["environment"]["dataset_revision"] == "fixture-v1"
    assert baseline["architectures"]["B_RETRIEVAL_JUDGE"]["evidence.evidence_f1"] == 0.75
    # Structural copies stay independent.
    mutated = copy.deepcopy(baseline)
    mutated["architectures"]["B_RETRIEVAL_JUDGE"]["claim.macro_f1"] = 0.0
    assert baseline["architectures"]["B_RETRIEVAL_JUDGE"]["claim.macro_f1"] == 0.4
