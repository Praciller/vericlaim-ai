from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from evals.metrics import calibration_metrics, compute_metrics
from evals.scifact.budget import (
    LiveBudgetGate,
    ProviderBudgetState,
    QuotaStatus,
    provider_budget_table,
)
from evals.scifact.cache import make_cache_key
from evals.scifact.dataset import (
    SciFactDataError,
    build_manifest,
    load_scifact,
    validate_manifest,
)
from evals.scifact.retrieval import BM25Retriever
from evals.scifact.runner import (
    ARCHITECTURE_CALL_UPPER_BOUNDS,
    ARCHITECTURES,
    FIXED_MODELS,
    ISOLATION_ARCHITECTURES,
    EvaluationError,
    OfflineBenchmarkProvider,
    _critic_risk_signals,
    build_live_providers,
    run_benchmark,
    run_dry_run,
    select_sample,
)
from evals.scifact.scope_quantifier import evaluate_fixture

from vericlaim.config import Settings
from vericlaim.domain.models import ProviderErrorCategory
from vericlaim.providers.base import ProviderException
from vericlaim.workflow import MIXED_VERDICT_HEURISTIC_CONFIDENCE


def _write_fixture(root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    data_dir.mkdir()
    documents = [
        {
            "doc_id": 1,
            "title": "Method effectiveness",
            "abstract": [
                "The method supports reliable evaluation results.",
                "The method works in tests.",
            ],
            "structured": False,
        },
        {
            "doc_id": 2,
            "title": "Method limitations",
            "abstract": [
                "The method has failures under evaluation.",
                "Limitations remain in deployment.",
            ],
            "structured": False,
        },
        {
            "doc_id": 3,
            "title": "Unrelated study",
            "abstract": ["This document discusses a different topic."],
            "structured": False,
        },
    ]
    claims = [
        {
            "id": 1,
            "claim": "The method supports reliable evaluation results.",
            "evidence": {"1": [{"label": "SUPPORT", "sentences": [0]}]},
            "cited_doc_ids": [1],
        },
        {
            "id": 2,
            "claim": "The method has failures under evaluation.",
            "evidence": {"2": [{"label": "CONTRADICT", "sentences": [0]}]},
            "cited_doc_ids": [2],
        },
        {
            "id": 3,
            "claim": "A claim without a matching abstract is known.",
            "evidence": {},
            "cited_doc_ids": [],
        },
        {
            "id": 4,
            "claim": "The method works in tests.",
            "evidence": {"1": [{"label": "SUPPORT", "sentences": [1]}]},
            "cited_doc_ids": [1],
        },
        {
            "id": 5,
            "claim": "Limitations remain in deployment.",
            "evidence": {"2": [{"label": "CONTRADICT", "sentences": [1]}]},
            "cited_doc_ids": [2],
        },
        {"id": 6, "claim": "A sixth claim has no evidence.", "evidence": {}, "cited_doc_ids": []},
    ]
    (data_dir / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8"
    )
    (data_dir / "claims_dev.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in claims), encoding="utf-8"
    )
    (data_dir / "claims_train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in claims), encoding="utf-8"
    )
    (data_dir / "claims_test.jsonl").write_text(
        "".join(json.dumps({"id": row["id"], "claim": row["claim"]}) + "\n" for row in claims),
        encoding="utf-8",
    )
    manifest = build_manifest(
        data_dir=data_dir,
        archive_path=None,
        source="local-test-fixture",
        revision="fixture-v1",
        downloaded_at="2026-01-01T00:00:00+00:00",
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return data_dir, manifest_path


class RecordingProvider(OfflineBenchmarkProvider):
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.tasks: list[str] = []

    def generate(self, request):  # type: ignore[no-untyped-def]
        self.prompts.append(request.prompt)
        self.tasks.append(request.task)
        return super().generate(request)


class BudgetProvider(RecordingProvider):
    name = "offline"
    model = "fixture-budget-v1"


class RateLimitedProvider(BudgetProvider):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def generate(self, request):  # type: ignore[no-untyped-def]
        self.attempts += 1
        raise ProviderException(
            "fixture rate limit",
            category=ProviderErrorCategory.RATE_LIMIT,
            fallback_allowed=False,
        )


class FailAfterFirstProvider(BudgetProvider):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def generate(self, request):  # type: ignore[no-untyped-def]
        self.attempts += 1
        if self.attempts == 2:
            raise ProviderException(
                "fixture timeout",
                category=ProviderErrorCategory.TIMEOUT,
                fallback_allowed=False,
            )
        return super().generate(request)


class DriftAfterFirstProvider(BudgetProvider):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def generate(self, request):  # type: ignore[no-untyped-def]
        self.attempts += 1
        response = super().generate(request)
        if self.attempts >= 1:
            return replace(response, actual_model="fixture-drift-v2")
        return response


class ChallengeProvider(RecordingProvider):
    def generate(self, request):  # type: ignore[no-untyped-def]
        response = super().generate(request)
        if request.task == "critic":
            return replace(
                response,
                text=json.dumps(
                    {
                        "decision": "CHALLENGE",
                        "reason": "scope mismatch requires one bounded recheck",
                    }
                ),
            )
        return response


class SubstitutingProvider(RecordingProvider):
    def generate(self, request):  # type: ignore[no-untyped-def]
        return replace(super().generate(request), actual_model="unexpected-model")


def test_manifest_hash_validation_and_split_loading(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    manifest = validate_manifest(manifest_path)
    corpus = load_scifact(data_dir, "dev")
    assert manifest["split_rows"]["dev"]["rows"] == 6
    assert corpus.label_distribution == {
        "CONTRADICT": 2,
        "NOT_ENOUGH_INFO": 2,
        "SUPPORT": 2,
    }
    assert corpus.claims[0].gold_sentence_ids == ("doc:1:sentence:0",)
    (data_dir / "corpus.jsonl").write_text(
        (data_dir / "corpus.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(SciFactDataError, match="hash mismatch"):
        validate_manifest(manifest_path)


def test_bm25_retrieval_is_deterministic_and_closed_corpus(tmp_path: Path) -> None:
    data_dir, _ = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    retriever = BM25Retriever(corpus)
    first = retriever.retrieve("method supports reliable evaluation")
    second = retriever.retrieve("method supports reliable evaluation")
    assert [item.document_id for item in first] == [item.document_id for item in second]
    assert first[0].document_id == "1"
    assert all(item.document_id in corpus.documents for item in first)


def test_same_stratified_sample_is_used_across_architectures(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    left = select_sample(corpus.claims, "smoke", 42)
    right = select_sample(corpus.claims, "smoke", 42)
    assert [claim.claim_id for claim in left] == [claim.claim_id for claim in right]
    result = run_dry_run(
        corpus=corpus,
        manifest=validate_manifest(manifest_path),
        profile="smoke",
        seed=42,
        architectures=ARCHITECTURES,
        cache_root=tmp_path / "dry-cache",
    )
    by_architecture = {
        architecture: {
            row["claim_id"] for row in result["predictions"] if row["architecture"] == architecture
        }
        for architecture in ARCHITECTURES
    }
    assert len({frozenset(value) for value in by_architecture.values()}) == 1


def test_architecture_contracts_and_gold_leakage_guard(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    providers = {"groq": RecordingProvider(), "gemini": RecordingProvider()}
    result = run_benchmark(
        corpus=corpus,
        manifest=validate_manifest(manifest_path),
        profile="smoke",
        seed=42,
        architectures=ARCHITECTURES,
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers=providers,
        model_profile={"mode": "OFFLINE_FIXTURE"},
    )
    calls_by_architecture = {
        architecture: sum(
            row["llm_calls"] for row in result["predictions"] if row["architecture"] == architecture
        )
        for architecture in ARCHITECTURES
    }
    assert calls_by_architecture == {
        "A_SINGLE_LLM": 6,
        "B_RETRIEVAL_JUDGE": 6,
        "C_SUPPORT_COUNTER": 12,
        "D_FULL_VERICLAIM": 18,
    }
    assert all(
        "gold_evidence" not in prompt and "gold_label" not in prompt
        for provider in providers.values()
        for prompt in provider.prompts
    )
    assert result["manifest"]["actual_models"] == ["deterministic-eval-fixture-v1"]


def test_isolated_architectures_keep_auditor_and_critic_roles_separate(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    providers = {"groq": RecordingProvider(), "gemini": RecordingProvider()}
    result = run_benchmark(
        corpus=corpus,
        manifest=validate_manifest(manifest_path),
        profile="smoke",
        seed=42,
        architectures=ISOLATION_ARCHITECTURES,
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers=providers,
        model_profile={"mode": "OFFLINE_FIXTURE"},
    )
    calls_by_architecture = {
        architecture: sum(
            row["stage_invocations"]
            for row in result["predictions"]
            if row["architecture"] == architecture
        )
        for architecture in ISOLATION_ARCHITECTURES
    }
    assert calls_by_architecture == {
        "C_SUPPORT_COUNTER": 12,
        "D1_AUDITOR": 18,
        "D2_CRITIC": 18,
        "D3_AUDITOR_CRITIC": 24,
        "D4_CONDITIONAL_CRITIC": 12,
    }
    assert all(
        "assessments array" in prompt and "Do not return a verdict" in prompt
        for task, prompt in zip(providers["gemini"].tasks, providers["gemini"].prompts, strict=True)
        if task == "auditor"
    )
    assert all(
        "decision PASS or CHALLENGE" in prompt and "Do not produce a verdict" in prompt
        for task, prompt in zip(providers["groq"].tasks, providers["groq"].prompts, strict=True)
        if task == "critic"
    )
    assert all(
        not row["assurance"]["critic_invoked"]
        for row in result["predictions"]
        if row["architecture"] == "D4_CONDITIONAL_CRITIC"
    )
    assert set(result["metrics"]["isolation_ablations"]) == {
        "C → D1",
        "C → D2",
        "C → D3",
        "C → D4",
    }


def test_critic_challenge_is_rechecked_at_most_once(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    provider = ChallengeProvider()
    result = run_benchmark(
        corpus=corpus,
        manifest=validate_manifest(manifest_path),
        profile="smoke",
        seed=42,
        architectures=("D2_CRITIC",),
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers={"groq": provider, "gemini": provider},
        model_profile={"mode": "OFFLINE_FIXTURE"},
    )
    rows = result["predictions"]
    assert all(row["assurance"]["recheck_performed"] for row in rows)
    assert all(row["stage_invocations"] == 4 for row in rows)
    assert sum(task == "judge" for task in provider.tasks) == len(rows) * 2
    assert sum(task == "critic" for task in provider.tasks) == len(rows)
    critic_metrics = result["metrics"]["D2_CRITIC"]["critic"]
    assert critic_metrics["rejudge_count"] == len(rows)
    assert critic_metrics["critic_challenge_count"] == len(rows)
    assert all(row["assurance"]["effect"] != "NOT_INVOKED" for row in rows)


def test_conditional_critic_risk_signals_are_deterministic() -> None:
    from evals.scifact.dataset import SciFactClaim

    claim = SciFactClaim("1", "This method always eliminates failures", "SUPPORT", (), ())
    signals = _critic_risk_signals(
        claim,
        [
            {"evidence_id": "a", "stance": "SUPPORTS"},
            {"evidence_id": "b", "stance": "CONTRADICTS"},
        ],
        {"confidence": 0.9},
        [],
        ["a"],
    )
    assert signals == [
        "SUPPORT_COUNTER_CLOSE",
        "EVIDENCE_DISAGREEMENT",
        "HIGH_CONFIDENCE_LOW_EVIDENCE_STRENGTH",
        "LOW_EVIDENCE_COUNT",
        "ABSOLUTE_OR_QUANTIFIED_WORDING",
    ]


def test_metrics_abstention_evidence_calibration_and_cache_key() -> None:
    rows = [
        {
            "gold_label": "SUPPORTED",
            "mapped_prediction": "SUPPORTED",
            "correct": True,
            "abstained": False,
            "confidence": 0.8,
            "gold_evidence_ids": ["doc:1:sentence:0"],
            "selected_evidence_ids": ["doc:1:sentence:0"],
            "gold_document_ids": ["1"],
            "retrieved_document_ids": ["1"],
            "retrieved_sentence_ids": ["doc:1:sentence:0"],
        },
        {
            "gold_label": "REFUTED",
            "mapped_prediction": "SUPPORTED",
            "correct": False,
            "abstained": False,
            "confidence": 0.9,
            "gold_evidence_ids": ["doc:2:sentence:0"],
            "selected_evidence_ids": [],
            "gold_document_ids": ["2"],
            "retrieved_document_ids": ["2"],
            "retrieved_sentence_ids": ["doc:2:sentence:0"],
        },
    ]
    metrics = compute_metrics(rows, evidence_applicable=True)
    assert metrics["claim"]["accuracy"] == 0.5
    assert metrics["evidence"]["evidence_recall"] == 0.5
    assert metrics["abstention"]["coverage"] == 1.0
    assert calibration_metrics(rows, minimum_sample_size=2)["status"] == "MEASURED"
    assert calibration_metrics(rows)["status"] == "CALIBRATION_SAMPLE_TOO_SMALL"
    first, _ = make_cache_key(
        architecture="B_RETRIEVAL_JUDGE",
        provider="gemini",
        configured_model="gemini-flash-lite-latest",
        prompt_version="judge_v1",
        generation_parameters={"temperature": 0},
        input_hash_value="input-a",
    )
    second, _ = make_cache_key(
        architecture="C_SUPPORT_COUNTER",
        provider="gemini",
        configured_model="gemini-flash-lite-latest",
        prompt_version="judge_v1",
        generation_parameters={"temperature": 0},
        input_hash_value="input-a",
    )
    assert first != second


def test_scope_fixture_and_confidence_heuristic_are_explicit() -> None:
    report = evaluate_fixture(Path("evals/fixtures/scope_quantifier.json"))
    assert report["accuracy"] == 1.0
    assert report["does_not_claim_sciFact_performance"] is True
    assert MIXED_VERDICT_HEURISTIC_CONFIDENCE == 0.65


def test_structured_cache_hit_avoids_second_provider_call(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    providers = {"groq": RecordingProvider(), "gemini": RecordingProvider()}
    first = run_benchmark(
        corpus=corpus,
        manifest=validate_manifest(manifest_path),
        profile="smoke",
        seed=42,
        architectures=("A_SINGLE_LLM",),
        output_root=tmp_path / "first",
        cache_root=tmp_path / "cache",
        providers=providers,
        model_profile={"mode": "OFFLINE_FIXTURE"},
    )
    first_calls = len(providers["gemini"].prompts)
    assert sum(row["provider_calls"] for row in first["predictions"]) == first_calls
    assert sum(row["stage_invocations"] for row in first["predictions"]) == first_calls
    second = run_benchmark(
        corpus=corpus,
        manifest=validate_manifest(manifest_path),
        profile="smoke",
        seed=42,
        architectures=("A_SINGLE_LLM",),
        output_root=tmp_path / "second",
        cache_root=tmp_path / "cache",
        providers=providers,
        model_profile={"mode": "OFFLINE_FIXTURE"},
    )
    assert first_calls == 6
    assert len(providers["gemini"].prompts) == first_calls
    assert sum(row["cache_hits"] for row in second["predictions"]) == 6
    assert sum(row["llm_calls"] for row in second["predictions"]) == 0
    assert sum(row["stage_invocations"] for row in second["predictions"]) == 6
    assert sum(row["input_tokens"] for row in second["predictions"]) == 0
    assert sum(row["cached_total_tokens"] for row in second["predictions"]) > 0
    assert sum(row["latency_ms"] for row in second["predictions"]) == 0
    assert first["manifest"]["cache_version"] == second["manifest"]["cache_version"]


def test_cache_context_changes_when_dataset_revision_changes(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    manifest = validate_manifest(manifest_path)
    provider = RecordingProvider()
    run_benchmark(
        corpus=corpus,
        manifest=manifest,
        profile="smoke",
        seed=42,
        architectures=("A_SINGLE_LLM",),
        output_root=tmp_path / "first",
        cache_root=tmp_path / "cache",
        providers={"gemini": provider},
        model_profile={"mode": "OFFLINE_FIXTURE"},
    )
    changed_manifest = {**manifest, "revision": "fixture-v2"}
    run_benchmark(
        corpus=corpus,
        manifest=changed_manifest,
        profile="smoke",
        seed=42,
        architectures=("A_SINGLE_LLM",),
        output_root=tmp_path / "second",
        cache_root=tmp_path / "cache",
        providers={"gemini": provider},
        model_profile={"mode": "OFFLINE_FIXTURE"},
    )
    assert len(provider.prompts) == 12


def test_fixed_live_model_lock_rejects_substitution(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        groq_api_key="unit-placeholder",
        groq_enabled=True,
        groq_model=FIXED_MODELS["groq"],
    )
    monkeypatch.setattr(
        "evals.scifact.runner._availability",
        lambda settings, providers=None: {
            provider: {"status": "AVAILABLE"} for provider in (providers or ("groq", "gemini"))
        },
    )
    providers, availability = build_live_providers(settings, ("groq",))
    assert set(providers) == {"groq"}
    assert availability["groq"]["status"] == "AVAILABLE"
    with pytest.raises(EvaluationError, match="fixed benchmark model mismatch"):
        build_live_providers(
            Settings(
                groq_api_key="unit-placeholder",
                groq_enabled=True,
                groq_model="wrong-model",
            ),
            ("groq",),
        )


def test_okmd_fixed_live_model_is_explicit_and_locked(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        okmd_api_key="unit-placeholder",
        okmd_enabled=True,
        okmd_model=FIXED_MODELS["okmd"],
    )
    monkeypatch.setattr(
        "evals.scifact.runner._availability",
        lambda settings, providers=None: {
            provider: {"status": "AVAILABLE"} for provider in (providers or ("okmd",))
        },
    )
    providers, availability = build_live_providers(settings, ("okmd",))
    assert set(providers) == {"okmd"}
    assert availability["okmd"]["status"] == "AVAILABLE"
    with pytest.raises(EvaluationError, match="fixed benchmark model mismatch"):
        build_live_providers(
            Settings(
                okmd_api_key="unit-placeholder",
                okmd_enabled=True,
                okmd_model="wrong-model",
            ),
            ("okmd",),
        )


def test_first_response_model_substitution_invalidates_paired_run(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    result = run_benchmark(
        corpus=corpus,
        manifest=validate_manifest(manifest_path),
        profile="smoke",
        seed=42,
        architectures=("C_SUPPORT_COUNTER",),
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers={"gemini": SubstitutingProvider()},
        model_profile={"mode": "LIVE_FIXTURE"},
        stage_providers={
            "single": "gemini",
            "judge": "gemini",
            "classifier": "gemini",
            "auditor": "gemini",
            "critic": "gemini",
        },
    )
    assert result["manifest"]["model_substitutions"] == 12
    assert result["manifest"]["valid_paired_run"] is False
    assert result["metrics"]["C_SUPPORT_COUNTER"]["valid_for_comparison"] is False


def test_live_budget_gate_enforces_unknown_and_known_quota_rules() -> None:
    unknown = ProviderBudgetState(
        provider="gemini",
        configured=True,
        enabled=True,
        model="fixture-budget-v1",
        quota_status=QuotaStatus.UNKNOWN,
    )
    gate = LiveBudgetGate(
        provider_states=(unknown,),
        global_max_calls=2,
        global_max_tokens=20,
        historical_average_tokens_per_call=10,
        safety_factor=0.5,
    )
    allowed = gate.plan(
        architecture_call_upper_bounds={"C_SUPPORT_COUNTER": 2},
        sample_size=1,
        provider_by_architecture={"C_SUPPORT_COUNTER": "gemini"},
    )
    assert allowed.decision == "ALLOW"
    gate.enforce(provider="gemini", estimated_tokens=10)
    gate.record(provider="gemini", tokens=10)
    assert gate.allow_call(provider="gemini", estimated_tokens=11) is False
    assert gate.denial_reason == "global token ceiling reached"

    known = ProviderBudgetState(
        provider="gemini",
        configured=True,
        enabled=True,
        model="fixture-budget-v1",
        quota_status=QuotaStatus.KNOWN,
        remaining_requests=10,
        remaining_tokens=100,
    )
    known_gate = LiveBudgetGate(
        provider_states=(known,),
        global_max_calls=100,
        global_max_tokens=10_000,
        historical_average_tokens_per_call=10,
        safety_factor=0.5,
    )
    known_plan = known_gate.plan(
        architecture_call_upper_bounds={"C_SUPPORT_COUNTER": 3},
        sample_size=2,
        provider_by_architecture={"C_SUPPORT_COUNTER": "gemini"},
    )
    assert known_plan.decision == "DENY"
    assert "50%" in known_plan.reason
    safe_plan = known_gate.plan(
        architecture_call_upper_bounds={"C_SUPPORT_COUNTER": 5},
        sample_size=1,
        provider_by_architecture={"C_SUPPORT_COUNTER": "gemini"},
    )
    assert safe_plan.decision == "ALLOW"
    table = provider_budget_table((known,), safe_plan)
    assert table[0]["allowed"] is True
    assert "api_key" not in json.dumps(table).casefold()


def test_budget_plan_credits_verified_cache_hits() -> None:
    state = ProviderBudgetState(
        provider="groq",
        configured=True,
        enabled=True,
        model="fixture-budget-v1",
        quota_status=QuotaStatus.UNKNOWN,
    )
    gate = LiveBudgetGate(
        provider_states=(state,),
        global_max_calls=100,
        global_max_tokens=100_000,
        historical_average_tokens_per_call=100,
        safety_factor=0.5,
    )
    plan = gate.plan(
        architecture_call_upper_bounds={
            architecture: ARCHITECTURE_CALL_UPPER_BOUNDS[architecture]
            for architecture in ISOLATION_ARCHITECTURES
        },
        sample_size=5,
        provider_by_architecture={architecture: "groq" for architecture in ISOLATION_ARCHITECTURES},
        cached_calls_by_provider={"groq": 2},
    )
    assert plan.estimated_cache_hits == 2
    assert plan.estimated_cache_misses == 88
    assert plan.estimated_total_calls == 88
    assert plan.provider_estimates["groq"]["logical_calls"] == 90


def test_budgeted_isolation_writes_manifest_and_causal_artifacts(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    state = ProviderBudgetState(
        provider="gemini",
        configured=True,
        enabled=True,
        model="fixture-budget-v1",
        quota_status=QuotaStatus.UNKNOWN,
    )
    gate = LiveBudgetGate(
        provider_states=(state,),
        global_max_calls=100,
        global_max_tokens=100_000,
        historical_average_tokens_per_call=100,
        safety_factor=0.5,
    )
    plan = gate.plan(
        architecture_call_upper_bounds={
            architecture: ARCHITECTURE_CALL_UPPER_BOUNDS[architecture]
            for architecture in ISOLATION_ARCHITECTURES
        },
        sample_size=5,
        provider_by_architecture={
            architecture: "gemini" for architecture in ISOLATION_ARCHITECTURES
        },
    )
    provider = BudgetProvider()
    result = run_benchmark(
        corpus=corpus,
        manifest=validate_manifest(manifest_path),
        profile="smoke",
        seed=42,
        architectures=ISOLATION_ARCHITECTURES,
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers={"gemini": provider},
        model_profile={"mode": "OFFLINE_FIXTURE"},
        stage_providers={
            "single": "gemini",
            "judge": "gemini",
            "classifier": "gemini",
            "auditor": "gemini",
            "critic": "gemini",
        },
        budget_gate=gate,
        budget_policy=plan.as_dict(),
        sample_claim_ids=tuple(claim.claim_id for claim in corpus.claims[:5]),
    )
    output = Path(result["output_directory"])
    required = {
        "manifest.json",
        "metrics.json",
        "predictions.jsonl",
        "provider_usage.json",
        "critic_effects.jsonl",
        "auditor_effects.jsonl",
        "errors.jsonl",
        "summary.md",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert required.issubset(set(result["manifest"]["result_artifacts"]))
    assert result["manifest"]["budget_policy"]["decision"] == "ALLOW"
    assert result["manifest"]["budget_gate_snapshot"]["actual_provider_calls"] > 0
    assert result["manifest"]["actual_provider_calls"] == gate.actual_calls
    assert result["manifest"]["actual_live_tokens"] == gate.actual_tokens
    assert result["metrics"]["D1_AUDITOR"]["auditor"]["claims_with_auditor"] == 5
    assert result["metrics"]["D2_CRITIC"]["critic"]["critic_invocation_count"] == 5
    assert output.joinpath("critic_effects.jsonl").read_text(encoding="utf-8").strip()
    assert output.joinpath("auditor_effects.jsonl").read_text(encoding="utf-8").strip()


def test_rate_limit_stops_live_run_before_the_next_provider_call(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    provider = RateLimitedProvider()
    gate = LiveBudgetGate(
        provider_states=(
            ProviderBudgetState(
                provider="gemini",
                configured=True,
                enabled=True,
                model=provider.model,
                quota_status=QuotaStatus.UNKNOWN,
            ),
        ),
        global_max_calls=100,
        global_max_tokens=100_000,
        historical_average_tokens_per_call=100,
        safety_factor=0.5,
    )
    result = run_benchmark(
        corpus=corpus,
        manifest=validate_manifest(manifest_path),
        profile="smoke",
        seed=42,
        architectures=("C_SUPPORT_COUNTER",),
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers={"gemini": provider},
        model_profile={"mode": "LIVE_FIXTURE"},
        stage_providers={
            "single": "gemini",
            "judge": "gemini",
            "classifier": "gemini",
            "auditor": "gemini",
            "critic": "gemini",
        },
        budget_gate=gate,
        budget_policy={"decision": "ALLOW"},
    )
    assert provider.attempts == 1
    assert result["metrics"]["C_SUPPORT_COUNTER"]["complete"] is False
    assert "rate_limit" in result["manifest"]["budget_stop_reason"]
    assert result["manifest"]["valid_paired_run"] is False
    assert result["manifest"]["valid_paired_predictions"] == 0
    assert result["metrics"]["isolation_ablations"]["C → D1"]["macro_f1_delta"] is None
    assert result["manifest"]["actual_provider_calls"] == gate.actual_calls
    assert result["manifest"]["provider_failures"] == gate.provider_failures
    assert result["provider_usage"]["unrecorded_provider_failures"] == 0
    assert result["manifest"]["unrecorded_provider_attempts"] == 0
    assert result["provider_usage"]["provider_attempt_record_count"] == gate.actual_calls
    assert [record["status"] for record in result["provider_usage"]["records"]] == [
        "FAILED",
        "BUDGET_STOP",
    ]


def test_resumable_window_reuses_stage_cache_and_completes_one_run(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    manifest = validate_manifest(manifest_path)
    run_id = "scifact-live-5-resumable-fixture"
    checkpoint = tmp_path / "results" / run_id / "checkpoint.json"
    identity = {"mode": "RESUMABLE_WINDOWED_LIVE5", "fixture": "resume-v1"}
    cache_extra = {"resumable_configuration_hash": "resume-v1"}
    stage_providers = {
        "single": "gemini",
        "judge": "gemini",
        "classifier": "gemini",
        "auditor": "gemini",
        "critic": "gemini",
    }

    def make_gate(max_calls: int = 10) -> LiveBudgetGate:
        return LiveBudgetGate(
            provider_states=(
                ProviderBudgetState(
                    provider="gemini",
                    configured=True,
                    enabled=True,
                    model="fixture-budget-v1",
                    quota_status=QuotaStatus.UNKNOWN,
                ),
            ),
            global_max_calls=max_calls,
            global_max_tokens=100_000,
            historical_average_tokens_per_call=100,
        )

    first = run_benchmark(
        corpus=corpus,
        manifest=manifest,
        profile="live-5",
        seed=42,
        architectures=("C_SUPPORT_COUNTER",),
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers={"gemini": FailAfterFirstProvider()},
        model_profile={
            "mode": "RESUMABLE_WINDOWED_LIVE5",
            "benchmark_provider": "gemini",
            "benchmark_configured_model": "fixture-budget-v1",
            "configuration_hash": "resume-v1",
            "resumable_identity": identity,
        },
        stage_providers=stage_providers,
        sample_claim_ids=("1", "2"),
        budget_gate=make_gate(),
        budget_policy={"decision": "ALLOW_WINDOW"},
        benchmark_run_id=run_id,
        checkpoint_path=checkpoint,
        resumable=True,
        cache_context_extra=cache_extra,
        window_id="window-001",
        window_policy={"configuration_hash": "resume-v1"},
    )
    assert first["manifest"]["valid_paired_run"] is False
    assert first["manifest"]["prediction_count"] == 0
    assert first["manifest"]["windows"][0]["provider_failures"] == 1
    assert first["manifest"]["resume_cursor"]["first_missing_stage"] == "judge"

    second = run_benchmark(
        corpus=corpus,
        manifest=manifest,
        profile="live-5",
        seed=42,
        architectures=("C_SUPPORT_COUNTER",),
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers={"gemini": BudgetProvider()},
        model_profile={
            "mode": "RESUMABLE_WINDOWED_LIVE5",
            "benchmark_provider": "gemini",
            "benchmark_configured_model": "fixture-budget-v1",
            "configuration_hash": "resume-v1",
            "resumable_identity": identity,
        },
        stage_providers=stage_providers,
        sample_claim_ids=("1", "2"),
        budget_gate=make_gate(),
        budget_policy={"decision": "ALLOW_WINDOW"},
        benchmark_run_id=run_id,
        checkpoint_path=checkpoint,
        resumable=True,
        cache_context_extra=cache_extra,
        window_id="window-002",
        window_policy={"configuration_hash": "resume-v1"},
    )
    assert second["manifest"]["valid_paired_run"] is True
    assert second["manifest"]["valid_paired_predictions"] == 2
    assert len(second["manifest"]["windows"]) == 2
    assert second["manifest"]["windows"][1]["cache_hits"] >= 1
    assert second["manifest"]["actual_provider_calls"] == 5
    checkpoint_value = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_value["benchmark_run_id"] == run_id
    assert checkpoint_value["resume_cursor"]["completed_pairs"] == 2
    assert checkpoint_value["resume_cursor"]["first_missing_pair"] is None


def test_resumable_run_rejects_actual_model_drift(tmp_path: Path) -> None:
    data_dir, manifest_path = _write_fixture(tmp_path)
    corpus = load_scifact(data_dir, "dev")
    manifest = validate_manifest(manifest_path)
    run_id = "scifact-live-5-resumable-drift-fixture"
    checkpoint = tmp_path / "results" / run_id / "checkpoint.json"
    identity = {"mode": "RESUMABLE_WINDOWED_LIVE5", "fixture": "drift-v1"}
    common = {
        "mode": "RESUMABLE_WINDOWED_LIVE5",
        "benchmark_provider": "gemini",
        "benchmark_configured_model": "fixture-budget-v1",
        "configuration_hash": "drift-v1",
        "resumable_identity": identity,
    }
    stage_providers = {
        stage: "gemini" for stage in ("single", "judge", "classifier", "auditor", "critic")
    }

    def make_gate(max_calls: int = 10) -> LiveBudgetGate:
        return LiveBudgetGate(
            provider_states=(
                ProviderBudgetState(
                    provider="gemini",
                    configured=True,
                    enabled=True,
                    model="fixture-budget-v1",
                    quota_status=QuotaStatus.UNKNOWN,
                ),
            ),
            global_max_calls=max_calls,
            global_max_tokens=100_000,
            historical_average_tokens_per_call=100,
        )

    run_benchmark(
        corpus=corpus,
        manifest=manifest,
        profile="live-5",
        seed=42,
        architectures=("C_SUPPORT_COUNTER",),
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers={"gemini": BudgetProvider()},
        model_profile=common,
        stage_providers=stage_providers,
        sample_claim_ids=("1",),
        budget_gate=make_gate(max_calls=1),
        benchmark_run_id=run_id,
        checkpoint_path=checkpoint,
        resumable=True,
        cache_context_extra={"resumable_configuration_hash": "drift-v1"},
        window_id="window-001",
    )
    drifted = run_benchmark(
        corpus=corpus,
        manifest=manifest,
        profile="live-5",
        seed=42,
        architectures=("C_SUPPORT_COUNTER",),
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        providers={"gemini": DriftAfterFirstProvider()},
        model_profile=common,
        stage_providers=stage_providers,
        sample_claim_ids=("1",),
        budget_gate=make_gate(),
        benchmark_run_id=run_id,
        checkpoint_path=checkpoint,
        resumable=True,
        cache_context_extra={"resumable_configuration_hash": "drift-v1"},
        window_id="window-002",
    )
    assert drifted["manifest"]["model_drift"] == 1
    assert drifted["manifest"]["valid_paired_run"] is False
    assert any(
        record["error_category"] == "model_drift" for record in drifted["provider_usage"]["records"]
    )
