from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx

from evals.metrics import ABSTENTION_LABELS, compute_metrics
from vericlaim.config import Settings, secret_value
from vericlaim.providers.base import (
    GeminiProvider,
    OpenAICompatibleProvider,
    ProviderException,
    ProviderRequest,
    ProviderResponse,
)

from .budget import BudgetGateDenied, LiveBudgetGate
from .cache import CACHE_VERSION, StructuredResponseCache, input_hash, make_cache_key
from .dataset import SciFactClaim, SciFactCorpus, SciFactDocument, validate_manifest
from .retrieval import BM25Retriever, RetrievalConfig, RetrievedDocument
from .scope_quantifier import evaluate_fixture

ARCHITECTURES = (
    "A_SINGLE_LLM",
    "B_RETRIEVAL_JUDGE",
    "C_SUPPORT_COUNTER",
    "D_FULL_VERICLAIM",
)
ISOLATION_ARCHITECTURES = (
    "C_SUPPORT_COUNTER",
    "D1_AUDITOR",
    "D2_CRITIC",
    "D3_AUDITOR_CRITIC",
    "D4_CONDITIONAL_CRITIC",
)
ARCHITECTURE_GROUPS = {
    "all": ARCHITECTURES,
    "primary": ARCHITECTURES,
    "isolation": ISOLATION_ARCHITECTURES,
}
ALL_ARCHITECTURES = tuple(dict.fromkeys(ARCHITECTURES + ISOLATION_ARCHITECTURES))
ARCHITECTURE_CALL_UPPER_BOUNDS = {
    "C_SUPPORT_COUNTER": 2,
    "D1_AUDITOR": 3,
    "D2_CRITIC": 4,
    "D3_AUDITOR_CRITIC": 5,
    "D4_CONDITIONAL_CRITIC": 4,
}
ARCHITECTURE_STAGE_UPPER_BOUNDS = {
    "C_SUPPORT_COUNTER": {"classifier": 1, "judge": 1},
    "D1_AUDITOR": {"classifier": 1, "auditor": 1, "judge": 1},
    "D2_CRITIC": {"classifier": 1, "judge": 2, "critic": 1},
    "D3_AUDITOR_CRITIC": {"classifier": 1, "auditor": 1, "judge": 2, "critic": 1},
    "D4_CONDITIONAL_CRITIC": {"classifier": 1, "judge": 2, "critic": 1},
}
PROFILE_SIZES = {"smoke": 10, "pilot": 50, "extended": 300}
PROMPT_VERSIONS = {
    "single_llm": "single_llm_v1",
    "judge": "judge_v2",
    "judge_recheck": "judge_recheck_v1",
    "evidence_classifier": "evidence_classifier_v2",
    "auditor": "auditor_v2",
    "critic": "critic_v2",
    "conditional_critic": "conditional_critic_v1",
}
GENERATION_PARAMETERS = {"temperature": 0, "max_tokens": 512}
TASK_GENERATION_PARAMETERS = {
    "evidence_classifier": {"temperature": 0, "max_tokens": 2048},
    "critic": {"temperature": 0, "max_tokens": 2048},
}
PROVIDER_MIN_INTERVAL_SECONDS = {"gemini": 4.2, "groq": 2.2, "okmd": 1.0}
OKMD_BASE_URL = "https://gen.ai.kku.ac.th/okmd/api/v1"
FIXED_MODELS = {
    "groq": "openai/gpt-oss-20b",
    "gemini": "gemini-flash-lite-latest",
    "okmd": "deepseek-v4-flash",
}
DEFAULT_STAGE_PROVIDERS = {
    "single": "gemini",
    "judge": "gemini",
    "classifier": "groq",
    "auditor": "gemini",
    "critic": "groq",
}
MAX_EXCERPT_CHARS = 300


class EvaluationError(RuntimeError):
    """Raised for a benchmark setup or artifact failure."""


class BenchmarkProvider(Protocol):
    name: str
    model: str

    def generate(self, request: ProviderRequest) -> ProviderResponse: ...


@dataclass(frozen=True)
class CallRecord:
    architecture: str
    task: str
    prompt_version: str
    provider: str
    configured_model: str
    actual_model: str
    calls: int
    cache_hits: int
    cache_misses: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    status: str
    error_category: str | None = None
    cached_input_tokens: int = 0
    cached_output_tokens: int = 0
    cached_latency_ms: int = 0
    latency_source: str = "provider"
    quota_limit_tokens: int | None = None
    quota_used_tokens: int | None = None
    quota_remaining_tokens: int | None = None


@dataclass(frozen=True)
class CallResult:
    parsed: dict[str, Any] | None
    record: CallRecord
    error_category: str | None = None


@dataclass(frozen=True)
class StageSpec:
    task: str
    prompt_version: str
    provider: str
    prompt: str
    input_payload: dict[str, Any]
    cache_scope: str | None = None


class OfflineBenchmarkProvider:
    """Deterministic provider used only for offline tests and dry-run planning."""

    name = "offline"
    model = "deterministic-eval-fixture-v1"

    @staticmethod
    def _claim(prompt: str) -> str:
        match = re.search(r'"claim"\s*:\s*"(.*?)"', prompt, re.DOTALL)
        return match.group(1).casefold() if match else prompt.casefold()

    @staticmethod
    def _candidate_ids(prompt: str) -> list[str]:
        return re.findall(r'"evidence_id"\s*:\s*"([^"]+)"', prompt)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        claim = self._claim(request.prompt)
        candidate_ids = self._candidate_ids(request.prompt)
        contradiction = any(marker in claim for marker in ("never", "no ", "lack", "eliminate"))
        if request.task == "evidence_classifier":
            items = [
                {
                    "evidence_id": evidence_id,
                    "stance": "CONTRADICTS" if contradiction else "SUPPORTS",
                }
                for evidence_id in candidate_ids[:2]
            ]
            payload: dict[str, Any] = {"items": items}
        elif request.task == "auditor":
            payload = {
                "assessments": [
                    {
                        "evidence_id": evidence_id,
                        "relevance": 0.9,
                        "directness": 0.8,
                        "scope_match": True,
                        "quantifier_match": True,
                        "temporal_match": True,
                        "usable": True,
                        "issues": [],
                    }
                    for evidence_id in candidate_ids[:2]
                ]
            }
        elif request.task == "critic":
            payload = {"decision": "PASS", "reason": "No concrete unsupportedness found."}
        else:
            payload = {
                "verdict": "REFUTED" if contradiction else "SUPPORTED",
                "confidence": 0.7,
                "selected_evidence_ids": candidate_ids[:2],
            }
        text = json.dumps(payload, separators=(",", ":"))
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            configured_model=self.model,
            actual_model=self.model,
            text=text,
            input_tokens=len(request.prompt.split()),
            output_tokens=len(text.split()),
            latency_ms=0,
            finish_reason="stop",
        )


class BenchmarkClient:
    def __init__(
        self,
        providers: dict[str, BenchmarkProvider],
        cache: StructuredResponseCache,
        *,
        dry_run: bool = False,
        budget_gate: LiveBudgetGate | None = None,
        cache_context: dict[str, Any] | None = None,
        locked_actual_model: str | None = None,
        pacing_seconds: float = 0.0,
        enforce_provider_min_interval: bool = True,
        checkpoint_callback: Callable[[], None] | None = None,
    ) -> None:
        self.providers = providers
        self.cache = cache
        self.dry_run = dry_run
        self.budget_gate = budget_gate
        self.cache_context = cache_context or {}
        self.locked_actual_model = locked_actual_model
        self.pacing_seconds = max(0.0, pacing_seconds)
        self.enforce_provider_min_interval = enforce_provider_min_interval
        self.checkpoint_callback = checkpoint_callback
        self._last_call_at: dict[str, float] = {}
        self._blocked_providers: set[str] = set()
        self._actual_models: dict[str, str] = {}
        self.telemetry_records: list[CallRecord] = []
        self._dry_run_cache: dict[str, dict[str, Any]] = {}

    def _record(self, record: CallRecord) -> CallRecord:
        """Append one authoritative event before returning it to the caller."""
        self.telemetry_records.append(record)
        if self.checkpoint_callback is not None:
            self.checkpoint_callback()
        return record

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(candidate[start : end + 1])
        if not isinstance(value, dict):
            raise json.JSONDecodeError("expected JSON object", candidate, 0)
        return value

    @staticmethod
    def _dry_run_payload(task: str) -> dict[str, Any]:
        if task == "evidence_classifier":
            return {"items": []}
        if task == "auditor":
            return {"assessments": []}
        if task == "critic":
            return {"decision": "PASS", "reason": "dry-run"}
        return {"verdict": "INSUFFICIENT_EVIDENCE", "confidence": 0.2, "selected_evidence_ids": []}

    def call(self, architecture: str, spec: StageSpec) -> CallResult:
        configured_model = getattr(self.providers.get(spec.provider), "model", spec.provider)
        value_hash = input_hash(
            {
                "cache_context": self.cache_context,
                "stage_input": spec.input_payload,
                "prompt": spec.prompt,
            }
        )
        generation_parameters = TASK_GENERATION_PARAMETERS.get(spec.task, GENERATION_PARAMETERS)
        key, identity = make_cache_key(
            architecture=spec.cache_scope or architecture,
            provider=spec.provider,
            configured_model=configured_model,
            prompt_version=spec.prompt_version,
            generation_parameters=generation_parameters,
            input_hash_value=value_hash,
        )
        cached = self.cache.get(key)
        if cached is not None:
            cached_model_error: str | None = None
            if (
                self.locked_actual_model is not None
                and cached.actual_model != self.locked_actual_model
            ):
                cached_model_error = "model_drift"
            elif cached.actual_model != configured_model:
                cached_model_error = "model_substitution"
            if cached_model_error is not None:
                if self.budget_gate is not None:
                    self.budget_gate.denial_reason = (
                        "benchmark actual model drift"
                        if cached_model_error == "model_drift"
                        else "cached response has unexpected provider model"
                    )
                record = CallRecord(
                    architecture=architecture,
                    task=spec.task,
                    prompt_version=spec.prompt_version,
                    provider=cached.provider,
                    configured_model=cached.configured_model,
                    actual_model=cached.actual_model,
                    calls=0,
                    cache_hits=0,
                    cache_misses=1,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0,
                    status="FAILED",
                    error_category=cached_model_error,
                    latency_source="cache",
                )
                return CallResult(
                    parsed=None,
                    record=self._record(record),
                    error_category=cached_model_error,
                )
            record = CallRecord(
                architecture=architecture,
                task=spec.task,
                prompt_version=spec.prompt_version,
                provider=cached.provider,
                configured_model=cached.configured_model,
                actual_model=cached.actual_model,
                calls=0,
                cache_hits=1,
                cache_misses=0,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                status="CACHE_HIT",
                cached_input_tokens=cached.input_tokens,
                cached_output_tokens=cached.output_tokens,
                cached_latency_ms=cached.latency_ms,
                latency_source="cache",
            )
            return CallResult(parsed=cached.parsed, record=self._record(record))
        if self.dry_run and key in self._dry_run_cache:
            record = CallRecord(
                architecture=architecture,
                task=spec.task,
                prompt_version=spec.prompt_version,
                provider=spec.provider,
                configured_model=configured_model,
                actual_model=configured_model,
                calls=0,
                cache_hits=1,
                cache_misses=0,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                status="CACHE_HIT",
                latency_source="dry-run-cache",
            )
            return CallResult(
                parsed=self._dry_run_cache[key],
                record=self._record(record),
            )
        if self.dry_run:
            parsed = self._dry_run_payload(spec.task)
            self._dry_run_cache[key] = parsed
            record = CallRecord(
                architecture=architecture,
                task=spec.task,
                prompt_version=spec.prompt_version,
                provider=spec.provider,
                configured_model=configured_model,
                actual_model=configured_model,
                calls=0,
                cache_hits=0,
                cache_misses=1,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                status="DRY_RUN",
            )
            return CallResult(parsed=parsed, record=self._record(record))
        if self.budget_gate is not None:
            try:
                self.budget_gate.enforce(
                    provider=spec.provider,
                    estimated_tokens=int(self.budget_gate.historical_average_tokens_per_call),
                )
            except BudgetGateDenied:
                self._record(
                    CallRecord(
                        architecture=architecture,
                        task=spec.task,
                        prompt_version=spec.prompt_version,
                        provider=spec.provider,
                        configured_model=configured_model,
                        actual_model=configured_model,
                        calls=0,
                        cache_hits=0,
                        cache_misses=1,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0,
                        status="BUDGET_STOP",
                        error_category="budget_denied",
                        latency_source="budget-gate",
                    )
                )
                raise
        if spec.provider in self._blocked_providers:
            record = CallRecord(
                architecture=architecture,
                task=spec.task,
                prompt_version=spec.prompt_version,
                provider=spec.provider,
                configured_model=configured_model,
                actual_model=configured_model,
                calls=0,
                cache_hits=0,
                cache_misses=1,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                status="SKIPPED_PROVIDER_BLOCK",
                error_category="rate_limit",
            )
            return CallResult(
                parsed=None,
                record=self._record(record),
                error_category="rate_limit",
            )
        provider = self.providers.get(spec.provider)
        if provider is None:
            record = CallRecord(
                architecture=architecture,
                task=spec.task,
                prompt_version=spec.prompt_version,
                provider=spec.provider,
                configured_model=configured_model,
                actual_model=configured_model,
                calls=0,
                cache_hits=0,
                cache_misses=1,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                status="FAILED",
                error_category="provider_unconfigured",
            )
            return CallResult(
                parsed=None,
                record=self._record(record),
                error_category="provider_unconfigured",
            )
        if getattr(provider, "name", spec.provider) == spec.provider:
            minimum_interval = self.pacing_seconds
            if self.enforce_provider_min_interval:
                minimum_interval = max(
                    PROVIDER_MIN_INTERVAL_SECONDS.get(spec.provider, 0.0),
                    minimum_interval,
                )
            elapsed = time.perf_counter() - self._last_call_at.get(spec.provider, 0.0)
            if elapsed < minimum_interval:
                time.sleep(minimum_interval - elapsed)
            self._last_call_at[spec.provider] = time.perf_counter()
        try:
            response = provider.generate(
                ProviderRequest(
                    task=spec.task,
                    prompt=spec.prompt,
                    max_tokens=generation_parameters["max_tokens"],
                )
            )
        except ProviderException as exc:
            category = exc.category.value
            if self.budget_gate is not None:
                self.budget_gate.record(provider=spec.provider, tokens=0, failed=True)
            self._blocked_providers.add(spec.provider)
            if self.budget_gate is not None:
                self.budget_gate.denial_reason = (
                    f"{spec.provider} provider returned {category}; live run stopped"
                )
            record = CallRecord(
                architecture=architecture,
                task=spec.task,
                prompt_version=spec.prompt_version,
                provider=spec.provider,
                configured_model=configured_model,
                actual_model=configured_model,
                calls=1,
                cache_hits=0,
                cache_misses=1,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                status="FAILED",
                error_category=category,
            )
            return CallResult(
                parsed=None,
                record=self._record(record),
                error_category=category,
            )
        except Exception:
            if self.budget_gate is not None:
                self.budget_gate.record(provider=spec.provider, tokens=0, failed=True)
            record = CallRecord(
                architecture=architecture,
                task=spec.task,
                prompt_version=spec.prompt_version,
                provider=spec.provider,
                configured_model=configured_model,
                actual_model=configured_model,
                calls=1,
                cache_hits=0,
                cache_misses=1,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                status="FAILED",
                error_category="provider_failure",
            )
            return CallResult(
                parsed=None,
                record=self._record(record),
                error_category="provider_failure",
            )
        try:
            parsed = self._parse_json(response.text)
        except json.JSONDecodeError:
            record = CallRecord(
                architecture=architecture,
                task=spec.task,
                prompt_version=spec.prompt_version,
                provider=response.provider,
                configured_model=response.configured_model or response.model,
                actual_model=response.actual_model or response.model,
                calls=1,
                cache_hits=0,
                cache_misses=1,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                status="FAILED",
                error_category="schema_parse_failure",
                quota_limit_tokens=response.quota_limit_tokens,
                quota_used_tokens=response.quota_used_tokens,
                quota_remaining_tokens=response.quota_remaining_tokens,
            )
            if self.budget_gate is not None:
                self.budget_gate.record(
                    provider=response.provider,
                    tokens=response.total_tokens,
                    failed=True,
                )
            return CallResult(
                parsed=None,
                record=self._record(record),
                error_category="schema_parse_failure",
            )
        actual_model = response.actual_model or response.model
        known_model = self._actual_models.get(response.provider)
        configured_response_model = response.configured_model or response.model
        if self.locked_actual_model is not None and actual_model != self.locked_actual_model:
            model_error_category = "model_drift"
        elif actual_model != configured_model or (
            known_model is not None and actual_model != known_model
        ):
            model_error_category = "model_substitution"
        else:
            model_error_category = None
        if model_error_category is not None:
            if self.budget_gate is not None:
                self.budget_gate.record(
                    provider=response.provider,
                    tokens=response.total_tokens,
                    failed=True,
                )
                self.budget_gate.denial_reason = (
                    "benchmark actual model drift"
                    if model_error_category == "model_drift"
                    else "unexpected provider model substitution"
                )
            record = CallRecord(
                architecture=architecture,
                task=spec.task,
                prompt_version=spec.prompt_version,
                provider=response.provider,
                configured_model=configured_response_model,
                actual_model=actual_model,
                calls=1,
                cache_hits=0,
                cache_misses=1,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                status="FAILED",
                error_category=model_error_category,
                quota_limit_tokens=response.quota_limit_tokens,
                quota_used_tokens=response.quota_used_tokens,
                quota_remaining_tokens=response.quota_remaining_tokens,
            )
            return CallResult(
                parsed=None,
                record=self._record(record),
                error_category=model_error_category,
            )
        self._actual_models[response.provider] = actual_model
        if self.budget_gate is not None:
            self.budget_gate.record(provider=response.provider, tokens=response.total_tokens)
        self.cache.put(
            key=key,
            identity=identity,
            parsed=parsed,
            provider=response.provider,
            configured_model=configured_response_model,
            actual_model=actual_model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )
        record = CallRecord(
            architecture=architecture,
            task=spec.task,
            prompt_version=spec.prompt_version,
            provider=response.provider,
            configured_model=configured_response_model,
            actual_model=actual_model,
            calls=1,
            cache_hits=0,
            cache_misses=1,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            status=response.status,
            quota_limit_tokens=response.quota_limit_tokens,
            quota_used_tokens=response.quota_used_tokens,
            quota_remaining_tokens=response.quota_remaining_tokens,
        )
        return CallResult(parsed=parsed, record=self._record(record))


def _gold_label(label: str) -> str:
    return {
        "SUPPORT": "SUPPORTED",
        "CONTRADICT": "REFUTED",
        "NOT_ENOUGH_INFO": "INSUFFICIENT_EVIDENCE",
        "MIXED": "MIXED",
    }.get(label, "NON_VERIFIABLE")


def _prediction_label(value: object) -> str:
    normalized = str(value or "").strip().upper().replace(" ", "_")
    return {
        "SUPPORT": "SUPPORTED",
        "SUPPORTS": "SUPPORTED",
        "SUPPORTED": "SUPPORTED",
        "CONTRADICT": "REFUTED",
        "CONTRADICTS": "REFUTED",
        "REFUTED": "REFUTED",
        "NOT_ENOUGH_INFO": "INSUFFICIENT_EVIDENCE",
        "INSUFFICIENT": "INSUFFICIENT_EVIDENCE",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
        "NON_VERIFIABLE": "NON_VERIFIABLE",
        "MIXED": "MIXED",
    }.get(normalized, "INSUFFICIENT_EVIDENCE")


def _candidate_payload(documents: tuple[RetrievedDocument, ...]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": sentence.evidence_id,
            "document_id": sentence.document_id,
            "sentence_index": sentence.sentence_index,
            "text": sentence.text[:MAX_EXCERPT_CHARS],
            "retrieval_score": round(sentence.score, 6),
        }
        for document in documents
        for sentence in document.sentences
    ]


def _uses_classifier(architecture: str) -> bool:
    return architecture in {
        "C_SUPPORT_COUNTER",
        "D_FULL_VERICLAIM",
        "D1_AUDITOR",
        "D2_CRITIC",
        "D3_AUDITOR_CRITIC",
        "D4_CONDITIONAL_CRITIC",
    }


def _uses_auditor(architecture: str) -> bool:
    return architecture in {"D_FULL_VERICLAIM", "D1_AUDITOR", "D3_AUDITOR_CRITIC"}


def _uses_critic(architecture: str) -> bool:
    return architecture in {
        "D_FULL_VERICLAIM",
        "D2_CRITIC",
        "D3_AUDITOR_CRITIC",
        "D4_CONDITIONAL_CRITIC",
    }


def _judge_prompt(
    claim: SciFactClaim,
    candidates: list[dict[str, Any]],
    *,
    context: str,
    classifier_items: list[dict[str, Any]] | None = None,
    audit_assessments: list[dict[str, Any]] | None = None,
    critic_challenge: dict[str, Any] | None = None,
    recheck: bool = False,
) -> str:
    instruction = (
        "Return JSON only with keys verdict, confidence, selected_evidence_ids, and limitations. "
        "verdict must be SUPPORTED, REFUTED, MIXED, or INSUFFICIENT_EVIDENCE. "
        "Use only the supplied evidence IDs; do not use outside knowledge. "
        "Classifier and auditor outputs are advisory; the judge owns the verdict. "
    )
    if recheck:
        instruction += "Recheck the proposed verdict against the critic challenge once. "
    payload = {
        "claim": claim.text,
        "candidates": candidates,
        "classifier_items": classifier_items or [],
        "audit_assessments": audit_assessments or [],
        "critic_challenge": critic_challenge,
    }
    return (
        instruction
        + f"CONTEXT={context}\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _judge_context(
    *,
    audit_assessments: list[dict[str, Any]] | None,
    critic_challenge: dict[str, Any] | None,
    recheck: bool,
) -> str:
    if recheck or critic_challenge is not None:
        return "EVIDENCE_JUDGE_RECHECK"
    if audit_assessments:
        return "EVIDENCE_JUDGE_WITH_AUDIT"
    return "EVIDENCE_JUDGE"


def _stage_specs(
    claim: SciFactClaim,
    architecture: str,
    documents: tuple[RetrievedDocument, ...],
    classifier_items: list[dict[str, Any]] | None = None,
    audit_assessments: list[dict[str, Any]] | None = None,
    stage_providers: dict[str, str] | None = None,
    critic_challenge: dict[str, Any] | None = None,
    recheck: bool = False,
) -> list[StageSpec]:
    providers = {**DEFAULT_STAGE_PROVIDERS, **(stage_providers or {})}
    candidates = _candidate_payload(documents)
    allowed_ids = {item["evidence_id"] for item in candidates}
    if classifier_items:
        stance_by_id = {item["evidence_id"]: item["stance"] for item in classifier_items}
        candidates = [
            {**item, "classified_stance": stance_by_id.get(item["evidence_id"], "NEUTRAL")}
            for item in candidates
        ]
    payload: list[StageSpec] = []
    if architecture == "A_SINGLE_LLM":
        input_payload = {"claim_id": claim.claim_id, "claim": claim.text}
        return [
            StageSpec(
                task="single_llm",
                prompt_version=PROMPT_VERSIONS["single_llm"],
                provider=providers["single"],
                prompt=(
                    "Return JSON only with keys verdict, confidence, selected_evidence_ids, "
                    "and limitations. No documents or external evidence are available. "
                    "verdict must be SUPPORTED, REFUTED, MIXED, or INSUFFICIENT_EVIDENCE. "
                    + json.dumps(input_payload, ensure_ascii=False, sort_keys=True)
                ),
                input_payload=input_payload,
            )
        ]
    if _uses_classifier(architecture) and classifier_items is None:
        input_payload = {"claim_id": claim.claim_id, "claim": claim.text, "candidates": candidates}
        payload.append(
            StageSpec(
                task="evidence_classifier",
                prompt=(
                    "JSON only. Return an items array with one object per supplied evidence ID. "
                    "Each object has evidence_id and stance SUPPORTS, CONTRADICTS, or NEUTRAL. "
                    "Use only supplied IDs; no prose. "
                    + json.dumps(input_payload, ensure_ascii=False, sort_keys=True)
                ),
                prompt_version=PROMPT_VERSIONS["evidence_classifier"],
                provider=providers["classifier"],
                input_payload=input_payload,
                cache_scope="SHARED_EVIDENCE_CLASSIFIER",
            )
        )
    if _uses_auditor(architecture) and audit_assessments is None:
        audit_input = {
            "claim_id": claim.claim_id,
            "claim": claim.text,
            "candidates": candidates,
            "classifier_items": classifier_items or [],
        }
        payload.append(
            StageSpec(
                task="auditor",
                prompt=(
                    "Return JSON only with an assessments array. Do not return a verdict or "
                    "select a final evidence set. For each evidence ID assess relevance, "
                    "directness, scope_match, quantifier_match, temporal_match, usable, and "
                    "bounded issues. Hard failures are limited to fabricated evidence, "
                    "missing provenance, metadata-only evidence, or invalid citation. "
                    + json.dumps(audit_input, ensure_ascii=False, sort_keys=True)
                ),
                prompt_version=PROMPT_VERSIONS["auditor"],
                provider=providers["auditor"],
                input_payload=audit_input,
                cache_scope="SHARED_AUDITOR",
            )
        )
    judge_input = {
        "claim_id": claim.claim_id,
        "claim": claim.text,
        "candidates": candidates,
        "allowed_evidence_ids": sorted(allowed_ids),
        "classifier_items": classifier_items or [],
        "audit_assessments": audit_assessments or [],
        "critic_challenge": critic_challenge,
        "recheck": recheck,
    }
    payload.append(
        StageSpec(
            task="judge",
            prompt=_judge_prompt(
                claim,
                candidates,
                context=_judge_context(
                    audit_assessments=audit_assessments,
                    critic_challenge=critic_challenge,
                    recheck=recheck,
                ),
                classifier_items=classifier_items,
                audit_assessments=audit_assessments,
                critic_challenge=critic_challenge,
                recheck=recheck,
            ),
            prompt_version=PROMPT_VERSIONS["judge_recheck" if recheck else "judge"],
            provider=providers["judge"],
            input_payload=judge_input,
            cache_scope=(
                "JUDGE_RECHECK"
                if recheck
                else "JUDGE_WITH_AUDIT"
                if audit_assessments
                else "JUDGE_NO_AUDIT"
            ),
        )
    )
    return payload


def _extract_classifier(value: dict[str, Any], allowed_ids: set[str]) -> list[dict[str, Any]]:
    raw = value.get("items", value.get("evidence", value.get("assessments", [])))
    if not isinstance(raw, list):
        return []
    items: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("evidence_id") not in allowed_ids:
            continue
        stance = str(item.get("stance", "NEUTRAL")).upper()
        stance = {
            "SUPPORT": "SUPPORTS",
            "CONTRADICT": "CONTRADICTS",
        }.get(stance, stance)
        if stance not in {"SUPPORTS", "CONTRADICTS", "NEUTRAL"}:
            stance = "NEUTRAL"
        items.append({"evidence_id": str(item["evidence_id"]), "stance": stance})
    return items


def _extract_auditor(value: dict[str, Any], allowed_ids: set[str]) -> list[dict[str, Any]]:
    raw = value.get("assessments", [])
    if not isinstance(raw, list):
        return []
    assessments: list[dict[str, Any]] = []

    def bounded_score(raw: object) -> float:
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return 0.0

    for item in raw:
        if not isinstance(item, dict) or str(item.get("evidence_id")) not in allowed_ids:
            continue
        issues = item.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        assessments.append(
            {
                "evidence_id": str(item["evidence_id"]),
                "relevance": bounded_score(item.get("relevance", 0.0)),
                "directness": bounded_score(item.get("directness", 0.0)),
                "scope_match": bool(item.get("scope_match", False)),
                "quantifier_match": bool(item.get("quantifier_match", False)),
                "temporal_match": bool(item.get("temporal_match", False)),
                "usable": bool(item.get("usable", False)),
                "issues": [str(issue)[:120] for issue in issues[:5]],
            }
        )
    return assessments


def _extract_critic(value: dict[str, Any], allowed_ids: set[str]) -> dict[str, Any]:
    decision = str(value.get("decision", "")).strip().upper()
    if not decision and bool(value.get("should_abstain")):
        decision = "CHALLENGE"
    if decision not in {"PASS", "CHALLENGE"}:
        decision = "CHALLENGE"
    reason = str(value.get("reason", "unspecified critic result"))[:240]
    return {
        "decision": decision,
        "reason": reason,
        "evidence_ids": _extract_ids(value, allowed_ids),
    }


def _extract_ids(value: dict[str, Any], allowed_ids: set[str]) -> list[str]:
    values: list[object] = []
    for key in (
        "selected_evidence_ids",
        "evidence_ids",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
    ):
        candidate = value.get(key, [])
        if isinstance(candidate, list):
            values.extend(candidate)
    return list(dict.fromkeys(str(item) for item in values if str(item) in allowed_ids))


def _extract_verdict(value: dict[str, Any], allowed_ids: set[str]) -> tuple[str, float, list[str]]:
    label = _prediction_label(value.get("verdict", value.get("label")))
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.2))))
    except (TypeError, ValueError):
        confidence = 0.2
    return label, confidence, _extract_ids(value, allowed_ids)


def _error_entry(
    *,
    claim: SciFactClaim,
    architecture: str,
    prediction: dict[str, Any],
    category: str,
    document_lookup: dict[str, SciFactDocument],
) -> dict[str, Any]:
    excerpts: list[str] = []
    for evidence_id in prediction.get("selected_evidence_ids", []):
        match = re.fullmatch(r"doc:(.+):sentence:(\d+)", evidence_id)
        if match:
            sentence = document_lookup.get(
                match.group(1), SciFactDocument("", "", (), False)
            ).sentence_text(int(match.group(2)))
            if sentence:
                excerpts.append(sentence[:MAX_EXCERPT_CHARS])
    return {
        "claim_id": claim.claim_id,
        "claim": claim.text,
        "gold_label": prediction["gold_label"],
        "predicted_verdict": prediction["mapped_prediction"],
        "architecture": architecture,
        "category": category,
        "gold_evidence_ids": list(claim.gold_sentence_ids),
        "selected_evidence_ids": list(prediction.get("selected_evidence_ids", [])),
        "confidence": prediction["confidence"],
        "selected_evidence_excerpts": excerpts[:5],
    }


def _classify_error(
    claim: SciFactClaim,
    prediction: dict[str, Any],
    architecture: str,
    document_lookup: dict[str, SciFactDocument],
) -> list[dict[str, Any]]:
    if prediction["correct"]:
        return []
    categories: list[str] = []
    if prediction.get("errors"):
        categories.extend(str(error) for error in prediction["errors"])
    gold_docs = set(prediction.get("gold_document_ids", []))
    retrieved_docs = set(prediction.get("retrieved_document_ids", []))
    if architecture != "A_SINGLE_LLM" and gold_docs and not gold_docs.intersection(retrieved_docs):
        categories.append("RETRIEVAL_MISS")
        prediction["retrieval_failure"] = True
    elif architecture != "A_SINGLE_LLM" and gold_docs:
        selected = set(prediction.get("selected_evidence_ids", []))
        if not selected.intersection(set(claim.gold_sentence_ids)):
            categories.append("EVIDENCE_SELECTION_MISS")
        prediction["reasoning_failure"] = True
    if prediction["abstained"] and prediction["gold_label"] != "INSUFFICIENT_EVIDENCE":
        categories.append("EXCESSIVE_ABSTENTION")
    elif not prediction["abstained"] and prediction["gold_label"] == "INSUFFICIENT_EVIDENCE":
        categories.append("FAILED_TO_ABSTAIN")
    if re.search(
        r"\b(always|never|all|none|every|eliminat|guarante|best|fastest)\b|\d+%|\b\d+x\b",
        claim.text,
        re.I,
    ):
        categories.append("QUANTIFIER_MISMATCH")
    if not categories:
        categories.append("STANCE_CLASSIFICATION_ERROR")
    return [
        _error_entry(
            claim=claim,
            architecture=architecture,
            prediction=prediction,
            category=category,
            document_lookup=document_lookup,
        )
        for category in sorted(set(categories))
    ]


def _critic_risk_signals(
    claim: SciFactClaim,
    classifier_items: list[dict[str, Any]],
    judge_value: dict[str, Any] | None,
    audit_assessments: list[dict[str, Any]],
    selected_ids: list[str],
) -> list[str]:
    signals: list[str] = []
    stances = [item["stance"] for item in classifier_items]
    support_count = stances.count("SUPPORTS")
    counter_count = stances.count("CONTRADICTS")
    if support_count and counter_count and abs(support_count - counter_count) <= 1:
        signals.append("SUPPORT_COUNTER_CLOSE")
    if support_count and counter_count:
        signals.append("EVIDENCE_DISAGREEMENT")
    confidence = 0.0
    if judge_value is not None:
        try:
            confidence = float(judge_value.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
    if confidence >= 0.8 and (
        len(selected_ids) <= 1
        or any(
            float(item.get("directness", 0.0)) < 0.5 or float(item.get("relevance", 0.0)) < 0.5
            for item in audit_assessments
        )
    ):
        signals.append("HIGH_CONFIDENCE_LOW_EVIDENCE_STRENGTH")
    if len(selected_ids) <= 1:
        signals.append("LOW_EVIDENCE_COUNT")
    if re.search(
        r"\b(always|never|all|none|every|eliminat|guarante|best|fastest)\b|\d+%|\b\d+x\b",
        claim.text,
        re.I,
    ):
        signals.append("ABSOLUTE_OR_QUANTIFIED_WORDING")
    if any(
        not bool(item.get("scope_match", True)) or not bool(item.get("quantifier_match", True))
        for item in audit_assessments
    ):
        signals.append("SCOPE_OR_QUANTIFIER_MISMATCH")
    if any(
        any(
            issue.casefold()
            in {
                "fabricated evidence",
                "missing provenance",
                "metadata-only evidence",
                "invalid citation",
            }
            for issue in item.get("issues", [])
        )
        for item in audit_assessments
    ):
        signals.append("HARD_EVIDENCE_FAILURE")
    return list(dict.fromkeys(signals))


def _critic_spec(
    claim: SciFactClaim,
    architecture: str,
    documents: tuple[RetrievedDocument, ...],
    classifier_items: list[dict[str, Any]],
    audit_assessments: list[dict[str, Any]],
    judge_value: dict[str, Any],
    risk_signals: list[str],
    stage_providers: dict[str, str] | None,
) -> StageSpec:
    providers = {**DEFAULT_STAGE_PROVIDERS, **(stage_providers or {})}
    candidates = _candidate_payload(documents)
    allowed_ids = {item["evidence_id"] for item in candidates}
    proposed_verdict, confidence, selected_ids = _extract_verdict(judge_value, allowed_ids)
    input_payload = {
        "claim_id": claim.claim_id,
        "claim": claim.text,
        "proposed_verdict": proposed_verdict,
        "confidence": confidence,
        "selected_evidence_ids": selected_ids,
        "classifier_items": classifier_items,
        "audit_assessments": audit_assessments,
        "candidates": candidates,
        "risk_signals": risk_signals,
    }
    return StageSpec(
        task="critic",
        prompt_version=(
            PROMPT_VERSIONS["conditional_critic"]
            if architecture == "D4_CONDITIONAL_CRITIC"
            else PROMPT_VERSIONS["critic"]
        ),
        provider=providers["critic"],
        prompt=(
            "Return JSON only with decision PASS or CHALLENGE, reason, and evidence_ids. "
            "Do not produce a verdict and do not rewrite the judgment. Challenge only when "
            "there is a concrete unsupportedness reason grounded in supplied evidence. "
            + json.dumps(input_payload, ensure_ascii=False, sort_keys=True)
        ),
        input_payload=input_payload,
    )


def _run_example(
    *,
    claim: SciFactClaim,
    architecture: str,
    retriever: BM25Retriever,
    client: BenchmarkClient,
    documents: dict[str, SciFactDocument],
    stage_providers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[CallRecord]]:
    retrieved = () if architecture == "A_SINGLE_LLM" else retriever.retrieve(claim.text)
    retrieved_sentence_ids = [
        sentence.evidence_id for document in retrieved for sentence in document.sentences
    ]
    allowed_ids = set(retrieved_sentence_ids)
    records: list[CallRecord] = []
    errors: list[str] = []
    classifier_items: list[dict[str, Any]] = []
    audit_assessments: list[dict[str, Any]] = []
    judge_value: dict[str, Any] | None = None
    critic_value: dict[str, Any] | None = None
    pre_critic_verdict: str | None = None
    pre_critic_correct: bool | None = None

    def call_and_parse(spec: StageSpec) -> dict[str, Any] | None:
        result = client.call(architecture, spec)
        records.append(result.record)
        if result.error_category:
            errors.append(
                "SCHEMA_PARSE_FAILURE"
                if result.error_category == "schema_parse_failure"
                else "PROVIDER_FAILURE"
            )
            return None
        return result.parsed

    if not _uses_classifier(architecture):
        first_spec = _stage_specs(claim, architecture, retrieved, stage_providers=stage_providers)[
            0
        ]
        judge_value = call_and_parse(first_spec)
    else:
        classifier_spec = next(
            spec
            for spec in _stage_specs(
                claim, architecture, retrieved, stage_providers=stage_providers
            )
            if spec.task == "evidence_classifier"
        )
        classifier_value = call_and_parse(classifier_spec)
        if classifier_value is not None:
            classifier_items = _extract_classifier(classifier_value, allowed_ids)

        if _uses_auditor(architecture):
            audit_spec = next(
                spec
                for spec in _stage_specs(
                    claim,
                    architecture,
                    retrieved,
                    classifier_items=classifier_items,
                    stage_providers=stage_providers,
                )
                if spec.task == "auditor"
            )
            audit_value = call_and_parse(audit_spec)
            if audit_value is not None:
                audit_assessments = _extract_auditor(audit_value, allowed_ids)

        judge_spec = next(
            spec
            for spec in _stage_specs(
                claim,
                architecture,
                retrieved,
                classifier_items=classifier_items,
                audit_assessments=audit_assessments,
                stage_providers=stage_providers,
            )
            if spec.task == "judge"
        )
        judge_value = call_and_parse(judge_spec)

    risk_signals: list[str] = []
    critic_invoked = False
    recheck_performed = False
    if _uses_critic(architecture) and judge_value is not None:
        _predicted, _confidence, selected_ids = _extract_verdict(judge_value, allowed_ids)
        pre_critic_verdict = _predicted
        pre_critic_correct = _predicted == _gold_label(claim.gold_label)
        risk_signals = _critic_risk_signals(
            claim, classifier_items, judge_value, audit_assessments, selected_ids
        )
        critic_invoked = architecture != "D4_CONDITIONAL_CRITIC" or bool(risk_signals)
    if critic_invoked and judge_value is not None:
        critic_spec = _critic_spec(
            claim,
            architecture,
            retrieved,
            classifier_items,
            audit_assessments,
            judge_value,
            risk_signals,
            stage_providers,
        )
        critic_raw = call_and_parse(critic_spec)
        if critic_raw is not None:
            critic_value = _extract_critic(critic_raw, allowed_ids)
        if critic_value is not None and critic_value["decision"] == "CHALLENGE":
            recheck_performed = True
            recheck_spec = _stage_specs(
                claim,
                architecture,
                retrieved,
                classifier_items=classifier_items,
                audit_assessments=audit_assessments,
                stage_providers=stage_providers,
                critic_challenge=critic_value,
                recheck=True,
            )[0]
            rechecked_value = call_and_parse(recheck_spec)
            if rechecked_value is not None:
                judge_value = rechecked_value
            else:
                judge_value = None

    if judge_value is None:
        errors.append("SCHEMA_PARSE_FAILURE")
        predicted, confidence, selected_ids = "INSUFFICIENT_EVIDENCE", 0.2, []
    else:
        predicted, confidence, selected_ids = _extract_verdict(judge_value, allowed_ids)
    if architecture == "A_SINGLE_LLM":
        selected_ids = []
    gold_label = _gold_label(claim.gold_label)
    critic_effect = "NOT_INVOKED"
    if critic_invoked and pre_critic_correct is not None:
        post_correct = predicted == gold_label
        if pre_critic_correct is False and post_correct:
            critic_effect = "CORRECTED"
        elif pre_critic_correct and not post_correct:
            critic_effect = "DAMAGED"
        elif post_correct:
            critic_effect = "UNCHANGED_CORRECT"
        else:
            critic_effect = "UNCHANGED_WRONG"
    auditor_effect: dict[str, Any] | None = None
    if _uses_auditor(architecture):
        downgraded = sum(
            item["relevance"] < 0.5
            or item["directness"] < 0.5
            or not item["scope_match"]
            or not item["quantifier_match"]
            or not item["temporal_match"]
            for item in audit_assessments
        )
        rejected = sum(not item["usable"] for item in audit_assessments)
        auditor_effect = {
            "before_evidence_ids": list(retrieved_sentence_ids),
            "after_evidence_ids": list(retrieved_sentence_ids),
            "gold_evidence_retained": set(claim.gold_sentence_ids).issubset(
                set(retrieved_sentence_ids)
            ),
            "non_gold_removed": 0,
            "items_audited": len(audit_assessments),
            "items_downgraded": downgraded,
            "items_rejected": rejected,
            "scope_mismatches": sum(not item["scope_match"] for item in audit_assessments),
            "quantifier_mismatches": sum(
                not item["quantifier_match"] for item in audit_assessments
            ),
            "temporal_mismatches": sum(not item["temporal_match"] for item in audit_assessments),
            "effect": "NO_EFFECT",
        }
    provider_calls = sum(record.calls for record in records)
    cache_hits = sum(record.cache_hits for record in records)
    cached_input_tokens = sum(record.cached_input_tokens for record in records)
    cached_output_tokens = sum(record.cached_output_tokens for record in records)
    cached_latency_ms = sum(record.cached_latency_ms for record in records)
    prediction = {
        "claim_id": claim.claim_id,
        "architecture": architecture,
        "claim": claim.text,
        "gold_label": gold_label,
        "predicted_verdict": predicted,
        "mapped_prediction": predicted,
        "abstained": predicted in ABSTENTION_LABELS,
        "confidence": confidence,
        "correct": predicted == gold_label,
        "retrieved_document_ids": [document.document_id for document in retrieved],
        "retrieved_sentence_ids": retrieved_sentence_ids,
        "selected_evidence_ids": selected_ids,
        "gold_evidence_ids": list(claim.gold_sentence_ids),
        "gold_document_ids": list(claim.gold_document_ids),
        "llm_calls": provider_calls,
        "provider_calls": provider_calls,
        "stage_invocations": len(records),
        "logical_llm_calls": provider_calls + cache_hits,
        "cache_hits": cache_hits,
        "cache_misses": sum(record.cache_misses for record in records),
        "input_tokens": sum(record.input_tokens for record in records),
        "output_tokens": sum(record.output_tokens for record in records),
        "total_tokens": sum(record.input_tokens + record.output_tokens for record in records),
        "cached_input_tokens": cached_input_tokens,
        "cached_output_tokens": cached_output_tokens,
        "cached_total_tokens": cached_input_tokens + cached_output_tokens,
        "latency_ms": sum(record.latency_ms for record in records),
        "cached_latency_ms": cached_latency_ms,
        "errors": sorted(set(errors)),
        "retrieval_failure": False,
        "reasoning_failure": False,
        "prompt_versions": sorted({record.prompt_version for record in records}),
        "selected_stances": {item["evidence_id"]: item["stance"] for item in classifier_items},
        "auditor_assessment_count": len(audit_assessments),
        "assurance": {
            "critic_invoked": critic_invoked,
            "pre_critic_verdict": pre_critic_verdict,
            "pre_critic_correct": pre_critic_correct,
            "critic_decision": critic_value["decision"] if critic_value else None,
            "critic_reason": critic_value["reason"] if critic_value else None,
            "risk_signals": risk_signals,
            "recheck_performed": recheck_performed,
            "verdict_changed": pre_critic_verdict is not None and pre_critic_verdict != predicted,
            "effect": critic_effect,
        },
        "auditor_effect": auditor_effect,
    }
    return prediction, records


def select_sample(
    claims: tuple[SciFactClaim, ...], profile: str, seed: int
) -> tuple[SciFactClaim, ...]:
    if profile not in PROFILE_SIZES:
        raise EvaluationError(f"unknown profile: {profile}")
    target = min(PROFILE_SIZES[profile], len(claims))
    grouped: dict[str, list[SciFactClaim]] = defaultdict(list)
    for claim in claims:
        grouped[claim.gold_label].append(claim)
    import random

    selected: list[SciFactClaim] = []
    rng = random.Random(seed)
    for label in sorted(grouped):
        bucket = sorted(grouped[label], key=lambda item: int(item.claim_id))
        rng.shuffle(bucket)
        count = min(len(bucket), target * len(bucket) // len(claims))
        selected.extend(bucket[:count])
    remaining = [claim for claim in claims if claim not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: target - len(selected)])
    return tuple(sorted(selected[:target], key=lambda item: int(item.claim_id)))


def _aggregate_usage(records: list[CallRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            record.architecture,
            record.provider,
            record.configured_model,
            record.actual_model,
            record.task,
        )
        entry = grouped.setdefault(
            key,
            {
                "architecture": record.architecture,
                "provider": record.provider,
                "configured_model": record.configured_model,
                "actual_model": record.actual_model,
                "task": record.task,
                "request_count": 0,
                "stage_invocations": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "cached_output_tokens": 0,
                "cached_total_tokens": 0,
                "latency_ms": 0,
                "cached_latency_ms": 0,
                "failures": 0,
                "error_categories": {},
                "prompt_versions": set(),
            },
        )
        entry["request_count"] += record.calls
        entry["stage_invocations"] += record.calls + record.cache_hits
        entry["cache_hits"] += record.cache_hits
        entry["cache_misses"] += record.cache_misses
        entry["input_tokens"] += record.input_tokens
        entry["output_tokens"] += record.output_tokens
        entry["total_tokens"] += record.input_tokens + record.output_tokens
        entry["cached_input_tokens"] += record.cached_input_tokens
        entry["cached_output_tokens"] += record.cached_output_tokens
        entry["cached_total_tokens"] += record.cached_input_tokens + record.cached_output_tokens
        entry["latency_ms"] += record.latency_ms
        entry["cached_latency_ms"] += record.cached_latency_ms
        entry["failures"] += int(record.status == "FAILED")
        if record.error_category:
            categories = entry["error_categories"]
            categories[record.error_category] = categories.get(record.error_category, 0) + 1
        entry["prompt_versions"].add(record.prompt_version)
    for entry in grouped.values():
        entry["prompt_versions"] = sorted(entry["prompt_versions"])
    return list(
        sorted(
            grouped.values(),
            key=lambda item: (
                str(item["architecture"]),
                str(item["provider"]),
                str(item["task"]),
            ),
        )
    )


def _error_analysis(
    *,
    predictions: list[dict[str, Any]],
    claims: dict[str, SciFactClaim],
    documents: dict[str, SciFactDocument],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    by_architecture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if prediction["correct"]:
            continue
        claim = claims[prediction["claim_id"]]
        entries = _classify_error(claim, prediction, prediction["architecture"], documents)
        errors.extend(entries)
        by_architecture[prediction["architecture"]].extend(entries)
    category_counts = Counter(entry["category"] for entry in errors)
    high_confidence = [
        prediction
        for prediction in predictions
        if float(prediction["confidence"]) >= 0.8 and not prediction["correct"]
    ]
    low_confidence_correct = [
        prediction
        for prediction in predictions
        if float(prediction["confidence"]) <= 0.4 and prediction["correct"]
    ]
    architecture_order = tuple(dict.fromkeys(row["architecture"] for row in predictions))
    grouped = {
        architecture: [row for row in predictions if row["architecture"] == architecture]
        for architecture in architecture_order
    }
    disagreements: list[dict[str, Any]] = []
    for index, left in enumerate(architecture_order):
        for right in architecture_order[index + 1 :]:
            right_by_id = {row["claim_id"]: row for row in grouped[right]}
            for row in grouped[left]:
                other = right_by_id.get(row["claim_id"])
                if other is None:
                    continue
                if row["mapped_prediction"] != other["mapped_prediction"]:
                    disagreements.append(
                        {
                            "claim_id": row["claim_id"],
                            "claim": row["claim"],
                            "left_architecture": left,
                            "left_prediction": row["mapped_prediction"],
                            "right_architecture": right,
                            "right_prediction": other["mapped_prediction"],
                        }
                    )
    retrieval_failures = [entry for entry in errors if entry["category"] == "RETRIEVAL_MISS"]
    reasoning_failures = [
        prediction
        for prediction in predictions
        if prediction.get("reasoning_failure") and not prediction["correct"]
    ]
    return {
        "errors": errors,
        "category_counts": dict(sorted(category_counts.items())),
        "by_architecture": {key: value for key, value in by_architecture.items()},
        "retrieval_failures": retrieval_failures,
        "reasoning_failures": reasoning_failures,
        "high_confidence_errors": high_confidence,
        "low_confidence_correct": low_confidence_correct,
        "architecture_disagreements": disagreements[:100],
    }


def _recommendations(metrics: dict[str, Any]) -> dict[str, dict[str, str]]:
    available = {
        architecture: metrics[architecture]["claim"]["macro_f1"]
        for architecture in metrics
        if architecture in ARCHITECTURES + ISOLATION_ARCHITECTURES
        and metrics[architecture].get("complete", True)
        and metrics[architecture].get("valid_for_comparison", True)
        and isinstance(metrics[architecture].get("claim", {}).get("macro_f1"), (float, int))
    }
    if not available:
        return {"_summary": {"status": "NOT_COMPUTED_FOR_PARTIAL_ARCHITECTURE_RUN"}}
    efficiency = {
        architecture: metrics[architecture]["efficiency"].get("avg_stage_invocations", 0.0)
        for architecture in available
    }
    recommendations: dict[str, dict[str, str]] = {
        "Claim Analyzer": {
            "decision": "OPTIONAL",
            "reason": "Not isolated; SciFact claims are already atomic.",
        },
        "Claim Decomposer": {
            "decision": "OPTIONAL",
            "reason": "Not isolated; SciFact claims are already atomic.",
        },
        "Judge": {
            "decision": "KEEP",
            "reason": "Shared decision stage in evidence-aware architectures.",
        },
        "Deterministic Validator": {
            "decision": "KEEP",
            "reason": "Safety and provenance gate; not an LLM quality component.",
        },
    }
    if all(architecture in available for architecture in ARCHITECTURES):
        recommendations.update(
            {
                "Evidence Retrieval": {
                    "decision": (
                        "KEEP"
                        if available["B_RETRIEVAL_JUDGE"] >= available["A_SINGLE_LLM"]
                        else "REWORK"
                    ),
                    "reason": "A-to-B Macro F1 delta is measured in this run.",
                },
                "Support/Counter Classifier": {
                    "decision": (
                        "KEEP"
                        if available["C_SUPPORT_COUNTER"] > available["B_RETRIEVAL_JUDGE"]
                        else "OPTIONAL"
                    ),
                    "reason": "B-to-C Macro F1 delta is measured in this run.",
                },
                "Evidence Auditor": {
                    "decision": "OPTIONAL",
                    "reason": "C-to-D combined audit/critic effect is not isolated yet.",
                },
                "Critic": {
                    "decision": "OPTIONAL",
                    "reason": "C-to-D combined audit/critic effect is not isolated yet.",
                },
            }
        )
    if all(architecture in available for architecture in ISOLATION_ARCHITECTURES):
        recommendations["Evidence Auditor"] = {
            "decision": "KEEP_FOR_ISOLATION",
            "reason": "D1 isolates auditor contribution; do not promote it to critical path yet.",
        }
        recommendations["Critic"] = {
            "decision": "KEEP_FOR_ISOLATION",
            "reason": "D2/D4 isolate always-on and conditional critic behavior.",
        }
    recommendations["_summary"] = {
        "best_measured_architecture": max(available, key=available.get),
        "lowest_stage_invocation_architecture": min(efficiency, key=efficiency.get),
    }
    return recommendations


def _summary_markdown(
    *,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    error_analysis: dict[str, Any],
    recommendations: dict[str, dict[str, str]],
    sample_size: int,
    architecture_order: tuple[str, ...],
    budget_policy: dict[str, Any],
    budget_snapshot: dict[str, Any] | None,
    budget_stop_reason: str | None,
) -> str:
    rows = [
        ("Accuracy", "claim", "accuracy"),
        ("Macro F1", "claim", "macro_f1"),
        ("Evidence Precision", "evidence", "evidence_precision"),
        ("Evidence Recall", "evidence", "evidence_recall"),
        ("Evidence F1", "evidence", "evidence_f1"),
        ("Coverage", "abstention", "coverage"),
        ("Abstention Rate", "abstention", "abstention_rate"),
        ("Selective Accuracy", "abstention", "selective_accuracy"),
        ("Unsupported Verdict Rate", "unsupported_verdict", "unsupported_verdict_rate"),
        ("ECE", "calibration", "ece"),
        ("Brier", "calibration", "brier_score"),
        ("Avg Stage Invocations", "efficiency", "avg_stage_invocations"),
        ("Avg Provider Calls", "efficiency", "avg_provider_calls"),
        ("Avg Cache Hits", "efficiency", "avg_cache_hits"),
        ("Avg Provider Tokens", "efficiency", "avg_total_tokens"),
        ("Avg Provider Latency ms", "efficiency", "avg_latency_ms"),
    ]

    def delta_text(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    actual_calls = (budget_snapshot or {}).get(
        "actual_provider_calls", manifest.get("actual_provider_calls", 0)
    )
    actual_tokens = (budget_snapshot or {}).get(
        "actual_live_tokens", manifest.get("actual_live_tokens", 0)
    )
    lines = [
        "# SciFact evaluation",
        "",
        (
            "PILOT/SMOKE result under the recorded configuration; this is not a universal "
            "claim about multi-agent systems."
        ),
        "",
        (
            f"- Dataset: {manifest['dataset_name']} split `{manifest['split']}` "
            f"({sample_size} examples)"
        ),
        f"- Revision: `{manifest['dataset_revision']}`",
        f"- Models: {json.dumps(manifest['model_profile'], sort_keys=True)}",
        (
            f"- Retrieval: `{manifest['retrieval_config']['algorithm']}`; "
            f"top-k={manifest['retrieval_config']['top_k_documents']}"
        ),
        "",
        "## Budget",
        "",
        f"- Gate decision: {budget_policy.get('decision', 'NOT_APPLIED')}",
        f"- Estimated calls/tokens: {budget_policy.get('estimated_total_calls', 'N/A')} / "
        f"{budget_policy.get('estimated_total_tokens', 'N/A')}",
        f"- Actual calls/tokens: {actual_calls} / {actual_tokens}",
        f"- Headroom calls/tokens: {budget_policy.get('headroom_calls', 'N/A')} / "
        f"{budget_policy.get('headroom_tokens', 'N/A')}",
        f"- Stop reason: {budget_stop_reason or 'none'}",
        "",
        "## Architecture comparison",
        "",
        "| Metric | " + " | ".join(architecture_order) + " |",
        "| --- | " + " | ".join("---:" for _ in architecture_order) + " |",
    ]
    for label, group, key in rows:
        values: list[str] = []
        for architecture in architecture_order:
            value = metrics[architecture][group]
            if value == "N/A":
                values.append("N/A")
            else:
                raw = value.get(key)
                values.append(
                    "N/A" if raw is None else f"{raw:.4f}" if isinstance(raw, float) else str(raw)
                )
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            "## Ablation deltas",
            "",
            (
                "| Comparison | Macro F1 delta | Evidence F1 delta | Abstention delta | "
                "Stage invocations delta | Provider calls delta | Cache hits delta | "
                "Tokens delta | Latency delta |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    comparisons = (
        ISOLATION_COMPARISONS
        if all(architecture in architecture_order for architecture in ISOLATION_ARCHITECTURES)
        else ABLATION_COMPARISONS
    )
    ablation_key = "isolation_ablations" if comparisons == ISOLATION_COMPARISONS else "ablations"
    for comparison, _left, _right in comparisons:
        delta = metrics[ablation_key][comparison]
        evidence_delta = (
            f"{delta['evidence_f1_delta']:.4f}" if delta["evidence_f1_delta"] is not None else "N/A"
        )
        lines.append(
            f"| {comparison} | {delta_text(delta['macro_f1_delta'])} | {evidence_delta} | "
            f"{delta_text(delta['abstention_rate_delta'])} | "
            f"{delta_text(delta['avg_llm_calls_delta'])} | "
            f"{delta_text(delta['avg_provider_calls_delta'])} | "
            f"{delta_text(delta['avg_cache_hits_delta'])} | "
            f"{delta_text(delta['avg_total_tokens_delta'])} | "
            f"{delta_text(delta['avg_latency_ms_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Error analysis",
            "",
            f"- Top categories: `{json.dumps(error_analysis['category_counts'], sort_keys=True)}`",
            f"- Retrieval failures: {len(error_analysis['retrieval_failures'])}",
            f"- Reasoning failures: {len(error_analysis['reasoning_failures'])}",
            f"- High-confidence wrong predictions: {len(error_analysis['high_confidence_errors'])}",
            f"- Low-confidence correct predictions: "
            f"{len(error_analysis['low_confidence_correct'])}",
            f"- Architecture disagreement cases retained: "
            f"{len(error_analysis['architecture_disagreements'])}",
            "",
            "## Recommendation",
            "",
        ]
    )
    for component, value in recommendations.items():
        if component == "_summary":
            continue
        lines.append(f"- {component}: **{value['decision']}** — {value['reason']}")
    for architecture in ("D1_AUDITOR", "D3_AUDITOR_CRITIC"):
        if architecture in metrics:
            auditor = metrics[architecture]["auditor"]
            lines.extend(
                [
                    "",
                    f"## Auditor effects ({architecture})",
                    "",
                    f"- Items audited/downgraded/rejected: {auditor['items_audited']} / "
                    f"{auditor['items_downgraded']} / {auditor['items_rejected']}",
                    f"- Helpful filters: {auditor['helpful_filters']}; harmful filters: "
                    f"{auditor['harmful_filters']}; no effect: {auditor['no_effect']}",
                    f"- Scope/quantifier/temporal mismatches: {auditor['scope_mismatches']} / "
                    f"{auditor['quantifier_mismatches']} / {auditor['temporal_mismatches']}",
                ]
            )
    for architecture in ("D2_CRITIC", "D3_AUDITOR_CRITIC", "D4_CONDITIONAL_CRITIC"):
        if architecture in metrics:
            critic = metrics[architecture]["critic"]
            lines.extend(
                [
                    "",
                    f"## Critic effects ({architecture})",
                    "",
                    f"- Invocations/PASS/CHALLENGE/rechecks: {critic['critic_invocation_count']} / "
                    f"{critic['critic_pass_count']} / {critic['critic_challenge_count']} / "
                    f"{critic['rejudge_count']}",
                    f"- Wrong-to-right/right-to-wrong/net: {critic['wrong_to_right']} / "
                    f"{critic['right_to_wrong']} / {critic['net_critic_corrections']}",
                ]
            )
            if architecture == "D4_CONDITIONAL_CRITIC":
                efficiency = metrics[architecture]["efficiency"]
                invocation_rate = critic["critic_invocation_count"] / max(
                    critic["claims_evaluated"], 1
                )
                lines.append(
                    "- D4 invocation rate: "
                    f"{invocation_rate:.4f}; "
                    "savings vs D2 calls/tokens/latency: "
                    f"{efficiency.get('calls_saved_vs_D2', 'N/A')} / "
                    f"{efficiency.get('tokens_saved_vs_D2', 'N/A')} / "
                    f"{efficiency.get('latency_saved_vs_D2', 'N/A')}"
                )
    lines.extend(
        [
            "",
            (
                "Calibration is reported as `CALIBRATION_SAMPLE_TOO_SMALL` below 20 examples; "
                "confidence remains a run-verdict heuristic, not an objective probability."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    _write_json(temporary, value)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_value(arguments: list[str], default: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return default
    return result.stdout.strip() or default


def _availability(settings: Settings, providers: tuple[str, ...] | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {}
    requested = tuple(dict.fromkeys(providers or ("groq", "gemini")))
    if "groq" in requested:
        groq_api_key = secret_value(settings.groq_api_key)
        if not groq_api_key or not settings.groq_enabled:
            results["groq"] = {"status": "UNCONFIGURED", "configured_model": settings.groq_model}
        else:
            try:
                response = httpx.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {groq_api_key}"},
                    timeout=10,
                )
                data = response.json() if response.status_code < 400 else {}
            except (httpx.HTTPError, ValueError):
                data = {}
            model_ids = {
                str(item.get("id")) for item in data.get("data", []) if isinstance(item, dict)
            }
            results["groq"] = {
                "status": (
                    "AVAILABLE"
                    if settings.groq_model in model_ids
                    else "UNAVAILABLE"
                    if not model_ids
                    else "MODEL_NOT_LISTED"
                ),
                "configured_model": settings.groq_model,
            }
    if "gemini" in requested:
        gemini_api_key = secret_value(settings.gemini_api_key)
        if not gemini_api_key or not settings.gemini_enabled:
            results["gemini"] = {
                "status": "UNCONFIGURED",
                "configured_model": settings.gemini_model,
            }
        else:
            try:
                response = httpx.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": gemini_api_key},
                    timeout=10,
                )
                data = response.json() if response.status_code < 400 else {}
            except (httpx.HTTPError, ValueError):
                data = {}
            model_ids = {
                str(item.get("name", "")).removeprefix("models/")
                for item in data.get("models", [])
                if isinstance(item, dict)
            }
            results["gemini"] = {
                "status": (
                    "AVAILABLE"
                    if settings.gemini_model in model_ids
                    else "UNAVAILABLE"
                    if not model_ids
                    else "MODEL_NOT_LISTED"
                ),
                "configured_model": settings.gemini_model,
            }
    if "okmd" in requested:
        okmd_api_key = secret_value(settings.okmd_api_key)
        if not okmd_api_key or not settings.okmd_enabled:
            results["okmd"] = {
                "status": "UNCONFIGURED",
                "configured_model": settings.okmd_model,
            }
        else:
            try:
                response = httpx.get(
                    f"{OKMD_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {okmd_api_key}"},
                    timeout=10,
                )
                data = response.json() if response.status_code < 400 else {}
            except (httpx.HTTPError, ValueError):
                data = {}
            model_ids = {
                str(item.get("id")) for item in data.get("data", []) if isinstance(item, dict)
            }
            results["okmd"] = {
                "status": (
                    "AVAILABLE"
                    if settings.okmd_model in model_ids
                    else "UNAVAILABLE"
                    if not model_ids
                    else "MODEL_NOT_LISTED"
                ),
                "configured_model": settings.okmd_model,
            }
    return results


def build_live_providers(
    settings: Settings, required_providers: tuple[str, ...] | None = None
) -> tuple[dict[str, BenchmarkProvider], dict[str, Any]]:
    requested = tuple(dict.fromkeys(required_providers or ("groq", "gemini")))
    if any(provider not in FIXED_MODELS for provider in requested):
        raise EvaluationError("unknown live provider")
    if any(
        getattr(settings, f"{provider}_model") != FIXED_MODELS[provider] for provider in requested
    ):
        raise EvaluationError(
            "fixed benchmark model mismatch; set the selected provider model to its "
            "fixed benchmark model explicitly"
        )
    availability = _availability(settings, requested)
    unavailable = [name for name, result in availability.items() if result["status"] != "AVAILABLE"]
    if unavailable:
        raise EvaluationError(f"fixed benchmark model unavailable: {','.join(unavailable)}")
    providers: dict[str, BenchmarkProvider] = {}
    if "groq" in requested:
        providers["groq"] = OpenAICompatibleProvider(
            "groq",
            settings.groq_model,
            secret_value(settings.groq_api_key) or "",
            "https://api.groq.com/openai/v1",
        )
    if "gemini" in requested:
        providers["gemini"] = GeminiProvider(
            "gemini", settings.gemini_model, secret_value(settings.gemini_api_key) or ""
        )
    if "okmd" in requested:
        providers["okmd"] = OpenAICompatibleProvider(
            "okmd",
            settings.okmd_model,
            secret_value(settings.okmd_api_key) or "",
            OKMD_BASE_URL,
        )
    return providers, availability


def _efficiency(predictions: list[dict[str, Any]]) -> dict[str, float]:
    count = len(predictions)
    return {
        "avg_llm_calls": sum(row["stage_invocations"] for row in predictions) / max(count, 1),
        "avg_stage_invocations": sum(row["stage_invocations"] for row in predictions)
        / max(count, 1),
        "avg_provider_calls": sum(row["provider_calls"] for row in predictions) / max(count, 1),
        "avg_cache_hits": sum(row["cache_hits"] for row in predictions) / max(count, 1),
        "avg_total_tokens": sum(row["total_tokens"] for row in predictions) / max(count, 1),
        "avg_latency_ms": sum(row["latency_ms"] for row in predictions) / max(count, 1),
        "provider_failures": sum("PROVIDER_FAILURE" in row["errors"] for row in predictions),
        "provider_calls": sum(row["provider_calls"] for row in predictions),
        "stage_invocations": sum(row["stage_invocations"] for row in predictions),
        "cache_hits": sum(row["cache_hits"] for row in predictions),
        "cache_misses": sum(row["cache_misses"] for row in predictions),
        "input_tokens": sum(row["input_tokens"] for row in predictions),
        "output_tokens": sum(row["output_tokens"] for row in predictions),
        "cached_total_tokens": sum(row["cached_total_tokens"] for row in predictions),
        "cached_latency_ms": sum(row["cached_latency_ms"] for row in predictions),
    }


ABLATION_COMPARISONS = (
    ("A → B", "A_SINGLE_LLM", "B_RETRIEVAL_JUDGE"),
    ("B → C", "B_RETRIEVAL_JUDGE", "C_SUPPORT_COUNTER"),
    ("C → D", "C_SUPPORT_COUNTER", "D_FULL_VERICLAIM"),
    ("A → D", "A_SINGLE_LLM", "D_FULL_VERICLAIM"),
)
ISOLATION_COMPARISONS = (
    ("C → D1", "C_SUPPORT_COUNTER", "D1_AUDITOR"),
    ("C → D2", "C_SUPPORT_COUNTER", "D2_CRITIC"),
    ("C → D3", "C_SUPPORT_COUNTER", "D3_AUDITOR_CRITIC"),
    ("C → D4", "C_SUPPORT_COUNTER", "D4_CONDITIONAL_CRITIC"),
)


def _compute_ablations(
    metrics: dict[str, Any],
    comparisons: tuple[tuple[str, str, str], ...] = ABLATION_COMPARISONS,
) -> dict[str, dict[str, float | None]]:
    names = tuple(name for name, _left, _right in comparisons)
    if any(
        architecture not in metrics
        or not metrics[architecture].get("complete", True)
        or not metrics[architecture].get("valid_for_comparison", True)
        or not isinstance(metrics[architecture].get("claim", {}).get("macro_f1"), (float, int))
        for architecture in {
            architecture for _name, left, right in comparisons for architecture in (left, right)
        }
    ):
        return {
            name: {
                "macro_f1_delta": None,
                "evidence_f1_delta": None,
                "abstention_rate_delta": None,
                "avg_llm_calls_delta": None,
                "avg_provider_calls_delta": None,
                "avg_cache_hits_delta": None,
                "avg_total_tokens_delta": None,
                "avg_latency_ms_delta": None,
            }
            for name in names
        }
    ablations: dict[str, dict[str, float | None]] = {}
    for name, left, right in comparisons:
        left_metrics, right_metrics = metrics[left], metrics[right]
        left_evidence = left_metrics["evidence"]
        right_evidence = right_metrics["evidence"]
        ablations[name] = {
            "macro_f1_delta": right_metrics["claim"]["macro_f1"]
            - left_metrics["claim"]["macro_f1"],
            "evidence_f1_delta": (
                None
                if left_evidence == "N/A" or right_evidence == "N/A"
                else right_evidence["evidence_f1"] - left_evidence["evidence_f1"]
            ),
            "abstention_rate_delta": right_metrics["abstention"]["abstention_rate"]
            - left_metrics["abstention"]["abstention_rate"],
            "avg_llm_calls_delta": right_metrics["efficiency"]["avg_llm_calls"]
            - left_metrics["efficiency"]["avg_llm_calls"],
            "avg_provider_calls_delta": right_metrics["efficiency"]["avg_provider_calls"]
            - left_metrics["efficiency"]["avg_provider_calls"],
            "avg_cache_hits_delta": right_metrics["efficiency"]["avg_cache_hits"]
            - left_metrics["efficiency"]["avg_cache_hits"],
            "avg_total_tokens_delta": right_metrics["efficiency"]["avg_total_tokens"]
            - left_metrics["efficiency"]["avg_total_tokens"],
            "avg_latency_ms_delta": right_metrics["efficiency"]["avg_latency_ms"]
            - left_metrics["efficiency"]["avg_latency_ms"],
        }
    return ablations


def run_benchmark(
    *,
    corpus: SciFactCorpus,
    manifest: dict[str, Any],
    profile: str,
    seed: int,
    architectures: tuple[str, ...],
    output_root: str | Path,
    cache_root: str | Path,
    providers: dict[str, BenchmarkProvider],
    model_profile: dict[str, Any],
    stage_providers: dict[str, str] | None = None,
    retrieval_config: RetrievalConfig | None = None,
    dry_run: bool = False,
    sample_claim_ids: tuple[str, ...] | None = None,
    budget_gate: LiveBudgetGate | None = None,
    budget_policy: dict[str, Any] | None = None,
    benchmark_run_id: str | None = None,
    checkpoint_path: str | Path | None = None,
    resumable: bool = False,
    cache_context_extra: dict[str, Any] | None = None,
    locked_actual_model: str | None = None,
    pacing_seconds: float = 0.0,
    enforce_provider_min_interval: bool = True,
    window_id: str | None = None,
    window_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if any(architecture not in ALL_ARCHITECTURES for architecture in architectures):
        raise EvaluationError("unknown architecture")
    if resumable and benchmark_run_id is None:
        raise EvaluationError("resumable benchmark requires a stable benchmark run ID")
    if sample_claim_ids is None:
        selected = select_sample(corpus.claims, profile, seed)
    else:
        claim_by_id = {claim.claim_id: claim for claim in corpus.claims}
        missing = [claim_id for claim_id in sample_claim_ids if claim_id not in claim_by_id]
        if missing:
            raise EvaluationError(f"sample claim IDs missing from dataset: {','.join(missing)}")
        selected = tuple(claim_by_id[claim_id] for claim_id in sample_claim_ids)
    cache = StructuredResponseCache(cache_root)
    retriever = BM25Retriever(corpus, retrieval_config)
    cache_context = {
        "dataset_revision": manifest.get("revision"),
        "retrieval_config": asdict(retriever.config),
        **(cache_context_extra or {}),
    }
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint: dict[str, Any] = {}
    if resumable and checkpoint_file is None:
        raise EvaluationError("resumable benchmark requires a checkpoint path")
    if resumable and checkpoint_file is not None and checkpoint_file.is_file():
        try:
            checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError("resumable checkpoint is unreadable") from exc
        expected_identity = model_profile.get("resumable_identity")
        if checkpoint.get("resumable_identity") != expected_identity:
            raise EvaluationError("resumable checkpoint identity mismatch; start a new run")
        if checkpoint.get("cache_context") != cache_context:
            raise EvaluationError("resumable cache context mismatch; start a new run")
        if checkpoint.get("benchmark_run_id") != benchmark_run_id:
            raise EvaluationError("resumable benchmark run ID mismatch")
    client = BenchmarkClient(
        providers,
        cache,
        dry_run=dry_run,
        budget_gate=budget_gate,
        cache_context=cache_context,
        locked_actual_model=locked_actual_model,
        pacing_seconds=pacing_seconds,
        enforce_provider_min_interval=enforce_provider_min_interval,
    )
    predictions: list[dict[str, Any]] = [
        dict(row) for row in checkpoint.get("predictions", []) if isinstance(row, dict)
    ]
    historical_records = [
        CallRecord(**row) for row in checkpoint.get("records", []) if isinstance(row, dict)
    ]
    window_history: list[dict[str, Any]] = [
        dict(row) for row in checkpoint.get("windows", []) if isinstance(row, dict)
    ]
    checkpoint_locked_actual_model = checkpoint.get("locked_actual_model")
    if locked_actual_model is not None and checkpoint_locked_actual_model not in {
        None,
        locked_actual_model,
    }:
        raise EvaluationError("resumable checkpoint actual-model lock mismatch")
    if checkpoint_locked_actual_model is not None:
        client.locked_actual_model = str(checkpoint_locked_actual_model)
    claim_lookup = {claim.claim_id: claim for claim in selected}
    budget_stop_reason: str | None = None
    completed_pairs = {
        (str(row.get("architecture")), str(row.get("claim_id")))
        for row in predictions
        if not row.get("errors")
    }
    checkpoint_cursor = checkpoint.get("resume_cursor", {})
    resume_pair_hint: tuple[str, str] | None = None
    resume_stage_hint: str | None = None
    if isinstance(checkpoint_cursor, dict):
        raw_pair = checkpoint_cursor.get("first_missing_pair")
        if isinstance(raw_pair, dict) and raw_pair.get("architecture") and raw_pair.get("claim_id"):
            resume_pair_hint = (
                str(raw_pair["architecture"]),
                str(raw_pair["claim_id"]),
            )
            raw_stage = checkpoint_cursor.get("first_missing_stage")
            if isinstance(raw_stage, str) and raw_stage not in {
                "",
                "cache-replayed-pipeline",
            }:
                resume_stage_hint = raw_stage

    def resume_cursor_snapshot() -> dict[str, Any]:
        first_missing: dict[str, str] | None = None
        for architecture in architectures:
            for claim in selected:
                if (architecture, claim.claim_id) not in completed_pairs:
                    first_missing = {
                        "architecture": architecture,
                        "claim_id": claim.claim_id,
                    }
                    break
            if first_missing is not None:
                break
        first_missing_key = (
            (first_missing["architecture"], first_missing["claim_id"])
            if first_missing is not None
            else None
        )
        return {
            "completed_pairs": len(completed_pairs),
            "expected_pairs": len(selected) * len(architectures),
            "first_missing_pair": first_missing,
            "first_missing_stage": (
                resume_stage_hint
                if first_missing_key is not None and first_missing_key == resume_pair_hint
                else "cache-replayed-pipeline"
                if first_missing is not None
                else None
            ),
            "resume_strategy": (
                "reuse exact structured stage cache, then call first uncached stage"
            ),
        }

    def persist_checkpoint() -> None:
        if not resumable or checkpoint_file is None:
            return
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        current_records = historical_records + client.telemetry_records
        actual_models_now = sorted(
            {
                record.actual_model
                for record in current_records
                if record.actual_model and record.status in {"success", "CACHE_HIT"}
            }
        )
        locked_now = client.locked_actual_model or (
            actual_models_now[0] if len(actual_models_now) == 1 else None
        )
        _write_json_atomic(
            checkpoint_file,
            {
                "checkpoint_version": 1,
                "benchmark_run_id": benchmark_run_id,
                "resumable_identity": model_profile.get("resumable_identity"),
                "cache_context": cache_context,
                "locked_actual_model": locked_now,
                "predictions": predictions,
                "records": [asdict(record) for record in current_records],
                "windows": window_history,
                "resume_cursor": resume_cursor_snapshot(),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    client.checkpoint_callback = persist_checkpoint
    persist_checkpoint()
    for architecture in architectures:
        for claim in selected:
            if resumable and (architecture, claim.claim_id) in completed_pairs:
                continue
            try:
                prediction, stage_records = _run_example(
                    claim=claim,
                    architecture=architecture,
                    retriever=retriever,
                    client=client,
                    documents=corpus.documents,
                    stage_providers=stage_providers,
                )
            except BudgetGateDenied as exc:
                budget_stop_reason = str(exc)
                if client.telemetry_records and (
                    resume_pair_hint is None or resume_pair_hint == (architecture, claim.claim_id)
                ):
                    resume_pair_hint = (architecture, claim.claim_id)
                    resume_stage_hint = client.telemetry_records[-1].task
                break
            if not resumable or not prediction.get("errors"):
                predictions.append(prediction)
            if resumable and not prediction.get("errors"):
                completed_pairs.add((architecture, claim.claim_id))
                if resume_pair_hint == (architecture, claim.claim_id):
                    resume_pair_hint = None
                    resume_stage_hint = None
                persist_checkpoint()
            elif resumable:
                failed_records = [
                    record
                    for record in reversed(stage_records)
                    if record.status in {"FAILED", "BUDGET_STOP", "SKIPPED_PROVIDER_BLOCK"}
                ]
                if failed_records and (
                    resume_pair_hint is None or resume_pair_hint == (architecture, claim.claim_id)
                ):
                    resume_pair_hint = (architecture, claim.claim_id)
                    resume_stage_hint = failed_records[0].task
        if budget_stop_reason is not None:
            break
    records = historical_records + client.telemetry_records
    if resumable:
        actual_models_now = sorted(
            {
                record.actual_model
                for record in records
                if record.actual_model and record.status in {"success", "CACHE_HIT"}
            }
        )
        if len(actual_models_now) == 1:
            client.locked_actual_model = client.locked_actual_model or actual_models_now[0]
        if window_id is not None:
            current_window = {
                **(window_policy or {}),
                "window_id": window_id,
                "finished_at": datetime.now(UTC).isoformat(),
                "provider": model_profile.get("benchmark_provider", "UNVERIFIED"),
                "configured_model": model_profile.get("benchmark_configured_model", "UNVERIFIED"),
                "actual_model": client.locked_actual_model,
                "configuration_hash": model_profile.get("configuration_hash"),
                "reliability_profile_hash": model_profile.get("reliability_profile_hash"),
                "calls_attempted": budget_gate.actual_calls if budget_gate else 0,
                "calls_succeeded": sum(
                    record.calls
                    for record in client.telemetry_records
                    if record.status == "success"
                ),
                "tokens_consumed": budget_gate.actual_tokens if budget_gate else 0,
                "cache_hits": sum(record.cache_hits for record in client.telemetry_records),
                "failure_categories": sorted(
                    {
                        record.error_category
                        for record in client.telemetry_records
                        if record.error_category is not None
                    }
                ),
                "provider_failures": sum(
                    record.status == "FAILED" for record in client.telemetry_records
                ),
                "resume_cursor": resume_cursor_snapshot(),
                "stop_reason": budget_stop_reason,
            }
            window_history.append(current_window)
        persist_checkpoint()
    by_architecture = {
        architecture: [row for row in predictions if row["architecture"] == architecture]
        for architecture in architectures
    }
    metrics: dict[str, Any] = {}
    for architecture, rows in by_architecture.items():
        measured = compute_metrics(rows, evidence_applicable=architecture != "A_SINGLE_LLM")
        measured["efficiency"] = _efficiency(rows)
        measured["complete"] = len(rows) == len(selected)
        measured["valid_for_comparison"] = measured["complete"] and all(
            not row.get("errors") for row in rows
        )
        metrics[architecture] = measured
    if "D2_CRITIC" in metrics and "D4_CONDITIONAL_CRITIC" in metrics:
        d2_efficiency = metrics["D2_CRITIC"]["efficiency"]
        d4_efficiency = metrics["D4_CONDITIONAL_CRITIC"]["efficiency"]
        d4_efficiency["tokens_saved_vs_D2"] = (
            d2_efficiency["avg_total_tokens"] - d4_efficiency["avg_total_tokens"]
        )
        d4_efficiency["calls_saved_vs_D2"] = (
            d2_efficiency["avg_provider_calls"] - d4_efficiency["avg_provider_calls"]
        )
        d4_efficiency["latency_saved_vs_D2"] = (
            d2_efficiency["avg_latency_ms"] - d4_efficiency["avg_latency_ms"]
        )
    metrics["ablations"] = _compute_ablations(metrics)
    metrics["isolation_ablations"] = _compute_ablations(metrics, ISOLATION_COMPARISONS)
    error_analysis = _error_analysis(
        predictions=predictions,
        claims=claim_lookup,
        documents=corpus.documents,
    )
    benchmark_id = benchmark_run_id or (
        f"scifact-{profile}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    )
    output = Path(output_root) / benchmark_id
    output.mkdir(parents=True, exist_ok=resumable)
    actual_models = sorted({record.actual_model for record in records if record.actual_model})
    configured_models = sorted(
        {record.configured_model for record in records if record.configured_model}
    )
    actual_providers = sorted({record.provider for record in records if record.calls})
    recorded_provider_failures = sum(record.status == "FAILED" for record in records)
    provider_failures = max(
        recorded_provider_failures,
        budget_gate.provider_failures if budget_gate is not None else 0,
    )
    model_substitutions = sum(record.error_category == "model_substitution" for record in records)
    model_drift = sum(record.error_category == "model_drift" for record in records)
    expected_prediction_count = len(selected) * len(architectures)
    valid_paired_run = (
        len(predictions) == expected_prediction_count
        and all(metrics[architecture]["valid_for_comparison"] for architecture in architectures)
        and model_substitutions == 0
        and model_drift == 0
        and budget_stop_reason is None
        and (resumable or provider_failures == 0)
    )
    benchmark_manifest = {
        "benchmark_run_id": benchmark_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_value(["rev-parse", "HEAD"], "NO_COMMIT"),
        "dirty_worktree": bool(_git_value(["status", "--porcelain"], "")),
        "dataset_name": manifest.get("dataset_name", "SciFact"),
        "dataset_source": manifest.get("source"),
        "dataset_revision": manifest.get("revision"),
        "dataset_hashes": {item["path"]: item["sha256"] for item in manifest.get("files", [])},
        "split": corpus.split,
        "sample_profile": profile,
        "sample_size": len(selected),
        "sample_ids": [claim.claim_id for claim in selected],
        "seed": seed,
        "architectures": list(architectures),
        "architecture_group": (
            "isolation" if tuple(architectures) == ISOLATION_ARCHITECTURES else "primary"
        ),
        "provider": model_profile,
        "configured_models": model_profile,
        "actual_models": actual_models,
        "benchmark_provider": model_profile.get(
            "benchmark_provider",
            actual_providers[0] if len(actual_providers) == 1 else "UNVERIFIED",
        ),
        "benchmark_configured_model": model_profile.get(
            "benchmark_configured_model",
            configured_models[0] if len(configured_models) == 1 else "UNVERIFIED",
        ),
        "benchmark_actual_model": actual_models[0] if len(actual_models) == 1 else "UNVERIFIED",
        "model_substitutions": model_substitutions,
        "provider_failures": provider_failures,
        "model_drift": model_drift,
        "expected_prediction_count": expected_prediction_count,
        "prediction_count": len(predictions),
        "valid_paired_predictions": expected_prediction_count if valid_paired_run else 0,
        "valid_paired_run": valid_paired_run,
        "resumable": resumable,
        "configuration_hash": model_profile.get("configuration_hash"),
        "locked_actual_model": client.locked_actual_model,
        "windows": window_history,
        "resume_cursor": resume_cursor_snapshot(),
        "model_profile": model_profile,
        "stage_providers": {**DEFAULT_STAGE_PROVIDERS, **(stage_providers or {})},
        "prompt_versions": PROMPT_VERSIONS,
        "generation_parameters": {
            "default": GENERATION_PARAMETERS,
            "task_overrides": TASK_GENERATION_PARAMETERS,
        },
        "retrieval_config": asdict(retriever.config),
        "cache_version": CACHE_VERSION,
        "cache_directory": str(Path(cache_root).resolve()),
        "free_tier_usage": {
            "api_cost_usd": "N/A",
            "classification": "FREE_TIER_USAGE",
            "openrouter_used": False,
            "paid_api_usage": "FORBIDDEN",
        },
        "budget_policy": budget_policy or {"decision": "NOT_APPLIED"},
        "budget_gate_snapshot": None,
        "budget_stop_reason": budget_stop_reason,
        "decision_rules": {
            "auditor": (
                "KEEP only with deterministic quality/safety benefit at defensible cost; "
                "otherwise OPTIONAL or REMOVE_FROM_CRITICAL_PATH."
            ),
            "always_on_critic": (
                "KEEP only when WRONG_TO_RIGHT exceeds RIGHT_TO_WRONG and added quality "
                "justifies cost; otherwise REMOVE or CONDITIONAL_ONLY."
            ),
            "conditional_critic": (
                "Prefer D4 over D2 only when quality is similar or better with lower "
                "invocation rate and meaningful savings."
            ),
            "production_candidate": (
                "Select at most one primary and one optional candidate; do not change "
                "production default on a sample that is too small."
            ),
        },
        "dry_run": dry_run,
    }
    recommendations = _recommendations(metrics) if not dry_run else {}
    provider_usage = {
        "records": [asdict(record) for record in records],
        "aggregate": _aggregate_usage(records),
        "provider_failures": provider_failures,
        "recorded_provider_failures": recorded_provider_failures,
        "unrecorded_provider_failures": provider_failures - recorded_provider_failures,
        "model_substitutions": model_substitutions,
        "model_drift": model_drift,
        "telemetry_invariant": (
            "every provider inference attempt has exactly one record; "
            "budget-stop and provider-block events have calls=0"
        ),
        "telemetry_record_count": len(records),
        "provider_attempt_record_count": sum(record.calls for record in records),
        "window_history": window_history,
        "retries": 0,
        "fallbacks": [],
    }
    if budget_gate is not None:
        benchmark_manifest["budget_gate_snapshot"] = budget_gate.snapshot()
    benchmark_manifest["actual_provider_calls"] = (
        sum(record.calls for record in records)
        if resumable
        else budget_gate.actual_calls
        if budget_gate is not None
        else sum(record.calls for record in records)
    )
    benchmark_manifest["actual_live_tokens"] = (
        sum(record.input_tokens + record.output_tokens for record in records)
        if resumable
        else budget_gate.actual_tokens
        if budget_gate is not None
        else sum(record.input_tokens + record.output_tokens for record in records)
    )
    benchmark_manifest["telemetry_record_count"] = len(records)
    benchmark_manifest["provider_attempt_record_count"] = sum(record.calls for record in records)
    benchmark_manifest["unrecorded_provider_attempts"] = max(
        0,
        benchmark_manifest["actual_provider_calls"]
        - benchmark_manifest["provider_attempt_record_count"],
    )
    scope_fixture = Path("evals/fixtures/scope_quantifier.json")
    metrics["scope_quantifier"] = (
        evaluate_fixture(scope_fixture)
        if scope_fixture.is_file()
        else {"status": "FIXTURE_NOT_FOUND"}
    )
    metrics["provider_usage"] = provider_usage
    metrics["recommendations"] = recommendations
    _write_json(output / "manifest.json", benchmark_manifest)
    _write_json(output / "metrics.json", metrics)
    _write_jsonl(output / "predictions.jsonl", predictions)
    _write_json(output / "provider_usage.json", provider_usage)
    _write_jsonl(output / "errors.jsonl", error_analysis["errors"])
    _write_json(output / "error_analysis.json", error_analysis)
    critic_effects = [
        row
        for row in predictions
        if row["assurance"].get("critic_invoked")
        or row["architecture"] in {"D2_CRITIC", "D3_AUDITOR_CRITIC", "D4_CONDITIONAL_CRITIC"}
    ]
    auditor_effects = [row for row in predictions if row.get("auditor_effect") is not None]
    _write_jsonl(output / "critic_effects.jsonl", critic_effects)
    _write_jsonl(output / "auditor_effects.jsonl", auditor_effects)
    _write_csv(
        output / "metrics.csv",
        [
            {
                "architecture": architecture,
                "macro_f1": value["claim"].get("macro_f1"),
                "accuracy": value["claim"].get("accuracy"),
                "evidence_f1": value["evidence"].get("evidence_f1")
                if value["evidence"] != "N/A"
                else "N/A",
                "coverage": value["abstention"].get("coverage"),
                "avg_llm_calls": value["efficiency"].get("avg_llm_calls"),
                "avg_stage_invocations": value["efficiency"].get("avg_stage_invocations"),
                "avg_provider_calls": value["efficiency"].get("avg_provider_calls"),
                "avg_cache_hits": value["efficiency"].get("avg_cache_hits"),
                "avg_total_tokens": value["efficiency"].get("avg_total_tokens"),
                "avg_latency_ms": value["efficiency"].get("avg_latency_ms"),
            }
            for architecture, value in metrics.items()
            if architecture in architectures
        ],
    )
    _write_csv(output / "provider_usage.csv", provider_usage["aggregate"])
    _write_csv(output / "errors.csv", error_analysis["errors"])
    _write_csv(output / "critic_effects.csv", critic_effects)
    _write_csv(output / "auditor_effects.csv", auditor_effects)
    _write_json(output / "scope_quantifier.json", metrics["scope_quantifier"])
    benchmark_manifest["result_artifacts"] = sorted(path.name for path in output.iterdir())
    _write_json(output / "manifest.json", benchmark_manifest)
    (output / "summary.md").write_text(
        _summary_markdown(
            manifest={
                **benchmark_manifest,
                "retrieval_config": benchmark_manifest["retrieval_config"],
            },
            metrics=metrics,
            error_analysis=error_analysis,
            recommendations=recommendations,
            sample_size=len(selected),
            architecture_order=architectures,
            budget_policy=benchmark_manifest["budget_policy"],
            budget_snapshot=benchmark_manifest["budget_gate_snapshot"],
            budget_stop_reason=budget_stop_reason,
        ),
        encoding="utf-8",
    )
    benchmark_manifest["result_artifacts"] = sorted(path.name for path in output.iterdir())
    _write_json(output / "manifest.json", benchmark_manifest)
    return {
        "benchmark_run_id": benchmark_id,
        "output_directory": str(output.resolve()),
        "manifest": benchmark_manifest,
        "metrics": metrics,
        "predictions": predictions,
        "provider_usage": provider_usage,
        "error_analysis": error_analysis,
    }


def load_and_validate_dataset(
    data_dir: str | Path, manifest_path: str | Path, split: str
) -> tuple[SciFactCorpus, dict[str, Any]]:
    manifest = validate_manifest(manifest_path)
    from .dataset import load_scifact

    corpus = load_scifact(data_dir, split=split)
    if manifest.get("split_rows", {}).get(split, {}).get("rows") != len(corpus.claims):
        raise EvaluationError("dataset manifest split row count does not match loaded split")
    return corpus, manifest


def run_dry_run(
    *,
    corpus: SciFactCorpus,
    manifest: dict[str, Any],
    profile: str,
    seed: int,
    architectures: tuple[str, ...],
    cache_root: str | Path,
    retrieval_config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    result = run_benchmark(
        corpus=corpus,
        manifest=manifest,
        profile=profile,
        seed=seed,
        architectures=architectures,
        output_root=Path(cache_root).parent / "dry-run-results",
        cache_root=cache_root,
        providers={"groq": OfflineBenchmarkProvider(), "gemini": OfflineBenchmarkProvider()},
        model_profile={
            "mode": "DRY_RUN",
            "groq": FIXED_MODELS["groq"],
            "gemini": FIXED_MODELS["gemini"],
        },
        stage_providers=DEFAULT_STAGE_PROVIDERS,
        retrieval_config=retrieval_config,
        dry_run=True,
    )
    return result
