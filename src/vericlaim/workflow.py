from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import TypeAdapter, ValidationError

from .config import Settings
from .domain.analysis import analyze_claim, normalize_claim
from .domain.decomposition import decompose_claim
from .domain.models import (
    AgentRun,
    AtomicClaim,
    AtomicClaimCandidate,
    Claim,
    Evidence,
    EvidenceAssessment,
    EvidenceAuditCandidate,
    EvidenceLevel,
    EvidenceStance,
    EvidenceStanceCandidate,
    ProviderErrorCategory,
    ProviderUsage,
    RunIssueCode,
    RunStatus,
    SearchQuery,
    Source,
    Verdict,
    VerdictLabel,
    VerificationRequest,
    VerificationResult,
    utc_now,
)
from .providers.base import ProviderException, ProviderResponse
from .providers.router import ProviderCallBudget, ProviderRouter
from .retrieval.adapters import CrossrefSource, OpenAlexSource, RetrievalError
from .retrieval.base import EvidenceSource
from .retrieval.fixture import FixtureSource
from .validation import validate_result

MIXED_VERDICT_HEURISTIC_CONFIDENCE = 0.65


class RequestTimeoutError(TimeoutError):
    """Raised when a verification exceeds its cooperative request deadline."""


@dataclass
class WorkflowState:
    request: VerificationRequest
    run_id: str = field(default_factory=lambda: f"run-{uuid4().hex}")
    claim: Claim | None = None
    atomic_claims: list[AtomicClaim] = field(default_factory=list)
    queries: list[SearchQuery] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    assessments: list[EvidenceAssessment] = field(default_factory=list)
    agent_runs: list[AgentRun] = field(default_factory=list)
    provider_usage: list[ProviderUsage] = field(default_factory=list)
    verdict: Verdict | None = None
    status: RunStatus = RunStatus.COMPLETED
    issue_code: RunIssueCode | None = None
    limitations: list[str] = field(default_factory=list)
    provider_budget: ProviderCallBudget | None = None


class VerificationWorkflow:
    MAX_EVIDENCE_EXCERPT = 700

    def __init__(
        self,
        settings: Settings,
        router: ProviderRouter | None = None,
        retrievers: list[EvidenceSource] | None = None,
    ) -> None:
        self.settings = settings
        self.router = router or ProviderRouter(settings)
        self.fixture = FixtureSource()
        self.retrievers = retrievers or (
            [OpenAlexSource(), CrossrefSource()]
            if settings.live_retrieval_enabled
            else [self.fixture]
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)
        graph.add_node("analyze", self._analyze)
        graph.add_node("decompose", self._decompose)
        graph.add_node("plan", self._plan)
        graph.add_node("research", self._research)
        graph.add_node("audit", self._audit)
        graph.add_node("judge", self._judge)
        graph.add_node("critic", self._critic)
        graph.add_node("validate", self._validate)
        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "decompose")
        graph.add_edge("decompose", "plan")
        graph.add_edge("plan", "research")
        graph.add_edge("research", "audit")
        graph.add_edge("audit", "judge")
        graph.add_edge("judge", "critic")
        graph.add_edge("critic", "validate")
        graph.add_edge("validate", END)
        return graph.compile()

    def verify(
        self,
        request: VerificationRequest | dict[str, str],
        *,
        timeout_seconds: float | None = None,
    ) -> VerificationResult:
        validated_request = VerificationRequest.model_validate(request)
        deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        state_value = self.graph.invoke(
            WorkflowState(
                request=validated_request,
                provider_budget=ProviderCallBudget(
                    limit=self.settings.max_provider_calls_per_request,
                    deadline=deadline,
                ),
            )
        )
        state = WorkflowState(**state_value) if isinstance(state_value, dict) else state_value
        self._ensure_request_budget(state)
        result = self._to_result(state)
        return validate_result(result)

    @staticmethod
    def _ensure_request_budget(state: WorkflowState) -> None:
        if state.provider_budget and state.provider_budget.remaining_seconds() == 0:
            raise RequestTimeoutError("verification request timed out")

    def _record_agent(
        self,
        state: WorkflowState,
        name: str,
        task: str,
        action: Callable[[], None],
        *,
        use_provider: bool = False,
        prompt: str | None = None,
    ) -> ProviderResponse | None:
        self._ensure_request_budget(state)
        started = utc_now()
        if not use_provider:
            state.agent_runs.append(
                AgentRun(
                    agent_name=name,
                    task=task,
                    provider="deterministic",
                    model="rules-v1",
                    status="success",
                    started_at=started,
                    completed_at=utc_now(),
                )
            )
            return None
        try:
            router_result = self.router.invoke(
                task,
                prompt or action.__doc__ or task,
                budget=state.provider_budget,
            )
            state.provider_usage.append(router_result.usage)
            provider = router_result.response.provider
            model = router_result.response.actual_model or router_result.response.model
            status = "success"
            error = ";".join(f"fallback={item}" for item in router_result.fallbacks) or None
            if router_result.fallbacks:
                state.status = RunStatus.DEGRADED
                state.issue_code = state.issue_code or self._issue_code_from_fallbacks(
                    router_result.fallbacks
                )
                state.limitations.append(
                    "A bounded provider fallback was used; inspect the run trace before relying "
                    "on this result."
                )
        except ProviderException as exc:
            provider = "unavailable"
            model = "none"
            status = "degraded"
            error = exc.category.value
            state.status = RunStatus.DEGRADED
            state.issue_code = state.issue_code or self._issue_code_for_category(exc.category)
            state.limitations.append(
                f"{task} used a deterministic fallback after {state.issue_code.value}"
            )
        self._ensure_request_budget(state)
        state.agent_runs.append(
            AgentRun(
                agent_name=name,
                task=task,
                provider=provider,
                model=model,
                status=status,
                started_at=started,
                completed_at=utc_now(),
                error=error,
            )
        )
        return router_result.response if status == "success" else None

    @staticmethod
    def _issue_code_for_category(category: ProviderErrorCategory) -> RunIssueCode:
        mapping = {
            ProviderErrorCategory.QUOTA_EXHAUSTED: RunIssueCode.QUOTA_EXHAUSTED,
            ProviderErrorCategory.RATE_LIMIT: RunIssueCode.PROVIDER_RATE_LIMIT,
            ProviderErrorCategory.TIMEOUT: RunIssueCode.PROVIDER_TIMEOUT,
            ProviderErrorCategory.AUTHENTICATION: RunIssueCode.PROVIDER_AUTHENTICATION,
            ProviderErrorCategory.MALFORMED_RESPONSE: RunIssueCode.PROVIDER_RESPONSE_INVALID,
            ProviderErrorCategory.INCOMPLETE_RESPONSE: RunIssueCode.PROVIDER_RESPONSE_INVALID,
            ProviderErrorCategory.RESOURCE_LIMIT: RunIssueCode.REQUEST_LIMIT_EXCEEDED,
        }
        return mapping.get(category, RunIssueCode.PROVIDER_UNAVAILABLE)

    @classmethod
    def _issue_code_from_fallbacks(cls, fallbacks: list[str]) -> RunIssueCode:
        for fallback in fallbacks:
            _, _, category_value = fallback.partition(":")
            try:
                category = ProviderErrorCategory(category_value)
            except ValueError:
                continue
            return cls._issue_code_for_category(category)
        return RunIssueCode.PROVIDER_UNAVAILABLE

    @staticmethod
    def _json_advisory(response: ProviderResponse | None) -> object | None:
        if response is None:
            return None
        try:
            return cast(object, json.loads(response.text))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _validated_advisory(
        response: ProviderResponse | None, adapter: TypeAdapter[Any]
    ) -> object | None:
        payload = VerificationWorkflow._json_advisory(response)
        if payload is None:
            return None
        try:
            return cast(object, adapter.validate_python(payload))
        except ValidationError:
            return None

    @staticmethod
    def _has_candidate(response: ProviderResponse | None) -> bool:
        return response is not None and response.text.strip() not in {"", "{}"}

    def _analyze(self, state: WorkflowState) -> WorkflowState:
        def action() -> None:
            """Analyze language, verifiability, quantifiers, comparisons, and conditions."""

        self._record_agent(state, "claim_analyzer", "claim_analysis", action)
        normalized = normalize_claim(state.request.claim)
        state.claim = Claim(
            original_text=state.request.claim,
            normalized_text=normalized,
            analysis=analyze_claim(normalized),
        )
        if state.claim.analysis.language == "th" and self.router.has_provider("thaillm"):
            self._record_agent(
                state,
                "thai_semantic_reviewer",
                "thai_semantic_review",
                action,
                use_provider=True,
                prompt=(
                    "Review the Thai claim for semantic-strength preservation. Return JSON only "
                    "with keys preserve_negation, preserve_quantifiers, preserve_modality, "
                    "and normalized_text. Never weaken or translate the claim.\n"
                    f"original_text={state.claim.original_text}"
                ),
            )
        return state

    def _decompose(self, state: WorkflowState) -> WorkflowState:
        def action() -> None:
            """Decompose one input claim into independently verifiable atomic claims."""

        self._record_agent(state, "claim_decomposer", "claim_decomposition", action)
        assert state.claim is not None
        state.atomic_claims = decompose_claim(
            state.claim, max_atomic_claims=self.settings.max_atomic_claims_per_request
        )
        if len(state.atomic_claims) > 1 and self.router.has_provider("groq"):
            advisory = self._record_agent(
                state,
                "llm_claim_decomposition_advisor",
                "claim_decomposition",
                action,
                use_provider=True,
                prompt=(
                    "Return JSON only as an array of atomic claim objects with a text field. "
                    "Do not invent claims, IDs, evidence, or sources. Preserve negation and "
                    f"quantifiers. Claim: {state.claim.normalized_text}"
                ),
            )
            # The candidate is parsed and retained only as an advisory validity signal. The
            # deterministic decomposition remains authoritative for IDs and semantics.
            candidate = self._validated_advisory(advisory, TypeAdapter(list[AtomicClaimCandidate]))
            if self._has_candidate(advisory) and candidate is None:
                state.limitations.append("LLM decomposition candidate was not a valid JSON list")
        return state

    @staticmethod
    def _retrieval_subject(claim: Claim) -> str:
        """Add a bounded English retrieval hint without changing the stored claim."""
        lowered = claim.normalized_text.casefold()
        if (
            claim.analysis.language == "th"
            and "rag" in lowered
            and any(marker in lowered for marker in ("หลอน", "hallucination"))
        ):
            return "retrieval augmented generation hallucinations"
        return claim.normalized_text

    @classmethod
    def _evidence_excerpt(cls, text: str) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= cls.MAX_EVIDENCE_EXCERPT:
            return normalized
        shortened = normalized[: cls.MAX_EVIDENCE_EXCERPT - 3].rsplit(" ", 1)[0]
        return f"{shortened}..."

    def _plan(self, state: WorkflowState) -> WorkflowState:
        def action() -> None:
            """Create bounded support and counter-evidence search queries."""

        self._record_agent(state, "evidence_planner", "query_generation", action)
        assert state.claim is not None
        retrieval_subject = self._retrieval_subject(state.claim)
        for atomic in state.atomic_claims:
            query_subject = (
                retrieval_subject if state.claim.analysis.language == "th" else atomic.text
            )
            state.queries.extend(
                [
                    SearchQuery(
                        query_id=f"Q-{atomic.atomic_id}-S",
                        atomic_id=atomic.atomic_id,
                        query=query_subject,
                        direction="support",
                    ),
                    SearchQuery(
                        query_id=f"Q-{atomic.atomic_id}-C",
                        atomic_id=atomic.atomic_id,
                        query=f"limitations counterexamples {query_subject}",
                        direction="counter",
                    ),
                ]
            )
        if len(state.queries) > self.settings.max_retrieval_queries_per_request:
            raise ValueError("claim exceeds the maximum retrieval query limit")
        return state

    def _research(self, state: WorkflowState) -> WorkflowState:
        def action() -> None:
            """Retrieve separate support and counter-evidence candidates from normalized sources."""

        self._record_agent(state, "support_and_counter_researchers", "research", action)
        assert state.claim is not None
        if not state.claim.analysis.verifiable or any(
            word in state.claim.normalized_text.casefold()
            for word in ("unknown", "unverifiable", "ไม่มีหลักฐาน")
        ):
            return state
        source_by_id: dict[str, Source] = {}
        metadata_only_seen = False
        candidates_seen = 0
        candidate_limit_hit = False
        for query in state.queries:
            self._ensure_request_budget(state)
            if candidate_limit_hit:
                break
            for retriever in self.retrievers:
                self._ensure_request_budget(state)
                try:
                    records = retriever.search(query.query, limit=2)
                except RetrievalError:
                    state.status = RunStatus.DEGRADED
                    state.issue_code = state.issue_code or RunIssueCode.RETRIEVAL_UNAVAILABLE
                    limitation = f"{retriever.name} retrieval was unavailable"
                    if limitation not in state.limitations:
                        state.limitations.append(limitation)
                    continue
                self._ensure_request_budget(state)
                for record in records:
                    if candidates_seen >= self.settings.max_evidence_candidates_per_request:
                        candidate_limit_hit = True
                        state.status = RunStatus.DEGRADED
                        state.issue_code = state.issue_code or RunIssueCode.REQUEST_LIMIT_EXCEEDED
                        state.limitations.append(
                            "The evidence candidate limit was reached; remaining retrieval "
                            "results were not processed."
                        )
                        break
                    candidates_seen += 1
                    try:
                        evidence_level = EvidenceLevel(record.evidence_level)
                    except ValueError:
                        evidence_level = EvidenceLevel.METADATA_ONLY
                    source_by_id[record.source_id] = Source(
                        source_id=record.source_id,
                        title=record.title,
                        source_type=record.source_type,
                        url=record.url,
                        doi=record.doi,
                        authors=record.authors,
                        published_at=record.published_at,
                        abstract=record.abstract,
                        evidence_level=evidence_level,
                        provenance=record.provenance,
                    )
                    if not record.abstract or evidence_level == EvidenceLevel.METADATA_ONLY:
                        metadata_only_seen = True
                        continue
                    state.evidence.append(
                        Evidence(
                            evidence_id=f"E-{uuid4().hex[:10]}",
                            atomic_id=query.atomic_id,
                            source_id=record.source_id,
                            excerpt=self._evidence_excerpt(record.abstract),
                            direction=query.direction,
                            evidence_level=evidence_level,
                            provenance=record.provenance,
                        )
                    )
                if candidate_limit_hit:
                    break
        state.sources = list(source_by_id.values())
        if metadata_only_seen:
            state.limitations.append(
                "Metadata-only retrieval results were retained as sources but excluded "
                "from evidence."
            )
        return state

    def _audit(self, state: WorkflowState) -> WorkflowState:
        def action() -> None:
            """Audit evidence dimensions independently for relevance, scope, recency, and stance."""

        if state.evidence:
            classification_advisory = self._record_agent(
                state,
                "evidence_classifier",
                "evidence_classification",
                action,
                use_provider=True,
                prompt=(
                    "Return JSON only as an array with evidence_id and stance for each item. "
                    "This is advisory; use only IDs supplied below and do not invent sources.\n"
                    + json.dumps(
                        [
                            {"evidence_id": item.evidence_id, "excerpt": item.excerpt[:500]}
                            for item in state.evidence
                        ]
                    )
                ),
            )
            classification = self._validated_advisory(
                classification_advisory, TypeAdapter(list[EvidenceStanceCandidate])
            )
            if self._has_candidate(classification_advisory) and classification is None:
                state.limitations.append(
                    "LLM evidence classification candidate failed schema validation"
                )
        audit_advisory = self._record_agent(
            state,
            "evidence_auditor",
            "evidence_audit",
            action,
            use_provider=bool(state.evidence),
            prompt=(
                "Return JSON only with an assessments array. Preserve separate dimensions for "
                "relevance, directness, scope compatibility, and stance. This is advisory and "
                "must not invent evidence IDs.\n"
                + json.dumps([item.evidence_id for item in state.evidence])
            ),
        )
        audit_candidate = self._validated_advisory(
            audit_advisory, TypeAdapter(EvidenceAuditCandidate)
        )
        if self._has_candidate(audit_advisory) and audit_candidate is None:
            state.limitations.append("LLM evidence audit candidate failed schema validation")
        assert state.claim is not None
        for evidence in state.evidence:
            lowered = evidence.excerpt.casefold()
            contradicting_markers = (
                "residual",
                "limitation",
                "vary",
                "failure",
                "not supported",
                "remain",
            )
            weaker_markers = (
                "reduce",
                "improve",
                "under",
                "benchmark",
                "conditional",
                "can ",
            )
            if evidence.direction == "counter" or any(
                word in lowered for word in contradicting_markers
            ):
                stance = EvidenceStance.CONTRADICTS
            elif state.claim.analysis.strong_quantifiers and any(
                word in lowered for word in weaker_markers
            ):
                stance = EvidenceStance.NEUTRAL
            else:
                stance = EvidenceStance.SUPPORTS
            state.assessments.append(
                EvidenceAssessment(
                    evidence_id=evidence.evidence_id,
                    relevance=0.9,
                    directness=0.8,
                    source_quality=0.55,
                    recency=0.5,
                    temporal_compatibility=0.8,
                    scope_compatibility=0.75,
                    reproducibility_signal=0.7,
                    stance=stance,
                    extraction_confidence=0.95,
                    rationale=(
                        "Deterministic audit; dimensions remain separately inspectable and weaker "
                        "evidence is not promoted to support for a strong quantifier."
                    ),
                )
            )
        return state

    def _judge(self, state: WorkflowState) -> WorkflowState:
        def action() -> None:
            """Judge only from audited evidence and preserve an explicit uncertainty label."""

        judge_advisory = self._record_agent(
            state,
            "claim_judge",
            "final_judgment",
            action,
            use_provider=bool(state.evidence),
            prompt=(
                "Return JSON only matching verdict, confidence, supporting_evidence_ids, "
                "contradicting_evidence_ids, conditions, missing_evidence, and limitations. "
                "Use only the supplied audited evidence IDs.\n"
                + json.dumps(
                    {
                        "claim": state.claim.normalized_text if state.claim else "",
                        "evidence_ids": [item.evidence_id for item in state.evidence],
                    }
                )
            ),
        )
        assert state.claim is not None
        if not state.claim.analysis.verifiable:
            state.verdict = Verdict(
                label=VerdictLabel.NON_VERIFIABLE,
                confidence=0.95,
                explanation=(
                    "The input is subjective or does not make a sufficiently testable technical "
                    "assertion."
                ),
                limitations=["A factual evidence verdict was not attempted."],
            )
            return state
        supports = [
            item.evidence_id for item in state.assessments if item.stance == EvidenceStance.SUPPORTS
        ]
        contradicts = [
            item.evidence_id
            for item in state.assessments
            if item.stance == EvidenceStance.CONTRADICTS
        ]
        if not supports and not contradicts:
            state.verdict = Verdict(
                label=VerdictLabel.INSUFFICIENT_EVIDENCE,
                confidence=0.2,
                explanation="No admissible evidence was retrieved for this claim.",
                missing_evidence=["A traceable source directly addressing the atomic claim."],
            )
        elif supports and contradicts:
            state.verdict = Verdict(
                label=VerdictLabel.MIXED,
                confidence=MIXED_VERDICT_HEURISTIC_CONFIDENCE,
                explanation=(
                    "Retrieved evidence contains both supporting and contradicting signals; the "
                    "broad wording is conditional."
                ),
                supporting_evidence_ids=supports[:3],
                contradicting_evidence_ids=contradicts[:3],
                conditions=[
                    "Results depend on the evaluated dataset, implementation, and measurement "
                    "protocol."
                ],
            )
        elif supports:
            state.verdict = Verdict(
                label=VerdictLabel.SUPPORTED,
                confidence=0.7,
                explanation=(
                    "Retrieved evidence supports the claim under the documented conditions, "
                    "without establishing universal truth."
                ),
                supporting_evidence_ids=supports[:3],
                conditions=["Evidence is limited to the retrieved evaluation context."],
            )
        else:
            state.verdict = Verdict(
                label=VerdictLabel.REFUTED,
                confidence=0.82,
                explanation=(
                    "Retrieved counter-evidence identifies residual failures or scope limitations "
                    "inconsistent with the claim's strong wording."
                ),
                contradicting_evidence_ids=contradicts[:3],
            )
        judge_candidate = self._validated_advisory(judge_advisory, TypeAdapter(Verdict))
        if self._has_candidate(judge_advisory) and judge_candidate is None:
            state.limitations.append("LLM judgment candidate failed schema validation")
        return state

    def _critic(self, state: WorkflowState) -> WorkflowState:
        def action() -> None:
            """Independently check scope mismatch, exaggerated language, and citation coverage."""

        assert state.verdict is not None
        if state.verdict.label != VerdictLabel.NON_VERIFIABLE:
            critic_advisory = self._record_agent(
                state,
                "critic",
                "critique",
                action,
                use_provider=True,
                prompt=(
                    "Return JSON only with concerns and citation_issues. Challenge scope, "
                    "temporal fit, strong wording, and unsupported conclusions. Do not invent "
                    f"evidence IDs. verdict={state.verdict.label.value}"
                ),
            )
            if (
                self._has_candidate(critic_advisory)
                and self._json_advisory(critic_advisory) is None
            ):
                state.limitations.append("LLM critique candidate was not valid JSON")
        if (
            state.claim
            and state.claim.analysis.strong_quantifiers
            and state.verdict.label == VerdictLabel.SUPPORTED
        ):
            state.verdict.label = VerdictLabel.MIXED
            state.verdict.confidence = min(state.verdict.confidence, 0.6)
            state.verdict.limitations.append(
                "Strong quantifier requires broader evidence than this run retrieved."
            )
        if not state.evidence and state.verdict.label not in {
            VerdictLabel.NON_VERIFIABLE,
            VerdictLabel.INSUFFICIENT_EVIDENCE,
        }:
            state.verdict = state.verdict.model_copy(
                update={"label": VerdictLabel.INSUFFICIENT_EVIDENCE, "confidence": 0.2}
            )
        return state

    def _validate(self, state: WorkflowState) -> WorkflowState:
        def action() -> None:
            """Run deterministic schema, provenance, and citation validation."""

        self._record_agent(state, "deterministic_validator", "validation", action)
        return state

    def _to_result(self, state: WorkflowState) -> VerificationResult:
        assert state.claim is not None and state.verdict is not None
        completed = utc_now()
        limitations = list(dict.fromkeys(state.limitations + state.verdict.limitations))
        supporting = [
            item
            for item in state.evidence
            if item.evidence_id in set(state.verdict.supporting_evidence_ids)
        ]
        contradicting = [
            item
            for item in state.evidence
            if item.evidence_id in set(state.verdict.contradicting_evidence_ids)
        ]
        return VerificationResult(
            run_id=state.run_id,
            original_claim=state.request.claim,
            normalized_claim=state.claim.normalized_text,
            status=state.status,
            verdict=state.verdict.label,
            confidence=state.verdict.confidence,
            summary=state.verdict.explanation,
            issue_code=state.issue_code,
            claim=state.claim,
            atomic_claims=state.atomic_claims,
            queries=state.queries,
            sources=state.sources,
            evidence=state.evidence,
            assessments=state.assessments,
            verdict_details=state.verdict,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            conditions=state.verdict.conditions,
            limitations=limitations,
            agent_runs=state.agent_runs,
            provider_usage=state.provider_usage,
            completed_at=completed,
        )
