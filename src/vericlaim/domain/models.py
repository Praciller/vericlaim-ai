from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class VerdictLabel(StrEnum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    MIXED = "MIXED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NON_VERIFIABLE = "NON_VERIFIABLE"


class EvidenceStance(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


class EvidenceLevel(StrEnum):
    METADATA_ONLY = "METADATA_ONLY"
    ABSTRACT_AVAILABLE = "ABSTRACT_AVAILABLE"
    FULL_TEXT_AVAILABLE = "FULL_TEXT_AVAILABLE"
    OFFICIAL_DOCUMENTATION = "OFFICIAL_DOCUMENTATION"


class RunStatus(StrEnum):
    COMPLETED = "COMPLETED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class RunIssueCode(StrEnum):
    """Safe, provider-neutral operational state for a verification run."""

    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    PROVIDER_RATE_LIMIT = "PROVIDER_RATE_LIMIT"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_AUTHENTICATION = "PROVIDER_AUTHENTICATION"
    PROVIDER_RESPONSE_INVALID = "PROVIDER_RESPONSE_INVALID"
    RETRIEVAL_UNAVAILABLE = "RETRIEVAL_UNAVAILABLE"
    REQUEST_LIMIT_EXCEEDED = "REQUEST_LIMIT_EXCEEDED"


class ClaimAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    claim_type: str
    verifiable: bool
    temporal_sensitivity: bool
    subjective_language: bool
    strong_quantifiers: list[str] = Field(default_factory=list)
    numerical_claims: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(default_factory=lambda: f"claim-{uuid4().hex[:12]}")
    original_text: str = Field(min_length=1, max_length=2000)
    normalized_text: str = Field(min_length=1, max_length=2000)
    analysis: ClaimAnalysis
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("original_text", "normalized_text")
    @classmethod
    def no_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
            raise ValueError("claim contains control characters")
        return value.strip()


class AtomicClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atomic_id: str
    claim_id: str
    text: str = Field(min_length=1)
    claim_type: str
    conditions: list[str] = Field(default_factory=list)


class SearchQuery(BaseModel):
    query_id: str
    atomic_id: str
    query: str = Field(min_length=1, max_length=500)
    direction: str
    source_names: list[str] = Field(default_factory=lambda: ["fixture", "openalex", "crossref"])


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    title: str
    source_type: str
    url: str | None = None
    doi: str | None = None
    authors: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    abstract: str | None = None
    evidence_level: EvidenceLevel = EvidenceLevel.METADATA_ONLY
    provenance: str


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    atomic_id: str
    source_id: str
    excerpt: str = Field(min_length=1)
    direction: str
    evidence_level: EvidenceLevel = EvidenceLevel.ABSTRACT_AVAILABLE
    retrieved_at: datetime = Field(default_factory=utc_now)
    provenance: str


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    relevance: float = Field(ge=0, le=1)
    directness: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    temporal_compatibility: float = Field(ge=0, le=1)
    scope_compatibility: float = Field(ge=0, le=1)
    reproducibility_signal: float = Field(ge=0, le=1)
    stance: EvidenceStance
    extraction_confidence: float = Field(ge=0, le=1)
    rationale: str


class AtomicClaimCandidate(BaseModel):
    """Untrusted LLM candidate; IDs are assigned by deterministic domain logic."""

    text: str = Field(min_length=1, max_length=2000)


class EvidenceStanceCandidate(BaseModel):
    """Untrusted batch-classification candidate constrained to known evidence IDs later."""

    evidence_id: str = Field(min_length=1)
    stance: EvidenceStance


class EvidenceAuditCandidate(BaseModel):
    assessments: list[EvidenceStanceCandidate] = Field(default_factory=list)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: VerdictLabel
    confidence: float = Field(ge=0, le=1)
    explanation: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AgentRun(BaseModel):
    agent_name: str
    task: str
    provider: str
    model: str
    status: str
    started_at: datetime
    completed_at: datetime
    error: str | None = None


class ProviderUsage(BaseModel):
    provider: str
    model: str
    configured_model: str | None = None
    actual_model: str | None = None
    task: str
    request_count: int = Field(ge=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    status: str
    error_code: str | None = None
    quota_limit_tokens: int | None = Field(default=None, ge=0)
    quota_used_tokens: int | None = Field(default=None, ge=0)
    quota_remaining_tokens: int | None = Field(default=None, ge=0)
    fallbacks: list[str] = Field(default_factory=list)


class ProviderErrorCategory(StrEnum):
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    PAYMENT_REQUIRED = "payment_required"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    MALFORMED_RESPONSE = "malformed_response"
    INCOMPLETE_RESPONSE = "incomplete_response"
    SERVER_ERROR = "server_error"
    RESOURCE_LIMIT = "resource_limit"
    UNKNOWN = "unknown"


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    original_claim: str
    normalized_claim: str
    status: RunStatus
    verdict: VerdictLabel
    confidence: float = Field(ge=0, le=1)
    summary: str
    issue_code: RunIssueCode | None = None
    claim: Claim
    atomic_claims: list[AtomicClaim]
    queries: list[SearchQuery]
    sources: list[Source]
    evidence: list[Evidence]
    assessments: list[EvidenceAssessment]
    verdict_details: Verdict
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    agent_runs: list[AgentRun] = Field(default_factory=list)
    provider_usage: list[ProviderUsage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class VerificationRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=2000)

    @field_validator("claim")
    @classmethod
    def validate_claim(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("claim must not be blank")
        return cleaned


class EvidenceListResponse(BaseModel):
    run_id: str
    evidence: list[Evidence]
    assessments: list[EvidenceAssessment]


class EvidenceGraphNode(BaseModel):
    node_id: str
    kind: Literal["claim", "atomic_claim", "evidence", "source"]
    label: str
    excerpt: str | None = None
    source_id: str | None = None
    direction: str | None = None
    stance: EvidenceStance | None = None
    evidence_level: EvidenceLevel | None = None
    provenance: str | None = None
    url: str | None = None


class EvidenceGraphEdge(BaseModel):
    edge_id: str
    source: str
    target: str
    relation: Literal["contains", "has_evidence", "cited_from"]


class EvidenceGraphResponse(BaseModel):
    run_id: str
    nodes: list[EvidenceGraphNode]
    edges: list[EvidenceGraphEdge]


class ProviderStatus(BaseModel):
    name: str
    configured: bool
    enabled: bool
    model: str
    supports_structured_output: bool
    note: str | None = None
    last_status: str | None = None
    quota_remaining_tokens: int | None = Field(default=None, ge=0)
    quota_limit_tokens: int | None = Field(default=None, ge=0)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    service: str
    version: str
    database: Literal["ok", "unavailable"]
    enabled_providers: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def model_dump_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value
