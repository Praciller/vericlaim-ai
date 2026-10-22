import json

from fastapi.testclient import TestClient

import vericlaim.api as api_module
from vericlaim.config import Settings
from vericlaim.domain.analysis import analyze_claim
from vericlaim.providers.base import ProviderRequest, ProviderResponse
from vericlaim.providers.router import ProviderRouter
from vericlaim.rate_limit import InMemoryRateLimiter
from vericlaim.retrieval.base import RetrievedRecord
from vericlaim.workflow import VerificationWorkflow


class ExternalRecords:
    name = "external-test"

    def __init__(self, excerpt: str):
        self.excerpt = excerpt

    def search(self, query: str, limit: int = 3) -> list[RetrievedRecord]:
        return [
            RetrievedRecord(
                source_id="https://example.test/paper",
                title="External test abstract",
                source_type="openalex",
                url="https://example.test/paper",
                doi=None,
                authors=[],
                published_at=None,
                abstract=self.excerpt,
                provenance="synthetic test source",
                evidence_level="ABSTRACT_AVAILABLE",
            )
        ]


class StanceProvider:
    name = "mock"
    model = "stance-test"
    supports_structured_output = True

    def __init__(self, stance: str = "SUPPORTS", invalid: bool = False):
        self.stance = stance
        self.invalid = invalid

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if request.task == "evidence_classification":
            rows = json.loads(request.prompt.split("\n", 1)[1])
            text = json.dumps(
                [
                    {
                        "evidence_id": "unrecognized" if self.invalid else row["evidence_id"],
                        "stance": self.stance,
                    }
                    for row in rows
                ]
            )
        else:
            text = "{}"
        return ProviderResponse(provider=self.name, model=self.model, text=text)


def external_result(claim: str, excerpt: str, stance: str = "SUPPORTS", invalid: bool = False):
    settings = Settings(mock_provider_enabled=True, live_retrieval_enabled=True)
    provider = StanceProvider(stance, invalid)
    router = ProviderRouter(settings, providers={"mock": provider})
    return VerificationWorkflow(
        settings, router=router, retrievers=[ExternalRecords(excerpt)]
    ).verify({"claim": claim})


def test_thai_factual_claim_without_spaces_is_verifiable():
    analysis = analyze_claim("ระบบนี้ไม่เคยล้มเหลว")
    assert analysis.language == "th"
    assert analysis.verifiable is True
    assert "ไม่เคย" in analysis.strong_quantifiers


def test_counter_query_does_not_convert_supporting_external_evidence_to_contradiction():
    result = external_result(
        "RAG improves outcomes",
        "RAG improves outcomes in controlled comparisons.",
    )
    assert result.verdict == "SUPPORTED"
    assert result.contradicting_evidence == []
    assert result.supporting_evidence


def test_support_query_does_not_convert_contradicting_evidence_to_support():
    result = external_result(
        "RAG eliminates hallucinations",
        "RAG exhibits residual hallucinations under benchmark conditions.",
        stance="CONTRADICTS",
    )
    assert result.verdict == "REFUTED"
    assert result.supporting_evidence == []


def test_irrelevant_external_evidence_cannot_support_claim_even_if_classifier_says_so():
    result = external_result("RAG improves outcomes", "Garden roses bloom in June.")
    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    assert not result.supporting_evidence


def test_invalid_or_partial_external_classification_abstains():
    result = external_result(
        "RAG improves outcomes", "RAG improves outcomes under conditions.", invalid=True
    )
    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    assert not result.contradicting_evidence
    assert any("classification was unavailable" in x for x in result.limitations)


def test_strong_universal_claim_does_not_accept_weaker_external_support():
    result = external_result(
        "RAG eliminates hallucinations",
        "RAG can reduce hallucinations under evaluated benchmark conditions.",
        stance="SUPPORTS",
    )
    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    assert result.supporting_evidence == []


def test_in_memory_rate_limiter_bounded_window():
    current = [100.0]
    limiter = InMemoryRateLimiter(max_requests=2, clock=lambda: current[0])
    assert limiter.allow("peer1") == (True, 0)
    assert limiter.allow("peer1") == (True, 0)
    assert limiter.allow("peer1") == (False, 60)
    assert limiter.allow("peer2") == (True, 0)
    current[0] = 160.0
    assert limiter.allow("peer1") == (True, 0)


def test_verify_endpoint_returns_sanitized_rate_limit(monkeypatch):
    monkeypatch.setattr(api_module, "verify_rate_limiter", InMemoryRateLimiter(max_requests=1))
    with TestClient(api_module.app) as client:
        first = client.post("/api/v1/claims/verify", json={"claim": "RAG improves results"})
        second = client.post(
            "/api/v1/claims/verify",
            json={"claim": "RAG improves results"},
            headers={"X-Forwarded-For": "203.0.113.55"},
        )
        health = client.get("/health")
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"]["code"] == "RATE_LIMITED"
    assert int(second.headers["Retry-After"]) >= 1
    assert health.status_code == 200


def test_public_run_api_does_not_expose_provider_traces_or_assessment_rationales():
    with TestClient(api_module.app) as client:
        created = client.post(
            "/api/v1/claims/verify", json={"claim": "RAG eliminates hallucinations"}
        ).json()
        run = client.get(f"/api/v1/runs/{created['run_id']}")
        evidence = client.get(f"/api/v1/runs/{created['run_id']}/evidence")
    assert run.status_code == evidence.status_code == 200
    assert run.json()["run_id"] == created["run_id"]
    assert "provider_usage" not in run.json()
    assert "agent_runs" not in run.json()
    assert "assessments" not in run.json()
    assert "queries" not in run.json()
    assert "assessments" not in evidence.json()
    assert evidence.json()["run_id"] == created["run_id"]
