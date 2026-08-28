import pytest
from fastapi.testclient import TestClient

import vericlaim.api as api_module
from vericlaim.api import app
from vericlaim.config import Settings
from vericlaim.domain.models import ProviderErrorCategory, RunIssueCode, RunStatus
from vericlaim.providers.base import (
    MockProvider,
    ProviderException,
    ProviderRequest,
    ProviderResponse,
)
from vericlaim.providers.router import ProviderRouter
from vericlaim.retrieval.base import RetrievedRecord
from vericlaim.workflow import VerificationWorkflow


class QuotaProvider:
    name = "gemini"
    model = "test-model"
    supports_structured_output = True

    def generate(self, request: ProviderRequest):  # type: ignore[no-untyped-def]
        raise ProviderException(
            "provider quota exhausted",
            category=ProviderErrorCategory.QUOTA_EXHAUSTED,
        )


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_endpoint_reports_safe_checks():
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["database"] == "ok"
    assert "mock" in payload["enabled_providers"]
    assert payload["issues"] == []


def test_readiness_endpoint_returns_sanitized_503_without_provider(monkeypatch):
    monkeypatch.setattr(api_module.router, "statuses", lambda: [])
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "SERVICE_NOT_READY"
    assert detail["checks"]["issues"] == ["no_provider_enabled"]


def test_verification_api_contract():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/claims/verify", json={"claim": "RAG eliminates hallucinations"}
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"]
    assert payload["verdict"] in {
        "SUPPORTED",
        "REFUTED",
        "MIXED",
        "INSUFFICIENT_EVIDENCE",
        "NON_VERIFIABLE",
    }
    assert "provider_usage" in payload


def test_verification_api_rejects_oversized_claim():
    with TestClient(app) as client:
        response = client.post("/api/v1/claims/verify", json={"claim": "x" * 2001})

    assert response.status_code == 422


def test_verification_api_rejects_excessive_atomic_claims():
    claim = " and ".join(f"Claim {index} is testable" for index in range(1, 10))

    with TestClient(app) as client:
        response = client.post("/api/v1/claims/verify", json={"claim": claim})

    assert response.status_code == 422
    assert "atomic claim" in response.json()["detail"]


def test_workflow_rejects_excessive_retrieval_queries():
    workflow = VerificationWorkflow(
        Settings(max_retrieval_queries_per_request=1, mock_provider_enabled=True)
    )

    with pytest.raises(ValueError, match="retrieval query"):
        workflow.verify({"claim": "RAG eliminates hallucinations"})


def test_workflow_bounds_evidence_candidates_per_request():
    class ManyRecords:
        name = "many-records"

        def search(self, query: str, limit: int = 3) -> list[RetrievedRecord]:
            return [
                RetrievedRecord(
                    source_id=f"source-{index}",
                    title=f"Source {index}",
                    source_type="test",
                    url=f"https://example.test/{index}",
                    doi=None,
                    authors=[],
                    published_at=None,
                    abstract="Evidence excerpt.",
                    provenance="test fixture",
                    evidence_level="ABSTRACT_AVAILABLE",
                )
                for index in range(4)
            ]

    workflow = VerificationWorkflow(
        Settings(max_evidence_candidates_per_request=1, mock_provider_enabled=True),
        retrievers=[ManyRecords()],
    )
    result = workflow.verify({"claim": "RAG eliminates hallucinations"})

    assert len(result.evidence) == 1
    assert result.issue_code.value == "REQUEST_LIMIT_EXCEEDED"


def test_workflow_enforces_provider_call_limit_per_request():
    class CountingProvider:
        name = "mock"
        model = "counting-test"
        supports_structured_output = True

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            self.calls += 1
            return ProviderResponse(
                provider=self.name,
                model=self.model,
                text="{}",
                finish_reason="stop",
            )

    provider = CountingProvider()
    settings = Settings(max_provider_calls_per_request=1, mock_provider_enabled=True)
    workflow = VerificationWorkflow(
        settings,
        router=ProviderRouter(settings, providers={"mock": provider}),
    )
    result = workflow.verify({"claim": "RAG eliminates hallucinations"})

    assert provider.calls == 1
    assert result.issue_code.value == "REQUEST_LIMIT_EXCEEDED"


def test_provider_status_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/providers/status")
    assert response.status_code == 200
    assert any(item["name"] == "mock" for item in response.json())


def test_run_persistence_and_evidence_endpoint():
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/claims/verify", json={"claim": "RAG eliminates hallucinations"}
        ).json()
        run_id = created["run_id"]
        fetched = client.get(f"/api/v1/runs/{run_id}")
        evidence = client.get(f"/api/v1/runs/{run_id}/evidence")
    assert fetched.status_code == 200
    assert evidence.status_code == 200
    assert evidence.json()["run_id"] == run_id


def test_evidence_graph_projects_persisted_run():
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/claims/verify", json={"claim": "RAG eliminates hallucinations"}
        ).json()
        graph = client.get(f"/api/v1/runs/{created['run_id']}/evidence-graph")
    assert graph.status_code == 200
    payload = graph.json()
    assert payload["run_id"] == created["run_id"]
    assert any(node["kind"] == "claim" for node in payload["nodes"])
    assert any(node["kind"] == "evidence" for node in payload["nodes"])
    assert any(edge["relation"] == "cited_from" for edge in payload["edges"])


def test_bounded_provider_fallback_exposes_safe_issue_code():
    settings = Settings(database_url="sqlite:///:memory:", mock_provider_enabled=True)
    router = ProviderRouter(
        settings,
        providers={"gemini": QuotaProvider(), "mock": MockProvider()},
    )
    result = VerificationWorkflow(settings, router=router).verify(
        {"claim": "RAG eliminates hallucinations"}
    )
    assert result.status == RunStatus.DEGRADED
    assert result.issue_code == RunIssueCode.QUOTA_EXHAUSTED
