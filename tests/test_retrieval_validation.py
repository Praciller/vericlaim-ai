import pytest

from vericlaim.config import Settings
from vericlaim.domain.models import VerificationRequest
from vericlaim.providers.router import ProviderRouter
from vericlaim.retrieval.adapters import CrossrefSource, OpenAlexSource
from vericlaim.retrieval.base import RetrievedRecord
from vericlaim.retrieval.fixture import FixtureSource
from vericlaim.validation import DeterministicValidationError, validate_result
from vericlaim.workflow import VerificationWorkflow


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_fixture_source_normalizes_records():
    records = FixtureSource().search("RAG reduces hallucinations")
    assert records
    assert records[0].source_id.startswith("fixture:")
    assert records[0].provenance


def test_openalex_reconstructs_abstract_from_inverted_index(monkeypatch):
    monkeypatch.setattr(
        "vericlaim.retrieval.adapters.httpx.get",
        lambda *args, **kwargs: FakeResponse(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "title": "RAG study",
                        "doi": "https://doi.org/10.1234/example",
                        "publication_date": "2024-01-02",
                        "primary_location": {"landing_page_url": "https://example.test/paper"},
                        "authorships": [],
                        "abstract_inverted_index": {"RAG": [0], "reduces": [1], "errors": [2]},
                    }
                ]
            }
        ),
    )
    record = OpenAlexSource().search("RAG", limit=1)[0]
    assert record.abstract == "RAG reduces errors"
    assert record.evidence_level == "ABSTRACT_AVAILABLE"


def test_crossref_strips_abstract_markup(monkeypatch):
    monkeypatch.setattr(
        "vericlaim.retrieval.adapters.httpx.get",
        lambda *args, **kwargs: FakeResponse(
            {
                "message": {
                    "items": [
                        {
                            "DOI": "10.1234/example",
                            "title": ["RAG study"],
                            "abstract": "<jats:p>RAG &amp; evaluation.</jats:p>",
                        }
                    ]
                }
            }
        ),
    )
    record = CrossrefSource().search("RAG", limit=1)[0]
    assert record.abstract == "RAG & evaluation."
    assert record.evidence_level == "ABSTRACT_AVAILABLE"


class MetadataOnlySource:
    name = "metadata-only"

    def search(self, query: str, limit: int = 3) -> list[RetrievedRecord]:
        return [
            RetrievedRecord(
                source_id="https://example.test/metadata",
                title="Metadata-only result",
                source_type="test",
                url="https://example.test/metadata",
                doi=None,
                authors=[],
                published_at=None,
                abstract=None,
                provenance="test fixture",
            )
        ]


class WeakAbstractSource:
    name = "weak-abstract"

    def search(self, query: str, limit: int = 3) -> list[RetrievedRecord]:
        return [
            RetrievedRecord(
                source_id="https://example.test/weak",
                title="Weak claim result",
                source_type="test",
                url="https://example.test/weak",
                doi=None,
                authors=[],
                published_at=None,
                abstract="The method can reduce hallucinations under benchmark conditions.",
                provenance="test fixture",
                evidence_level="ABSTRACT_AVAILABLE",
            )
        ]


def test_live_mode_keeps_metadata_only_records_out_of_evidence():
    settings = Settings(live_retrieval_enabled=True, mock_provider_enabled=True)
    workflow = VerificationWorkflow(
        settings,
        ProviderRouter(settings),
        retrievers=[MetadataOnlySource()],
    )
    result = workflow.verify(VerificationRequest(claim="RAG eliminates hallucinations"))
    assert result.sources
    assert result.evidence == []
    assert any("metadata-only" in item.casefold() for item in result.limitations)


def test_weaker_abstract_does_not_support_strong_quantifier():
    settings = Settings(live_retrieval_enabled=True, mock_provider_enabled=True)
    workflow = VerificationWorkflow(
        settings,
        ProviderRouter(settings),
        retrievers=[WeakAbstractSource()],
    )
    result = workflow.verify(VerificationRequest(claim="RAG eliminates hallucinations"))
    assert result.supporting_evidence == []
    assert result.verdict in {"REFUTED", "INSUFFICIENT_EVIDENCE", "MIXED"}


def test_thai_retrieval_hint_preserves_original_claim(workflow):
    result = workflow.verify(VerificationRequest(claim="การใช้ RAG ทำให้ AI ไม่หลอนเลย"))

    assert result.original_claim == "การใช้ RAG ทำให้ AI ไม่หลอนเลย"
    assert result.normalized_claim == result.original_claim
    assert {query.query for query in result.queries} == {
        "retrieval augmented generation hallucinations",
        "limitations counterexamples retrieval augmented generation hallucinations",
    }


def test_evidence_excerpt_is_bounded_for_user_facing_results(workflow):
    excerpt = workflow._evidence_excerpt("word " * 500)

    assert len(excerpt) <= workflow.MAX_EVIDENCE_EXCERPT
    assert excerpt.endswith("...")


def test_evidence_stance_is_separate_from_evidence():
    result = FixtureSource().search("limitations counterexamples RAG eliminates hallucinations")
    assert result[0].source_type == "deterministic_fixture"


def test_invalid_evidence_citation_is_detected(workflow):
    result = workflow.verify(VerificationRequest(claim="RAG eliminates hallucinations"))
    invalid_details = result.verdict_details.model_copy(
        update={"supporting_evidence_ids": ["missing-evidence"]}
    )
    invalid = result.model_copy(update={"verdict_details": invalid_details})
    with pytest.raises(DeterministicValidationError):
        validate_result(invalid)


def test_insufficient_evidence_behavior(workflow):
    result = workflow.verify(VerificationRequest(claim="Unknown system has no evidence"))
    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    assert result.evidence == []
