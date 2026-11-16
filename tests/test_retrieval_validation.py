import pytest

from vericlaim.config import Settings
from vericlaim.domain.models import EvidenceStance, VerdictLabel, VerificationRequest
from vericlaim.evidence_graph import build_evidence_graph
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
    with pytest.raises(DeterministicValidationError, match="verdict cites nonexistent evidence"):
        validate_result(invalid)


@pytest.fixture
def refuted_result(workflow):
    return workflow.verify({"claim": "RAG eliminates hallucinations."})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("atomic_claims", "duplicate atomic claim ID"),
        ("sources", "duplicate source ID"),
        ("evidence", "duplicate evidence ID"),
        ("assessments", "duplicate assessment for evidence ID"),
    ],
)
def test_duplicate_graph_identity_is_detected(refuted_result, field, message):
    entries = getattr(refuted_result, field)
    invalid = refuted_result.model_copy(update={field: [*entries, entries[0]]})

    with pytest.raises(DeterministicValidationError, match=message):
        validate_result(invalid)


def test_assessment_must_reference_run_evidence(refuted_result):
    unknown = refuted_result.assessments[0].model_copy(update={"evidence_id": "unknown"})
    invalid = refuted_result.model_copy(update={"assessments": [unknown]})

    with pytest.raises(
        DeterministicValidationError, match="assessment references missing evidence"
    ):
        validate_result(invalid)


@pytest.mark.parametrize("stance", [None, EvidenceStance.SUPPORTS, EvidenceStance.NEUTRAL])
def test_verdict_citations_require_matching_assessed_stance(refuted_result, stance):
    assessments = (
        []
        if stance is None
        else [refuted_result.assessments[0].model_copy(update={"stance": stance})]
    )
    invalid = refuted_result.model_copy(update={"assessments": assessments})

    with pytest.raises(DeterministicValidationError, match="contradicting citations require"):
        validate_result(invalid)


@pytest.mark.parametrize("field", ["supporting_evidence_ids", "contradicting_evidence_ids"])
def test_duplicate_verdict_citation_is_detected(refuted_result, field):
    details = refuted_result.verdict_details
    invalid = refuted_result.model_copy(
        update={
            "verdict_details": details.model_copy(
                update={field: getattr(details, field) + details.contradicting_evidence_ids}
            )
        }
    )

    with pytest.raises(DeterministicValidationError, match="duplicate verdict citation"):
        validate_result(invalid)


@pytest.mark.parametrize("change", ["omit", "duplicate", "change_provenance"])
def test_verdict_evidence_list_matches_cited_records(refuted_result, change):
    evidence = refuted_result.contradicting_evidence[0]
    altered = {
        "omit": [],
        "duplicate": [evidence, evidence],
        "change_provenance": [evidence.model_copy(update={"provenance": "different provenance"})],
    }
    invalid = refuted_result.model_copy(update={"contradicting_evidence": altered[change]})

    with pytest.raises(DeterministicValidationError, match="contradicting evidence list"):
        validate_result(invalid)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"confidence": 0.1}, "verdict confidence"),
        ({"verdict": VerdictLabel.SUPPORTED}, "verdict label"),
        ({"summary": ""}, "verdict explanation"),
    ],
)
def test_public_verdict_matches_explained_score(refuted_result, changes, message):
    invalid = refuted_result.model_copy(update=changes)

    with pytest.raises(DeterministicValidationError, match=message):
        validate_result(invalid)


def test_score_requires_explanation_even_when_both_fields_match(refuted_result):
    invalid = refuted_result.model_copy(
        update={
            "summary": " ",
            "verdict_details": refuted_result.verdict_details.model_copy(
                update={"explanation": " "}
            ),
        }
    )

    with pytest.raises(DeterministicValidationError, match="verdict explanation"):
        validate_result(invalid)


def test_same_source_and_excerpt_remain_distinct_across_atomic_claims(workflow):
    atomic_text = "The intervention improves measured outcomes in the evaluated setting"
    result = workflow.verify({"claim": " and ".join([atomic_text] * 4)})
    graph = build_evidence_graph(result)

    assert len(result.sources) == 1
    assert len(result.evidence) == len({item.evidence_id for item in result.evidence}) == 4
    assert {item.atomic_id for item in result.evidence} == {
        item.atomic_id for item in result.atomic_claims
    }
    assert len({(item.source_id, item.excerpt) for item in result.evidence}) == 1
    assert sum(node.kind == "source" for node in graph.nodes) == 1
    assert sum(edge.relation == "cited_from" for edge in graph.edges) == 4
    assert len(result.supporting_evidence) == 3
    assert validate_result(result) == result


def test_insufficient_evidence_behavior(workflow):
    result = workflow.verify(VerificationRequest(claim="Unknown system has no evidence"))
    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    assert result.evidence == []
