import pytest

from vericlaim.domain.analysis import analyze_claim, detect_strong_quantifiers, normalize_claim
from vericlaim.domain.decomposition import decompose_claim
from vericlaim.domain.models import Claim


def test_strong_quantifiers_are_detected():
    assert "eliminates" in detect_strong_quantifiers("RAG eliminates hallucinations")
    assert "always" in detect_strong_quantifiers("It always works")


def test_claim_analysis_has_structured_fields():
    result = analyze_claim("Model A is 20% faster than Model B")
    assert result.verifiable is True
    assert result.numerical_claims == ["20%"]
    assert result.comparisons


def test_decomposition_schema_preserves_atomic_ids():
    claim = Claim(
        original_text="A is cheaper and B is faster",
        normalized_text="A is cheaper and B is faster",
        analysis=analyze_claim("A is cheaper and B is faster"),
    )
    atomic = decompose_claim(claim)
    assert [item.atomic_id for item in atomic] == ["C1", "C2"]
    assert all(item.claim_id == claim.claim_id for item in atomic)


def test_thai_normalization_preserves_semantic_strength():
    original = "การใช้ RAG ทำให้ AI ไม่หลอนเลย"
    normalized = normalize_claim(original)
    assert normalized == original
    assert "ไม่หลอนเลย" in normalized
    assert "reduce" not in normalized.casefold()


@pytest.mark.parametrize(
    ("text", "marker"),
    [
        ("ระบบนี้ทำงานทุกครั้ง", "ทุกครั้ง"),
        ("ระบบนี้ทำงานเสมอ", "เสมอ"),
        ("ระบบนี้ไม่เคยล้มเหลว", "ไม่เคย"),
        ("ระบบนี้ไม่มีเลย", "ไม่มีเลย"),
        ("ระบบ A มากกว่า ระบบ B", "มากกว่า"),
        ("ระบบ A น้อยกว่า ระบบ B", "น้อยกว่า"),
        ("ข้อผิดพลาดลดลง 50%", "ลดลง 50%"),
        ("ระบบนี้เร็วกว่า 2 เท่า", "เร็วกว่า 2 เท่า"),
        ("ปัจจุบันระบบนี้ใช้งานได้", "ปัจจุบัน"),
    ],
)
def test_thai_semantic_markers_survive_normalization(text, marker):
    normalized = normalize_claim(text)

    assert normalized == text
    assert marker in normalized


def test_thai_semantic_markers_are_analyzed_without_weakening():
    result = analyze_claim("ระบบนี้ไม่เคยล้มเหลวและเร็วกว่า 2 เท่า ปัจจุบัน")

    assert {"ไม่เคย"}.issubset(result.strong_quantifiers)
    assert "เร็วกว่า" in result.comparisons
    assert result.numerical_claims == ["2"]
    assert result.temporal_sensitivity is True


def test_subjective_claim_is_non_verifiable(workflow):
    result = workflow.verify({"claim": "The best AI model is beautiful"})
    assert result.verdict == "NON_VERIFIABLE"
