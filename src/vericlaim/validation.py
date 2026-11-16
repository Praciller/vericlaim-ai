from __future__ import annotations

from pydantic import ValidationError

from .domain.models import EvidenceStance, VerificationResult


class DeterministicValidationError(ValueError):
    pass


def validate_result(result: VerificationResult) -> VerificationResult:
    evidence_by_id = {item.evidence_id: item for item in result.evidence}
    atomic_ids = {item.atomic_id for item in result.atomic_claims}
    source_ids = {item.source_id for item in result.sources}
    stance_by_id = {item.evidence_id: item.stance for item in result.assessments}
    if len(atomic_ids) != len(result.atomic_claims):
        raise DeterministicValidationError("duplicate atomic claim ID")
    if len(source_ids) != len(result.sources):
        raise DeterministicValidationError("duplicate source ID")
    if len(evidence_by_id) != len(result.evidence):
        raise DeterministicValidationError("duplicate evidence ID")
    if len(stance_by_id) != len(result.assessments):
        raise DeterministicValidationError("duplicate assessment for evidence ID")
    if stance_by_id.keys() - evidence_by_id.keys():
        raise DeterministicValidationError("assessment references missing evidence")
    if any(item.source_id not in source_ids for item in result.evidence):
        raise DeterministicValidationError("evidence references a missing source")
    if any(item.atomic_id not in atomic_ids for item in result.evidence):
        raise DeterministicValidationError("evidence references a missing atomic claim")

    details = result.verdict_details
    cited = details.supporting_evidence_ids + details.contradicting_evidence_ids
    missing = sorted(set(cited) - evidence_by_id.keys())
    if missing:
        raise DeterministicValidationError(f"verdict cites nonexistent evidence: {missing}")
    if len(set(cited)) != len(cited):
        raise DeterministicValidationError("duplicate verdict citation across or within lists")
    for side, stance, ids, records in (
        (
            "supporting",
            EvidenceStance.SUPPORTS,
            details.supporting_evidence_ids,
            result.supporting_evidence,
        ),
        (
            "contradicting",
            EvidenceStance.CONTRADICTS,
            details.contradicting_evidence_ids,
            result.contradicting_evidence,
        ),
    ):
        if any(stance_by_id.get(evidence_id) != stance for evidence_id in ids):
            raise DeterministicValidationError(f"{side} citations require {stance.value} stance")
        expected = {evidence_id: evidence_by_id[evidence_id] for evidence_id in ids}
        if (
            len(records) != len(expected)
            or {item.evidence_id: item for item in records} != expected
        ):
            raise DeterministicValidationError(f"{side} evidence list differs from cited records")

    if result.verdict != details.label:
        raise DeterministicValidationError("verdict label differs from verdict details")
    if result.confidence != details.confidence:
        raise DeterministicValidationError("verdict confidence differs from verdict details")
    if not details.explanation.strip() or result.summary != details.explanation:
        raise DeterministicValidationError("verdict explanation is missing or differs from summary")

    if result.completed_at is not None and result.completed_at.tzinfo is None:
        raise DeterministicValidationError("completed_at must include timezone information")
    for evidence in result.evidence:
        if not evidence.provenance or not evidence.excerpt:
            raise DeterministicValidationError("evidence provenance and excerpt are required")
    try:
        VerificationResult.model_validate(result.model_dump())
    except ValidationError as exc:
        raise DeterministicValidationError("result schema validation failed") from exc
    return result
