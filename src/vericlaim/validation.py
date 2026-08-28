from __future__ import annotations

from pydantic import ValidationError

from .domain.models import VerificationResult


class DeterministicValidationError(ValueError):
    pass


def validate_result(result: VerificationResult) -> VerificationResult:
    evidence_ids = {item.evidence_id for item in result.evidence}
    atomic_ids = {item.atomic_id for item in result.atomic_claims}
    source_ids = {item.source_id for item in result.sources}
    if any(item.source_id not in source_ids for item in result.evidence):
        raise DeterministicValidationError("evidence references a missing source")
    if any(item.atomic_id not in atomic_ids for item in result.evidence):
        raise DeterministicValidationError("evidence references a missing atomic claim")
    cited = set(
        result.verdict_details.supporting_evidence_ids
        + result.verdict_details.contradicting_evidence_ids
    )
    if not cited.issubset(evidence_ids):
        missing = sorted(cited - evidence_ids)
        raise DeterministicValidationError(f"verdict cites nonexistent evidence: {missing}")
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
