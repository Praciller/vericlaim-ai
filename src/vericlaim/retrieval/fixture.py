from __future__ import annotations

import hashlib

from .base import RetrievedRecord


class FixtureSource:
    """Small deterministic offline corpus used by tests and local development."""

    name = "fixture"

    def search(self, query: str, limit: int = 3) -> list[RetrievedRecord]:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        lowered = query.casefold()
        if any(term in lowered for term in ("unknown", "unverifiable", "no evidence")):
            return []
        if any(
            term in lowered
            for term in (
                "eliminate",
                "eliminates",
                "never",
                "always",
                "guarantee",
                "รับประกัน",
                "ไม่หลอนเลย",
            )
        ):
            return [
                RetrievedRecord(
                    source_id=f"fixture:counter:{digest}",
                    title="Controlled evaluations report residual hallucinations",
                    source_type="deterministic_fixture",
                    url=f"fixture://vericlaim/counter/{digest}",
                    doi=None,
                    authors=["VeriClaim fixture corpus"],
                    published_at=None,
                    abstract=(
                        "Controlled retrieval-augmented generation evaluations continue to observe "
                        "residual hallucinations and scope limitations."
                    ),
                    provenance="bundled deterministic fixture; not an external publication",
                    evidence_level="ABSTRACT_AVAILABLE",
                )
            ]
        return [
            RetrievedRecord(
                source_id=f"fixture:support:{digest}",
                title="Technical evaluation reports measurable improvement under stated conditions",
                source_type="deterministic_fixture",
                url=f"fixture://vericlaim/support/{digest}",
                doi=None,
                authors=["VeriClaim fixture corpus"],
                published_at=None,
                abstract=(
                    "The intervention improves measured outcomes in the evaluated setting, but the "
                    "result is conditional on the benchmark and implementation."
                ),
                provenance="bundled deterministic fixture; not an external publication",
                evidence_level="ABSTRACT_AVAILABLE",
            ),
            RetrievedRecord(
                source_id=f"fixture:counter:{digest}",
                title="Technical evaluation reports limitations and residual errors",
                source_type="deterministic_fixture",
                url=f"fixture://vericlaim/counter/{digest}",
                doi=None,
                authors=["VeriClaim fixture corpus"],
                published_at=None,
                abstract=(
                    "Results vary across datasets and failure cases remain, so broad universal "
                    "wording is not supported."
                ),
                provenance="bundled deterministic fixture; not an external publication",
                evidence_level="ABSTRACT_AVAILABLE",
            ),
        ]
