from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class RetrievedRecord:
    source_id: str
    title: str
    source_type: str
    url: str | None
    doi: str | None
    authors: list[str]
    published_at: datetime | None
    abstract: str | None
    provenance: str
    evidence_level: str = "METADATA_ONLY"


class EvidenceSource(Protocol):
    name: str

    def search(self, query: str, limit: int = 3) -> list[RetrievedRecord]: ...
