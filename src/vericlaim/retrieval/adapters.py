from __future__ import annotations

import html
import re
from datetime import datetime

import httpx

from .base import RetrievedRecord


class RetrievalError(Exception):
    pass


def _safe_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return re.sub(r"\s+", " ", without_tags).strip() or None


def _abstract_from_inverted_index(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    positions: dict[int, str] = {}
    for word, indexes in value.items():
        if not isinstance(word, str) or not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int) and index >= 0:
                positions[index] = word
    if not positions:
        return None
    return _safe_text(" ".join(positions[index] for index in sorted(positions)))


class OpenAlexSource:
    name = "openalex"
    endpoint = "https://api.openalex.org/works"

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 3) -> list[RetrievedRecord]:
        try:
            response = httpx.get(
                self.endpoint,
                params={"search": query, "per-page": min(limit, 10)},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise RetrievalError("OpenAlex retrieval failed") from exc
        if not isinstance(data, dict):
            raise RetrievalError("OpenAlex returned an invalid response")
        records: list[RetrievedRecord] = []
        for item in data.get("results", []):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            primary = item.get("primary_location") or {}
            source = primary.get("source") or {}
            abstract = _abstract_from_inverted_index(item.get("abstract_inverted_index"))
            records.append(
                RetrievedRecord(
                    source_id=str(item["id"]),
                    title=_safe_text(item.get("title")) or "Untitled OpenAlex work",
                    source_type="openalex",
                    url=_safe_text(primary.get("landing_page_url")) or _safe_text(item.get("doi")),
                    doi=_safe_text(item.get("doi")),
                    authors=[
                        str((author.get("author") or {}).get("display_name"))
                        for author in item.get("authorships", [])
                        if isinstance(author, dict)
                        and (author.get("author") or {}).get("display_name")
                    ],
                    published_at=datetime.fromisoformat(item["publication_date"]).replace(
                        tzinfo=None
                    )
                    if item.get("publication_date")
                    else None,
                    abstract=abstract,
                    provenance=f"OpenAlex API; host={source.get('display_name', 'unknown')}",
                    evidence_level=("ABSTRACT_AVAILABLE" if abstract else "METADATA_ONLY"),
                )
            )
        return records


class CrossrefSource:
    name = "crossref"
    endpoint = "https://api.crossref.org/works"

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 3) -> list[RetrievedRecord]:
        try:
            response = httpx.get(
                self.endpoint,
                params={"query.bibliographic": query, "rows": min(limit, 10)},
                timeout=self.timeout_seconds,
                headers={"User-Agent": "vericlaim-ai/0.1 (mailto:research@example.invalid)"},
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise RetrievalError("Crossref retrieval failed") from exc
        if not isinstance(data, dict):
            raise RetrievalError("Crossref returned an invalid response")
        records: list[RetrievedRecord] = []
        message = data.get("message")
        if not isinstance(message, dict):
            raise RetrievalError("Crossref returned an invalid message")
        for item in message.get("items", []):
            if not isinstance(item, dict) or not item.get("DOI"):
                continue
            published = item.get("published-print") or item.get("published-online") or {}
            parts = (published.get("date-parts") or [[]])[0]
            try:
                published_at = (
                    datetime(
                        int(parts[0]),
                        int(parts[1]) if len(parts) > 1 else 1,
                        int(parts[2]) if len(parts) > 2 else 1,
                    )
                    if parts
                    else None
                )
            except (TypeError, ValueError, IndexError):
                published_at = None
            abstract = _safe_text(item.get("abstract"))
            records.append(
                RetrievedRecord(
                    source_id=f"https://doi.org/{item['DOI']}",
                    title=_safe_text((item.get("title") or [None])[0]) or "Untitled Crossref work",
                    source_type="crossref",
                    url=f"https://doi.org/{item['DOI']}",
                    doi=str(item["DOI"]),
                    authors=[
                        f"{author.get('given', '')} {author.get('family', '')}".strip()
                        for author in item.get("author", [])
                        if isinstance(author, dict)
                    ],
                    published_at=published_at,
                    abstract=abstract,
                    provenance="Crossref REST API",
                    evidence_level=("ABSTRACT_AVAILABLE" if abstract else "METADATA_ONLY"),
                )
            )
        return records


class ArxivSource:
    """Placeholder adapter boundary; arXiv XML parsing is intentionally deferred."""

    name = "arxiv"

    def search(self, query: str, limit: int = 3) -> list[RetrievedRecord]:
        return []
