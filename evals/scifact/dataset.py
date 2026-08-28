from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SciFactDataError(ValueError):
    """Raised when a SciFact snapshot is missing, malformed, or changed."""


SUPPORTED_ANNOTATION_LABELS = {"SUPPORT", "CONTRADICT"}
LABEL_SUPPORT = "SUPPORT"
LABEL_CONTRADICT = "CONTRADICT"
LABEL_NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
LABEL_MIXED = "MIXED"


@dataclass(frozen=True)
class SciFactGoldEvidence:
    document_id: str
    sentence_indices: tuple[int, ...]
    label: str

    @property
    def document_evidence_id(self) -> str:
        return f"doc:{self.document_id}"

    @property
    def sentence_evidence_ids(self) -> tuple[str, ...]:
        return tuple(f"doc:{self.document_id}:sentence:{index}" for index in self.sentence_indices)


@dataclass(frozen=True)
class SciFactClaim:
    claim_id: str
    text: str
    gold_label: str
    gold_evidence: tuple[SciFactGoldEvidence, ...]
    cited_document_ids: tuple[str, ...]

    @property
    def gold_document_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.document_id for item in self.gold_evidence}))

    @property
    def gold_sentence_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    sentence_id
                    for evidence in self.gold_evidence
                    for sentence_id in evidence.sentence_evidence_ids
                }
            )
        )


@dataclass(frozen=True)
class SciFactDocument:
    document_id: str
    title: str
    abstract_sentences: tuple[str, ...]
    structured: bool

    def sentence_text(self, index: int) -> str | None:
        if 0 <= index < len(self.abstract_sentences):
            return self.abstract_sentences[index]
        return None


@dataclass(frozen=True)
class SciFactExample:
    claim: SciFactClaim
    document_ids: tuple[str, ...]


@dataclass(frozen=True)
class SciFactCorpus:
    claims: tuple[SciFactClaim, ...]
    documents: dict[str, SciFactDocument]
    split: str
    data_dir: str

    @property
    def examples(self) -> tuple[SciFactExample, ...]:
        return tuple(
            SciFactExample(claim=claim, document_ids=self.document_ids) for claim in self.claims
        )

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.documents))

    @property
    def label_distribution(self) -> dict[str, int]:
        return dict(sorted(Counter(claim.gold_label for claim in self.claims).items()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SciFactDataError(f"missing SciFact file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SciFactDataError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise SciFactDataError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _string_id(value: object, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise SciFactDataError(f"{field} must be an integer-like ID")
    return str(value)


def _parse_evidence(raw: object, claim_id: str) -> tuple[SciFactGoldEvidence, ...]:
    if not isinstance(raw, dict):
        raise SciFactDataError(f"claim {claim_id} evidence must be an object")
    parsed: list[SciFactGoldEvidence] = []
    for document_id, evidence_annotations in raw.items():
        if not isinstance(evidence_annotations, list):
            raise SciFactDataError(f"claim {claim_id} evidence annotations must be a list")
        for annotation in evidence_annotations:
            if not isinstance(annotation, dict):
                raise SciFactDataError(f"claim {claim_id} has a malformed evidence annotation")
            label = annotation.get("label")
            if label not in SUPPORTED_ANNOTATION_LABELS:
                raise SciFactDataError(f"claim {claim_id} has unknown evidence label: {label}")
            sentences = annotation.get("sentences")
            if not isinstance(sentences, list) or not all(
                isinstance(index, int) and not isinstance(index, bool) and index >= 0
                for index in sentences
            ):
                raise SciFactDataError(f"claim {claim_id} has invalid evidence sentences")
            parsed.append(
                SciFactGoldEvidence(
                    document_id=_string_id(document_id, "document_id"),
                    sentence_indices=tuple(sorted(set(sentences))),
                    label=str(label),
                )
            )
    return tuple(parsed)


def _claim_label(evidence: tuple[SciFactGoldEvidence, ...]) -> str:
    if not evidence:
        return LABEL_NOT_ENOUGH_INFO
    labels = {item.label for item in evidence}
    if labels == {LABEL_SUPPORT}:
        return LABEL_SUPPORT
    if labels == {LABEL_CONTRADICT}:
        return LABEL_CONTRADICT
    return LABEL_MIXED


def _parse_claim(row: dict[str, Any], *, require_evidence: bool = True) -> SciFactClaim:
    if "id" not in row or not isinstance(row.get("claim"), str):
        raise SciFactDataError("SciFact claim requires id and claim")
    evidence_raw = row.get("evidence", {})
    if require_evidence and "evidence" not in row:
        raise SciFactDataError(f"claim {row['id']} has no gold evidence field")
    evidence = _parse_evidence(evidence_raw, _string_id(row["id"], "id"))
    cited = row.get("cited_doc_ids", [])
    if not isinstance(cited, list):
        raise SciFactDataError(f"claim {row['id']} cited_doc_ids must be a list")
    return SciFactClaim(
        claim_id=_string_id(row["id"], "id"),
        text=row["claim"].strip(),
        gold_label=_claim_label(evidence),
        gold_evidence=evidence,
        cited_document_ids=tuple(_string_id(value, "cited_doc_ids") for value in cited),
    )


def _parse_document(row: dict[str, Any]) -> SciFactDocument:
    required = ("doc_id", "title", "abstract", "structured")
    if any(field not in row for field in required):
        raise SciFactDataError("SciFact corpus row is missing a required field")
    abstract = row["abstract"]
    if (
        not isinstance(row["title"], str)
        or not isinstance(abstract, list)
        or not all(isinstance(sentence, str) for sentence in abstract)
    ):
        raise SciFactDataError(f"document {row.get('doc_id')} has an invalid abstract")
    if not isinstance(row["structured"], bool):
        raise SciFactDataError(f"document {row.get('doc_id')} has an invalid structured flag")
    return SciFactDocument(
        document_id=_string_id(row["doc_id"], "doc_id"),
        title=row["title"].strip(),
        abstract_sentences=tuple(sentence.strip() for sentence in abstract),
        structured=row["structured"],
    )


def load_scifact(data_dir: str | Path, split: str = "dev") -> SciFactCorpus:
    """Load a labeled SciFact split without exposing gold data to inference code."""

    root = Path(data_dir)
    if split not in {"train", "dev", "test"}:
        raise SciFactDataError(f"unsupported SciFact split: {split}")
    claim_rows = _read_jsonl(root / f"claims_{split}.jsonl")
    claims = tuple(_parse_claim(row, require_evidence=split != "test") for row in claim_rows)
    if split == "test":
        raise SciFactDataError("SciFact test labels are not public; use train or dev")
    document_rows = _read_jsonl(root / "corpus.jsonl")
    documents = {}
    for row in document_rows:
        document = _parse_document(row)
        if document.document_id in documents:
            raise SciFactDataError(f"duplicate corpus document: {document.document_id}")
        documents[document.document_id] = document
    missing = sorted(
        {
            evidence.document_id
            for claim in claims
            for evidence in claim.gold_evidence
            if evidence.document_id not in documents
        }
    )
    if missing:
        raise SciFactDataError(f"gold evidence documents missing from corpus: {missing[:5]}")
    return SciFactCorpus(
        claims=claims,
        documents=documents,
        split=split,
        data_dir=str(root.resolve()),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_distribution(path: Path) -> dict[str, int]:
    rows = _read_jsonl(path)
    if path.name == "claims_test.jsonl":
        return {}
    labels = Counter(_parse_claim(row).gold_label for row in rows)
    return dict(sorted(labels.items()))


def build_manifest(
    *,
    data_dir: str | Path,
    archive_path: str | Path | None,
    source: str,
    revision: str,
    downloaded_at: str | None = None,
) -> dict[str, Any]:
    root = Path(data_dir)
    if not root.is_dir():
        raise SciFactDataError(f"dataset directory does not exist: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl")):
        item: dict[str, Any] = {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if path.name.startswith("claims_"):
            item["rows"] = len(_read_jsonl(path))
            item["label_distribution"] = _label_distribution(path)
        files.append(item)
    archive = None
    if archive_path is not None:
        archive_file = Path(archive_path)
        if archive_file.is_file():
            archive = {
                "path": archive_file.name,
                "sha256": sha256_file(archive_file),
                "bytes": archive_file.stat().st_size,
            }
    split_rows = {}
    for split in ("train", "dev", "test"):
        split_path = root / f"claims_{split}.jsonl"
        split_rows[split] = {
            "rows": len(_read_jsonl(split_path)),
            "label_distribution": _label_distribution(split_path),
        }
    return {
        "dataset_name": "SciFact",
        "source": source,
        "revision": revision,
        "downloaded_at": downloaded_at or datetime.now(UTC).isoformat(),
        "data_directory": root.name,
        "archive": archive,
        "files": files,
        "split_names": ["train", "dev", "test"],
        "split_rows": split_rows,
        "corpus_rows": len(_read_jsonl(root / "corpus.jsonl")),
        "label_distribution": {
            split: values["label_distribution"] for split, values in split_rows.items()
        },
    }


def validate_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate every recorded hash and row count; never silently accept drift."""

    path = Path(manifest_path)
    if not path.is_file():
        raise SciFactDataError(f"missing dataset manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SciFactDataError(f"invalid dataset manifest: {path}") from exc
    required = {
        "dataset_name",
        "source",
        "revision",
        "downloaded_at",
        "files",
        "split_names",
        "label_distribution",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise SciFactDataError(f"dataset manifest missing fields: {missing}")
    root = Path(manifest.get("data_directory", path.parent / "data"))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    for item in manifest["files"]:
        file_path = root / item["path"]
        if not file_path.is_file():
            raise SciFactDataError(f"manifest file is missing: {file_path}")
        actual_hash = sha256_file(file_path)
        if actual_hash != item["sha256"]:
            raise SciFactDataError(f"dataset hash mismatch: {item['path']}")
        if "bytes" in item and file_path.stat().st_size != item["bytes"]:
            raise SciFactDataError(f"dataset byte count mismatch: {item['path']}")
        if "rows" in item and len(_read_jsonl(file_path)) != item["rows"]:
            raise SciFactDataError(f"dataset row count mismatch: {item['path']}")
        if (
            "label_distribution" in item
            and _label_distribution(file_path) != item["label_distribution"]
        ):
            raise SciFactDataError(f"dataset label distribution mismatch: {item['path']}")
    archive = manifest.get("archive")
    if isinstance(archive, dict) and archive.get("path"):
        archive_path = Path(str(archive["path"]))
        if not archive_path.is_absolute():
            archive_path = path.parent / archive_path
        if not archive_path.is_file():
            raise SciFactDataError(f"manifest archive is missing: {archive_path}")
        if sha256_file(archive_path) != archive.get("sha256"):
            raise SciFactDataError("dataset archive hash mismatch")
    return manifest
