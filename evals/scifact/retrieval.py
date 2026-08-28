from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .dataset import SciFactCorpus, SciFactDocument

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text.casefold()))


@dataclass(frozen=True)
class RetrievedSentence:
    evidence_id: str
    document_id: str
    sentence_index: int
    text: str
    score: float


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    title: str
    score: float
    rank: int
    sentences: tuple[RetrievedSentence, ...]


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_documents: int = 5
    top_k_sentences_per_document: int = 3
    k1: float = 1.2
    b: float = 0.75
    algorithm: str = "bm25_lexical_v1"


def _document_text(document: SciFactDocument) -> str:
    return " ".join((document.title, *document.abstract_sentences))


class BM25Retriever:
    """Transparent closed-corpus BM25 with deterministic tie-breaking."""

    def __init__(self, corpus: SciFactCorpus, config: RetrievalConfig | None = None) -> None:
        self.corpus = corpus
        self.config = config or RetrievalConfig()
        self._tokens = {
            document_id: tokenize(_document_text(document))
            for document_id, document in corpus.documents.items()
        }
        self._sentence_tokens = {
            document_id: tuple(tokenize(sentence) for sentence in document.abstract_sentences)
            for document_id, document in corpus.documents.items()
        }
        self._doc_lengths = {
            document_id: len(tokens) for document_id, tokens in self._tokens.items()
        }
        self._average_length = sum(self._doc_lengths.values()) / max(len(self._doc_lengths), 1)
        self._document_frequency: dict[str, int] = {}
        for tokens in self._tokens.values():
            for token in set(tokens):
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1

    def _score(self, query_tokens: tuple[str, ...], document_id: str) -> float:
        tokens = self._tokens[document_id]
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        document_count = len(self._tokens)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            df = self._document_frequency.get(token, 0)
            inverse_document_frequency = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + self.config.k1 * (
                1 - self.config.b + self.config.b * len(tokens) / max(self._average_length, 1)
            )
            score += inverse_document_frequency * (
                frequency * (self.config.k1 + 1) / max(denominator, 1e-12)
            )
        return score

    @staticmethod
    def _sentence_score(query_tokens: tuple[str, ...], tokens: tuple[str, ...]) -> float:
        if not tokens:
            return 0.0
        query_set = set(query_tokens)
        return sum(token in query_set for token in tokens) / len(tokens)

    def retrieve(self, claim: str) -> tuple[RetrievedDocument, ...]:
        query_tokens = tokenize(claim)
        scored = [
            (document_id, self._score(query_tokens, document_id))
            for document_id in self.corpus.documents
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        documents: list[RetrievedDocument] = []
        for rank, (document_id, score) in enumerate(scored[: self.config.top_k_documents], 1):
            document = self.corpus.documents[document_id]
            sentence_scores = [
                (index, self._sentence_score(query_tokens, tokens))
                for index, tokens in enumerate(self._sentence_tokens[document_id])
            ]
            sentence_scores.sort(key=lambda item: (-item[1], item[0]))
            sentences = tuple(
                RetrievedSentence(
                    evidence_id=f"doc:{document_id}:sentence:{index}",
                    document_id=document_id,
                    sentence_index=index,
                    text=document.abstract_sentences[index],
                    score=sentence_score,
                )
                for index, sentence_score in sentence_scores[
                    : self.config.top_k_sentences_per_document
                ]
            )
            documents.append(
                RetrievedDocument(
                    document_id=document_id,
                    title=document.title,
                    score=score,
                    rank=rank,
                    sentences=sentences,
                )
            )
        return tuple(documents)
