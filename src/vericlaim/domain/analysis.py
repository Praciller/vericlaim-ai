from __future__ import annotations

import re

from .models import ClaimAnalysis

THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
STRONG_QUANTIFIERS = (
    "always",
    "never",
    "all",
    "none",
    "every",
    "eliminates",
    "eliminate",
    "guarantees",
    "guarantee",
    "fastest",
    "best",
    "perfect",
    "ไม่มีวัน",
    "ทั้งหมด",
    "ทุก",
    "ทุกครั้ง",
    "เสมอ",
    "ไม่เคย",
    "ไม่มีเลย",
    "ไม่หลอนเลย",
    "รับประกัน",
)
SUBJECTIVE_MARKERS = ("best", "beautiful", "should", "better", "ดีที่สุด", "ควร")
THAI_COMPARISON_MARKERS = ("มากกว่า", "น้อยกว่า", "ลดลง", "เร็วกว่า")


def detect_strong_quantifiers(text: str) -> list[str]:
    lowered = text.casefold()
    detected: list[str] = []
    for term in STRONG_QUANTIFIERS:
        folded_term = term.casefold()
        found = (
            bool(re.search(rf"(?<!\w){re.escape(folded_term)}(?!\w)", lowered))
            if folded_term.isascii()
            else folded_term in lowered
        )
        if found:
            detected.append(term)
    return detected


def analyze_claim(text: str) -> ClaimAnalysis:
    language = "th" if THAI_RE.search(text) else "en"
    quantifiers = detect_strong_quantifiers(text)
    numbers = re.findall(r"(?<!\w)\d+(?:\.\d+)?%?", text)
    lowered_text = text.casefold()
    comparisons = re.findall(r"\b(?:faster|slower|cheaper|higher|lower|more|less)\b", lowered_text)
    comparisons.extend(marker for marker in THAI_COMPARISON_MARKERS if marker in lowered_text)
    subjective = any(marker in text.casefold() for marker in SUBJECTIVE_MARKERS)
    temporal = bool(
        re.search(r"\b(?:today|current|latest|now|since)\b", lowered_text)
        or any(marker in lowered_text for marker in ("ก่อน", "ปัจจุบัน", "ล่าสุด"))
    )
    claim_type = (
        "scientific_computing"
        if any(
            marker in text.casefold()
            for marker in (
                "model",
                "ai",
                "ml",
                "rag",
                "algorithm",
                "software",
                "hallucination",
                "อัลกอริทึม",
            )
        )
        else "technical"
    )
    entities = [
        word.strip(".,:;()") for word in text.split() if word[:1].isupper() and len(word) > 1
    ]
    verifiable = not subjective and len(text.split()) >= 2
    return ClaimAnalysis(
        language=language,
        claim_type=claim_type,
        verifiable=verifiable,
        temporal_sensitivity=temporal,
        subjective_language=subjective,
        strong_quantifiers=quantifiers,
        numerical_claims=numbers,
        comparisons=comparisons,
        entities=entities[:12],
        conditions=[],
    )


def normalize_claim(text: str) -> str:
    """Normalize layout only; translation and strength-changing paraphrase are forbidden."""
    return " ".join(text.strip().split())
