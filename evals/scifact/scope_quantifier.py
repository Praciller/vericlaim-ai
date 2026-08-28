from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCOPE_TERMS = (
    "always",
    "never",
    "all",
    "none",
    "every",
    "eliminates",
    "guarantees",
    "best",
    "fastest",
)


def detect_scope_features(claim: str) -> dict[str, Any]:
    lowered = claim.casefold()
    quantifiers = [term for term in SCOPE_TERMS if re.search(rf"\b{term}\b", lowered)]
    percentages = re.findall(r"\b\d+(?:\.\d+)?%", claim)
    multipliers = re.findall(r"\b\d+(?:\.\d+)?x\b", lowered)
    return {
        "quantifiers": quantifiers,
        "percentages": percentages,
        "multipliers": multipliers,
        "conditional": bool(re.search(r"\b(if|when|under|provided|conditional)\b", lowered)),
    }


def evaluate_fixture(path: str | Path) -> dict[str, Any]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    for row in rows:
        features = detect_scope_features(row["claim"])
        expected = row["expected"]
        results.append(
            {
                "id": row["id"],
                "claim": row["claim"],
                "features": features,
                "correct": features == expected,
            }
        )
    return {
        "name": "CUSTOM_SCOPE_QUANTIFIER_FIXTURE",
        "sample_size": len(results),
        "accuracy": sum(item["correct"] for item in results) / max(len(results), 1),
        "results": results,
        "does_not_claim_sciFact_performance": True,
    }
