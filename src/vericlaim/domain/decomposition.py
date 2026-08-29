from __future__ import annotations

import re

from .analysis import analyze_claim
from .models import AtomicClaim, Claim


def decompose_claim(claim: Claim, *, max_atomic_claims: int | None = None) -> list[AtomicClaim]:
    # Split only on explicit conjunctions. The original wording, including
    # negation and quantifiers, is retained in each atomic statement.
    pieces = [
        piece.strip(" .")
        for piece in re.split(
            r"\s+(?:and|while|และ|และยัง)\s+", claim.normalized_text, flags=re.IGNORECASE
        )
    ]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        pieces = [claim.normalized_text]
    if max_atomic_claims is not None and len(pieces) > max_atomic_claims:
        raise ValueError("claim exceeds the maximum atomic claim limit")
    return [
        AtomicClaim(
            atomic_id=f"C{index}",
            claim_id=claim.claim_id,
            text=piece,
            claim_type=analyze_claim(piece).claim_type,
        )
        for index, piece in enumerate(pieces, start=1)
    ]
