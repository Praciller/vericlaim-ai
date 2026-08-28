from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_VERSION = "scifact-structured-cache-v2"


@dataclass(frozen=True)
class CacheLookup:
    key: str
    parsed: dict[str, Any]
    provider: str
    configured_model: str
    actual_model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


def input_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def make_cache_key(
    *,
    architecture: str,
    provider: str,
    configured_model: str,
    prompt_version: str,
    generation_parameters: dict[str, Any],
    input_hash_value: str,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "cache_version": CACHE_VERSION,
        "architecture": architecture,
        "provider": provider,
        "configured_model": configured_model,
        "prompt_version": prompt_version,
        "generation_parameters": generation_parameters,
        "input_hash": input_hash_value,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), identity


class StructuredResponseCache:
    """Cache parsed response objects only; prompts, headers, and raw responses never persist."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> CacheLookup | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if value.get("cache_version") != CACHE_VERSION or not isinstance(value.get("parsed"), dict):
            return None
        return CacheLookup(
            key=key,
            parsed=value["parsed"],
            provider=str(value["provider"]),
            configured_model=str(value["configured_model"]),
            actual_model=str(value["actual_model"]),
            input_tokens=int(value.get("input_tokens", 0)),
            output_tokens=int(value.get("output_tokens", 0)),
            latency_ms=int(value.get("latency_ms", 0)),
        )

    def put(
        self,
        *,
        key: str,
        identity: dict[str, Any],
        parsed: dict[str, Any],
        provider: str,
        configured_model: str,
        actual_model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
    ) -> None:
        value = {
            "cache_version": CACHE_VERSION,
            **identity,
            "parsed": parsed,
            "provider": provider,
            "configured_model": configured_model,
            "actual_model": actual_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        }
        self._path(key).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def count(self) -> int:
        return len(tuple(self.root.glob("*.json")))
