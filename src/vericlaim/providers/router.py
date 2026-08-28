from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import Settings
from ..domain.models import ProviderErrorCategory, ProviderStatus, ProviderUsage
from .base import (
    GeminiProvider,
    LLMProvider,
    MockProvider,
    OKMDProvider,
    OpenAICompatibleProvider,
    ProviderException,
    ProviderRequest,
    ProviderResponse,
    ThaiLLMProvider,
)

TASK_DEFAULTS = {
    "claim_analysis": "groq",
    "claim_decomposition": "groq",
    "query_generation": "groq",
    "evidence_classification": "groq",
    "evidence_audit": "gemini",
    "final_judgment": "gemini",
    "critique": "okmd",
    "thai_normalization": "thaillm",
    "thai_semantic_review": "thaillm",
}

TASK_FALLBACKS = {
    "critique": ("okmd", "openrouter"),
}

PROVIDER_ORDER = ("mock", "groq", "gemini", "okmd", "thaillm", "openrouter", "cerebras")


@dataclass
class RouterResult:
    response: ProviderResponse
    usage: ProviderUsage
    fallbacks: list[str]


class ProviderRouter:
    def __init__(self, settings: Settings, providers: dict[str, LLMProvider] | None = None) -> None:
        self.settings = settings
        self._disabled_notes: dict[str, str] = {}
        self.providers: dict[str, LLMProvider] = (
            providers if providers is not None else self._build_providers()
        )
        self.fallbacks = [name for name in PROVIDER_ORDER if name in self.providers]
        self._last_status: dict[str, str] = {}
        self._quota: dict[str, tuple[int | None, int | None]] = {}

    @staticmethod
    def _is_free_openrouter_model(model: str) -> bool:
        return model == "openrouter/free" or model.endswith(":free")

    def _build_providers(self) -> dict[str, LLMProvider]:
        providers: dict[str, LLMProvider] = {}
        if self.settings.mock_provider_enabled:
            providers["mock"] = MockProvider()
        if self.settings.groq_enabled and self.settings.groq_api_key:
            providers["groq"] = OpenAICompatibleProvider(
                "groq",
                self.settings.groq_model,
                self.settings.groq_api_key,
                "https://api.groq.com/openai/v1",
            )
        if self.settings.gemini_enabled and self.settings.gemini_api_key:
            providers["gemini"] = GeminiProvider(
                "gemini", self.settings.gemini_model, self.settings.gemini_api_key
            )
        if self.settings.okmd_enabled and self.settings.okmd_api_key:
            providers["okmd"] = OKMDProvider(
                "okmd",
                self.settings.okmd_model,
                self.settings.okmd_api_key,
                "https://gen.ai.kku.ac.th/okmd/api/v1",
            )
        if self.settings.thaillm_enabled and self.settings.thaillm_api_key:
            providers["thaillm"] = ThaiLLMProvider(
                "thaillm",
                self.settings.thaillm_model,
                self.settings.thaillm_api_key,
                "https://thaillm.or.th/api/v1",
            )
        if self.settings.cerebras_enabled and self.settings.cerebras_api_key:
            providers["cerebras"] = OpenAICompatibleProvider(
                "cerebras",
                self.settings.cerebras_model,
                self.settings.cerebras_api_key,
                "https://api.cerebras.ai/v1",
            )
        if self.settings.openrouter_api_key and not self._is_free_openrouter_model(
            self.settings.openrouter_model
        ):
            self._disabled_notes["openrouter"] = (
                "OpenRouter disabled because the configured route is not allowlisted as free-only."
            )
        elif self.settings.openrouter_enabled and self.settings.openrouter_api_key:
            providers["openrouter"] = OpenAICompatibleProvider(
                "openrouter",
                self.settings.openrouter_model,
                self.settings.openrouter_api_key,
                "https://openrouter.ai/api/v1",
            )
        return providers

    def has_provider(self, name: str) -> bool:
        return name in self.providers

    def _candidates(self, task: str, reproducible: bool) -> list[str]:
        preferred = TASK_DEFAULTS.get(task, self.settings.default_provider)
        task_fallbacks = TASK_FALLBACKS.get(task, ())
        candidates = [preferred]
        candidates.extend(name for name in task_fallbacks if name != preferred)
        candidates.extend(
            name for name in self.fallbacks if name != preferred and name not in candidates
        )
        if reproducible and not self.settings.allow_non_reproducible_openrouter:
            candidates = [name for name in candidates if name != "openrouter"]
        return candidates

    @staticmethod
    def _can_retry_same_provider(error: ProviderException, attempt: int) -> bool:
        if attempt >= 1 or not error.retryable:
            return False
        if error.category == ProviderErrorCategory.RATE_LIMIT:
            return error.retry_after is None or error.retry_after <= 1.0
        return error.category in {
            ProviderErrorCategory.NETWORK,
            ProviderErrorCategory.TIMEOUT,
            ProviderErrorCategory.SERVER_ERROR,
        }

    def invoke(self, task: str, prompt: str, *, reproducible: bool = False) -> RouterResult:
        last_error: ProviderException | None = None
        fallbacks: list[str] = []
        for name in self._candidates(task, reproducible):
            provider = self.providers.get(name)
            if provider is None:
                continue
            attempt = 0
            while True:
                try:
                    response = provider.generate(ProviderRequest(task=task, prompt=prompt))
                    self._last_status[name] = response.status
                    self._quota[name] = (
                        response.quota_remaining_tokens,
                        response.quota_limit_tokens,
                    )
                    usage = ProviderUsage(
                        provider=response.provider,
                        model=response.actual_model or response.model,
                        configured_model=response.configured_model or response.model,
                        actual_model=response.actual_model or response.model,
                        task=task,
                        request_count=1,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        total_tokens=response.total_tokens,
                        latency_ms=response.latency_ms,
                        status=response.status,
                        error_code=response.error_code,
                        quota_limit_tokens=response.quota_limit_tokens,
                        quota_used_tokens=response.quota_used_tokens,
                        quota_remaining_tokens=response.quota_remaining_tokens,
                        fallbacks=fallbacks,
                    )
                    return RouterResult(response=response, usage=usage, fallbacks=fallbacks)
                except ProviderException as exc:
                    self._last_status[name] = exc.category.value
                    last_error = exc
                    fallbacks.append(f"{name}:{exc.category.value}")
                    if self._can_retry_same_provider(exc, attempt):
                        if exc.retry_after:
                            time.sleep(min(exc.retry_after, 1.0))
                        attempt += 1
                        continue
                    break
            if last_error and not last_error.fallback_allowed:
                raise last_error
        if last_error:
            raise last_error
        raise ProviderException("no configured provider", retryable=False)

    def _configured_and_model(self, name: str) -> tuple[bool, str]:
        if name == "mock":
            return True, "deterministic-fixture-v1"
        key = getattr(self.settings, f"{name}_api_key", None)
        model = getattr(self.settings, f"{name}_model", "not configured")
        return bool(key), model

    def statuses(self) -> list[ProviderStatus]:
        statuses: list[ProviderStatus] = []
        for name in PROVIDER_ORDER:
            configured, model = self._configured_and_model(name)
            enabled = name in self.providers
            provider = self.providers.get(name)
            note = self._disabled_notes.get(name)
            if name == "cerebras" and configured and not enabled:
                note = "Inference disabled locally; enable only after a current free-tier probe."
            if name == "openrouter" and enabled:
                note = "Last-resort free fallback; not reproducible across underlying models."
            if name == "okmd" and configured and not enabled:
                note = "Configured but disabled; set OKMD_ENABLED=true for explicit use."
            if name == "thaillm" and configured and not enabled:
                note = "Configured but disabled; set THAILLM_ENABLED=true for explicit use."
            remaining, limit = self._quota.get(name, (None, None))
            statuses.append(
                ProviderStatus(
                    name=name,
                    configured=configured,
                    enabled=enabled,
                    model=getattr(provider, "model", model),
                    supports_structured_output=getattr(
                        provider, "supports_structured_output", False
                    ),
                    note=note,
                    last_status=self._last_status.get(name),
                    quota_remaining_tokens=remaining,
                    quota_limit_tokens=limit,
                )
            )
        return statuses
