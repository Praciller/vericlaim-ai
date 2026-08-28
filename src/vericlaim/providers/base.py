from __future__ import annotations

import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import httpx

from ..domain.models import ProviderErrorCategory


def _category_for_status(status_code: int | None) -> ProviderErrorCategory:
    if status_code == 401:
        return ProviderErrorCategory.AUTHENTICATION
    if status_code == 403:
        return ProviderErrorCategory.PERMISSION
    if status_code == 402:
        return ProviderErrorCategory.PAYMENT_REQUIRED
    if status_code == 429:
        return ProviderErrorCategory.RATE_LIMIT
    if status_code is not None and status_code >= 500:
        return ProviderErrorCategory.SERVER_ERROR
    return ProviderErrorCategory.UNKNOWN


def _category_for_error(
    status_code: int | None, error_code: object = None, message: object = None
) -> ProviderErrorCategory:
    code = str(error_code or "").casefold()
    detail = str(message or "").casefold()
    quota_markers = (
        "quota",
        "daily limit",
        "budget limit",
        "credits exhausted",
        "credit exhausted",
        "insufficient quota",
    )
    if any(marker in code or marker in detail for marker in quota_markers):
        return ProviderErrorCategory.QUOTA_EXHAUSTED
    authentication_markers = (
        "api key",
        "authentication",
        "unauthorized",
        "invalid credential",
    )
    if any(marker in code or marker in detail for marker in authentication_markers):
        return ProviderErrorCategory.AUTHENTICATION
    return _category_for_status(status_code)


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            timestamp = parsedate_to_datetime(value).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0.0, timestamp - time.time())


class ProviderException(Exception):
    """Sanitized, provider-neutral failure used by routing decisions."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool | None = None,
        status_code: int | None = None,
        category: ProviderErrorCategory | None = None,
        error_code: str | None = None,
        retry_after: float | None = None,
        fallback_allowed: bool = True,
    ) -> None:
        self.category = category or _category_for_status(status_code)
        self.status_code = status_code
        self.error_code = error_code
        self.retry_after = retry_after
        self.fallback_allowed = fallback_allowed
        self.retryable = (
            retryable
            if retryable is not None
            else self.category
            in {
                ProviderErrorCategory.NETWORK,
                ProviderErrorCategory.TIMEOUT,
                ProviderErrorCategory.RATE_LIMIT,
                ProviderErrorCategory.SERVER_ERROR,
            }
        )
        super().__init__(message)


def extract_final_content(text: str, finish_reason: str | None = None) -> str:
    """Remove only a complete leading <think> block; preserve ordinary text."""
    candidate = text.strip()
    if not candidate:
        raise ProviderException(
            "provider response was incomplete"
            if finish_reason == "length"
            else "provider returned empty content",
            category=(
                ProviderErrorCategory.INCOMPLETE_RESPONSE
                if finish_reason == "length"
                else ProviderErrorCategory.MALFORMED_RESPONSE
            ),
        )
    if candidate.lower().startswith("<think>"):
        closing = candidate.lower().find("</think>")
        if closing < 0:
            raise ProviderException(
                "provider response ended inside reasoning",
                category=ProviderErrorCategory.INCOMPLETE_RESPONSE,
            )
        final = candidate[closing + len("</think>") :].strip()
        if not final:
            raise ProviderException(
                "provider returned reasoning without final content",
                category=ProviderErrorCategory.INCOMPLETE_RESPONSE,
            )
        return final
    return candidate


@dataclass(frozen=True)
class ProviderRequest:
    task: str
    prompt: str
    max_tokens: int = 512
    timeout_seconds: float | None = None


@dataclass
class ProviderResponse:
    provider: str
    model: str
    text: str
    configured_model: str | None = None
    actual_model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str | None = None
    status: str = "success"
    error_code: str | None = None
    quota_limit_tokens: int | None = None
    quota_used_tokens: int | None = None
    quota_remaining_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.configured_model is None:
            self.configured_model = self.model
        if self.actual_model is None:
            self.actual_model = self.model

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(Protocol):
    name: str
    model: str
    supports_structured_output: bool

    def generate(self, request: ProviderRequest) -> ProviderResponse: ...


@dataclass
class MockProvider:
    name: str = "mock"
    model: str = "deterministic-fixture-v1"
    supports_structured_output: bool = True
    responses: dict[str, str] = field(default_factory=dict)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        text = self.responses.get(request.task, "{}")
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            configured_model=self.model,
            actual_model=self.model,
            text=text,
            input_tokens=len(request.prompt.split()),
            output_tokens=len(text.split()),
            latency_ms=0,
            finish_reason="stop",
        )


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _quota_values(data: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    quota = data.get("model_quota")
    if not isinstance(quota, dict):
        return None, None, None
    return (
        _int_or_none(quota.get("daily_quota_tokens")),
        _int_or_none(quota.get("daily_usage_tokens")),
        _int_or_none(quota.get("daily_remaining_tokens")),
    )


@dataclass
class OpenAICompatibleProvider:
    name: str
    model: str
    api_key: str
    base_url: str
    supports_structured_output: bool = True
    timeout_seconds: float = 20.0

    def _generate_once(
        self, request: ProviderRequest, max_tokens: int | None = None
    ) -> ProviderResponse:
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": request.prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens or request.max_tokens,
                },
                timeout=(
                    min(self.timeout_seconds, request.timeout_seconds)
                    if request.timeout_seconds is not None
                    else self.timeout_seconds
                ),
            )
        except httpx.TimeoutException as exc:
            raise ProviderException(
                "provider request timed out", category=ProviderErrorCategory.TIMEOUT
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderException(
                "provider network request failed", category=ProviderErrorCategory.NETWORK
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        retry_after = retry_after_seconds(response.headers.get("Retry-After"))
        try:
            data = response.json()
        except (ValueError, TypeError):
            data = None
        error_payload = data.get("error") if isinstance(data, dict) else None
        error_code_value = (
            data.get("code") or data.get("status") or error_payload.get("code")
            if isinstance(data, dict) and isinstance(error_payload, dict)
            else data.get("code") or data.get("status")
            if isinstance(data, dict)
            else None
        )
        error_message = error_payload.get("message") if isinstance(error_payload, dict) else None
        if response.status_code >= 400:
            raise ProviderException(
                "provider request was rejected",
                status_code=response.status_code,
                category=_category_for_error(response.status_code, error_code_value, error_message),
                error_code=str(error_code_value) if error_code_value is not None else None,
                retry_after=retry_after,
            )
        try:
            if isinstance(data, dict) and isinstance(data.get("error"), dict):
                embedded_code = data.get("code") or data.get("status") or data["error"].get("code")
                embedded_message = data["error"].get("message")
                try:
                    embedded_status = int(embedded_code)
                except (TypeError, ValueError):
                    embedded_status = None
                raise ProviderException(
                    "provider returned an error envelope",
                    status_code=embedded_status,
                    category=_category_for_error(embedded_status, embedded_code, embedded_message),
                    error_code=str(embedded_code) if embedded_code is not None else None,
                )
            choice = data["choices"][0]
            message = choice["message"]
            content = _as_text(message.get("content"))
            if not content and message.get("reasoning_content"):
                content = f"<think>{message['reasoning_content']}"
            finish_reason = choice.get("finish_reason")
            text = extract_final_content(content, finish_reason)
        except ProviderException:
            raise
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderException(
                "provider response was malformed",
                category=ProviderErrorCategory.MALFORMED_RESPONSE,
            ) from exc
        usage = data.get("usage") or {}
        quota_limit, quota_used, quota_remaining = _quota_values(data)
        actual_model = data.get("model") if isinstance(data.get("model"), str) else self.model
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            configured_model=self.model,
            actual_model=actual_model,
            text=text,
            input_tokens=_int_or_none(usage.get("prompt_tokens")) or 0,
            output_tokens=_int_or_none(usage.get("completion_tokens")) or 0,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            quota_limit_tokens=quota_limit,
            quota_used_tokens=quota_used,
            quota_remaining_tokens=quota_remaining,
        )

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return self._generate_once(request)


@dataclass
class GeminiProvider:
    name: str
    model: str
    api_key: str
    supports_structured_output: bool = True
    timeout_seconds: float = 20.0

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        started = time.perf_counter()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        try:
            response = httpx.post(
                url,
                params={"key": self.api_key},
                json={
                    "contents": [{"parts": [{"text": request.prompt}]}],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": request.max_tokens,
                    },
                },
                timeout=(
                    min(self.timeout_seconds, request.timeout_seconds)
                    if request.timeout_seconds is not None
                    else self.timeout_seconds
                ),
            )
        except httpx.TimeoutException as exc:
            raise ProviderException(
                "provider request timed out", category=ProviderErrorCategory.TIMEOUT
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderException(
                "provider network request failed", category=ProviderErrorCategory.NETWORK
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        retry_after = retry_after_seconds(response.headers.get("Retry-After"))
        if response.status_code >= 400:
            try:
                data = response.json()
            except (ValueError, TypeError):
                data = None
            error_payload = data.get("error") if isinstance(data, dict) else None
            error_code = (
                error_payload.get("status") or error_payload.get("code")
                if isinstance(error_payload, dict)
                else None
            )
            error_message = (
                error_payload.get("message") if isinstance(error_payload, dict) else None
            )
            raise ProviderException(
                "provider request was rejected",
                status_code=response.status_code,
                category=_category_for_error(response.status_code, error_code, error_message),
                error_code=str(error_code) if error_code is not None else None,
                retry_after=retry_after,
            )
        try:
            data = response.json()
            candidate = data["candidates"][0]
            content = _as_text(candidate["content"]["parts"])
            finish_reason = candidate.get("finishReason")
            text = extract_final_content(content, finish_reason)
        except ProviderException:
            raise
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderException(
                "provider response was malformed",
                category=ProviderErrorCategory.MALFORMED_RESPONSE,
            ) from exc
        usage = data.get("usageMetadata") or {}
        actual_model = data.get("modelVersion")
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            configured_model=self.model,
            actual_model=actual_model if isinstance(actual_model, str) else self.model,
            text=text,
            input_tokens=_int_or_none(usage.get("promptTokenCount")) or 0,
            output_tokens=_int_or_none(usage.get("candidatesTokenCount")) or 0,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )


@dataclass
class ThaiLLMProvider(OpenAICompatibleProvider):
    def __post_init__(self) -> None:
        if not self.base_url.lower().startswith("https://"):
            raise ValueError("ThaiLLM must use HTTPS")

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        try:
            return self._generate_once(request, max_tokens=request.max_tokens)
        except ProviderException as exc:
            if exc.category != ProviderErrorCategory.INCOMPLETE_RESPONSE:
                raise
            # One bounded retry for models that spend the first budget inside <think>.
            return self._generate_once(request, max_tokens=min(request.max_tokens * 2, 1024))


class OKMDProvider(OpenAICompatibleProvider):
    """OKMD's documented OpenAI-compatible chat endpoint."""
