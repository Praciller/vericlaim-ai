from dataclasses import dataclass

import pytest

from vericlaim.config import Settings
from vericlaim.domain.models import ProviderErrorCategory
from vericlaim.providers.base import (
    GeminiProvider,
    MockProvider,
    OKMDProvider,
    OpenAICompatibleProvider,
    ProviderException,
    ProviderRequest,
    ProviderResponse,
    ThaiLLMProvider,
    extract_final_content,
)
from vericlaim.providers.router import ProviderRouter


@dataclass
class FailingProvider:
    name: str = "groq"
    model: str = "test"
    supports_structured_output: bool = True
    retryable: bool = True
    category: ProviderErrorCategory = ProviderErrorCategory.RATE_LIMIT
    calls: int = 0

    def generate(self, request):
        self.calls += 1
        raise ProviderException(
            "provider failure", retryable=self.retryable, category=self.category
        )


@dataclass
class HealthyProvider:
    name: str = "mock"
    model: str = "test"
    supports_structured_output: bool = True

    def generate(self, request):
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            text="{}",
            input_tokens=2,
            output_tokens=1,
        )


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload


def test_gemini_invalid_api_key_400_is_authentication(monkeypatch):
    monkeypatch.setattr(
        "vericlaim.providers.base.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            {
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "message": "API key not valid. Please pass a valid API key.",
                }
            },
            status_code=400,
        ),
    )
    with pytest.raises(ProviderException) as raised:
        GeminiProvider("gemini", "gemini-flash-lite-latest", "unit-placeholder").generate(
            ProviderRequest("smoke", "Return exactly: OK", max_tokens=16)
        )
    assert raised.value.category == ProviderErrorCategory.AUTHENTICATION
    assert raised.value.error_code == "INVALID_ARGUMENT"
    assert raised.value.status_code == 400


def test_gemini_success_parses_fixed_actual_model_and_usage(monkeypatch):
    monkeypatch.setattr(
        "vericlaim.providers.base.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            {
                "modelVersion": "gemini-3.5-flash-lite",
                "candidates": [
                    {
                        "content": {"parts": [{"text": "OK"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 1,
                },
            }
        ),
    )
    response = GeminiProvider("gemini", "gemini-3.5-flash-lite", "unit-placeholder").generate(
        ProviderRequest("smoke", "Return exactly: OK", max_tokens=16)
    )
    assert response.text == "OK"
    assert response.configured_model == "gemini-3.5-flash-lite"
    assert response.actual_model == "gemini-3.5-flash-lite"
    assert response.input_tokens == 4
    assert response.output_tokens == 1
    assert response.finish_reason == "STOP"


def test_provider_enable_flag_is_required():
    disabled = ProviderRouter(
        Settings(mock_provider_enabled=False, groq_api_key="unit-placeholder", groq_enabled=False)
    )
    assert "groq" not in disabled.providers
    enabled = ProviderRouter(
        Settings(mock_provider_enabled=False, groq_api_key="unit-placeholder", groq_enabled=True)
    )
    assert "groq" in enabled.providers


def test_configured_but_disabled_provider_state_has_no_secret():
    router = ProviderRouter(
        Settings(
            mock_provider_enabled=False,
            cerebras_api_key="unit-placeholder",
            cerebras_enabled=False,
        )
    )
    status = next(item for item in router.statuses() if item.name == "cerebras")
    assert status.configured is True
    assert status.enabled is False
    assert "unit-placeholder" not in status.model_dump_json()


def test_provider_fallback_and_usage():
    settings = Settings(mock_provider_enabled=True)
    failing = FailingProvider()
    router = ProviderRouter(settings, providers={"groq": failing, "mock": HealthyProvider()})
    result = router.invoke("claim_analysis", "prompt")
    assert result.response.provider == "mock"
    assert result.usage.total_tokens == 3


def test_provider_unavailable_is_explicit():
    settings = Settings(mock_provider_enabled=False)
    router = ProviderRouter(settings, providers={})
    with pytest.raises(ProviderException, match="no configured provider"):
        router.invoke("claim_analysis", "prompt")


def test_payment_required_falls_back_without_retrying():
    failing = FailingProvider(retryable=False, category=ProviderErrorCategory.PAYMENT_REQUIRED)
    router = ProviderRouter(
        Settings(mock_provider_enabled=True), providers={"groq": failing, "mock": HealthyProvider()}
    )
    result = router.invoke("claim_analysis", "prompt")
    assert result.response.provider == "mock"
    assert failing.calls == 1


def test_authentication_failure_falls_back_without_retrying():
    failing = FailingProvider(retryable=False, category=ProviderErrorCategory.AUTHENTICATION)
    router = ProviderRouter(
        Settings(mock_provider_enabled=True), providers={"groq": failing, "mock": HealthyProvider()}
    )
    assert router.invoke("claim_analysis", "prompt").response.provider == "mock"
    assert failing.calls == 1


def test_retry_is_bounded_to_one_same_provider_attempt():
    failing = FailingProvider()
    router = ProviderRouter(Settings(mock_provider_enabled=False), providers={"groq": failing})
    with pytest.raises(ProviderException):
        router.invoke("claim_analysis", "prompt")
    assert failing.calls == 2


def test_rate_limit_retry_after_is_parsed():
    response = FakeResponse({}, status_code=429, headers={"Retry-After": "3"})
    provider = OpenAICompatibleProvider("groq", "model", "unit-placeholder", "https://example.test")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("vericlaim.providers.base.httpx.post", lambda *args, **kwargs: response)
        with pytest.raises(ProviderException) as raised:
            provider.generate(ProviderRequest("task", "prompt"))
    assert raised.value.category == ProviderErrorCategory.RATE_LIMIT
    assert raised.value.retry_after == 3.0


def test_okmd_adapter_parses_optional_quota(monkeypatch):
    monkeypatch.setattr(
        "vericlaim.providers.base.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            {
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                "model_quota": {
                    "daily_quota_tokens": 180000,
                    "daily_usage_tokens": 11,
                    "daily_remaining_tokens": 179989,
                },
            }
        ),
    )
    response = OKMDProvider(
        "okmd", "deepseek-v4-flash", "unit-placeholder", "https://example.test/api/v1"
    ).generate(ProviderRequest("smoke", "Reply OK"))
    assert response.text == "OK"
    assert response.quota_remaining_tokens == 179989


def test_okmd_embedded_daily_limit_is_quota_exhausted(monkeypatch):
    monkeypatch.setattr(
        "vericlaim.providers.base.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            {
                "status": 403,
                "code": 403,
                "error": {"code": 403, "message": "Budget limit exceeded (daily limit)."},
            }
        ),
    )
    with pytest.raises(ProviderException) as raised:
        OKMDProvider(
            "okmd", "deepseek-v4-flash", "unit-placeholder", "https://example.test/api/v1"
        ).generate(ProviderRequest("smoke", "Reply OK"))
    assert raised.value.category == ProviderErrorCategory.QUOTA_EXHAUSTED
    assert raised.value.error_code == "403"


def test_thaillm_requires_https():
    with pytest.raises(ValueError, match="HTTPS"):
        ThaiLLMProvider("thaillm", "model", "unit-placeholder", "http://thaillm.or.th/api/v1")


def test_think_extraction_is_conservative():
    assert extract_final_content("<think>reasoning</think>FINAL") == "FINAL"
    assert extract_final_content("A sentence that mentions think without tags") == (
        "A sentence that mentions think without tags"
    )


def test_unclosed_think_block_is_incomplete():
    with pytest.raises(ProviderException) as raised:
        extract_final_content("<think>reasoning", "length")
    assert raised.value.category == ProviderErrorCategory.INCOMPLETE_RESPONSE


def test_empty_http_200_is_not_success(monkeypatch):
    monkeypatch.setattr(
        "vericlaim.providers.base.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        ),
    )
    with pytest.raises(ProviderException) as raised:
        OpenAICompatibleProvider(
            "groq", "model", "unit-placeholder", "https://example.test"
        ).generate(ProviderRequest("smoke", "Reply"))
    assert raised.value.category == ProviderErrorCategory.MALFORMED_RESPONSE


def test_reasoning_only_openai_response_is_incomplete(monkeypatch):
    monkeypatch.setattr(
        "vericlaim.providers.base.httpx.post",
        lambda *args, **kwargs: FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": "", "reasoning_content": "thinking"},
                        "finish_reason": "length",
                    }
                ]
            }
        ),
    )
    with pytest.raises(ProviderException) as raised:
        OpenAICompatibleProvider(
            "groq", "model", "unit-placeholder", "https://example.test"
        ).generate(ProviderRequest("smoke", "Reply"))
    assert raised.value.category == ProviderErrorCategory.INCOMPLETE_RESPONSE


def test_thaillm_retries_one_incomplete_response(monkeypatch):
    responses = iter(
        [
            FakeResponse(
                {
                    "choices": [
                        {"message": {"content": "<think>unfinished"}, "finish_reason": "length"}
                    ]
                }
            ),
            FakeResponse(
                {
                    "model": "thai-actual",
                    "choices": [
                        {
                            "message": {"content": "<think>done</think>ผลลัพธ์"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "vericlaim.providers.base.httpx.post", lambda *args, **kwargs: next(responses)
    )
    response = ThaiLLMProvider(
        "thaillm", "thai-configured", "unit-placeholder", "https://thaillm.or.th/api/v1"
    ).generate(ProviderRequest("thai_semantic_review", "ตรวจสอบ", max_tokens=64))
    assert response.text == "ผลลัพธ์"
    assert response.actual_model == "thai-actual"


def test_openrouter_records_actual_model():
    provider = OpenAICompatibleProvider(
        "openrouter", "openrouter/free", "unit-placeholder", "https://example.test"
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "vericlaim.providers.base.httpx.post",
            lambda *args, **kwargs: FakeResponse(
                {
                    "model": "provider/free-model",
                    "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                }
            ),
        )
        response = provider.generate(ProviderRequest("smoke", "Reply OK"))
    assert response.configured_model == "openrouter/free"
    assert response.actual_model == "provider/free-model"


def test_openrouter_is_excluded_from_reproducible_route():
    router = ProviderRouter(
        Settings(mock_provider_enabled=True),
        providers={"openrouter": HealthyProvider(name="openrouter"), "mock": HealthyProvider()},
    )
    result = router.invoke("claim_analysis", "prompt", reproducible=True)
    assert result.response.provider == "mock"


def test_critique_prefers_openrouter_after_okmd_failure():
    failing = FailingProvider(
        name="okmd", retryable=False, category=ProviderErrorCategory.PERMISSION
    )
    router = ProviderRouter(
        Settings(mock_provider_enabled=False),
        providers={"okmd": failing, "openrouter": HealthyProvider(name="openrouter")},
    )
    result = router.invoke("critique", "prompt")
    assert result.response.provider == "openrouter"
    assert failing.calls == 1


def test_mock_provider_is_deterministic():
    provider = MockProvider(responses={"critique": '{"ok":true}'})
    response = provider.generate(type("Request", (), {"task": "critique", "prompt": "x"})())
    assert response.text == '{"ok":true}'
