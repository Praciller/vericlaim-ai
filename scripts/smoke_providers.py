from __future__ import annotations

import argparse
from typing import Any

import httpx

from vericlaim.config import Settings, secret_value
from vericlaim.domain.models import ProviderErrorCategory
from vericlaim.providers.base import (
    OpenAICompatibleProvider,
    ProviderException,
    ProviderRequest,
)
from vericlaim.providers.router import ProviderRouter

ROLES = {
    "gemini": "evidence audit / final judgment",
    "groq": "claim analysis / decomposition / batched classification",
    "cerebras": "disabled until free inference is available",
    "openrouter": "last-resort free runtime fallback",
    "okmd": "independent critic / reasoning fallback",
    "thaillm": "Thai semantic review",
}
PROVIDERS = ("gemini", "groq", "cerebras", "openrouter", "okmd", "thaillm")


def model_check(name: str, provider: Any) -> str:
    try:
        if name == "gemini":
            response = httpx.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": provider.api_key},
                timeout=10,
            )
            payload: Any = response.json()
            names = [item.get("name", "") for item in payload.get("models", [])]
            return "PASS" if any(name.endswith(provider.model) for name in names) else "UNKNOWN"
        base_url = provider.base_url
        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {provider.api_key}"},
            timeout=10,
        )
        if response.status_code >= 400:
            return f"FAIL_HTTP_{response.status_code}"
        payload = response.json()
        models = payload.get("data", []) if isinstance(payload, dict) else []
        model_ids = [item.get("id", "") for item in models if isinstance(item, dict)]
        if provider.model == "openrouter/free":
            return "PASS" if response.status_code == 200 else "UNKNOWN"
        return "PASS" if provider.model in model_ids else "UNKNOWN"
    except (AttributeError, httpx.HTTPError, ValueError, TypeError):
        return "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal, opt-in provider smoke tests")
    parser.add_argument("--provider", choices=[*PROVIDERS, "all"], default="all")
    args = parser.parse_args()

    settings = Settings()
    router = ProviderRouter(settings)
    names = PROVIDERS if args.provider == "all" else (args.provider,)
    failures = 0
    for name in names:
        status = next(item for item in router.statuses() if item.name == name)
        provider = router.providers.get(name)
        if not status.enabled or provider is None:
            disabled_model_status = "not_checked"
            cerebras_api_key = secret_value(settings.cerebras_api_key)
            if name == "cerebras" and cerebras_api_key:
                disabled_probe = OpenAICompatibleProvider(
                    "cerebras",
                    settings.cerebras_model,
                    cerebras_api_key,
                    "https://api.cerebras.ai/v1",
                )
                disabled_model_status = model_check(name, disabled_probe)
            print(
                f"{name} configured={status.configured} enabled={status.enabled} "
                f"model={status.model} actual_model=none live_status=DISABLED "
                f"model_check={disabled_model_status} "
                f"role={ROLES[name]} note={status.note or 'none'}"
            )
            continue
        model_status = model_check(name, provider)
        try:
            response = provider.generate(
                ProviderRequest(
                    task="smoke",
                    prompt="Reply with OK only.",
                    max_tokens=128,
                )
            )
            print(
                f"{name} configured=true enabled=true model={response.configured_model} "
                f"actual_model={response.actual_model} live_status=PASS "
                f"model_check={model_status} "
                f"latency_ms={response.latency_ms} input_tokens={response.input_tokens} "
                f"output_tokens={response.output_tokens} total_tokens={response.total_tokens} "
                f"quota_remaining={response.quota_remaining_tokens} "
                f"quota_limit={response.quota_limit_tokens} finish_reason={response.finish_reason} "
                f"role={ROLES[name]}"
            )
        except ProviderException as exc:
            status_name = (
                "INFERENCE_UNAVAILABLE"
                if exc.category
                in {ProviderErrorCategory.PAYMENT_REQUIRED, ProviderErrorCategory.QUOTA_EXHAUSTED}
                else "FAIL"
            )
            failures += 1
            print(
                f"{name} configured=true enabled=true model={status.model} actual_model=none "
                f"live_status={status_name} model_check={model_status} "
                f"error_category={exc.category.value} "
                f"retry_after={exc.retry_after} role={ROLES[name]}"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
