from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QuotaStatus(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    EXHAUSTED = "EXHAUSTED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class ProviderBudgetState:
    provider: str
    configured: bool
    enabled: bool
    model: str
    quota_status: QuotaStatus
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_at: str | None = None
    source: str = "not_observed"


@dataclass(frozen=True)
class BudgetPlan:
    global_max_calls: int
    global_max_tokens: int
    estimated_total_calls: int
    estimated_total_tokens: int
    historical_average_tokens_per_call: float
    safety_factor: float
    decision: str
    reason: str
    provider_estimates: dict[str, dict[str, int]] = field(default_factory=dict)
    estimated_cache_hits: int = 0
    estimated_cache_misses: int = 0

    @property
    def headroom_calls(self) -> int:
        return self.global_max_calls - self.estimated_total_calls

    @property
    def headroom_tokens(self) -> int:
        return self.global_max_tokens - self.estimated_total_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_max_calls": self.global_max_calls,
            "global_max_tokens": self.global_max_tokens,
            "estimated_total_calls": self.estimated_total_calls,
            "estimated_total_tokens": self.estimated_total_tokens,
            "historical_average_tokens_per_call": self.historical_average_tokens_per_call,
            "safety_factor": self.safety_factor,
            "headroom_calls": self.headroom_calls,
            "headroom_tokens": self.headroom_tokens,
            "decision": self.decision,
            "reason": self.reason,
            "provider_estimates": self.provider_estimates,
            "estimated_cache_hits": self.estimated_cache_hits,
            "estimated_cache_misses": self.estimated_cache_misses,
        }


class BudgetGateDenied(RuntimeError):
    """Raised when another live provider call would exceed the hard budget."""


class LiveBudgetGate:
    """Hard call/token budget for bounded free-tier evaluation runs."""

    def __init__(
        self,
        *,
        provider_states: tuple[ProviderBudgetState, ...],
        global_max_calls: int = 100,
        global_max_tokens: int | None = None,
        historical_average_tokens_per_call: float = 91_723 / 58,
        safety_factor: float = 0.98,
        max_provider_failure_rate: float = 0.25,
    ) -> None:
        if global_max_calls <= 0:
            raise ValueError("global_max_calls must be positive")
        if not 0 < safety_factor < 1:
            raise ValueError("safety_factor must be between zero and one")
        self.provider_states = provider_states
        self.global_max_calls = global_max_calls
        self.historical_average_tokens_per_call = historical_average_tokens_per_call
        self.safety_factor = safety_factor
        self.global_max_tokens = global_max_tokens or int(
            historical_average_tokens_per_call * global_max_calls * safety_factor
        )
        self.max_provider_failure_rate = max_provider_failure_rate
        self.actual_calls = 0
        self.actual_tokens = 0
        self.provider_calls: dict[str, int] = {}
        self.provider_tokens: dict[str, int] = {}
        self.provider_failures = 0
        self.denial_reason: str | None = None

    def plan(
        self,
        *,
        architecture_call_upper_bounds: dict[str, int],
        sample_size: int,
        provider_by_architecture: dict[str, str | dict[str, int]],
        cached_calls_by_provider: dict[str, int] | None = None,
        provider_call_upper_bounds: dict[str, int] | None = None,
    ) -> BudgetPlan:
        cached_calls_by_provider = cached_calls_by_provider or {}
        provider_estimates: dict[str, dict[str, int]] = {}
        for architecture, calls_per_claim in architecture_call_upper_bounds.items():
            assignment = provider_by_architecture.get(architecture, "unknown")
            assignments = (
                {assignment: calls_per_claim} if isinstance(assignment, str) else assignment
            )
            for provider, calls_per_claim_for_provider in assignments.items():
                estimate = provider_estimates.setdefault(
                    provider,
                    {
                        "calls": 0,
                        "tokens": 0,
                        "logical_calls": 0,
                        "cache_hits": 0,
                        "cache_misses": 0,
                    },
                )
                logical_calls = calls_per_claim_for_provider * sample_size
                available_hits = max(0, cached_calls_by_provider.get(provider, 0))
                cache_hits = min(logical_calls, available_hits)
                cached_calls_by_provider[provider] = available_hits - cache_hits
                cache_misses = logical_calls - cache_hits
                estimate["logical_calls"] += logical_calls
                estimate["cache_hits"] += cache_hits
                estimate["cache_misses"] += cache_misses
                estimate["calls"] += cache_misses
                estimate["tokens"] += int(cache_misses * self.historical_average_tokens_per_call)
        if provider_call_upper_bounds:
            for provider, call_upper_bound in provider_call_upper_bounds.items():
                estimate = provider_estimates.get(provider)
                if estimate is None:
                    continue
                optimized_calls = min(estimate["calls"], max(0, call_upper_bound))
                estimate["calls"] = optimized_calls
                estimate["cache_misses"] = optimized_calls
                estimate["cache_hits"] = estimate["logical_calls"] - optimized_calls
                estimate["tokens"] = int(optimized_calls * self.historical_average_tokens_per_call)
        estimated_calls = sum(value["calls"] for value in provider_estimates.values())
        estimated_tokens = sum(value["tokens"] for value in provider_estimates.values())
        estimated_cache_hits = sum(value["cache_hits"] for value in provider_estimates.values())
        estimated_cache_misses = sum(value["cache_misses"] for value in provider_estimates.values())
        unavailable = {
            state.provider
            for state in self.provider_states
            if not state.configured
            or not state.enabled
            or state.quota_status == QuotaStatus.UNAVAILABLE
        }
        if unavailable:
            decision = "DENY"
            reason = f"provider unavailable or disabled: {','.join(sorted(unavailable))}"
        elif any(state.quota_status == QuotaStatus.EXHAUSTED for state in self.provider_states):
            decision = "DENY"
            reason = "provider quota is exhausted"
        elif any(
            state.quota_status == QuotaStatus.KNOWN
            and (
                (
                    state.remaining_requests is not None
                    and provider_estimates.get(state.provider, {}).get("calls", 0)
                    > state.remaining_requests // 2
                )
                or (
                    state.remaining_tokens is not None
                    and provider_estimates.get(state.provider, {}).get("tokens", 0)
                    > state.remaining_tokens // 2
                )
            )
            for state in self.provider_states
        ):
            decision = "DENY"
            reason = "estimated use exceeds 50% of known provider quota"
        elif estimated_calls > self.global_max_calls:
            decision = "DENY"
            reason = "estimated provider calls exceed global hard ceiling"
        elif estimated_tokens > self.global_max_tokens:
            decision = "DENY"
            reason = "estimated tokens exceed global hard ceiling"
        else:
            decision = "ALLOW"
            reason = "within conservative free-tier hard ceilings"
        return BudgetPlan(
            global_max_calls=self.global_max_calls,
            global_max_tokens=self.global_max_tokens,
            estimated_total_calls=estimated_calls,
            estimated_total_tokens=estimated_tokens,
            historical_average_tokens_per_call=self.historical_average_tokens_per_call,
            safety_factor=self.safety_factor,
            decision=decision,
            reason=reason,
            provider_estimates=provider_estimates,
            estimated_cache_hits=estimated_cache_hits,
            estimated_cache_misses=estimated_cache_misses,
        )

    def allow_call(
        self, *, provider: str | None = None, estimated_tokens: int | None = None
    ) -> bool:
        next_calls = self.actual_calls + 1
        next_tokens = self.actual_tokens + max(0, estimated_tokens or 0)
        if next_calls > self.global_max_calls:
            self.denial_reason = "global provider-call ceiling reached"
            return False
        if next_tokens > self.global_max_tokens:
            self.denial_reason = "global token ceiling reached"
            return False
        if provider is not None:
            state = next((item for item in self.provider_states if item.provider == provider), None)
            if state is not None and state.quota_status == QuotaStatus.KNOWN:
                safe_requests = (
                    state.remaining_requests // 2 if state.remaining_requests is not None else None
                )
                safe_tokens = (
                    state.remaining_tokens // 2 if state.remaining_tokens is not None else None
                )
                if (
                    safe_requests is not None
                    and self.provider_calls.get(provider, 0) + 1 > safe_requests
                ):
                    self.denial_reason = f"known {provider} request headroom exhausted"
                    return False
                if (
                    safe_tokens is not None
                    and self.provider_tokens.get(provider, 0) + max(0, estimated_tokens or 0)
                    > safe_tokens
                ):
                    self.denial_reason = f"known {provider} token headroom exhausted"
                    return False
        return True

    def record(self, *, provider: str, tokens: int, failed: bool = False) -> None:
        self.actual_calls += 1
        self.actual_tokens += max(0, tokens)
        self.provider_calls[provider] = self.provider_calls.get(provider, 0) + 1
        self.provider_tokens[provider] = self.provider_tokens.get(provider, 0) + max(0, tokens)
        if failed:
            self.provider_failures += 1
        if self.actual_tokens > self.global_max_tokens:
            self.denial_reason = "global token ceiling reached"
        if (
            self.actual_calls >= 4
            and self.provider_failures / self.actual_calls > self.max_provider_failure_rate
        ):
            self.denial_reason = "provider failure rate exceeded safe threshold"

    def enforce(self, *, provider: str | None = None, estimated_tokens: int | None = None) -> None:
        if self.denial_reason is not None or not self.allow_call(
            provider=provider, estimated_tokens=estimated_tokens
        ):
            raise BudgetGateDenied(self.denial_reason or "budget gate denied")

    def snapshot(self) -> dict[str, Any]:
        return {
            "actual_provider_calls": self.actual_calls,
            "actual_live_tokens": self.actual_tokens,
            "provider_calls": dict(sorted(self.provider_calls.items())),
            "provider_tokens": dict(sorted(self.provider_tokens.items())),
            "provider_failures": self.provider_failures,
            "denial_reason": self.denial_reason,
            "global_max_calls": self.global_max_calls,
            "global_max_tokens": self.global_max_tokens,
        }


def provider_budget_table(
    states: tuple[ProviderBudgetState, ...], plan: BudgetPlan
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in states:
        estimate = plan.provider_estimates.get(state.provider, {"calls": 0, "tokens": 0})
        allowed = (
            state.configured
            and state.enabled
            and state.quota_status
            not in {
                QuotaStatus.EXHAUSTED,
                QuotaStatus.UNAVAILABLE,
            }
        )
        rows.append(
            {
                "provider": state.provider,
                "model": state.model,
                "quota_status": state.quota_status.value,
                "known_remaining_tokens": state.remaining_tokens,
                "known_remaining_requests": state.remaining_requests,
                "estimated_calls": estimate["calls"],
                "estimated_tokens": estimate["tokens"],
                "allowed": allowed and plan.decision == "ALLOW",
                "reason": plan.reason if allowed else "provider is not configured/enabled",
                "source": state.source,
            }
        )
    return rows
