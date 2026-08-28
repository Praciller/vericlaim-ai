from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.scifact.budget import (  # noqa: E402
    LiveBudgetGate,
    ProviderBudgetState,
    QuotaStatus,
    provider_budget_table,
)
from evals.scifact.cache import (  # noqa: E402
    CACHE_VERSION,
    StructuredResponseCache,
    input_hash,
    make_cache_key,
)
from evals.scifact.dataset import SciFactDataError  # noqa: E402
from evals.scifact.retrieval import BM25Retriever, RetrievalConfig  # noqa: E402
from evals.scifact.runner import (  # noqa: E402
    ARCHITECTURE_CALL_UPPER_BOUNDS,
    ARCHITECTURE_STAGE_UPPER_BOUNDS,
    FIXED_MODELS,
    GENERATION_PARAMETERS,
    ISOLATION_ARCHITECTURES,
    TASK_GENERATION_PARAMETERS,
    EvaluationError,
    OfflineBenchmarkProvider,
    _extract_classifier,
    _stage_specs,
    build_live_providers,
    load_and_validate_dataset,
    run_benchmark,
    select_sample,
)

from vericlaim.config import Settings, secret_value  # noqa: E402
from vericlaim.providers.base import ProviderException, ProviderRequest  # noqa: E402

DEFAULT_SAMPLE_MANIFEST = "evals/results/scifact-smoke-20260825T165453Z-e1fd1ba6/manifest.json"


def _historical_average_tokens_per_call(output_root: str) -> tuple[float, str]:
    total_tokens = 0
    total_calls = 0
    for path in sorted(Path(output_root).glob("scifact-*/provider_usage.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("dry_run") or int(value.get("provider_failures", 0)) > 0:
            continue
        for record in value.get("records", []):
            if (
                record.get("provider") != "offline"
                and record.get("status") == "success"
                and int(record.get("calls", 0)) > 0
            ):
                total_calls += int(record["calls"])
                total_tokens += int(record.get("input_tokens", 0)) + int(
                    record.get("output_tokens", 0)
                )
    if total_calls:
        return total_tokens / total_calls, f"successful live records: {total_tokens}/{total_calls}"
    return 91723 / 58, "prior validated live smoke baseline"


def _rekey_verified_legacy_cache(
    corpus,
    manifest: dict[str, object],
    sample_ids: tuple[str, ...],
    provider: str,
    cache_root: str,
    output_root: str,
) -> int:
    """Re-key only exact old live responses whose provenance matches this replay."""
    current_retrieval = asdict(RetrievalConfig())
    legacy_run_verified = False
    for path in Path(output_root).glob("scifact-live-5-*/manifest.json"):
        try:
            legacy = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            legacy.get("sample_ids") == list(sample_ids)
            and legacy.get("dataset_revision") == manifest.get("revision")
            and legacy.get("retrieval_config") == current_retrieval
        ):
            legacy_run_verified = True
            break
    if not legacy_run_verified:
        return 0

    cache = StructuredResponseCache(cache_root)
    legacy_entries: dict[str, dict[str, object]] = {}
    for path in Path(cache_root).glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            value.get("cache_version") == CACHE_VERSION
            and value.get("provider") == provider
            and value.get("configured_model") == FIXED_MODELS[provider]
            and value.get("actual_model") == FIXED_MODELS[provider]
            and value.get("prompt_version") in {"evidence_classifier_v2", "judge_v2"}
            and isinstance(value.get("parsed"), dict)
            and isinstance(value.get("input_hash"), str)
        ):
            legacy_entries[str(value["input_hash"])] = value

    cache_context = {
        "dataset_revision": manifest.get("revision"),
        "retrieval_config": current_retrieval,
    }
    rekeyed = 0
    retriever = BM25Retriever(corpus, RetrievalConfig())
    for claim in (item for item in corpus.claims if item.claim_id in set(sample_ids)):
        retrieved = retriever.retrieve(claim.text)
        specs = _stage_specs(
            claim,
            "C_SUPPORT_COUNTER",
            retrieved,
            stage_providers={
                "classifier": provider,
                "judge": provider,
            },
        )
        classifier_spec = next(spec for spec in specs if spec.task == "evidence_classifier")
        classifier_entry = legacy_entries.get(input_hash(classifier_spec.input_payload))
        if classifier_entry is None:
            continue
        for spec, entry in ((classifier_spec, classifier_entry),):
            value_hash = input_hash(
                {
                    "cache_context": cache_context,
                    "stage_input": spec.input_payload,
                    "prompt": spec.prompt,
                }
            )
            key, identity = make_cache_key(
                architecture=spec.cache_scope or "C_SUPPORT_COUNTER",
                provider=provider,
                configured_model=FIXED_MODELS[provider],
                prompt_version=spec.prompt_version,
                generation_parameters=TASK_GENERATION_PARAMETERS.get(
                    spec.task, GENERATION_PARAMETERS
                ),
                input_hash_value=value_hash,
            )
            if cache.get(key) is None:
                cache.put(
                    key=key,
                    identity=identity,
                    parsed=entry["parsed"],
                    provider=provider,
                    configured_model=FIXED_MODELS[provider],
                    actual_model=FIXED_MODELS[provider],
                    input_tokens=int(entry.get("input_tokens", 0)),
                    output_tokens=int(entry.get("output_tokens", 0)),
                    latency_ms=int(entry.get("latency_ms", 0)),
                )
                rekeyed += 1
        allowed_ids = {
            sentence.evidence_id for document in retrieved for sentence in document.sentences
        }
        classifier_items = _extract_classifier(classifier_entry["parsed"], allowed_ids)
        judge_spec = next(
            spec
            for spec in _stage_specs(
                claim,
                "C_SUPPORT_COUNTER",
                retrieved,
                classifier_items=classifier_items,
                stage_providers={"classifier": provider, "judge": provider},
            )
            if spec.task == "judge"
        )
        judge_entry = legacy_entries.get(input_hash(judge_spec.input_payload))
        if judge_entry is None:
            continue
        value_hash = input_hash(
            {
                "cache_context": cache_context,
                "stage_input": judge_spec.input_payload,
                "prompt": judge_spec.prompt,
            }
        )
        key, identity = make_cache_key(
            architecture=judge_spec.cache_scope or "C_SUPPORT_COUNTER",
            provider=provider,
            configured_model=FIXED_MODELS[provider],
            prompt_version=judge_spec.prompt_version,
            generation_parameters=TASK_GENERATION_PARAMETERS.get(
                judge_spec.task, GENERATION_PARAMETERS
            ),
            input_hash_value=value_hash,
        )
        if cache.get(key) is None:
            cache.put(
                key=key,
                identity=identity,
                parsed=judge_entry["parsed"],
                provider=provider,
                configured_model=FIXED_MODELS[provider],
                actual_model=FIXED_MODELS[provider],
                input_tokens=int(judge_entry.get("input_tokens", 0)),
                output_tokens=int(judge_entry.get("output_tokens", 0)),
                latency_ms=int(judge_entry.get("latency_ms", 0)),
            )
            rekeyed += 1
    return rekeyed


def _cache_probe(
    corpus,
    manifest: dict[str, object],
    sample_ids: tuple[str, ...],
    provider: str,
    cache_root: str,
) -> tuple[int, int, int]:
    probe_provider = OfflineBenchmarkProvider()
    probe_provider.name = provider
    probe_provider.model = FIXED_MODELS[provider]
    with tempfile.TemporaryDirectory(prefix="vericlaim-cache-probe-") as temporary_root:
        result = run_benchmark(
            corpus=corpus,
            manifest=manifest,
            profile="live-5",
            seed=42,
            architectures=ISOLATION_ARCHITECTURES,
            output_root=temporary_root,
            cache_root=cache_root,
            providers={provider: probe_provider},
            model_profile={"mode": "DRY_RUN_CACHE_PROBE"},
            stage_providers={
                "single": provider,
                "judge": provider,
                "classifier": provider,
                "auditor": provider,
                "critic": provider,
            },
            retrieval_config=RetrievalConfig(),
            sample_claim_ids=sample_ids,
            dry_run=True,
        )
    predictions = result["predictions"]
    cache_hits = sum(row["cache_hits"] for row in predictions)
    cache_misses = sum(row["cache_misses"] for row in predictions)
    conditional_reserve = sum(
        int(
            row["architecture"] == "D4_CONDITIONAL_CRITIC"
            and not row["assurance"]["critic_invoked"]
        )
        + int(
            row["architecture"] in {"D2_CRITIC", "D3_AUDITOR_CRITIC", "D4_CONDITIONAL_CRITIC"}
            and not row["assurance"]["recheck_performed"]
        )
        for row in predictions
    )
    return cache_hits, cache_misses, conditional_reserve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a budget-gated paired live SciFact agent-isolation evaluation"
    )
    parser.add_argument("--profile", choices=("live-5", "live-10"), default="live-5")
    parser.add_argument("--dataset-dir", default="evals/data/scifact/data")
    parser.add_argument("--manifest", default="evals/data/scifact/manifest.json")
    parser.add_argument("--sample-manifest", default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--output-root", default="evals/results")
    parser.add_argument("--cache-root", default="evals/cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--core-provider", choices=("groq", "gemini", "okmd"), default="gemini")
    parser.add_argument(
        "--classifier-provider", choices=("groq", "gemini", "okmd"), default="gemini"
    )
    parser.add_argument("--critic-provider", choices=("groq", "gemini", "okmd"), default="gemini")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sample_ids(path: str, claims, size: int, seed: int) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
    corpus_claim_ids = {claim.claim_id for claim in claims}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        ids = value.get("sample_ids", [])
    except (OSError, json.JSONDecodeError):
        ids = []
    if not isinstance(ids, list) or not all(str(item) in corpus_claim_ids for item in ids):
        ids = []
    if len(ids) < size:
        fallback = select_sample(
            claims,
            "smoke",
            seed,
        )
        ids = [claim.claim_id for claim in fallback]
    return tuple(str(item) for item in ids[:size])


def _provider_states(
    settings: Settings,
    providers: tuple[str, ...],
    availability: dict[str, object] | None = None,
    quota_by_provider: dict[str, dict[str, object]] | None = None,
) -> tuple[ProviderBudgetState, ...]:
    values: list[ProviderBudgetState] = []
    for provider in providers:
        enabled = bool(getattr(settings, f"{provider}_enabled", False))
        configured = bool(secret_value(getattr(settings, f"{provider}_api_key", None)))
        model = str(getattr(settings, f"{provider}_model", FIXED_MODELS[provider]))
        availability_status = (
            str((availability or {}).get(provider, {}).get("status"))
            if isinstance((availability or {}).get(provider), dict)
            else None
        )
        unavailable = availability_status not in {None, "AVAILABLE"}
        quota = (quota_by_provider or {}).get(provider, {})
        quota_status = quota.get("quota_status")
        if not isinstance(quota_status, QuotaStatus):
            quota_status = (
                QuotaStatus(str(quota_status))
                if quota_status in {item.value for item in QuotaStatus}
                else QuotaStatus.UNKNOWN
            )
        values.append(
            ProviderBudgetState(
                provider=provider,
                configured=configured,
                enabled=enabled,
                model=model,
                quota_status=(
                    QuotaStatus.UNAVAILABLE
                    if not configured or not enabled or unavailable
                    else quota_status
                ),
                remaining_requests=(
                    int(quota["remaining_requests"])
                    if quota.get("remaining_requests") is not None
                    else None
                ),
                remaining_tokens=(
                    int(quota["remaining_tokens"])
                    if quota.get("remaining_tokens") is not None
                    else None
                ),
                reset_at=str(quota["reset_at"]) if quota.get("reset_at") else None,
                source=(
                    str(quota["source"])
                    if quota.get("source")
                    else "model availability check"
                    if availability_status is not None
                    else "provider quota endpoint not exposed"
                ),
            )
        )
    return tuple(values)


def _okmd_reproducibility_probes(
    provider,
) -> tuple[bool, list[dict[str, object]], dict[str, object]]:  # type: ignore[no-untyped-def]
    """Run two identical bounded probes and return only safe metadata."""
    configured_model = FIXED_MODELS["okmd"]
    request = ProviderRequest(
        task="smoke",
        prompt="Reply with OK only.",
        max_tokens=128,
    )
    probes: list[dict[str, object]] = []
    for index in (1, 2):
        try:
            response = provider.generate(request)
        except ProviderException as exc:
            print(
                f"OKMD_PROBE_{index}=FAIL error_category={exc.category.value} "
                f"retry_after={exc.retry_after}"
            )
            return False, probes, {"quota_status": QuotaStatus.UNAVAILABLE}
        actual_model = response.actual_model or response.model
        row: dict[str, object] = {
            "probe": index,
            "status": "PASS",
            "configured_model": response.configured_model or configured_model,
            "actual_model": actual_model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "quota_remaining_tokens": response.quota_remaining_tokens,
            "quota_limit_tokens": response.quota_limit_tokens,
            "finish_reason": response.finish_reason,
        }
        probes.append(row)
        print(
            f"OKMD_PROBE_{index}=PASS configured_model={row['configured_model']} "
            f"actual_model={row['actual_model']} input_tokens={row['input_tokens']} "
            f"output_tokens={row['output_tokens']} total_tokens={row['total_tokens']} "
            f"quota_remaining={row['quota_remaining_tokens']} "
            f"quota_limit={row['quota_limit_tokens']} finish_reason={row['finish_reason']}"
        )
        if row["configured_model"] != configured_model or actual_model != configured_model:
            return False, probes, {"quota_status": QuotaStatus.UNAVAILABLE}
    remaining_values = [
        int(row["quota_remaining_tokens"])
        for row in probes
        if row.get("quota_remaining_tokens") is not None
    ]
    limit_values = [
        int(row["quota_limit_tokens"])
        for row in probes
        if row.get("quota_limit_tokens") is not None
    ]
    if len(remaining_values) != 2 or not limit_values:
        return False, probes, {"quota_status": QuotaStatus.UNKNOWN}
    return (
        True,
        probes,
        {
            "quota_status": QuotaStatus.KNOWN,
            "remaining_tokens": min(remaining_values),
            "remaining_requests": None,
            "source": "OKMD two identical fixed-model probes",
        },
    )


def _print_budget(plan, rows, cache_hits: int, cache_misses: int) -> None:  # type: ignore[no-untyped-def]
    print("BUDGET_GATE_TABLE")
    print(
        "provider model quota_status known_remaining_tokens known_remaining_requests "
        "estimated_calls estimated_tokens allowed reason"
    )
    for row in rows:
        print(
            f"{row['provider']} {row['model']} {row['quota_status']} "
            f"{row['known_remaining_tokens']} {row['known_remaining_requests']} "
            f"{row['estimated_calls']} {row['estimated_tokens']} {row['allowed']} "
            f"{row['reason']}"
        )
    print(f"GLOBAL_MAX_CALLS={plan.global_max_calls}")
    print(f"GLOBAL_MAX_TOKENS={plan.global_max_tokens}")
    print(f"ESTIMATED_TOTAL_CALLS={plan.estimated_total_calls}")
    print(f"ESTIMATED_TOTAL_TOKENS={plan.estimated_total_tokens}")
    print(f"HEADROOM_CALLS={plan.headroom_calls}")
    print(f"HEADROOM_TOKENS={plan.headroom_tokens}")
    print(f"CACHE_HITS={cache_hits}")
    print(f"CACHE_MISSES={cache_misses}")
    print(f"OPTIMIZED_CALL_UPPER_BOUND={plan.estimated_total_calls}")
    print(f"OPTIMIZED_TOKEN_UPPER_BOUND={plan.estimated_total_tokens}")
    print(f"BENCHMARK_PROVIDER={rows[0]['provider'] if rows else 'UNVERIFIED'}")
    print(f"BENCHMARK_CONFIGURED_MODEL={rows[0]['model'] if rows else 'UNVERIFIED'}")
    print(f"BUDGET_GATE={plan.decision} reason={plan.reason}")


def main() -> int:
    args = parse_args()
    sample_size = int(args.profile.removeprefix("live-"))
    try:
        corpus, manifest = load_and_validate_dataset(args.dataset_dir, args.manifest, "dev")
        settings = Settings()
        selected_provider_set = {
            args.core_provider,
            args.classifier_provider,
            args.critic_provider,
        }
        if len(selected_provider_set) != 1:
            raise EvaluationError(
                "exactly one provider must be selected for the fixed live benchmark"
            )
        required_providers = (args.core_provider,)
        sample_ids = _sample_ids(
            args.sample_manifest,
            corpus.claims,
            sample_size,
            args.seed,
        )
        if len(sample_ids) != sample_size:
            raise EvaluationError("unable to resolve a stable paired sample")
        try:
            live_providers, availability = build_live_providers(settings, required_providers)
        except EvaluationError:
            live_providers, availability = (
                {},
                {provider: {"status": "UNAVAILABLE"} for provider in required_providers},
            )
        quota_by_provider: dict[str, dict[str, object]] = {}
        probe_metadata: list[dict[str, object]] = []
        if args.core_provider == "okmd" and live_providers.get("okmd") is not None:
            probes_passed, probe_metadata, okmd_quota = _okmd_reproducibility_probes(
                live_providers["okmd"]
            )
            availability.setdefault("okmd", {})["reproducibility_probes"] = probe_metadata
            if not probes_passed:
                print("OKMD_FIXED_MODEL_PROBES=BLOCKED")
                return 2
            quota_by_provider["okmd"] = okmd_quota
            print("OKMD_FIXED_MODEL_PROBES=PASS count=2")
        states = _provider_states(
            settings,
            required_providers,
            availability,
            quota_by_provider=quota_by_provider,
        )
        stage_providers = {
            "single": args.core_provider,
            "judge": args.core_provider,
            "classifier": args.classifier_provider,
            "auditor": args.core_provider,
            "critic": args.critic_provider,
        }
        provider_by_architecture = {}
        for architecture in ISOLATION_ARCHITECTURES:
            provider_calls: dict[str, int] = {}
            for stage, calls_per_claim in ARCHITECTURE_STAGE_UPPER_BOUNDS[architecture].items():
                provider = stage_providers[stage]
                provider_calls[provider] = provider_calls.get(provider, 0) + calls_per_claim
            provider_by_architecture[architecture] = provider_calls
        rekeyed_cache_entries = _rekey_verified_legacy_cache(
            corpus,
            manifest,
            sample_ids,
            args.core_provider,
            args.cache_root,
            args.output_root,
        )
        cache_hits, cache_misses, conditional_reserve = _cache_probe(
            corpus,
            manifest,
            sample_ids,
            args.core_provider,
            args.cache_root,
        )
        print(f"VERIFIED_LEGACY_CACHE_REKEYED={rekeyed_cache_entries}")
        historical_average, historical_source = _historical_average_tokens_per_call(
            args.output_root
        )
        gate = LiveBudgetGate(
            provider_states=states,
            historical_average_tokens_per_call=historical_average,
        )
        plan = gate.plan(
            architecture_call_upper_bounds=ARCHITECTURE_CALL_UPPER_BOUNDS,
            sample_size=sample_size,
            provider_by_architecture=provider_by_architecture,
            cached_calls_by_provider={args.core_provider: cache_hits},
            provider_call_upper_bounds={args.core_provider: cache_misses + conditional_reserve},
        )
        _print_budget(plan, provider_budget_table(states, plan), cache_hits, cache_misses)
        print(f"CONDITIONAL_RECHECK_RESERVE={conditional_reserve}")
        if args.core_provider == "okmd":
            okmd_state = next(state for state in states if state.provider == "okmd")
            print(f"OKMD_REMAINING_TOKENS={okmd_state.remaining_tokens}")
            safe_ceiling = (
                okmd_state.remaining_tokens // 2
                if okmd_state.remaining_tokens is not None
                else "UNKNOWN"
            )
            print(f"OKMD_SAFE_TOKEN_CEILING={safe_ceiling}")
        if plan.decision == "DENY":
            return 2
        if (
            plan.headroom_calls / plan.global_max_calls < 0.10
            or plan.headroom_tokens / plan.global_max_tokens < 0.10
        ):
            print("LIVE5_SKIPPED_BUDGET_HEADROOM=minimum 10% call/token headroom not met")
            return 2
        print(f"SAMPLE_IDS={','.join(sample_ids)}")
        if args.dry_run:
            print("DRY_RUN_PROVIDER_CALLS=0")
            return 0
        providers = live_providers
        model_profile = {
            "mode": "LIVE_FREE_TIER_ISOLATION",
            "benchmark_provider": args.core_provider,
            "benchmark_configured_model": FIXED_MODELS[args.core_provider],
            "historical_average_tokens_per_call": historical_average,
            "groq": FIXED_MODELS["groq"],
            "gemini": FIXED_MODELS["gemini"],
            "okmd": FIXED_MODELS["okmd"],
            "okmd_reproducibility_probes": probe_metadata,
            "availability": availability,
            "stage_providers": stage_providers,
        }
        budget_policy = plan.as_dict()
        budget_policy["cache_hits"] = cache_hits
        budget_policy["cache_misses"] = cache_misses
        budget_policy["verified_legacy_cache_rekeyed"] = rekeyed_cache_entries
        budget_policy["historical_average_source"] = historical_source
        budget_policy["provider_states"] = [
            {
                "provider": state.provider,
                "configured": state.configured,
                "enabled": state.enabled,
                "model": state.model,
                "quota_status": state.quota_status.value,
                "remaining_requests": state.remaining_requests,
                "remaining_tokens": state.remaining_tokens,
                "reset_at": state.reset_at,
                "source": state.source,
            }
            for state in states
        ]
        result = run_benchmark(
            corpus=corpus,
            manifest=manifest,
            profile=args.profile,
            seed=args.seed,
            architectures=ISOLATION_ARCHITECTURES,
            output_root=args.output_root,
            cache_root=args.cache_root,
            providers=providers,
            model_profile=model_profile,
            stage_providers=stage_providers,
            retrieval_config=RetrievalConfig(),
            sample_claim_ids=sample_ids,
            budget_gate=gate,
            budget_policy=budget_policy,
        )
    except (SciFactDataError, EvaluationError, OSError) as exc:
        print(f"LIVE_ISOLATION=FAIL reason={exc}")
        return 1
    architectures_complete = all(
        result["metrics"].get(architecture, {}).get("complete", False)
        for architecture in ISOLATION_ARCHITECTURES
    )
    provider_failures = int(result["provider_usage"].get("provider_failures", 0))
    stop_reason = result["manifest"].get("budget_stop_reason")
    valid_paired_run = bool(result["manifest"].get("valid_paired_run"))
    model_substitutions = int(result["manifest"].get("model_substitutions", 0))
    if not architectures_complete or provider_failures or stop_reason or not valid_paired_run:
        reason = stop_reason or "paired isolation run incomplete or provider failure recorded"
        if model_substitutions:
            reason = "fixed model substitution detected"
        elif not valid_paired_run:
            reason = "paired predictions are not valid for comparison"
        print(f"LIVE_ISOLATION=BLOCKED reason={reason}")
        print(f"RESULT_DIRECTORY={result['output_directory']}")
        return 2
    print(f"LIVE_ISOLATION=PASS run_id={result['benchmark_run_id']}")
    print(f"RESULT_DIRECTORY={result['output_directory']}")
    print(f"ACTUAL_PROVIDER_CALLS={result['manifest']['actual_provider_calls']}")
    print(f"ACTUAL_LIVE_TOKENS={result['manifest']['actual_live_tokens']}")
    print(f"OPENROUTER_USED={result['manifest']['free_tier_usage']['openrouter_used']}")
    print("API_COST_USD=N/A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
