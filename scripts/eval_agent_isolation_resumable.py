from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.scifact.budget import (  # noqa: E402
    LiveBudgetGate,
    ProviderBudgetState,
    QuotaStatus,
)
from evals.scifact.cache import input_hash  # noqa: E402
from evals.scifact.dataset import SciFactDataError  # noqa: E402
from evals.scifact.retrieval import BM25Retriever, RetrievalConfig  # noqa: E402
from evals.scifact.runner import (  # noqa: E402
    FIXED_MODELS,
    GENERATION_PARAMETERS,
    ISOLATION_ARCHITECTURES,
    PROMPT_VERSIONS,
    TASK_GENERATION_PARAMETERS,
    EvaluationError,
    _critic_spec,
    _stage_specs,
    build_live_providers,
    load_and_validate_dataset,
    run_benchmark,
)
from scripts.eval_agent_isolation import (  # noqa: E402
    DEFAULT_SAMPLE_MANIFEST,
    _historical_average_tokens_per_call,
)

from vericlaim.config import Settings  # noqa: E402
from vericlaim.providers.base import ProviderException, ProviderRequest  # noqa: E402

TARGET_SAMPLE_IDS = ("13", "208", "268", "314", "549")
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_PACING_SECONDS = 1.0
LOW_PROGRESS_FAILURE_WINDOW_LIMIT = 3


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sample_ids(
    path: str,
    claims,
    size: int,
    manifest: dict[str, object],
) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
    corpus_ids = {claim.claim_id for claim in claims}
    payload: object = {}
    try:
        parsed = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = parsed if isinstance(parsed, dict) else {}
        values = payload.get("sample_ids", [])
    except (OSError, json.JSONDecodeError):
        values = []
    if (
        not isinstance(values, list)
        or len(values) < size
        or not all(str(value) in corpus_ids for value in values)
    ):
        raise EvaluationError("sample manifest does not contain a valid immutable sample")
    selected_values = tuple(str(value) for value in values if str(value) in TARGET_SAMPLE_IDS)
    if set(selected_values) != set(TARGET_SAMPLE_IDS) or len(selected_values) != len(
        TARGET_SAMPLE_IDS
    ):
        raise EvaluationError("resumable live-5 requires the locked sample IDs 13,208,268,314,549")
    selected = TARGET_SAMPLE_IDS
    expected_hashes = {item["path"]: item["sha256"] for item in manifest.get("files", [])}
    if not isinstance(payload, dict):
        raise EvaluationError("sample manifest must be a JSON object")
    if payload.get("dataset_revision") != manifest.get("revision"):
        raise EvaluationError("sample manifest dataset revision does not match the loaded dataset")
    if payload.get("dataset_hashes") != expected_hashes:
        raise EvaluationError("sample manifest dataset hashes do not match the loaded dataset")
    return selected


def _stage_payloads(
    corpus,
    claim,
    provider: str,
    retriever: BM25Retriever | None = None,
) -> dict[str, tuple[str, int, int]]:  # type: ignore[no-untyped-def]
    retriever = retriever or BM25Retriever(corpus, RetrievalConfig())
    retrieved = retriever.retrieve(claim.text)
    providers = {
        "single": provider,
        "judge": provider,
        "classifier": provider,
        "auditor": provider,
        "critic": provider,
    }
    classifier = _stage_specs(claim, "C_SUPPORT_COUNTER", retrieved, stage_providers=providers)[0]
    judge = _stage_specs(
        claim,
        "C_SUPPORT_COUNTER",
        retrieved,
        classifier_items=[],
        stage_providers=providers,
    )[0]
    critic = _critic_spec(
        claim,
        "D2_CRITIC",
        retrieved,
        [],
        [],
        {"verdict": "SUPPORTED", "confidence": 0.7, "selected_evidence_ids": []},
        ["CALIBRATION_PAYLOAD"],
        providers,
    )
    return {
        "classifier": (
            classifier.prompt,
            len(classifier.prompt),
            len(classifier.prompt.split()),
        ),
        "judge": (judge.prompt, len(judge.prompt), len(judge.prompt.split())),
        "critic": (critic.prompt, len(critic.prompt), len(critic.prompt.split())),
    }


def _select_reliability_claims(corpus, sample_ids: tuple[str, ...], provider: str):  # type: ignore[no-untyped-def]
    excluded = set(sample_ids)
    candidates = [claim for claim in corpus.claims if claim.claim_id not in excluded]
    if len(candidates) < 3:
        raise EvaluationError("not enough non-evaluation claims for reliability calibration")
    retriever = BM25Retriever(corpus, RetrievalConfig())
    target_claims = [claim for claim in corpus.claims if claim.claim_id in set(sample_ids)]
    payloads = {
        claim.claim_id: _stage_payloads(corpus, claim, provider, retriever)
        for claim in (*target_claims, *candidates)
    }
    target_sizes = {
        stage: max(payloads[claim.claim_id][stage][1] for claim in target_claims)
        for stage in ("classifier", "judge", "critic")
    }
    selected = {}
    remaining = list(candidates)
    for stage in ("classifier", "judge", "critic"):
        candidate = min(
            remaining,
            key=lambda claim: abs(payloads[claim.claim_id][stage][1] - target_sizes[stage]),
        )
        selected[stage] = candidate
        remaining.remove(candidate)
    return selected


def _reliability_configuration(
    manifest: dict[str, object],
    sample_ids: tuple[str, ...],
    provider: str,
    timeout_seconds: float,
    pacing_seconds: float,
) -> dict[str, object]:
    configuration: dict[str, object] = {
        "mode": "PROVIDER_RELIABILITY_ENVELOPE",
        "provider": provider,
        "configured_model": FIXED_MODELS[provider],
        "excluded_evaluation_sample_ids": list(sample_ids),
        "selection_rule": "three non-target claims nearest to target max payload per stage",
        "dataset_revision": manifest.get("revision"),
        "retrieval_config": asdict(RetrievalConfig()),
        "prompt_versions": PROMPT_VERSIONS,
        "generation_parameters": {
            "default": GENERATION_PARAMETERS,
            "task_overrides": TASK_GENERATION_PARAMETERS,
        },
        "timeout_seconds": timeout_seconds,
        "pacing_seconds": pacing_seconds,
        "concurrency": 1,
        "no_immediate_retry": True,
        "provider_min_interval_policy": "configured_pacing_only",
    }
    return configuration


def run_reliability_calibration(
    *,
    corpus,
    manifest: dict[str, object],
    sample_ids: tuple[str, ...],
    provider_name: str,
    output_path: Path,
    timeout_seconds: float,
    pacing_seconds: float,
) -> tuple[bool, dict[str, object]]:
    configuration = _reliability_configuration(
        manifest, sample_ids, provider_name, timeout_seconds, pacing_seconds
    )
    configuration_hash = input_hash(configuration)
    started_at = datetime.now(UTC).isoformat()
    providers, availability = build_live_providers(Settings(), (provider_name,))
    provider = providers[provider_name]
    if hasattr(provider, "timeout_seconds"):
        provider.timeout_seconds = timeout_seconds
    selected = _select_reliability_claims(corpus, sample_ids, provider_name)
    records: list[dict[str, object]] = []
    locked_actual_model: str | None = None
    previous_call_at = 0.0
    for stage in ("classifier", "judge", "critic"):
        claim = selected[stage]
        prompt, prompt_chars, prompt_tokens = _stage_payloads(corpus, claim, provider_name)[stage]
        task_by_stage = {
            "classifier": "evidence_classifier",
            "judge": "judge",
            "critic": "critic",
        }
        task = task_by_stage[stage]
        generation_parameters = TASK_GENERATION_PARAMETERS.get(task, GENERATION_PARAMETERS)
        if previous_call_at:
            elapsed = time.perf_counter() - previous_call_at
            if elapsed < pacing_seconds:
                time.sleep(pacing_seconds - elapsed)
        call_started_at = time.perf_counter()
        previous_call_at = call_started_at
        record: dict[str, object] = {
            "claim_id": claim.claim_id,
            "stage": stage,
            "task": task,
            "prompt_chars": prompt_chars,
            "prompt_tokens_approx": prompt_tokens,
            "input_context_chars": prompt_chars,
            "max_output_tokens": generation_parameters["max_tokens"],
            "timeout_seconds": timeout_seconds,
            "status": "FAIL",
        }
        try:
            response = provider.generate(
                ProviderRequest(
                    task=task,
                    prompt=prompt,
                    max_tokens=generation_parameters["max_tokens"],
                )
            )
        except ProviderException as exc:
            record.update(
                {
                    "failure_category": exc.category.value,
                    "latency_ms": int((time.perf_counter() - call_started_at) * 1000),
                    "response_complete": False,
                }
            )
            records.append(record)
            break
        except Exception:
            record.update(
                {
                    "failure_category": "provider_failure",
                    "latency_ms": int((time.perf_counter() - call_started_at) * 1000),
                    "response_complete": False,
                }
            )
            records.append(record)
            break
        actual_model = response.actual_model or response.model
        complete = bool(response.text.strip()) and response.finish_reason != "length"
        record.update(
            {
                "status": "PASS" if complete else "FAIL",
                "configured_model": response.configured_model or provider.model,
                "actual_model": actual_model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "latency_ms": response.latency_ms,
                "finish_reason": response.finish_reason,
                "response_complete": complete,
            }
        )
        if response.quota_remaining_tokens is not None:
            record["quota_remaining_tokens"] = response.quota_remaining_tokens
        if response.quota_limit_tokens is not None:
            record["quota_limit_tokens"] = response.quota_limit_tokens
        if response.quota_used_tokens is not None:
            record["quota_used_tokens"] = response.quota_used_tokens
        if actual_model != FIXED_MODELS[provider_name]:
            record["status"] = "FAIL"
            record["failure_category"] = "model_substitution"
        elif locked_actual_model is not None and actual_model != locked_actual_model:
            record["status"] = "FAIL"
            record["failure_category"] = "model_drift"
        else:
            locked_actual_model = actual_model
        records.append(record)
        if record["status"] != "PASS":
            break
    passed = (
        len(records) == 3
        and all(record["status"] == "PASS" for record in records)
        and locked_actual_model == FIXED_MODELS[provider_name]
    )
    profile: dict[str, object] = {
        "profile_version": 1,
        "profile_id": (
            f"reliability-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        ),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "provider": provider_name,
        "configured_model": FIXED_MODELS[provider_name],
        "actual_model": locked_actual_model,
        "availability": availability,
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "excluded_evaluation_sample_ids": list(sample_ids),
        "calibration_claim_ids": {stage: claim.claim_id for stage, claim in selected.items()},
        "records": records,
        "max_calls": 3,
        "concurrency": 1,
        "no_immediate_retry": True,
        "provider_min_interval_policy": "configured_pacing_only",
    }
    _write_json_atomic(output_path, profile)
    print(f"PROVIDER_RELIABILITY_PROFILE={profile['status']}")
    print(f"RELIABILITY_PROFILE_PATH={output_path.resolve()}")
    print(f"RELIABILITY_CONFIGURATION_HASH={configuration_hash}")
    print(f"RELIABILITY_CALLS={len(records)}")
    if records and records[-1].get("failure_category"):
        print(f"RELIABILITY_FAILURE_CATEGORY={records[-1]['failure_category']}")
    return passed, profile


def _load_profile(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("reliability profile is unreadable") from exc
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise EvaluationError("provider reliability profile is not PASS")
    if value.get("configured_model") != value.get("actual_model"):
        raise EvaluationError("reliability profile actual model is not locked")
    configuration = value.get("configuration")
    if not isinstance(configuration, dict) or input_hash(configuration) != value.get(
        "configuration_hash"
    ):
        raise EvaluationError("reliability profile configuration hash is invalid")
    return value


def _identity(
    *,
    manifest: dict[str, object],
    sample_ids: tuple[str, ...],
    provider: str,
    profile: dict[str, object],
    timeout_seconds: float,
    pacing_seconds: float,
    window_max_calls: int,
) -> dict[str, object]:
    return {
        "mode": "RESUMABLE_WINDOWED_LIVE5",
        "dataset_revision": manifest.get("revision"),
        "dataset_hashes": {item["path"]: item["sha256"] for item in manifest.get("files", [])},
        "sample_ids": list(sample_ids),
        "retrieval_config": asdict(RetrievalConfig()),
        "architectures": list(ISOLATION_ARCHITECTURES),
        "provider": provider,
        "configured_model": FIXED_MODELS[provider],
        "actual_model": profile["actual_model"],
        "temperature": 0,
        "prompt_versions": PROMPT_VERSIONS,
        "generation_parameters": {
            "default": GENERATION_PARAMETERS,
            "task_overrides": TASK_GENERATION_PARAMETERS,
        },
        "reliability_profile_hash": profile["configuration_hash"],
        "timeout_seconds": timeout_seconds,
        "pacing_seconds": pacing_seconds,
        "window_max_calls": window_max_calls,
        "concurrency": 1,
        "no_immediate_retry": True,
        "provider_min_interval_policy": "configured_pacing_only",
        "fallbacks": False,
    }


def _read_checkpoint(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("resumable checkpoint is unreadable") from exc
    if not isinstance(value, dict):
        raise EvaluationError("resumable checkpoint must be an object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one bounded resumable SciFact live-5 window")
    parser.add_argument("--provider", choices=tuple(FIXED_MODELS), default="okmd")
    parser.add_argument("--dataset-dir", default="evals/data/scifact/data")
    parser.add_argument("--manifest", default="evals/data/scifact/manifest.json")
    parser.add_argument("--sample-manifest", default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--output-root", default="evals/results")
    parser.add_argument("--cache-root", default="evals/cache")
    parser.add_argument("--reliability-profile", default="")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="explicit checkpoint.json path for --resume when more than one run exists",
    )
    parser.add_argument("--window-max-calls", type=int, default=8)
    parser.add_argument("--window-pacing-seconds", type=float, default=DEFAULT_PACING_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--min-progress", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 5 <= args.window_max_calls <= 10:
        print("RESUMABLE_LIVE5=FAIL reason=window-max-calls must be between 5 and 10")
        return 1
    if args.timeout_seconds <= 0 or args.window_pacing_seconds < 0:
        print("RESUMABLE_LIVE5=FAIL reason=timeout/pacing values are invalid")
        return 1
    if args.max_windows <= 0 or args.min_progress <= 0:
        print("RESUMABLE_LIVE5=FAIL reason=window stop-rule values are invalid")
        return 1
    if args.max_windows < LOW_PROGRESS_FAILURE_WINDOW_LIMIT:
        print("RESUMABLE_LIVE5=FAIL reason=max-windows must allow three failure windows")
        return 1
    try:
        corpus, manifest = load_and_validate_dataset(args.dataset_dir, args.manifest, "dev")
        sample_ids = _sample_ids(args.sample_manifest, corpus.claims, 5, manifest)
        if args.calibrate_only:
            output_path = (
                Path(args.reliability_profile)
                if args.reliability_profile
                else Path(args.output_root)
                / (
                    f"scifact-reliability-live-5-{args.provider}-"
                    f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
                )
            )
            passed, _ = run_reliability_calibration(
                corpus=corpus,
                manifest=manifest,
                sample_ids=sample_ids,
                provider_name=args.provider,
                output_path=output_path,
                timeout_seconds=args.timeout_seconds,
                pacing_seconds=args.window_pacing_seconds,
            )
            return 0 if passed else 2
        if not args.reliability_profile:
            raise EvaluationError("--reliability-profile is required for benchmark windows")
        profile = _load_profile(Path(args.reliability_profile))
        if profile.get("provider") != args.provider:
            raise EvaluationError("reliability profile provider mismatch")
        expected_configuration = _reliability_configuration(
            manifest,
            sample_ids,
            args.provider,
            args.timeout_seconds,
            args.window_pacing_seconds,
        )
        if input_hash(expected_configuration) != profile.get("configuration_hash"):
            raise EvaluationError("reliability settings changed; create a new profile and run")
        identity = _identity(
            manifest=manifest,
            sample_ids=sample_ids,
            provider=args.provider,
            profile=profile,
            timeout_seconds=args.timeout_seconds,
            pacing_seconds=args.window_pacing_seconds,
            window_max_calls=args.window_max_calls,
        )
        configuration_hash = input_hash(identity)
        run_id = ""
        checkpoint_file: Path
        if args.resume:
            if args.checkpoint:
                checkpoint_file = Path(args.checkpoint)
                if not checkpoint_file.is_file():
                    raise EvaluationError("--checkpoint does not point to a readable checkpoint")
            else:
                root = Path(args.output_root)
                candidates = sorted(root.glob("scifact-live-5-resumable-*/checkpoint.json"))
                if not candidates:
                    raise EvaluationError("--resume requested but no resumable checkpoint exists")
                if len(candidates) > 1:
                    raise EvaluationError(
                        "multiple resumable checkpoints exist; pass --checkpoint explicitly"
                    )
                checkpoint_file = candidates[0]
            checkpoint = _read_checkpoint(checkpoint_file)
            run_id = str(checkpoint.get("benchmark_run_id", ""))
            if checkpoint.get("resumable_identity") != identity:
                raise EvaluationError("resumable identity changed; start a new benchmark run")
            if checkpoint.get("status") in {
                "COMPLETE",
                "BENCHMARK_INVALID_MODEL_DRIFT",
                "PROVIDER_UNSUITABLE_FOR_BENCHMARK",
            }:
                raise EvaluationError("checkpoint is terminal; start a new benchmark run")
        else:
            root = Path(args.output_root)
            run_id = (
                f"scifact-live-5-resumable-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid4().hex[:8]}"
            )
            checkpoint_file = root / run_id / "checkpoint.json"
            if checkpoint_file.exists():
                raise EvaluationError("run ID collision; choose a new run directory")
            _write_json_atomic(
                checkpoint_file,
                {
                    "checkpoint_version": 1,
                    "benchmark_run_id": run_id,
                    "resumable_identity": identity,
                    "cache_context": {
                        "dataset_revision": manifest.get("revision"),
                        "retrieval_config": asdict(RetrievalConfig()),
                        "resumable_configuration_hash": configuration_hash,
                        "reliability_profile_hash": profile["configuration_hash"],
                    },
                    "locked_actual_model": None,
                    "predictions": [],
                    "records": [],
                    "windows": [],
                },
            )
        settings = Settings()
        providers, availability = build_live_providers(settings, (args.provider,))
        provider = providers[args.provider]
        if hasattr(provider, "timeout_seconds"):
            provider.timeout_seconds = args.timeout_seconds
        state = _read_checkpoint(checkpoint_file)
        windows = state.get("windows", [])
        next_window = len(windows) + 1 if isinstance(windows, list) else 1
        if next_window > args.max_windows:
            print("PROVIDER_UNSUITABLE_FOR_BENCHMARK reason=max bounded windows reached")
            return 2
        historical_average, historical_source = _historical_average_tokens_per_call(
            args.output_root
        )
        gate = LiveBudgetGate(
            provider_states=(
                ProviderBudgetState(
                    provider=args.provider,
                    configured=True,
                    enabled=True,
                    model=FIXED_MODELS[args.provider],
                    quota_status=QuotaStatus.UNKNOWN,
                    source="resumable bounded window; no quota inference",
                ),
            ),
            global_max_calls=args.window_max_calls,
            global_max_tokens=int(historical_average * args.window_max_calls * 0.98),
            historical_average_tokens_per_call=historical_average,
        )
        stage_providers = {
            stage: args.provider for stage in ("single", "judge", "classifier", "auditor", "critic")
        }
        model_profile = {
            "mode": "RESUMABLE_WINDOWED_LIVE5",
            "benchmark_provider": args.provider,
            "benchmark_configured_model": FIXED_MODELS[args.provider],
            "benchmark_actual_model": profile["actual_model"],
            "historical_average_tokens_per_call": historical_average,
            "historical_average_source": historical_source,
            "configuration_hash": configuration_hash,
            "resumable_identity": identity,
            "reliability_profile_hash": profile["configuration_hash"],
            "availability": availability,
            "stage_providers": stage_providers,
        }
        started_at = datetime.now(UTC).isoformat()
        print("LIVE5_MODE=RESUMABLE")
        print(f"WINDOW_ID=window-{next_window:03d}")
        print(f"SAMPLE_IDS={','.join(sample_ids)}")
        print(f"WINDOW_MAX_CALLS={args.window_max_calls}")
        print(f"WINDOW_PACING_SECONDS={args.window_pacing_seconds}")
        print(f"TIMEOUT_SECONDS={args.timeout_seconds}")
        result = run_benchmark(
            corpus=corpus,
            manifest=manifest,
            profile="live-5",
            seed=42,
            architectures=ISOLATION_ARCHITECTURES,
            output_root=args.output_root,
            cache_root=args.cache_root,
            providers=providers,
            model_profile=model_profile,
            stage_providers=stage_providers,
            retrieval_config=RetrievalConfig(),
            sample_claim_ids=sample_ids,
            budget_gate=gate,
            budget_policy={
                "decision": "ALLOW_WINDOW",
                "window_max_calls": args.window_max_calls,
                "window_pacing_seconds": args.window_pacing_seconds,
                "timeout_seconds": args.timeout_seconds,
                "no_immediate_retry": True,
                "provider_min_interval_policy": "configured_pacing_only",
                "historical_average_source": historical_source,
            },
            benchmark_run_id=run_id,
            checkpoint_path=checkpoint_file,
            resumable=True,
            cache_context_extra={
                "resumable_configuration_hash": configuration_hash,
                "reliability_profile_hash": profile["configuration_hash"],
            },
            locked_actual_model=(
                str(state["locked_actual_model"]) if state.get("locked_actual_model") else None
            ),
            pacing_seconds=args.window_pacing_seconds,
            enforce_provider_min_interval=False,
            window_id=f"window-{next_window:03d}",
            window_policy={
                "started_at": started_at,
                "max_calls": args.window_max_calls,
                "pacing_seconds": args.window_pacing_seconds,
                "timeout_seconds": args.timeout_seconds,
                "no_immediate_retry": True,
                "provider_min_interval_policy": "configured_pacing_only",
                "concurrency": 1,
            },
        )
        manifest_result = result["manifest"]
        window = manifest_result.get("windows", [])[-1]
        previous_pairs = int(state.get("resume_cursor", {}).get("completed_pairs", 0))
        current_pairs = int(manifest_result.get("resume_cursor", {}).get("completed_pairs", 0))
        new_pairs = current_pairs - previous_pairs
        window_failed = int(window.get("provider_failures", 0)) > 0
        prior_streak = 0
        prior_pairs = 0
        for prior in state.get("windows", []):
            completed_pairs = int(prior.get("resume_cursor", {}).get("completed_pairs", 0))
            prior_progress = completed_pairs - prior_pairs
            if int(prior.get("provider_failures", 0)) > 0 and prior_progress < args.min_progress:
                prior_streak += 1
            else:
                prior_streak = 0
            prior_pairs = completed_pairs
        failure_streak = prior_streak + 1 if window_failed and new_pairs < args.min_progress else 0
        state = _read_checkpoint(checkpoint_file)
        state["failure_streak"] = failure_streak
        state["last_window_progress"] = new_pairs
        state["provider_reliability_profile"] = str(Path(args.reliability_profile).resolve())
        if int(manifest_result.get("model_drift", 0)) > 0:
            state["status"] = "BENCHMARK_INVALID_MODEL_DRIFT"
            _write_json_atomic(checkpoint_file, state)
            print("BENCHMARK_INVALID_MODEL_DRIFT")
            print(f"RESULT_DIRECTORY={result['output_directory']}")
            print("LIVE5_STATUS=INVALID")
            return 2
        if failure_streak >= LOW_PROGRESS_FAILURE_WINDOW_LIMIT:
            state["status"] = "PROVIDER_UNSUITABLE_FOR_BENCHMARK"
            _write_json_atomic(checkpoint_file, state)
            print("PROVIDER_UNSUITABLE_FOR_BENCHMARK reason=three bounded low-progress failures")
            print(f"RESULT_DIRECTORY={result['output_directory']}")
            return 2
        if manifest_result.get("valid_paired_run"):
            state["status"] = "COMPLETE"
        _write_json_atomic(checkpoint_file, state)
        print("PROVIDER_RELIABILITY_PROFILE=PASS")
        print(f"WINDOW_PROGRESS={new_pairs}")
        print(f"WINDOWS_COMPLETED={len(manifest_result.get('windows', []))}")
        print(f"RESULT_DIRECTORY={result['output_directory']}")
        if manifest_result.get("valid_paired_run"):
            print("VALID_PAIRED_PREDICTIONS=25/25")
            print("MODEL_SUBSTITUTIONS=0")
            print("MODEL_DRIFT=0")
            print("LIVE5_STATUS=PASS")
            return 0
        print(f"VALID_PAIRED_PREDICTIONS={manifest_result.get('valid_paired_predictions', 0)}/25")
        print(f"WINDOW_STOP_REASON={window.get('stop_reason') or 'window incomplete'}")
        print("LIVE5_STATUS=WINDOW_STOPPED")
        return 2
    except (SciFactDataError, EvaluationError, OSError, KeyError) as exc:
        print(f"RESUMABLE_LIVE5=FAIL reason={exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
