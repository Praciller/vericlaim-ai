from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.scifact.dataset import SciFactDataError  # noqa: E402
from evals.scifact.retrieval import RetrievalConfig  # noqa: E402
from evals.scifact.runner import (  # noqa: E402
    ARCHITECTURE_GROUPS,
    ARCHITECTURES,
    FIXED_MODELS,
    ISOLATION_ARCHITECTURES,
    EvaluationError,
    OfflineBenchmarkProvider,
    build_live_providers,
    load_and_validate_dataset,
    run_benchmark,
    run_dry_run,
)

from vericlaim.config import Settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded SciFact primary or agent-isolation evaluation"
    )
    parser.add_argument(
        "--architecture",
        default="all",
        help="all, isolation, or one architecture name",
    )
    parser.add_argument("--profile", choices=("smoke", "pilot", "extended"), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="dev", choices=("train", "dev"))
    parser.add_argument("--dataset-dir", default="evals/data/scifact/data")
    parser.add_argument("--manifest", default="evals/data/scifact/manifest.json")
    parser.add_argument("--output-root", default="evals/results")
    parser.add_argument("--cache-root", default="evals/cache")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-extended", action="store_true")
    parser.add_argument("--classifier-provider", choices=("groq", "gemini"), default="groq")
    parser.add_argument("--critic-provider", choices=("groq", "gemini"), default="groq")
    parser.add_argument("--resume", action="store_true", help="Reuse structured cache entries")
    return parser.parse_args()


def architectures(value: str) -> tuple[str, ...]:
    if value in ARCHITECTURE_GROUPS:
        return ARCHITECTURE_GROUPS[value]
    valid_architectures = ARCHITECTURES + tuple(
        architecture
        for architecture in ISOLATION_ARCHITECTURES
        if architecture not in ARCHITECTURES
    )
    if value not in valid_architectures:
        raise EvaluationError(f"unknown architecture: {value}")
    return (value,)


def main() -> int:
    args = parse_args()
    if args.profile == "extended" and not args.allow_extended:
        print("EXTENDED_RUN=EXTENDED_SKIPPED_FREE_TIER_BUDGET reason=explicit allow flag required")
        return 2
    try:
        corpus, manifest = load_and_validate_dataset(args.dataset_dir, args.manifest, args.split)
        selected_architectures = architectures(args.architecture)
        retrieval_config = RetrievalConfig()
        stage_providers = {
            "single": "gemini",
            "judge": "gemini",
            "classifier": args.classifier_provider,
            "auditor": "gemini",
            "critic": args.critic_provider,
        }
    except (SciFactDataError, EvaluationError) as exc:
        print(f"EVALUATION_SETUP=FAIL reason={exc}")
        return 1
    if args.dry_run:
        result = run_dry_run(
            corpus=corpus,
            manifest=manifest,
            profile=args.profile,
            seed=args.seed,
            architectures=selected_architectures,
            cache_root=args.cache_root,
            retrieval_config=retrieval_config,
        )
        predictions = result["predictions"]
        cached = sum(row["cache_hits"] for row in predictions)
        uncached = sum(row["cache_misses"] for row in predictions)
        print(f"DATASET=SciFact split={args.split}")
        sample_size = len({row["claim_id"] for row in predictions})
        print(f"SAMPLE_PROFILE={args.profile} sample_size={sample_size}")
        print(f"ARCHITECTURES={','.join(selected_architectures)}")
        models = {"groq": FIXED_MODELS["groq"], "gemini": FIXED_MODELS["gemini"]}
        print(f"MODELS={json.dumps(models, sort_keys=True)}")
        print(f"ESTIMATED_LLM_CALLS={cached + uncached}")
        print(f"CACHED_CALLS={cached}")
        print(f"ESTIMATED_UNCACHED_CALLS={uncached}")
        print("DRY_RUN_PROVIDER_CALLS=0")
        return 0
    try:
        if args.offline:
            providers = {"groq": OfflineBenchmarkProvider(), "gemini": OfflineBenchmarkProvider()}
            model_profile = {
                "mode": "OFFLINE_FIXTURE",
                **FIXED_MODELS,
                "stage_providers": stage_providers,
            }
            availability = {}
        else:
            providers, availability = build_live_providers(Settings())
            model_profile = {
                "mode": "LIVE_FREE_TIER",
                "groq": FIXED_MODELS["groq"],
                "gemini": FIXED_MODELS["gemini"],
                "availability": availability,
                "stage_providers": stage_providers,
            }
        result = run_benchmark(
            corpus=corpus,
            manifest=manifest,
            profile=args.profile,
            seed=args.seed,
            architectures=selected_architectures,
            output_root=args.output_root,
            cache_root=args.cache_root,
            providers=providers,
            model_profile=model_profile,
            stage_providers=stage_providers,
            retrieval_config=retrieval_config,
        )
    except (SciFactDataError, EvaluationError, OSError) as exc:
        print(f"EVALUATION_RUN=FAIL reason={exc}")
        return 1
    print(f"EVALUATION_RUN=PASS run_id={result['benchmark_run_id']}")
    print(f"RESULT_DIRECTORY={result['output_directory']}")
    print(f"SAMPLE_SIZE={result['manifest']['sample_size']}")
    print(f"LABEL_DISTRIBUTION={json.dumps(corpus.label_distribution, sort_keys=True)}")
    for architecture in selected_architectures:
        metric = result["metrics"][architecture]
        print(
            f"{architecture} macro_f1={metric['claim']['macro_f1']:.4f} "
            f"accuracy={metric['claim']['accuracy']:.4f} "
            f"calls={metric['efficiency']['avg_llm_calls']:.2f} "
            f"tokens={metric['efficiency']['avg_total_tokens']:.2f} "
            f"latency_ms={metric['efficiency']['avg_latency_ms']:.2f}"
        )
    print(f"OPENROUTER_USED={result['manifest']['free_tier_usage']['openrouter_used']}")
    print(f"API_COST_USD={result['manifest']['free_tier_usage']['api_cost_usd']}")
    print(f"CACHE_VERSION={result['manifest']['cache_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
