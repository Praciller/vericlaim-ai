"""Deterministic regression gate over the checked-in SciFact fixture.

Runs the offline fixture benchmark end to end, compares the resulting metrics
against the checked-in baseline, and exits non-zero on regression, baseline
drift, or an invalid paired run. Intended for CI and local verification; makes
no network or provider calls.

Exit codes:
    0  PASS
    1  REGRESSION, BASELINE_DRIFT, or INVALID_RUN
    2  setup or benchmark failure

Use --update-baseline to regenerate the checked-in baseline from a fresh
verified run. This is an explicit opt-in: commit the new baseline separately
and record why the metrics changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.scifact.dataset import SciFactDataError  # noqa: E402
from evals.scifact.regression import (  # noqa: E402
    GateStatus,
    build_baseline_from_result,
    compare_to_baseline,
    describe_report,
    load_baseline,
)
from evals.scifact.retrieval import RetrievalConfig  # noqa: E402
from evals.scifact.runner import (  # noqa: E402
    ARCHITECTURES,
    EvaluationError,
    OfflineBenchmarkProvider,
    load_and_validate_dataset,
    run_benchmark,
)

DEFAULT_DATASET_DIR = "evals/fixtures/scifact/data"
DEFAULT_MANIFEST = "evals/fixtures/scifact/manifest.json"
DEFAULT_BASELINE = "evals/baselines/scifact-offline-fixture-smoke.json"
BASELINE_ID = "scifact-offline-fixture-smoke"
OFFLINE_MODELS = {
    "groq": "deterministic-eval-fixture-v1",
    "gemini": "deterministic-eval-fixture-v1",
}
STAGE_PROVIDERS = {
    "single": "gemini",
    "judge": "gemini",
    "classifier": "groq",
    "auditor": "gemini",
    "critic": "groq",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate the offline SciFact fixture benchmark against a checked-in baseline"
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="dev", choices=("train", "dev"))
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Regenerate the checked-in baseline from a fresh offline run",
    )
    return parser.parse_args()


def run_offline_fixture_benchmark(args: argparse.Namespace) -> dict:
    corpus, manifest = load_and_validate_dataset(args.dataset_dir, args.manifest, args.split)
    with tempfile.TemporaryDirectory(prefix="vericlaim-regression-gate-") as temp:
        return run_benchmark(
            corpus=corpus,
            manifest=manifest,
            profile="smoke",
            seed=args.seed,
            architectures=ARCHITECTURES,
            output_root=str(Path(temp) / "results"),
            cache_root=str(Path(temp) / "cache"),
            providers={
                "groq": OfflineBenchmarkProvider(),
                "gemini": OfflineBenchmarkProvider(),
            },
            model_profile={
                "mode": "OFFLINE_FIXTURE",
                **OFFLINE_MODELS,
                "stage_providers": STAGE_PROVIDERS,
            },
            stage_providers=STAGE_PROVIDERS,
            retrieval_config=RetrievalConfig(),
        )


def main() -> int:
    args = parse_args()
    baseline_path = Path(args.baseline)
    try:
        result = run_offline_fixture_benchmark(args)
    except (SciFactDataError, EvaluationError, OSError) as exc:
        print(f"REGRESSION_GATE_SETUP=FAIL reason={exc}")
        return 2

    if args.update_baseline:
        baseline = build_baseline_from_result(result, baseline_id=BASELINE_ID)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"BASELINE_UPDATED path={baseline_path}")
        print(f"BASELINE_ARCHITECTURES={','.join(sorted(baseline['architectures']))}")
        return 0

    try:
        baseline = load_baseline(baseline_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REGRESSION_GATE_SETUP=FAIL reason=baseline unavailable: {exc}")
        return 2

    report = compare_to_baseline(result, baseline)
    print(describe_report(report))
    run_id = result["manifest"]["benchmark_run_id"]
    print(f"BENCHMARK_RUN_ID={run_id}")
    if report["status"] is GateStatus.PASS and report["improvements"]:
        print(
            "NOTE=metrics improved against the baseline; consider refreshing it with "
            "--update-baseline in a separate reviewed commit"
        )
    return 0 if report["status"] is GateStatus.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
