from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from urllib.parse import urlparse

from vericlaim.config import Settings
from vericlaim.db import Database
from vericlaim.domain.models import VerificationRequest
from vericlaim.providers.router import ProviderRouter
from vericlaim.workflow import VerificationWorkflow

PRESETS = {
    "english": "RAG eliminates hallucinations.",
    "thai": "การใช้ RAG ทำให้ AI ไม่หลอนเลย",
}


def _domains(urls: Iterable[str | None]) -> list[str]:
    return sorted({parsed.netloc for url in urls if url and (parsed := urlparse(url)).netloc})


def _fallbacks(result) -> list[str]:
    return sorted(
        {
            item.error
            for item in result.agent_runs
            if item.error and item.error.startswith("fallback=")
        }
    )


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Run one minimal live VeriClaim verification")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--claim", help="Claim to verify")
    group.add_argument("--preset", choices=sorted(PRESETS), default="english")
    args = parser.parse_args()
    claim = args.claim or PRESETS[args.preset]

    # The script is the explicit live boundary: normal settings remain fixture/offline-first.
    settings = Settings(live_retrieval_enabled=True, mock_provider_enabled=False)
    router = ProviderRouter(settings)
    workflow = VerificationWorkflow(settings, router=router)
    result = workflow.verify(VerificationRequest(claim=claim))

    database = Database(settings.database_url)
    database.init()
    database.save(result)
    persisted = database.get(result.run_id)
    source_urls = [source.url for source in result.sources]
    dois = sorted({source.doi for source in result.sources if source.doi})
    providers = sorted(
        {f"{usage.provider}/{usage.actual_model or usage.model}" for usage in result.provider_usage}
    )
    tokens = sum(usage.total_tokens for usage in result.provider_usage)
    llm_calls = sum(1 + len(usage.fallbacks) for usage in result.provider_usage)
    print(f"run_id={result.run_id}")
    print(f"claim={result.original_claim}")
    print(
        f"status={result.status.value} verdict={result.verdict} confidence={result.confidence:.2f}"
    )
    print(f"atomic_claim_count={len(result.atomic_claims)}")
    print(f"sources_retrieved={len(result.sources)} evidence_used={len(result.evidence)}")
    print(
        "support_count="
        f"{len(result.supporting_evidence)} contradict_count={len(result.contradicting_evidence)}"
    )
    print(f"source_domains={','.join(_domains(source_urls)) or 'none'}")
    print(f"dois={','.join(dois) or 'none'}")
    print(f"providers_used={','.join(providers) or 'none'}")
    print(f"llm_call_count={llm_calls} tokens_used={tokens}")
    print(f"latency_ms={sum(usage.latency_ms for usage in result.provider_usage)}")
    print(f"fallbacks={','.join(_fallbacks(result)) or 'none'}")
    print(f"persisted_again={persisted is not None}")
    passed = bool(result.sources and result.evidence and persisted is not None)
    print(f"live_e2e_status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
