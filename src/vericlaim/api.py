from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import get_settings, secret_value
from .db import Database
from .domain.models import (
    EvidenceGraphResponse,
    HealthResponse,
    ProviderStatus,
    PublicEvidenceResponse,
    PublicRunResponse,
    ReadinessResponse,
    VerificationRequest,
    VerificationResult,
)
from .evidence_graph import build_evidence_graph
from .providers.router import ProviderRouter
from .rate_limit import DatabaseRateLimiter, InMemoryRateLimiter
from .workflow import VerificationWorkflow

settings = get_settings()
logger = logging.getLogger(__name__)
database = Database(settings.database_url, retention_days=settings.run_retention_days)
router = ProviderRouter(settings)
workflow = VerificationWorkflow(settings, router=router)
DEMO_FIXTURE_CLAIM = "RAG eliminates hallucinations."
verify_rate_limiter = (
    DatabaseRateLimiter(
        database.engine,
        max_requests=settings.max_verify_requests_per_minute,
        peer_secret=secret_value(settings.rate_limit_peer_secret) or "",
    )
    if settings.rate_limit_backend == "database"
    else InMemoryRateLimiter(max_requests=settings.max_verify_requests_per_minute)
)

_SAFE_VALIDATION_MESSAGES = frozenset(
    {
        "claim contains control characters",
        "claim must not be blank",
        "claim exceeds the maximum atomic claim limit",
        "claim exceeds the maximum retrieval query limit",
    }
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    database.init()
    yield


app = FastAPI(title="VeriClaim AI API", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip() for origin in settings.cors_allowed_origins.split(",") if origin.strip()
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="vericlaim-api", version=__version__)


@app.get("/ready", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    issues: list[str] = []
    database_status: Literal["ok", "unavailable"] = "ok"
    try:
        database.check()
    except Exception:
        database_status = "unavailable"
        issues.append("database_unavailable")

    enabled_providers = [item.name for item in router.statuses() if item.enabled]
    if not enabled_providers:
        issues.append("no_provider_enabled")

    response = ReadinessResponse(
        status="ready" if not issues else "not_ready",
        service="vericlaim-api",
        version=__version__,
        database=database_status,
        enabled_providers=enabled_providers,
        issues=issues,
    )
    if issues:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "SERVICE_NOT_READY",
                "message": "service readiness checks failed",
                "checks": response.model_dump(mode="json"),
            },
        )
    return response


@app.get("/api/v1/capabilities")
def public_capabilities() -> dict[str, bool | str]:
    """Expose public mode; never expose provider secrets or internal settings."""
    return {"demo_only": settings.public_demo_only, "demo_claim": DEMO_FIXTURE_CLAIM}


@app.post("/api/v1/claims/verify", response_model=VerificationResult)
async def verify_claim(request: VerificationRequest, http_request: Request) -> VerificationResult:
    # Apply the public-demo data collection policy before inference, retrieval,
    # and persistence. This is not a substitute for multi-user authorization.
    if settings.public_demo_only and request.claim != DEMO_FIXTURE_CLAIM:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "DEMO_ONLY",
                "message": "This public demo accepts only the built-in deterministic example.",
            },
        )
    # Trust the direct connection peer, not client-supplied forwarding headers.
    peer = http_request.client.host if http_request.client is not None else "unknown"
    try:
        allowed, retry_after = verify_rate_limiter.allow(peer)
    except Exception as exc:
        # Fail closed if the shared database is unavailable, without exposing
        # peer identity or SQL details to clients or logs.
        logger.error("rate_limiter_unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "RATE_LIMIT_UNAVAILABLE",
                "message": "verification temporarily unavailable",
            },
        ) from None
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"code": "RATE_LIMITED", "message": "too many verification requests"},
            headers={"Retry-After": str(retry_after)},
        )
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                workflow.verify,
                request,
                timeout_seconds=settings.request_timeout_seconds,
            ),
            timeout=settings.request_timeout_seconds + 0.25,
        )
        database.save(result)
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail={"code": "REQUEST_TIMEOUT", "message": "verification timed out safely"},
        ) from None
    except ValueError as exc:
        message = str(exc)
        raise HTTPException(
            status_code=422,
            detail=(
                message
                if message in _SAFE_VALIDATION_MESSAGES
                else "verification request is invalid"
            ),
        ) from None
    except Exception as exc:
        # Do not expose provider prompts or vendor error payloads.
        database_error = getattr(exc, "orig", None)
        diagnostic = getattr(database_error, "diag", None)
        logger.error(
            "verification_failed error_type=%s cause_type=%s database_table=%s "
            "database_constraint=%s",
            type(exc).__name__,
            type(exc.__cause__).__name__ if exc.__cause__ is not None else "none",
            getattr(diagnostic, "table_name", "none"),
            getattr(diagnostic, "constraint_name", "none"),
        )
        raise HTTPException(
            status_code=500, detail="verification failed safely; inspect server logs"
        ) from None


def _public_run_or_404(run_id: str) -> VerificationResult:
    result = database.get(run_id)
    if result is None or (
        settings.public_demo_only and result.original_claim != DEMO_FIXTURE_CLAIM
    ):
        # Do not disclose the existence of historical non-demo runs while
        # operating in publicly accessible demo-only mode.
        raise HTTPException(status_code=404, detail="run not found")
    return result


@app.get("/api/v1/runs/{run_id}", response_model=PublicRunResponse)
def get_run(run_id: str) -> PublicRunResponse:
    return PublicRunResponse.from_result(_public_run_or_404(run_id))


@app.get("/api/v1/runs/{run_id}/evidence", response_model=PublicEvidenceResponse)
def get_evidence(run_id: str) -> PublicEvidenceResponse:
    result = _public_run_or_404(run_id)
    return PublicEvidenceResponse(run_id=result.run_id, evidence=result.evidence)


@app.get("/api/v1/runs/{run_id}/evidence-graph", response_model=EvidenceGraphResponse)
def get_evidence_graph(run_id: str) -> EvidenceGraphResponse:
    return build_evidence_graph(_public_run_or_404(run_id))


@app.get("/api/v1/providers/status", response_model=list[ProviderStatus])
def provider_status() -> list[ProviderStatus]:
    return router.statuses()
