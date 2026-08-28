from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import get_settings
from .db import Database
from .domain.models import (
    EvidenceGraphResponse,
    EvidenceListResponse,
    HealthResponse,
    ProviderStatus,
    ReadinessResponse,
    VerificationRequest,
    VerificationResult,
)
from .evidence_graph import build_evidence_graph
from .providers.router import ProviderRouter
from .workflow import VerificationWorkflow

settings = get_settings()
database = Database(settings.database_url)
router = ProviderRouter(settings)
workflow = VerificationWorkflow(settings, router=router)


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


@app.post("/api/v1/claims/verify", response_model=VerificationResult)
def verify_claim(request: VerificationRequest) -> VerificationResult:
    try:
        result = workflow.verify(request)
        database.save(result)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Do not expose provider prompts or vendor error payloads.
        raise HTTPException(
            status_code=500, detail="verification failed safely; inspect server logs"
        ) from exc


@app.get("/api/v1/runs/{run_id}", response_model=VerificationResult)
def get_run(run_id: str) -> VerificationResult:
    result = database.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    return result


@app.get("/api/v1/runs/{run_id}/evidence", response_model=EvidenceListResponse)
def get_evidence(run_id: str) -> EvidenceListResponse:
    result = database.get_evidence(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    return result


@app.get("/api/v1/runs/{run_id}/evidence-graph", response_model=EvidenceGraphResponse)
def get_evidence_graph(run_id: str) -> EvidenceGraphResponse:
    result = database.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    return build_evidence_graph(result)


@app.get("/api/v1/providers/status", response_model=list[ProviderStatus])
def provider_status() -> list[ProviderStatus]:
    return router.statuses()
