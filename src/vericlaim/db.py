from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, PrimaryKeyConstraint, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .domain.models import EvidenceListResponse, VerificationResult


class Base(DeclarativeBase):
    pass


class VerificationRunRow(Base):
    __tablename__ = "verification_runs"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    original_claim: Mapped[str] = mapped_column(Text)
    normalized_claim: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    verdict: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column()
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ClaimRow(Base):
    __tablename__ = "claims"
    claim_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.run_id"))
    payload_json: Mapped[str] = mapped_column(Text)


class AtomicClaimRow(Base):
    __tablename__ = "atomic_claims"
    __table_args__ = (PrimaryKeyConstraint("run_id", "atomic_id"),)
    atomic_id: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.run_id"))
    payload_json: Mapped[str] = mapped_column(Text)


class SearchQueryRow(Base):
    __tablename__ = "search_queries"
    __table_args__ = (PrimaryKeyConstraint("run_id", "query_id"),)
    query_id: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.run_id"))
    payload_json: Mapped[str] = mapped_column(Text)


class SourceRow(Base):
    __tablename__ = "sources"
    __table_args__ = (PrimaryKeyConstraint("run_id", "source_id"),)
    source_id: Mapped[str] = mapped_column(String(512))
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.run_id"))
    payload_json: Mapped[str] = mapped_column(Text)


class EvidenceRow(Base):
    __tablename__ = "evidence"
    evidence_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.run_id"))
    payload_json: Mapped[str] = mapped_column(Text)


class EvidenceAssessmentRow(Base):
    __tablename__ = "evidence_assessments"
    assessment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.run_id"))
    payload_json: Mapped[str] = mapped_column(Text)


class VerdictRow(Base):
    __tablename__ = "verdicts"
    verdict_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.run_id"))
    payload_json: Mapped[str] = mapped_column(Text)


class AgentRunRow(Base):
    __tablename__ = "agent_runs"
    agent_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.run_id"))
    payload_json: Mapped[str] = mapped_column(Text)


class ProviderUsageRow(Base):
    __tablename__ = "provider_usage"
    usage_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("verification_runs.run_id"))
    payload_json: Mapped[str] = mapped_column(Text)


class Database:
    def __init__(self, url: str) -> None:
        if url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url.removeprefix("postgres://")
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url.removeprefix("postgresql://")
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args)

    def init(self) -> None:
        if self.engine.url.database and self.engine.url.database != ":memory:":
            Path(self.engine.url.database).parent.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(self.engine)

    def check(self) -> bool:
        """Run a bounded database liveness check for deployment readiness."""
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True

    def save(self, result: VerificationResult) -> None:
        payload = result.model_dump(mode="json")
        with Session(self.engine) as session:
            session.add(
                VerificationRunRow(
                    run_id=result.run_id,
                    original_claim=result.original_claim,
                    normalized_claim=result.normalized_claim,
                    status=result.status.value,
                    verdict=result.verdict.value,
                    confidence=result.confidence,
                    result_json=json.dumps(payload),
                    created_at=result.created_at,
                    completed_at=result.completed_at,
                )
            )
            session.add(
                ClaimRow(
                    claim_id=result.claim.claim_id,
                    run_id=result.run_id,
                    payload_json=json.dumps(result.claim.model_dump(mode="json")),
                )
            )
            for atomic in result.atomic_claims:
                session.add(
                    AtomicClaimRow(
                        atomic_id=atomic.atomic_id,
                        run_id=result.run_id,
                        payload_json=json.dumps(atomic.model_dump(mode="json")),
                    )
                )
            for query in result.queries:
                session.add(
                    SearchQueryRow(
                        query_id=query.query_id,
                        run_id=result.run_id,
                        payload_json=json.dumps(query.model_dump(mode="json")),
                    )
                )
            for source in result.sources:
                session.add(
                    SourceRow(
                        source_id=source.source_id,
                        run_id=result.run_id,
                        payload_json=json.dumps(source.model_dump(mode="json")),
                    )
                )
            for evidence in result.evidence:
                session.add(
                    EvidenceRow(
                        evidence_id=evidence.evidence_id,
                        run_id=result.run_id,
                        payload_json=json.dumps(evidence.model_dump(mode="json")),
                    )
                )
            for assessment in result.assessments:
                session.add(
                    EvidenceAssessmentRow(
                        assessment_id=f"{result.run_id}:{assessment.evidence_id}",
                        run_id=result.run_id,
                        payload_json=json.dumps(assessment.model_dump(mode="json")),
                    )
                )
            session.add(
                VerdictRow(
                    verdict_id=result.run_id,
                    run_id=result.run_id,
                    payload_json=json.dumps(result.verdict_details.model_dump(mode="json")),
                )
            )
            for index, agent_run in enumerate(result.agent_runs):
                session.add(
                    AgentRunRow(
                        agent_run_id=f"{result.run_id}:{index}",
                        run_id=result.run_id,
                        payload_json=json.dumps(agent_run.model_dump(mode="json")),
                    )
                )
            for index, usage in enumerate(result.provider_usage):
                session.add(
                    ProviderUsageRow(
                        usage_id=f"{result.run_id}:{index}",
                        run_id=result.run_id,
                        payload_json=json.dumps(usage.model_dump(mode="json")),
                    )
                )
            session.commit()

    def get(self, run_id: str) -> VerificationResult | None:
        with Session(self.engine) as session:
            row = session.get(VerificationRunRow, run_id)
            return VerificationResult.model_validate(json.loads(row.result_json)) if row else None

    def get_evidence(self, run_id: str) -> EvidenceListResponse | None:
        result = self.get(run_id)
        if result is None:
            return None
        return EvidenceListResponse(
            run_id=run_id, evidence=result.evidence, assessments=result.assessments
        )
