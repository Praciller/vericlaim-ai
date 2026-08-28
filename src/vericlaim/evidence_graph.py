from __future__ import annotations

from .domain.models import (
    EvidenceGraphEdge,
    EvidenceGraphNode,
    EvidenceGraphResponse,
    VerificationResult,
)


def build_evidence_graph(result: VerificationResult) -> EvidenceGraphResponse:
    """Project one stored run into an inspectable claim-evidence-source graph."""
    nodes: list[EvidenceGraphNode] = [
        EvidenceGraphNode(
            node_id=f"claim:{result.claim.claim_id}",
            kind="claim",
            label=result.original_claim,
            provenance="stored claim from this verification run",
        )
    ]
    edges: list[EvidenceGraphEdge] = []

    for atomic in result.atomic_claims:
        atomic_node_id = f"atomic_claim:{atomic.atomic_id}"
        nodes.append(
            EvidenceGraphNode(
                node_id=atomic_node_id,
                kind="atomic_claim",
                label=atomic.text,
                provenance="deterministic claim decomposition",
            )
        )
        edges.append(
            EvidenceGraphEdge(
                edge_id=f"contains:{result.claim.claim_id}:{atomic.atomic_id}",
                source=f"claim:{result.claim.claim_id}",
                target=atomic_node_id,
                relation="contains",
            )
        )

    assessments = {item.evidence_id: item for item in result.assessments}
    sources = {item.source_id: item for item in result.sources}
    for evidence in result.evidence:
        evidence_node_id = f"evidence:{evidence.evidence_id}"
        assessment = assessments.get(evidence.evidence_id)
        nodes.append(
            EvidenceGraphNode(
                node_id=evidence_node_id,
                kind="evidence",
                label=f"{evidence.direction.title()} evidence",
                excerpt=evidence.excerpt,
                source_id=evidence.source_id,
                direction=evidence.direction,
                stance=assessment.stance if assessment else None,
                evidence_level=evidence.evidence_level,
                provenance=evidence.provenance,
            )
        )
        edges.append(
            EvidenceGraphEdge(
                edge_id=f"has_evidence:{evidence.atomic_id}:{evidence.evidence_id}",
                source=f"atomic_claim:{evidence.atomic_id}",
                target=evidence_node_id,
                relation="has_evidence",
            )
        )

        source = sources.get(evidence.source_id)
        if source is None:
            continue
        source_node_id = f"source:{source.source_id}"
        if not any(node.node_id == source_node_id for node in nodes):
            nodes.append(
                EvidenceGraphNode(
                    node_id=source_node_id,
                    kind="source",
                    label=source.title,
                    source_id=source.source_id,
                    evidence_level=source.evidence_level,
                    provenance=source.provenance,
                    url=source.url,
                )
            )
        edges.append(
            EvidenceGraphEdge(
                edge_id=f"cited_from:{evidence.evidence_id}:{source.source_id}",
                source=evidence_node_id,
                target=source_node_id,
                relation="cited_from",
            )
        )

    return EvidenceGraphResponse(run_id=result.run_id, nodes=nodes, edges=edges)
