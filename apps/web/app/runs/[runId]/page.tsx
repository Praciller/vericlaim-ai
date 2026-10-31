import Link from "next/link";
import { notFound } from "next/navigation";
import { headers } from "next/headers";
import { backendAccessHeaders, backendEndpoint } from "../../lib/backend";
import { ownerAuthorized, privateModeEnabled } from "../../lib/owner-auth";
import { sanitizeRunArtifact } from "../../lib/run-artifact";

type AtomicClaim = { atomic_id: string; text: string };
type Evidence = {
  evidence_id: string; excerpt: string; provenance: string;
  evidence_level: string; direction: string; source_id: string;
};
type Source = {
  source_id: string; title: string; url?: string | null;
  source_type: string; evidence_level: string; provenance: string;
};
type PublicRun = {
  run_id: string; status: string; verdict: string; confidence: number;
  original_claim: string; normalized_claim: string; summary: string;
  issue_code?: string | null; conditions: string[]; limitations: string[];
  atomic_claims: AtomicClaim[]; supporting_evidence: Evidence[];
  contradicting_evidence: Evidence[]; sources: Source[];
};
type GraphNode = {
  node_id: string; kind: "claim" | "atomic_claim" | "evidence" | "source";
  label: string; excerpt?: string | null; source_id?: string | null;
  stance?: string | null; evidence_level?: string | null;
  provenance?: string | null; url?: string | null;
};
type GraphEdge = { edge_id: string; source: string; target: string; relation: "contains" | "has_evidence" | "cited_from" };
type EvidenceGraph = { run_id: string; nodes: GraphNode[]; edges: GraphEdge[] };

// Only the server handles the backend bearer token. It is never serialized to a page.
async function loadJson<T>(path: string): Promise<T | null> {
  const response = await fetch(backendEndpoint(path), {
    cache: "no-store",
    headers: backendAccessHeaders(),
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Run artifact request failed (${response.status})`);
  return response.json() as Promise<T>;
}

export default async function RunArtifactPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  const requestHeaders = await headers();
  if (privateModeEnabled() && !ownerAuthorized(requestHeaders.get("authorization"))) notFound();
  if (!/^[A-Za-z0-9_-]{8,128}$/.test(runId)) notFound();
  const encoded = encodeURIComponent(runId);
  const [rawRun, graph] = await Promise.all([
    loadJson<Record<string, unknown> & { run_id: string }>(`/api/v1/runs/${encoded}`),
    loadJson<EvidenceGraph>(`/api/v1/runs/${encoded}/evidence-graph`),
  ]);
  if (!rawRun) notFound();
  const run = sanitizeRunArtifact(rawRun) as unknown as PublicRun;
  const atomics = graph?.nodes.filter((node) => node.kind === "atomic_claim") ?? [];
  const evidenceNodes = graph?.nodes.filter((node) => node.kind === "evidence") ?? [];
  const sourceNodes = graph?.nodes.filter((node) => node.kind === "source") ?? [];
  const graphNodesById = new Map((graph?.nodes ?? []).map((node) => [node.node_id, node]));
  const evidenceByAtomicId = new Map<string, GraphNode[]>();
  const sourceNodesById = new Map(sourceNodes.map((node) => [node.node_id, node]));
  const citedSourceByEvidenceId = new Map<string, GraphNode>();
  for (const edge of graph?.edges ?? []) {
    if (edge.relation === "has_evidence") {
      const evidence = graphNodesById.get(edge.target);
      if (evidence) {
        const linkedEvidence = evidenceByAtomicId.get(edge.source) ?? [];
        linkedEvidence.push(evidence);
        evidenceByAtomicId.set(edge.source, linkedEvidence);
      }
    }
    if (edge.relation === "cited_from") {
      const source = sourceNodesById.get(edge.target);
      if (source) citedSourceByEvidenceId.set(edge.source, source);
    }
  }

  return <main className="shell"><div className="container">
    <header className="hero artifact-hero">
      <div className="eyebrow">Persisted / sanitized / read only</div>
      <h1>Run artifact.<br /><span>{run.verdict}</span></h1>
      <p className="lede">This page loads stored run evidence. Opening it does not rerun retrieval or inference. Owner-only deployments require authentication to open this link.</p>
      <div className="hero-tags"><span>{run.status}</span><span>{run.run_id}</span><span>{Math.round(run.confidence * 100)}% run confidence</span></div>
    </header>
    <section className="panel result-panel">
      <div className="section-heading"><div><span className="eyebrow">Claim</span><h2>{run.original_claim}</h2></div><Link href="/">Start another run</Link></div>
      {run.issue_code ? <div className="degraded-banner"><strong>{run.issue_code}</strong><span>This operational limitation was persisted with the run.</span></div> : null}
      <p className="lede result-summary">{run.summary}</p>
      <div className="grid overview-grid">
        <div className="card"><h3>Atomic claims</h3><ul>{run.atomic_claims.map((item) => <li key={item.atomic_id}>{item.text}</li>)}</ul></div>
        <div className="card"><h3>Conditions</h3><ul>{(run.conditions.length ? run.conditions : ["No additional conditions recorded."]).map((item) => <li key={item}>{item}</li>)}</ul></div>
        <div className="card"><h3>Limitations</h3><ul>{(run.limitations.length ? run.limitations : ["No additional limitations recorded."]).map((item) => <li key={item}>{item}</li>)}</ul></div>
      </div>
      <section className="card graph-card" aria-labelledby="artifact-graph-title">
        <div className="section-heading"><div><span className="eyebrow">Persisted projection</span><h2 id="artifact-graph-title">Evidence graph</h2></div><span className="graph-count">{atomics.length} claims / {evidenceNodes.length} evidence / {sourceNodes.length} sources</span></div>
        <div className="grid artifact-graph-grid">
          <div><h3>Atomic claims</h3>{atomics.map((node) => {
            const linkedEvidence = evidenceByAtomicId.get(node.node_id) ?? [];
            return <article id={node.node_id} className="graph-node atomic-node" key={node.node_id}><strong>{node.label}</strong>{linkedEvidence.length ? <ul className="graph-link-list">{linkedEvidence.map((item) => <li key={item.node_id}><a href={`#${item.node_id}`}>{item.label}</a></li>)}</ul> : <span>No linked evidence</span>}</article>;
          })}</div>
          <div><h3>Evidence</h3>{evidenceNodes.map((node) => {
            const source = citedSourceByEvidenceId.get(node.node_id);
            return <article id={node.node_id} className="graph-node evidence-node" key={node.node_id}><strong>{node.label}</strong><p>{node.excerpt ?? "No excerpt was stored for this evidence node."}</p><span>{node.stance ?? "UNASSESSED"} / {node.evidence_level ?? "level not recorded"} / {node.provenance ?? "provenance not recorded"}</span><p className="source-citation">Cited source: {source ? <a href={`#${source.node_id}`}>{source.label}</a> : "No source link recorded"}</p></article>;
          })}</div>
          <div><h3>Sources</h3>{sourceNodes.map((node) => <article id={node.node_id} className="graph-node source-node" key={node.node_id}><strong>{node.label}</strong><span>{node.evidence_level ?? "level not recorded"}</span><span>{node.provenance ?? "provenance not recorded"}</span>{node.url?.startsWith("http") ? <a href={node.url} target="_blank" rel="noreferrer">Open source</a> : null}</article>)}</div>
        </div>
      </section>
      <div className="grid evidence-grid">
        <div className="card"><h3>Supporting evidence</h3>{run.supporting_evidence.length ? run.supporting_evidence.map((item) => <div className="source" key={item.evidence_id}><p>{item.excerpt}</p><small>{item.provenance} / {item.evidence_level}</small></div>) : <p>No supporting evidence was cited.</p>}</div>
        <div className="card"><h3>Contradicting evidence</h3>{run.contradicting_evidence.length ? run.contradicting_evidence.map((item) => <div className="source" key={item.evidence_id}><p>{item.excerpt}</p><small>{item.provenance} / {item.evidence_level}</small></div>) : <p>No contradicting evidence was cited.</p>}</div>
        <div className="card"><h3>Sources</h3>{run.sources.length ? run.sources.map((source) => <article className="source" key={source.source_id}><strong>{source.title}</strong><small>{source.source_type} / {source.evidence_level} / {source.provenance}</small>{source.url?.startsWith("http") ? <a href={source.url} target="_blank" rel="noreferrer">Open source</a> : null}</article>) : <p>No source records were retained.</p>}</div>
      </div>
      <p className="meta claim-metadata">Normalized claim: {run.normalized_claim}</p>
    </section>
  </div></main>;
}
