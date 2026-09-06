import Link from "next/link";
import { notFound } from "next/navigation";
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
  label: string; stance?: string | null; provenance?: string | null;
};
type EvidenceGraph = { run_id: string; nodes: GraphNode[]; edges: unknown[] };

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function endpoint(path: string): string {
  const url = new URL(path, apiUrl);
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Unsupported API protocol');
  return url.toString();
}
async function loadJson<T>(path: string): Promise<T | null> {
  const response = await fetch(endpoint(path), { cache: "no-store" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Run artifact request failed (${response.status})`);
  return response.json() as Promise<T>;
}

export default async function RunArtifactPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
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

  return <main className="shell"><div className="container">
    <header className="hero artifact-hero">
      <div className="eyebrow">Persisted / sanitized / read only</div>
      <h1>Run artifact.<br /><span>{run.verdict}</span></h1>
      <p className="lede">This page loads stored run evidence. Opening it does not rerun retrieval or inference.</p>
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
          <div><h3>Atomic claims</h3>{atomics.map((node) => <article className="graph-node atomic-node" key={node.node_id}><strong>{node.label}</strong></article>)}</div>
          <div><h3>Evidence</h3>{evidenceNodes.map((node) => <article className="graph-node evidence-node" key={node.node_id}><strong>{node.label}</strong><span>{node.stance ?? "UNASSESSED"} / {node.provenance ?? "stored provenance"}</span></article>)}</div>
          <div><h3>Sources</h3>{sourceNodes.map((node) => <article className="graph-node source-node" key={node.node_id}><strong>{node.label}</strong><span>{node.provenance ?? "stored provenance"}</span></article>)}</div>
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
