"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { FIXTURE_DEMO_CLAIM, runArtifactHref } from "./lib/run-artifact";

type EvidenceStance = "SUPPORTS" | "CONTRADICTS" | "NEUTRAL";
type Evidence = {
  evidence_id: string;
  excerpt: string;
  source_id: string;
  provenance: string;
  evidence_level: string;
  direction: string;
};
type Source = {
  source_id: string;
  title: string;
  url?: string | null;
  source_type: string;
  evidence_level: string;
  authors: string[];
  published_at?: string | null;
  abstract?: string | null;
  provenance: string;
};
type Result = {
  run_id: string;
  status: string;
  issue_code?: string | null;
  original_claim: string;
  normalized_claim: string;
  verdict: string;
  confidence: number;
  summary: string;
  conditions: string[];
  limitations: string[];
  atomic_claims: { atomic_id: string; text: string }[];
  supporting_evidence: Evidence[];
  contradicting_evidence: Evidence[];
  sources: Source[];
};
type PublicCapabilities = { demo_only: boolean; demo_claim: string };
type GraphNode = {
  node_id: string;
  kind: "claim" | "atomic_claim" | "evidence" | "source";
  label: string;
  excerpt?: string | null;
  source_id?: string | null;
  direction?: string | null;
  stance?: EvidenceStance | null;
  evidence_level?: string | null;
  provenance?: string | null;
  url?: string | null;
};
type GraphEdge = { edge_id: string; source: string; target: string; relation: "contains" | "has_evidence" | "cited_from" };
type EvidenceGraph = { run_id: string; nodes: GraphNode[]; edges: GraphEdge[] };

// All browser API calls pass through the same-origin Next.js server, which
// can hold a server-only bearer token in owner mode. Never put it in NEXT_PUBLIC_*.
function apiEndpoint(path: string): string {
  return `/api/backend${path}`;
}

class ApiError extends Error {
  code: string;

  constructor(message: string, code: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
  }
}

async function parseApiResponse<T>(response: Response): Promise<T> {
  if (response.ok) return response.json() as Promise<T>;
  const payload = await response.json().catch(() => null) as { detail?: unknown } | null;
  const detail = payload?.detail;
  if (detail && typeof detail === "object") {
    const structured = detail as { code?: unknown; message?: unknown };
    throw new ApiError(
      typeof structured.message === "string" ? structured.message : "The service could not complete this request.",
      typeof structured.code === "string" ? structured.code : `HTTP_${response.status}`,
    );
  }
  throw new ApiError(
    typeof detail === "string" ? detail : "The service could not complete this request.",
    response.status === 422 ? "INVALID_REQUEST" : `HTTP_${response.status}`,
  );
}

const issueDescriptions: Record<string, string> = {
  PROVIDER_UNAVAILABLE: "The configured inference provider is unavailable. This result used a bounded deterministic path.",
  QUOTA_EXHAUSTED: "The inference provider reported exhausted quota. No provider switch was attempted.",
  PROVIDER_RATE_LIMIT: "The inference provider rate-limited this run. The result is marked degraded.",
  PROVIDER_TIMEOUT: "The inference provider timed out. Review the recorded limitations before relying on the advisory output.",
  PROVIDER_AUTHENTICATION: "The configured inference credential was rejected. No credential details are shown here.",
  PROVIDER_RESPONSE_INVALID: "The inference provider returned an incomplete or invalid response.",
  RETRIEVAL_UNAVAILABLE: "One or more evidence sources were unavailable; review the retained source provenance.",
};

function issueDescription(code: string | null | undefined): string {
  return code ? issueDescriptions[code] ?? "This run completed with a recorded operational limitation." : "";
}

function GraphColumn({ title, eyebrow, children }: { title: string; eyebrow: string; children: React.ReactNode }) {
  return <div className="graph-column"><div className="graph-column-head"><span className="eyebrow">{eyebrow}</span><h3>{title}</h3></div>{children}</div>;
}

function EvidenceGraphView({ graph }: { graph: EvidenceGraph }) {
  const atomics = graph.nodes.filter((node) => node.kind === "atomic_claim");
  const evidence = graph.nodes.filter((node) => node.kind === "evidence");
  const sources = graph.nodes.filter((node) => node.kind === "source");
  const evidenceByAtomic = new Map<string, GraphNode[]>();
  const sourcesByNodeId = new Map(sources.map((node) => [node.node_id, node]));
  const sourceByEvidenceId = new Map<string, GraphNode>();
  graph.edges.filter((edge) => edge.relation === "has_evidence").forEach((edge) => {
    const items = evidenceByAtomic.get(edge.source) ?? [];
    const item = evidence.find((node) => node.node_id === edge.target);
    if (item) items.push(item);
    evidenceByAtomic.set(edge.source, items);
  });
  graph.edges.filter((edge) => edge.relation === "cited_from").forEach((edge) => {
    const source = sourcesByNodeId.get(edge.target);
    if (source) sourceByEvidenceId.set(edge.source, source);
  });

  return <section className="card graph-card" aria-labelledby="evidence-graph-title">
    <div className="section-heading"><div><span className="eyebrow">Traceable structure</span><h2 id="evidence-graph-title">Evidence graph</h2></div><span className="graph-count">{atomics.length} claims · {evidence.length} evidence · {sources.length} sources</span></div>
    <p className="helper">Follow each atomic claim to the evidence retrieved in this run and then to its stored source. The graph is a projection of persisted run data, not a separate research result.</p>
    <div className="graph-layout">
      <GraphColumn title="Atomic claims" eyebrow="01 · claim scope">
        {atomics.length ? atomics.map((node) => {
          const linkedEvidence = evidenceByAtomic.get(node.node_id) ?? [];
          return <article id={node.node_id} className="graph-node atomic-node" key={node.node_id}><strong>{node.label}</strong><span>{linkedEvidence.length} linked evidence item{linkedEvidence.length === 1 ? "" : "s"}</span>{linkedEvidence.length ? <ul className="graph-link-list">{linkedEvidence.map((item) => <li key={item.node_id}><a href={`#${item.node_id}`}>{item.label}</a></li>)}</ul> : null}</article>;
        }) : <p className="empty">No atomic claims were recorded.</p>}
      </GraphColumn>
      <div className="graph-connector" aria-hidden="true">→</div>
      <GraphColumn title="Evidence" eyebrow="02 · inspect excerpts">
        {evidence.length ? evidence.map((node) => <details id={node.node_id} className={`graph-node evidence-node ${node.stance === "CONTRADICTS" ? "is-counter" : node.stance === "SUPPORTS" ? "is-support" : "is-neutral"}`} key={node.node_id}>
          <summary><strong>{node.label}</strong><span className="status-chip">{node.stance ?? "UNASSESSED"}</span></summary>
          <p>{node.excerpt}</p>
          <dl className="compact-dl"><div><dt>Direction</dt><dd>{node.direction ?? "not recorded"}</dd></div><div><dt>Level</dt><dd>{node.evidence_level ?? "not recorded"}</dd></div><div><dt>Provenance</dt><dd>{node.provenance ?? "not recorded"}</dd></div><div><dt>Cited source</dt><dd>{sourceByEvidenceId.get(node.node_id) ? <a href={`#${sourceByEvidenceId.get(node.node_id)?.node_id}`}>{sourceByEvidenceId.get(node.node_id)?.label}</a> : "No source link recorded"}</dd></div></dl>
        </details>) : <p className="empty">No evidence was cited.</p>}
      </GraphColumn>
      <div className="graph-connector" aria-hidden="true">→</div>
      <GraphColumn title="Sources" eyebrow="03 · verify origin">
        {sources.length ? sources.map((node) => <article id={node.node_id} className="graph-node source-node" key={node.node_id}><strong>{node.label}</strong><span>{node.evidence_level ?? "source"}</span><span>{node.provenance ?? "Provenance not recorded"}</span>{node.url?.startsWith("http") ? <a href={node.url} target="_blank" rel="noreferrer">Open source</a> : null}</article>) : <p className="empty">No source records were retained.</p>}
      </GraphColumn>
    </div>
    <div className="graph-legend" aria-label="Evidence graph legend"><span><i className="legend-dot support" /> Supports</span><span><i className="legend-dot counter" /> Contradicts</span><span><i className="legend-dot neutral" /> Neutral or unassessed</span></div>
  </section>;
}

export default function Home() {
  const [claim, setClaim] = useState("");
  const errorSummaryRef = useRef<HTMLDivElement>(null);
  const capabilitiesQuery = useQuery<PublicCapabilities>({
    queryKey: ["public-capabilities"],
    queryFn: async () => parseApiResponse<PublicCapabilities>(await fetch(apiEndpoint("/api/v1/capabilities"))),
    staleTime: 60_000,
    retry: false,
  });
  const mutation = useMutation<Result, Error, string>({
    mutationFn: async (claimToVerify): Promise<Result> => parseApiResponse<Result>(await fetch(apiEndpoint("/api/v1/claims/verify"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim: claimToVerify }) })),
  });
  const graphQuery = useQuery<EvidenceGraph>({
    queryKey: ["evidence-graph", mutation.data?.run_id],
    queryFn: async () => parseApiResponse<EvidenceGraph>(await fetch(apiEndpoint(`/api/v1/runs/${encodeURIComponent(mutation.data?.run_id ?? "")}/evidence-graph`))),
    enabled: Boolean(mutation.data?.run_id),
    staleTime: Infinity,
    retry: false,
  });
  const claimTooLong = claim.length > 2000;
  // Do not enable arbitrary input until the backend confirms its current mode.
  const demoOnly = capabilitiesQuery.data?.demo_only ?? true;
  const demoClaim = capabilitiesQuery.data?.demo_claim ?? FIXTURE_DEMO_CLAIM;
  const mode = capabilitiesQuery.data
    ? capabilitiesQuery.data.demo_only ? "fixture" : "non-demo"
    : capabilitiesQuery.isError ? "unavailable" : "checking";
  const modeTitle = mode === "fixture"
    ? "Fixture-only public demo"
    : mode === "non-demo"
      ? "Non-demo mode · live AI may be configured"
      : mode === "unavailable" ? "Runtime mode unavailable" : "Checking runtime mode";
  const modeDescription = mode === "fixture"
    ? `This deployment accepts only the built-in example “${demoClaim}”. Public demo settings require deterministic mock inference and disable live retrieval. This walkthrough demonstrates evidence handling, not live AI performance.`
    : mode === "non-demo"
      ? "This deployment accepts custom claims. Live model or retrieval calls depend on server-side configuration and may fall back. Avoid confidential claims."
      : mode === "unavailable"
        ? "Custom claim input stays locked until the API confirms its mode. The demo button sends only the built-in example; provider traces and token usage are not sent to this page."
        : "Custom claim input stays locked while the API mode is checked. The demo button sends only the built-in example.";
  const claimHelpId = mode === "fixture" ? "claim-demo-help" : mode === "non-demo" ? claimTooLong ? "claim-error" : "claim-help" : "claim-mode-help";

  useEffect(() => {
    if (mutation.isError) errorSummaryRef.current?.focus();
  }, [mutation.isError]);

  return <main className="shell"><div className="container">
    <header className="hero"><div className="eyebrow">Evidence-driven claim verification</div><h1>Verify the claim.<br /><span>Inspect the evidence.</span></h1><p className="lede">VeriClaim AI decomposes technical claims, separates support from counter-evidence, tracks source provenance, and reports uncertainty.</p><div className="hero-tags"><span>Bounded workflow</span><span>Stored evidence graph</span><span>Deterministic validation</span></div></header>
    <section className={`mode-banner mode-${mode}`} aria-labelledby="runtime-mode-title" role="status" aria-live="polite">
      <div className="mode-heading"><span className="eyebrow">Current deployment</span><span className="mode-chip">{mode === "fixture" ? "No live AI" : mode === "non-demo" ? "Configuration dependent" : "Input locked"}</span></div>
      <h2 id="runtime-mode-title">{modeTitle}</h2><p>{modeDescription}</p>
    </section>
    <section className="panel walkthrough" aria-labelledby="walkthrough-title">
      <div className="section-heading"><div><span className="eyebrow">For reviewers</span><h2 id="walkthrough-title">Evidence walkthrough</h2></div></div>
      <ol className="walkthrough-list">
        <li><span className="step-number">01</span><h3>Run the built-in example</h3><p>Use the deterministic fixture to see the claim and verdict flow.</p><a href="#claim-form">Go to demo</a></li>
        <li><span className="step-number">02</span><h3>Follow the evidence</h3><p>Compare supporting and contradicting evidence, then follow each cited source and its provenance.</p></li>
        <li><span className="step-number">03</span><h3>Inspect what was saved</h3><p>Open the persisted run artifact to review the graph, source records, and limitations without rerunning it.</p></li>
      </ol>
    </section>
    <section id="claim-form" className="panel verify-panel" aria-labelledby="claim-form-title">
      <div className="section-heading"><div><span className="eyebrow">Start a run</span><h2 id="claim-form-title">Claim to verify</h2></div><span className="char-count">{claim.length}/2000</span></div>
      <label htmlFor="claim">Technical or scientific claim</label>
      <textarea id="claim" value={claim} disabled={demoOnly} onChange={(event) => setClaim(event.target.value)} aria-describedby={claimHelpId} aria-invalid={claimTooLong} placeholder="Example: RAG eliminates hallucinations." />
      {mode === "fixture" ? <p id="claim-demo-help" className="helper" role="status">Public demo mode accepts only the built-in synthetic example. Do not enter confidential claims.</p> : mode === "non-demo" ? claimTooLong ? <p id="claim-error" className="field-error" role="alert">Keep the claim under 2,000 characters.</p> : <p id="claim-help" className="helper">Preserved as entered; only whitespace is normalized for analysis.</p> : <p id="claim-mode-help" className="helper" role="status">Custom claim input stays locked until the API confirms its mode. The built-in example remains available.</p>}
      <div className="actions"><button type="button" disabled={demoOnly || !claim.trim() || claimTooLong || mutation.isPending} onClick={() => mutation.mutate(claim)}>{mutation.isPending ? "Verifying…" : "Verify claim"}</button><button className="secondary-button" type="button" disabled={mutation.isPending} onClick={() => { setClaim(demoClaim); mutation.mutate(demoClaim); }}>{mode === "fixture" ? "Try deterministic demo" : mode === "non-demo" ? "Use example claim" : "Try built-in example"}</button><span className="action-hint">{mode === "fixture" ? "Runs only the built-in fixture; no live provider quota is used." : mode === "non-demo" ? "Provider and retrieval use depend on server-side configuration and availability." : "Only the built-in example can be run until mode is confirmed."}</span></div>
      {mutation.isError ? <div className="error-summary" ref={errorSummaryRef} role="alert" tabIndex={-1} aria-labelledby="error-title"><h2 id="error-title">Verification needs attention</h2><p><strong>{mutation.error instanceof ApiError ? mutation.error.code : "REQUEST_FAILED"}</strong> · {mutation.error.message}</p><button className="secondary-button" type="button" onClick={() => mutation.mutate(claim)}>Try again</button></div> : null}
    </section>
    {mutation.data && <section id="run-result" className="panel result-panel" aria-live="polite" aria-labelledby="run-result-title">
      <div className="result-head"><div><div className="eyebrow">Persisted run {mutation.data.run_id}</div><h2 id="run-result-title" className="verdict">{mutation.data.verdict}</h2></div><div className="result-meta"><span className={`run-status ${mutation.data.status.toLowerCase()}`}>{mutation.data.status}</span><span>Confidence in this run: {(mutation.data.confidence * 100).toFixed(0)}%</span></div></div>
      {mutation.data.issue_code ? <div className="degraded-banner" role="status"><strong>{mutation.data.issue_code}</strong><span>{issueDescription(mutation.data.issue_code)}</span></div> : null}
      <p className="lede result-summary">{mutation.data.summary}</p><p className="artifact-link"><a href={runArtifactHref(mutation.data.run_id)}>{mode === "fixture" ? "Open shareable demo artifact" : "Open persisted run artifact"}</a><span>Loads persisted evidence without rerunning verification.</span></p>
      <div className="grid overview-grid"><div className="card"><h3>Atomic claims</h3><ul>{mutation.data.atomic_claims.map((item) => <li key={item.atomic_id}>{item.text}</li>)}</ul></div><div className="card"><h3>Conditions</h3><ul>{(mutation.data.conditions.length ? mutation.data.conditions : ["No additional conditions recorded."]).map((item) => <li key={item}>{item}</li>)}</ul></div><div className="card"><h3>Limitations</h3><ul>{(mutation.data.limitations.length ? mutation.data.limitations : ["No additional limitations recorded."]).map((item) => <li key={item}>{item}</li>)}</ul></div></div>
      {graphQuery.data ? <EvidenceGraphView graph={graphQuery.data} /> : <section className="card graph-loading" aria-live="polite">{graphQuery.isLoading ? "Loading persisted evidence graph…" : "Evidence graph could not be loaded; the run result remains available above."}</section>}
      <div className="grid evidence-grid"><div className="card"><h3>Supporting evidence</h3>{mutation.data.supporting_evidence.length ? mutation.data.supporting_evidence.map((item) => <div className="source" key={item.evidence_id}><p>{item.excerpt}</p><small>{item.provenance} · {item.evidence_level}</small></div>) : <p>No supporting evidence was cited.</p>}</div><div className="card"><h3>Contradicting evidence</h3>{mutation.data.contradicting_evidence.length ? mutation.data.contradicting_evidence.map((item) => <div className="source" key={item.evidence_id}><p>{item.excerpt}</p><small>{item.provenance} · {item.evidence_level}</small></div>) : <p>No contradicting evidence was cited.</p>}</div><div className="card"><h3>Sources</h3>{mutation.data.sources.length ? mutation.data.sources.map((source) => <details className="source" key={source.source_id}><summary>{source.title}</summary><p>{source.abstract ?? "No abstract was stored for this source."}</p><small>{source.source_type} · {source.evidence_level} · {source.provenance}</small>{source.url?.startsWith("http") ? <a href={source.url} target="_blank" rel="noreferrer">Open source</a> : null}</details>) : <p>No source records were retained.</p>}</div></div>
      <p className="meta claim-metadata">Original: {mutation.data.original_claim}<br />Normalized: {mutation.data.normalized_claim}</p>
    </section>}
  </div></main>;
}
