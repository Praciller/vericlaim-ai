"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

type EvidenceStance = "SUPPORTS" | "CONTRADICTS" | "NEUTRAL";
type Evidence = {
  evidence_id: string;
  excerpt: string;
  source_id: string;
  provenance: string;
  evidence_level: string;
  direction: string;
};
type Assessment = {
  evidence_id: string;
  relevance: number;
  directness: number;
  source_quality: number;
  recency: number;
  temporal_compatibility: number;
  scope_compatibility: number;
  reproducibility_signal: number;
  stance: EvidenceStance;
  extraction_confidence: number;
  rationale: string;
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
  evidence: Evidence[];
  assessments: Assessment[];
  supporting_evidence: Evidence[];
  contradicting_evidence: Evidence[];
  sources: Source[];
  agent_runs: { agent_name: string; task: string; provider: string; model: string; status: string; error?: string | null }[];
  provider_usage: { provider: string; configured_model?: string | null; actual_model?: string | null; task: string; total_tokens: number; latency_ms: number; fallbacks: string[] }[];
};
type ProviderStatus = {
  name: string;
  configured: boolean;
  enabled: boolean;
  model: string;
  supports_structured_output: boolean;
  note?: string | null;
  last_status?: string | null;
  quota_remaining_tokens?: number | null;
  quota_limit_tokens?: number | null;
};
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

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
  PROVIDER_TIMEOUT: "The inference provider timed out. Inspect the trace before relying on the advisory output.",
  PROVIDER_AUTHENTICATION: "The configured inference credential was rejected. No credential details are shown here.",
  PROVIDER_RESPONSE_INVALID: "The inference provider returned an incomplete or invalid response.",
  RETRIEVAL_UNAVAILABLE: "One or more evidence sources were unavailable; inspect the retained provenance.",
};

function issueDescription(code: string | null | undefined): string {
  return code ? issueDescriptions[code] ?? "This run completed with a recorded operational limitation." : "";
}

function formatQuota(value: number | null | undefined): string {
  return value === null || value === undefined ? "not reported" : value.toLocaleString();
}

function GraphColumn({ title, eyebrow, children }: { title: string; eyebrow: string; children: React.ReactNode }) {
  return <div className="graph-column"><div className="graph-column-head"><span className="eyebrow">{eyebrow}</span><h3>{title}</h3></div>{children}</div>;
}

function EvidenceGraphView({ graph }: { graph: EvidenceGraph }) {
  const atomics = graph.nodes.filter((node) => node.kind === "atomic_claim");
  const evidence = graph.nodes.filter((node) => node.kind === "evidence");
  const sources = graph.nodes.filter((node) => node.kind === "source");
  const evidenceByAtomic = new Map<string, GraphNode[]>();
  graph.edges.filter((edge) => edge.relation === "has_evidence").forEach((edge) => {
    const items = evidenceByAtomic.get(edge.source) ?? [];
    const item = evidence.find((node) => node.node_id === edge.target);
    if (item) items.push(item);
    evidenceByAtomic.set(edge.source, items);
  });

  return <section className="card graph-card" aria-labelledby="evidence-graph-title">
    <div className="section-heading"><div><span className="eyebrow">Traceable structure</span><h2 id="evidence-graph-title">Evidence graph</h2></div><span className="graph-count">{atomics.length} claims · {evidence.length} evidence · {sources.length} sources</span></div>
    <p className="helper">Follow each atomic claim to the evidence retrieved in this run and then to its stored source. The graph is a projection of persisted run data, not a separate research result.</p>
    <div className="graph-layout">
      <GraphColumn title="Atomic claims" eyebrow="01 · claim scope">
        {atomics.length ? atomics.map((node) => <article className="graph-node atomic-node" key={node.node_id}><strong>{node.label}</strong><span>{(evidenceByAtomic.get(node.node_id) ?? []).length} linked evidence item{(evidenceByAtomic.get(node.node_id) ?? []).length === 1 ? "" : "s"}</span></article>) : <p className="empty">No atomic claims were recorded.</p>}
      </GraphColumn>
      <div className="graph-connector" aria-hidden="true">→</div>
      <GraphColumn title="Evidence" eyebrow="02 · inspect excerpts">
        {evidence.length ? evidence.map((node) => <details className={`graph-node evidence-node ${node.stance === "CONTRADICTS" ? "is-counter" : node.stance === "SUPPORTS" ? "is-support" : "is-neutral"}`} key={node.node_id}>
          <summary><strong>{node.label}</strong><span className="status-chip">{node.stance ?? "UNASSESSED"}</span></summary>
          <p>{node.excerpt}</p>
          <dl className="compact-dl"><div><dt>Direction</dt><dd>{node.direction ?? "not recorded"}</dd></div><div><dt>Level</dt><dd>{node.evidence_level ?? "not recorded"}</dd></div><div><dt>Provenance</dt><dd>{node.provenance ?? "not recorded"}</dd></div></dl>
        </details>) : <p className="empty">No evidence was cited.</p>}
      </GraphColumn>
      <div className="graph-connector" aria-hidden="true">→</div>
      <GraphColumn title="Sources" eyebrow="03 · verify origin">
        {sources.length ? sources.map((node) => <article className="graph-node source-node" key={node.node_id}><strong>{node.label}</strong><span>{node.evidence_level ?? "source"}</span>{node.url?.startsWith("http") ? <a href={node.url} target="_blank" rel="noreferrer">Open source</a> : null}</article>) : <p className="empty">No source records were retained.</p>}
      </GraphColumn>
    </div>
    <div className="graph-legend" aria-label="Evidence graph legend"><span><i className="legend-dot support" /> Supports</span><span><i className="legend-dot counter" /> Contradicts</span><span><i className="legend-dot neutral" /> Neutral or unassessed</span></div>
  </section>;
}

function ProviderPanel({ providers, isLoading }: { providers: ProviderStatus[]; isLoading: boolean }) {
  return <section className="panel provider-panel" aria-labelledby="provider-status-title">
    <div className="section-heading"><div><span className="eyebrow">Operational view</span><h2 id="provider-status-title">Inference providers</h2></div><span className="status-chip">{isLoading ? "Checking" : `${providers.filter((item) => item.enabled).length} enabled`}</span></div>
    <p className="helper">Safe metadata only: configuration, model label, last status, and provider-reported quota. Keys and response bodies never appear here.</p>
    {providers.length ? <div className="provider-grid">{providers.map((provider) => <article className={`provider-row ${provider.enabled ? "is-enabled" : "is-disabled"}`} key={provider.name}><div><strong>{provider.name}</strong><span>{provider.model}</span></div><div className="provider-meta"><span>{provider.enabled ? "enabled" : provider.configured ? "configured, disabled" : "not configured"}</span><span>{provider.last_status ?? "not checked"}</span><span>quota {formatQuota(provider.quota_remaining_tokens)}</span></div></article>)}</div> : <p className="empty">Provider status is unavailable.</p>}
  </section>;
}

export default function Home() {
  const [claim, setClaim] = useState("");
  const errorSummaryRef = useRef<HTMLDivElement>(null);
  const providersQuery = useQuery<ProviderStatus[]>({
    queryKey: ["provider-status"],
    queryFn: async () => parseApiResponse<ProviderStatus[]>(await fetch(`${apiUrl}/api/v1/providers/status`)),
    staleTime: 30_000,
    retry: false,
  });
  const mutation = useMutation<Result, Error>({
    mutationFn: async (): Promise<Result> => parseApiResponse<Result>(await fetch(`${apiUrl}/api/v1/claims/verify`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim }) })),
  });
  const graphQuery = useQuery<EvidenceGraph>({
    queryKey: ["evidence-graph", mutation.data?.run_id],
    queryFn: async () => parseApiResponse<EvidenceGraph>(await fetch(`${apiUrl}/api/v1/runs/${mutation.data?.run_id}/evidence-graph`)),
    enabled: Boolean(mutation.data?.run_id),
    staleTime: Infinity,
    retry: false,
  });
  const claimTooLong = claim.length > 2000;

  useEffect(() => {
    if (mutation.isError) errorSummaryRef.current?.focus();
  }, [mutation.isError]);

  return <main className="shell"><div className="container">
    <header className="hero"><div className="eyebrow">Evidence-driven claim verification</div><h1>Verify the claim.<br /><span>Inspect the evidence.</span></h1><p className="lede">VeriClaim AI decomposes technical claims, searches for support and counter-evidence, audits provenance, and reports uncertainty instead of reducing a research question to a model vote.</p><div className="hero-tags"><span>Bounded agents</span><span>Stored provenance</span><span>Deterministic validation</span></div></header>
    <section className="panel verify-panel" aria-labelledby="claim-form-title">
      <div className="section-heading"><div><span className="eyebrow">Start a run</span><h2 id="claim-form-title">Claim to verify</h2></div><span className="char-count">{claim.length}/2000</span></div>
      <label htmlFor="claim">Technical or scientific claim</label>
      <textarea id="claim" value={claim} onChange={(event) => setClaim(event.target.value)} aria-describedby={claimTooLong ? "claim-error" : "claim-help"} aria-invalid={claimTooLong} placeholder="Example: RAG eliminates hallucinations." />
      {claimTooLong ? <p id="claim-error" className="field-error" role="alert">Keep the claim under 2,000 characters.</p> : <p id="claim-help" className="helper">Preserved as entered; only whitespace is normalized for analysis.</p>}
      <div className="actions"><button disabled={!claim.trim() || claimTooLong || mutation.isPending} onClick={() => mutation.mutate()}>{mutation.isPending ? "Verifying…" : "Verify claim"}</button><span className="action-hint">Offline fixture mode is safe for local demos.</span></div>
      {mutation.isError ? <div className="error-summary" ref={errorSummaryRef} role="alert" tabIndex={-1} aria-labelledby="error-title"><h2 id="error-title">Verification needs attention</h2><p><strong>{mutation.error instanceof ApiError ? mutation.error.code : "REQUEST_FAILED"}</strong> · {mutation.error.message}</p><button className="secondary-button" type="button" onClick={() => mutation.mutate()}>Try again</button></div> : null}
    </section>
    <ProviderPanel providers={providersQuery.data ?? []} isLoading={providersQuery.isLoading} />
    {mutation.data && <section className="panel result-panel" aria-live="polite">
      <div className="result-head"><div><div className="eyebrow">Run {mutation.data.run_id}</div><div className="verdict">{mutation.data.verdict}</div></div><div className="result-meta"><span className={`run-status ${mutation.data.status.toLowerCase()}`}>{mutation.data.status}</span><span>Confidence in this run: {(mutation.data.confidence * 100).toFixed(0)}%</span></div></div>
      {mutation.data.issue_code ? <div className="degraded-banner" role="status"><strong>{mutation.data.issue_code}</strong><span>{issueDescription(mutation.data.issue_code)}</span></div> : null}
      <p className="lede result-summary">{mutation.data.summary}</p>
      <div className="grid overview-grid"><div className="card"><h3>Atomic claims</h3><ul>{mutation.data.atomic_claims.map((item) => <li key={item.atomic_id}>{item.text}</li>)}</ul></div><div className="card"><h3>Conditions</h3><ul>{(mutation.data.conditions.length ? mutation.data.conditions : ["No additional conditions recorded."]).map((item) => <li key={item}>{item}</li>)}</ul></div><div className="card"><h3>Limitations</h3><ul>{(mutation.data.limitations.length ? mutation.data.limitations : ["No additional limitations recorded."]).map((item) => <li key={item}>{item}</li>)}</ul></div></div>
      {graphQuery.data ? <EvidenceGraphView graph={graphQuery.data} /> : <section className="card graph-loading" aria-live="polite">{graphQuery.isLoading ? "Loading persisted evidence graph…" : "Evidence graph could not be loaded; the run result remains available above."}</section>}
      <div className="grid evidence-grid"><div className="card"><h3>Supporting evidence</h3>{mutation.data.supporting_evidence.length ? mutation.data.supporting_evidence.map((item) => <div className="source" key={item.evidence_id}><p>{item.excerpt}</p><small>{item.provenance} · {item.evidence_level}</small></div>) : <p>No supporting evidence was cited.</p>}</div><div className="card"><h3>Contradicting evidence</h3>{mutation.data.contradicting_evidence.length ? mutation.data.contradicting_evidence.map((item) => <div className="source" key={item.evidence_id}><p>{item.excerpt}</p><small>{item.provenance} · {item.evidence_level}</small></div>) : <p>No contradicting evidence was cited.</p>}</div><div className="card"><h3>Sources</h3>{mutation.data.sources.length ? mutation.data.sources.map((source) => <details className="source" key={source.source_id}><summary>{source.title}</summary><p>{source.abstract ?? "No abstract was stored for this source."}</p><small>{source.source_type} · {source.evidence_level} · {source.provenance}</small>{source.url?.startsWith("http") ? <a href={source.url} target="_blank" rel="noreferrer">Open source</a> : null}</details>) : <p>No source records were retained.</p>}</div></div>
      <details className="card trace-card"><summary>Run trace and provider usage</summary><p className="meta">Status: {mutation.data.status}</p><ul>{mutation.data.agent_runs.map((run) => <li key={`${run.agent_name}-${run.task}`}>{run.agent_name}: {run.provider} / {run.model} ({run.status}){run.error ? ` · ${run.error}` : ""}</li>)}</ul>{mutation.data.provider_usage.length ? <p className="meta">Usage: {mutation.data.provider_usage.map((usage) => `${usage.task}=${usage.provider}/${usage.actual_model ?? usage.configured_model ?? "unknown"}, ${usage.total_tokens} tokens, ${usage.latency_ms}ms`).join(" · ")}</p> : <p className="meta">No live provider usage recorded.</p>}</details>
      <p className="meta claim-metadata">Original: {mutation.data.original_claim}<br />Normalized: {mutation.data.normalized_claim}</p>
    </section>}
  </div></main>;
}
