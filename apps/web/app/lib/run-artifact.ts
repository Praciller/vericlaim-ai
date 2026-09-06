export const FIXTURE_DEMO_CLAIM = "RAG eliminates hallucinations.";

export function runArtifactHref(runId: string): string {
  return `/runs/${encodeURIComponent(runId)}`;
}

type RunLike = Record<string, unknown> & { run_id: string };

const PUBLIC_RUN_FIELDS = [
  "run_id", "status", "issue_code", "original_claim", "normalized_claim",
  "verdict", "confidence", "summary", "conditions", "limitations",
  "atomic_claims", "supporting_evidence",
  "contradicting_evidence", "sources",
] as const;

export function sanitizeRunArtifact(run: RunLike): Record<string, unknown> & { run_id: string } {
  const artifact: Record<string, unknown> = {};
  for (const field of PUBLIC_RUN_FIELDS) {
    if (field in run) artifact[field] = run[field];
  }
  return artifact as Record<string, unknown> & { run_id: string };
}
