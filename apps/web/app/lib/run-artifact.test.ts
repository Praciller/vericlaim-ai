import test from "node:test";
import assert from "node:assert/strict";
import { FIXTURE_DEMO_CLAIM, runArtifactHref, sanitizeRunArtifact, sanitizeVerificationResponse } from "./run-artifact.ts";

const unsafeRun = {
  run_id: "run-123", status: "COMPLETED", original_claim: "RAG eliminates hallucinations.",
  normalized_claim: "RAG eliminates hallucinations.", verdict: "REFUTED", confidence: 0.9,
  summary: "Evidence contradicts the absolute claim.", conditions: ["Fixture evidence only."],
  limitations: ["Bounded demo."], atomic_claims: [{ atomic_id: "a1", text: "RAG eliminates hallucinations." }],
  evidence: [{ internal: "raw evidence aggregate" }], supporting_evidence: [], contradicting_evidence: [], sources: [],
  assessments: [{ rationale: "internal assessment rationale" }],
  agent_runs: [{ provider: "secret-provider", error: "internal detail" }],
  provider_usage: [{ provider: "secret-provider", total_tokens: 99 }],
};

test("fixture demo uses the deterministic claim", () => {
  assert.equal(FIXTURE_DEMO_CLAIM, "RAG eliminates hallucinations.");
});

test("shareable paths encode run ids", () => {
  assert.equal(runArtifactHref("run/a b"), "/runs/run%2Fa%20b");
});

test("public artifacts drop provider and internal trace fields", () => {
  const artifact = sanitizeRunArtifact(unsafeRun);
  assert.equal(Object.hasOwn(artifact, "evidence"), false);
  assert.equal(Object.hasOwn(artifact, "assessments"), false);
  assert.equal(Object.hasOwn(artifact, "agent_runs"), false);
  assert.equal(Object.hasOwn(artifact, "provider_usage"), false);
  assert.equal(JSON.stringify(artifact).includes("secret-provider"), false);
  assert.equal(artifact.run_id, "run-123");
});

test("browser verification responses use the public allowlist", () => {
  const response = sanitizeVerificationResponse(unsafeRun);
  assert.ok(response);
  assert.equal(response.run_id, "run-123");
  assert.equal(Object.hasOwn(response, "agent_runs"), false);
  assert.equal(Object.hasOwn(response, "provider_usage"), false);
  assert.equal(Object.hasOwn(response, "assessments"), false);
  assert.equal(Object.hasOwn(response, "evidence"), false);
  assert.equal(JSON.stringify(response).includes("secret-provider"), false);
  assert.equal(JSON.stringify(response).includes("99"), false);
});

test("browser verification sanitizer rejects responses without a run id", () => {
  assert.equal(sanitizeVerificationResponse({ provider_usage: [{ total_tokens: 99 }] }), null);
  assert.equal(sanitizeVerificationResponse(null), null);
});
