import assert from "node:assert/strict";
import { test } from "node:test";
import { trustedOwnerBearerBackend } from "./backend-origin.ts";

const vercel = { VERCEL: "1", VERCEL_ENV: "production" };
const local = {};
function permitted(target: string, environment: typeof vercel | typeof local = vercel): boolean {
  return trustedOwnerBearerBackend(new URL(target), environment);
}

test("bearer may reach only the canonical VeriClaim API HTTPS origin in Vercel", () => {
  assert.equal(permitted("https://vericlaim-api.vercel.app/api/v1/claims/verify"), true);
  assert.equal(permitted("https://VERICLAIM-API.VERCEL.APP:443/health"), true);
  for (const target of [
    "https://vericlaim-web.vercel.app/",
    "https://vericlaim-api.vercel.app.evil.example/",
    "https://evil.example/",
    "https://vericlaim-x-pracillatlove-3370s-projects.vercel.app/",
    "https://vericlaim-api.vercel.app:444/",
    "http://vericlaim-api.vercel.app/",
    "https://example.com@evil.example/",
    "https://user:password@vericlaim-api.vercel.app/",
  ]) {
    assert.equal(permitted(target), false, target);
  }
});

test("non-Vercel local loopback and explicit Compose service are supported", () => {
  const standalone = { VERCEL: undefined, VERCEL_ENV: undefined, INTERNAL_API_URL: "http://api:8000" };
  for (const target of [
    "http://localhost:8000/",
    "http://127.0.0.1:8000/",
    "http://[::1]:8000/",
    "http://api:8000/",
  ]) {
    assert.equal(permitted(target, standalone), true, target);
  }
  assert.equal(permitted("http://api:8000/", local), false);
  assert.equal(permitted("http://api.evil.example:8000/", standalone), false);
  assert.equal(permitted("http://internal-other-service:8000/", standalone), false);
});

test("Vercel Preview and Production both reject local URLs with owner bearer", () => {
  for (const runtime of [{ VERCEL: "1", VERCEL_ENV: "preview" }, { VERCEL_ENV: "production" }]) {
    assert.equal(permitted("http://localhost:8000/", runtime), false);
    assert.equal(permitted("http://api:8000/", { ...runtime, INTERNAL_API_URL: "http://api:8000" }), false);
  }
});

test("backend URLs with embedded credentials are always refused", () => {
  const localRuntime = { INTERNAL_API_URL: "http://api:8000" };
  assert.equal(permitted("http://u:p@localhost:8000/", localRuntime), false);
  assert.equal(permitted("http://u:p@api:8000/", localRuntime), false);
});
