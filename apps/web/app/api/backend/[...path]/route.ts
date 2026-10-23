import type { NextRequest } from "next/server";
import { backendAccessHeaders, backendEndpoint } from "../../../lib/backend";
import { accessDenied, ownerAuthorized, privateModeConfigured, privateModeEnabled } from "../../../lib/owner-auth";

type Context = { params: Promise<{ path: string[] }> };
const PUBLIC_READS = new Set(["api/v1/capabilities", "api/v1/providers/status"]);
const RUN_READ = /^api\/v1\/runs\/[A-Za-z0-9_-]{8,128}(?:\/(?:evidence|evidence-graph))?$/;

function apiError(status: number, code: string, message: string): Response {
  return Response.json({ detail: { code, message } }, { status, headers: { "Cache-Control": "no-store" } });
}
async function handle(request: NextRequest, context: Context, method: "GET" | "POST"): Promise<Response> {
  const { path } = await context.params;
  const requested = path.join("/");
  const permitted = method === "GET" ? PUBLIC_READS.has(requested) || RUN_READ.test(requested) : requested === "api/v1/claims/verify";
  if (!permitted || request.nextUrl.search) return apiError(404, "NOT_FOUND", "Not found");
  if (privateModeEnabled()) {
    if (!privateModeConfigured()) return apiError(503, "PRIVATE_NOT_CONFIGURED", "Owner access is not configured");
    if (!ownerAuthorized(request.headers.get("authorization"))) return accessDenied();
  } else if (process.env.VERICLAIM_API_ACCESS_TOKEN) {
    return apiError(503, "UNSAFE_CONFIGURATION", "Owner protection must be enabled");
  }
  if (method === "POST") {
    const origin = request.headers.get("origin");
    if (!origin || origin !== request.nextUrl.origin) return apiError(403, "ORIGIN_DENIED", "Same-origin request required");
    const length = Number(request.headers.get("content-length") ?? "0");
    if (!Number.isFinite(length) || length > 8192) return apiError(413, "REQUEST_TOO_LARGE", "Claim request too large");
  }
  try {
    const outboundHeaders: Record<string, string> = { ...backendAccessHeaders() };
    let body: string | undefined;
    if (method === "POST") {
      body = await request.text();
      if (body.length > 8192) return apiError(413, "REQUEST_TOO_LARGE", "Claim request too large");
      outboundHeaders["Content-Type"] = "application/json";
    }
    const upstream = await fetch(backendEndpoint(`/${requested}`), {
      method, headers: outboundHeaders, body, cache: "no-store", redirect: "manual",
      signal: AbortSignal.timeout(35_000),
    });
    const type = upstream.headers.get("content-type") ?? "";
    if (!type.includes("application/json")) return apiError(502, "BACKEND_PROTOCOL", "Invalid backend response");
    const text = await upstream.text();
    if (text.length > 2_000_000) return apiError(502, "BACKEND_RESPONSE_TOO_LARGE", "Backend response too large");
    const headers: Record<string, string> = { "Content-Type": "application/json", "Cache-Control": "no-store" };
    const retryAfter = upstream.headers.get("retry-after");
    if (retryAfter && /^\d{1,4}$/.test(retryAfter)) headers["Retry-After"] = retryAfter;
    return new Response(text, { status: upstream.status, headers });
  } catch {
    return apiError(502, "BACKEND_UNAVAILABLE", "The verification service is unavailable");
  }
}
export async function GET(request: NextRequest, context: Context): Promise<Response> {
  return handle(request, context, "GET");
}
export async function POST(request: NextRequest, context: Context): Promise<Response> {
  return handle(request, context, "POST");
}
