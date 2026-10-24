import { privateModeConfigured, privateModeEnabled } from "./owner-auth";
import { trustedOwnerBearerBackend } from "./backend-origin";

export function backendEndpoint(path: string): string {
  const base = process.env.VERICLAIM_BACKEND_URL ?? process.env.INTERNAL_API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  const url = new URL(path, base);
  // The sole HTTP exception outside loopback is our isolated Compose
  // service. Never allow arbitrary internal or remote HTTP hosts.
  const composeService = process.env.INTERNAL_API_URL === "http://api:8000"
    && url.hostname === "api" && url.port === "8000";
  const allowed = url.protocol === "https:" || (url.protocol === "http:" && (
    ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) || composeService
  ));
  if (!allowed) throw new Error("Backend must use HTTPS, loopback, or the configured Compose service");
  if (process.env.VERICLAIM_API_ACCESS_TOKEN && !trustedOwnerBearerBackend(url)) {
    throw new Error("Refusing to transmit an owner API bearer to an untrusted backend origin");
  }
  return url.toString();
}

export function backendAccessHeaders(): Record<string, string> {
  const token = process.env.VERICLAIM_API_ACCESS_TOKEN;
  if (privateModeEnabled()) {
    if (!privateModeConfigured()) throw new Error("Owner-only deployment is not configured");
    return { Authorization: `Bearer ${token}` };
  }
  if (token) throw new Error("Refusing to expose an authenticated backend without owner-only protection");
  return {};
}
