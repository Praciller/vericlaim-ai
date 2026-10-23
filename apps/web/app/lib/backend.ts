import { privateModeConfigured, privateModeEnabled } from "./owner-auth";

export function backendEndpoint(path: string): string {
  const base = process.env.VERICLAIM_BACKEND_URL ?? process.env.INTERNAL_API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  const url = new URL(path, base);
  const allowed = url.protocol === "https:" || (url.protocol === "http:" && ["localhost", "127.0.0.1", "::1"].includes(url.hostname));
  if (!allowed) throw new Error("Backend must use HTTPS or local loopback");
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
