import { createHash, timingSafeEqual } from "node:crypto";

export function privateModeEnabled(): boolean {
  return process.env.VERICLAIM_PRIVATE_MODE === "true";
}

export function privateModeConfigured(): boolean {
  const user = process.env.VERICLAIM_OWNER_USER;
  const password = process.env.VERICLAIM_OWNER_PASSWORD;
  const token = process.env.VERICLAIM_API_ACCESS_TOKEN;
  return Boolean(user && /^[A-Za-z0-9._-]{1,64}$/.test(user) && password && password.length >= 24 && token && token.length >= 32);
}

function safeMatch(a: string, b: string): boolean {
  const first = createHash("sha256").update(a, "utf8").digest();
  const second = createHash("sha256").update(b, "utf8").digest();
  return timingSafeEqual(first, second);
}

export function ownerAuthorized(authorization: string | null): boolean {
  if (!privateModeEnabled() || !privateModeConfigured()) return false;
  if (!authorization || authorization.length > 1024 || !/^Basic [A-Za-z0-9+/=]+$/.test(authorization)) return false;
  const credentials = Buffer.from(authorization.slice(6), "base64").toString("utf8");
  const separator = credentials.indexOf(":");
  if (separator <= 0) return false;
  const username = credentials.slice(0, separator);
  const password = credentials.slice(separator + 1);
  const expectedUser = process.env.VERICLAIM_OWNER_USER ?? "";
  const expectedPassword = process.env.VERICLAIM_OWNER_PASSWORD ?? "";
  const userOk = safeMatch(username, expectedUser);
  const passwordOk = safeMatch(password, expectedPassword);
  return userOk && passwordOk;
}

export function accessDenied(): Response {
  return new Response("Owner access required", {
    status: 401,
    headers: {
      "WWW-Authenticate": 'Basic realm="VeriClaim AI Owner", charset="UTF-8"',
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
