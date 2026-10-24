/** Permit sending owner-only API bearer credentials only to a controlled origin.
 *
 * A valid HTTPS scheme alone is not proof that a backend belongs to VeriClaim.
 * Local loopback / Compose targets are allowed outside Vercel for development.
 */
export function trustedOwnerBearerBackend(
  url: URL,
  runtime: { VERCEL?: string; VERCEL_ENV?: string; INTERNAL_API_URL?: string } = process.env,
): boolean {
  if (url.username || url.password) return false;
  if (url.origin === "https://vericlaim-api.vercel.app") return true;

  // A Vercel function must never send credentials to loopback, Compose, or
  // another team preview's hostname, even if it uses HTTPS.
  if (runtime.VERCEL === "1" || runtime.VERCEL_ENV) return false;
  if (url.protocol !== "http:") return false;
  if (["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) return true;
  return runtime.INTERNAL_API_URL === "http://api:8000" && url.origin === "http://api:8000";
}
