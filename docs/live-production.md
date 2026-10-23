# VeriClaim AI — Owner-only Live Production Cutover

This is the **real-claim** deployment plan; it is separate from the existing
anonymous, deterministic public portfolio demo. It does **not** require
completion of the portable recovery ZIP before release. It also does **not**
authorize deleting old records.

## Current blocker

As of the 2026-10-24 inspection, Vercel API Production has **no** `API_ACCESS_TOKEN`
or live `GEMINI_API_KEY`/`GROQ_API_KEY` entries, and Web has no private owner
credentials or server-side API bearer token. Do not represent the project as a
real-claim service until these are configured and a bounded live smoke passes.

## Release approach: one owner, free tier only

The first release is owner-only, not an anonymous/public multi-tenant service.
The Web enforces HTTP Basic authentication on all dynamic pages and API routes
in private mode. Server-side route handlers forward allowlisted API requests
to the existing FastAPI bearer-protected endpoint; the browser never receives
the API bearer token or inference provider key. Run artifacts also require
owner access. A server-only backend URL is used for those calls.

Use a **unique generated owner password 24+ characters** and a **separate
API bearer token 32+ characters**. Save owner credentials only in your password
manager and encrypted Vercel environment settings; never write them to GitHub,
issues, logs, NEXT_PUBLIC_* variables, or ChatGPT messages. HTTPS is required
outside loopback. Note that Basic authentication has no logout button: close
the browser or clear its cached HTTP authentication to end the session.

### Vercel API project (`vericlaim-api`) — Production environment

Retain the current Neon `DATABASE_URL` and existing schema; do not reset
branches or export/restore to Production.

| Environment variable | Required value or policy |
| --- | --- |
| `APP_ENV` | `production` |
| `PUBLIC_DEMO_ONLY` | `false` |
| `API_ACCESS_TOKEN` | unique 32+ character secret |
| `MOCK_PROVIDER_ENABLED` | `false` for real evidence mode |
| `DEFAULT_PROVIDER` | `gemini` when enabled |
| `LIVE_RETRIEVAL_ENABLED` | `true` |
| `GEMINI_ENABLED` | `true`, after a live free-quota check |
| `GEMINI_API_KEY` | server-only API key from a genuinely ongoing free tier |
| `GEMINI_MODEL` | verified free-tier model (e.g. `gemini-2.5-flash-lite`) |
| `GROQ_ENABLED`, other live providers | `false` unless separately verified |
| `MAX_PROVIDER_CALLS_PER_REQUEST` | `4` initially; adjust only with measured headroom |
| `MAX_VERIFY_REQUESTS_PER_MINUTE` | `3` initially |
| `RATE_LIMIT_BACKEND` | `database` |
| `RATE_LIMIT_PEER_SECRET` | separate random secret 32+ characters |
| `RUN_RETENTION_PURGE_ENABLED` | **`false`**, preserve current rows pending issue #29 |
| `CORS_ALLOWED_ORIGINS` | exact https://vericlaim-web.vercel.app origin |

The free Gemini API is not universal or guaranteed by this runbook. Confirm
your account's applicable free quota, that billing is not enabled or cannot
charge, that the selected model is available, and that responses work before
enabling live calls. The free tier's data-use terms may allow API inputs to be
used to improve the provider's products. **Do not submit confidential claims.**

### Vercel Web project (`vericlaim-web`) — Production environment

| Environment variable | Required value or policy |
| --- | --- |
| `VERICLAIM_PRIVATE_MODE` | `true` |
| `VERICLAIM_OWNER_USER` | private login username (letters/numbers/._-) |
| `VERICLAIM_OWNER_PASSWORD` | unique generated owner password 24+ characters |
| `VERICLAIM_API_ACCESS_TOKEN` | **same value** as API project's `API_ACCESS_TOKEN` |
| `VERICLAIM_BACKEND_URL` | `https://vericlaim-api.vercel.app` |
| `NEXT_PUBLIC_API_URL` | existing public API origin, URL only, never a secret |

Do not enable Web server bearer access before owner protection is configured.
A missing secret fails closed (503); wrong/missing Basic credentials get 401.
The internal backend proxy forwards only the explicit claim verification,
capability/provider-status, and run/evidence/graph endpoints. It rejects
unexpected paths, query strings, cross-origin POSTs, large payloads, and
non-JSON upstream responses. Never turn this into an unrestricted HTTP proxy.

## Two-phase release gate

1. Commit, PR and test all owner-only Web changes using local fixture-mode
   tests plus frontend build, backend tests, CodeQL, Gitleaks and a manual
   unauthenticated 401/private authenticated 200 check on a Preview.
2. Obtain credentials directly into Vercel encrypted environments (not chat).
   Check private API readiness and provider quotas. Stage API Production without
   changing the public alias, verify live processing on a bounded safe
   non-sensitive claim, check the real model, external evidence provenance,
   latency, storage/readback, errors and 401 rejection. Then stage Web with
   private owner credentials, verify responsive owner login and full UI flow.
   Promote each tested deployment only after both services pass. If the Vercel
   plan or connector cannot stage/promote this way, use a reviewed manual
   deployment with immediate read-only smoke and a rollback plan.

Keep the existing public demo intact until this gate passes. Do not
advertise a guaranteed fact-checking verdict or public multi-user service;
this is technical/scientific evidence assistance with explicit uncertainty.

## Backup and retention are separate gates

The existing AES-256-GCM encrypted PostgreSQL backup has passed independent
offline restore on the original Windows user profile. Portable key escrow,
cross-profile restore and off-device backup remain issue #29 follow-up work.
They are **not** prerequisites for a non-destructive owner-only launch, but
loss of the current Windows profile could still make the local archive
unrecoverable. Keep the nonexpiring Neon backup branch and do not turn
`RUN_RETENTION_PURGE_ENABLED=true` without a fresh independent recovery
verification and a separate explicit deletion approval.
