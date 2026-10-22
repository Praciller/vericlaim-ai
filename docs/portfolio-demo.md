# Portfolio demo release contract

## Chosen scope

VeriClaim AI is an **anonymous, deterministic portfolio exhibit**, not a
multi-user fact-checking service. A visitor can use only the synthetic example
`RAG eliminates hallucinations.`, inspect the evidence/provenance graph, and
open a shareable run. Do **not** promise unrestricted claim entry, factual
correctness, calibrated confidence, or a production uptime SLA.

For Docker, the Web service sets `INTERNAL_API_URL=http://api:8000` for
server-rendered shareable-run pages. Browser-side requests use
`NEXT_PUBLIC_API_URL=http://localhost:8000` instead. Never publish the Docker
service hostname as a `NEXT_PUBLIC_` variable. This separation is required:
inside the Web container, `localhost` does not point at the API container.

The active Compose configuration runs `APP_ENV=production`,
`PUBLIC_DEMO_ONLY=true`, `MOCK_PROVIDER_ENABLED=true`,
`LIVE_RETRIEVAL_ENABLED=false`, and `DEFAULT_PROVIDER=mock`.
Backend configuration validation rejects live providers/retrieval in public-demo
mode. The demo allows no externally supplied URLs or provider/API credentials.
It is not authentication for arbitrary user data.

## Repeatable local/CI verification

From the repo root with Docker available:

```bash
docker compose build
docker compose up -d --wait
python3 scripts/smoke_public_demo.py
docker compose down
```

The dedicated GitHub Actions **containers** job now builds both images, starts
the production-shaped Compose services, calls the smoke, and always tears down
the disposable stack and volume (including on a failed job). The smoke runs
without dependencies beyond the Python standard library. It tests:

- API and Web readiness;
- the **required** `demo_only=true` capability contract and the canonical
  demo claim;
- rejection of an arbitrary claim (HTTP 403, `DEMO_ONLY`), even with a
  spoofed `X-Forwarded-For` header;
- fixture-only verification and a persisted run ID;
- retrieval of saved Run, Evidence, and Evidence Graph;
- HTTP 200 for the shareable Next.js run page, and 404 for an unknown run.

The smoke intentionally **refuses non-loopback URLs** to prevent mistakenly
submitting write requests to a real publicly hosted service. Never modify it
to send private claims or credentials to external domains.

Backend pytest/Ruff/mypy, frontend test/lint/typecheck/build and SciFact
offline regression remain independent CI requirements.

## Hosted Demo — verified staging, production awaiting database safeguard

Verified on **2026-10-23** using the authenticated Vercel account. Both
production projects are owned by Vercel Hobby team
`pracillatlove-3370s-projects`, and their existing deployments came from the
Vercel CLI. They are not currently linked to GitHub auto-deployment:

| Target | URL | Deployed commit | Status |
| --- | --- | --- | --- |
| API production | https://vericlaim-api.vercel.app | `e299bd8` | Old build; `/api/v1/capabilities` returned 404 |
| Web production | https://vericlaim-web.vercel.app | `e299bd8` | Old build; home page returned 200 |
| Reviewed GitHub main | https://github.com/Praciller/vericlaim-ai | `955c42c` | CI and security checks passed |

The Vercel production **API** project now contains the following explicit
configuration keys for the *next* deployment:
`APP_ENV=production`, `PUBLIC_DEMO_ONLY=true`,
`MOCK_PROVIDER_ENABLED=true`, `DEFAULT_PROVIDER=mock`,
and `LIVE_RETRIEVAL_ENABLED=false`. These are non-sensitive mode switches,
not provider API keys. The existing production PostgreSQL environment
variables were **not edited**; values were not read or copied. Updating
environment variables alone does not upgrade a running Vercel deployment.

### Isolated Vercel Preview verification

Both projects were built from the **same main commit** `955c42c` in Vercel
Preview, leaving production aliases unchanged:

- Web preview:
  https://vericlaim-web-git-main-pracillatlove-3370s-projects.vercel.app
- API preview:
  https://vericlaim-api-git-main-pracillatlove-3370s-projects.vercel.app

The API preview is explicitly **fixture-only** and uses a disposable
`sqlite:////tmp/vericlaim-preview.db` database, **not** the existing
production PostgreSQL connection. Web preview points to the API preview URL,
and API preview CORS permits only the specific Web preview origin. This
ephemeral preview storage is **not suitable for durable production runs**.

Verified results:

- API `/health`, `/ready`, and `/api/v1/capabilities`: HTTP 200,
  `demo_only=true`, enabled providers only `mock`.
- Arbitrary non-fixture claim: HTTP 403 `DEMO_ONLY`.
- Built-in synthetic fixture: successful POST with persisted Run, Evidence,
  and Evidence Graph endpoints returning 200.
- Web browser E2E: 1440px desktop and 390px mobile; manual input disabled,
  fixture button works, shareable Run/Evidence Graph renders with HTTP 200,
  and no browser JavaScript errors.
- Preview API CORS preflight for Web preview: HTTP 200, allow-origin restricted
  to the tested preview Web domain.

### Production release blocker: preserve or explicitly expire old records

**Do not promote a Preview deployment to Production.** It was built using an
intentionally temporary, isolated database and preview environment. Its
passing smoke tests do not establish the safety of existing production data.

The current backend initializes SQL tables and opportunistically **purges
verification runs older than `RUN_RETENTION_DAYS` (default 30)**, including
linked claim/evidence/provider trace rows. Switching the production API to
this version could irreversibly delete legacy rows on startup. A rollback
cannot restore those rows.

Before switching production, the authorized owner must **verify a restorable
backup/snapshot of the production PostgreSQL database or explicitly accept
the documented retention cleanup and possible loss of old Run links**.
The available Neon connector did not expose the Vercel-linked database
project, so its backup/PITR status could **not** be verified through this
connection. Do not infer that the Hobby/Neon plans offer backup guarantees.

Once this prerequisite is satisfied, deploy **both existing Vercel production
projects from reviewed commit `955c42c` (or a later approved commit)**,
using the existing production database configuration and fixture-only
environment values. Check production API readiness and CORS first, then
Web. Confirm `NEXT_PUBLIC_API_URL` points to the intended production HTTPS
API origin; never expose private database URLs or tokens to the browser.
Retain the previous deployment IDs for code rollback, but remember that
rollback does **not** undo database changes. GitHub Actions is not currently
configured to deploy to Vercel, and Git auto-deploy must not be assumed.

### Post-deployment acceptance

Before linking the demo as current in a resume or portfolio:

1. Identify the exact source commit deployed for **both** Web and API.
2. Confirm `GET /health` and `GET /ready` return 200.
3. Confirm `GET /api/v1/capabilities` returns exactly
   `{"demo_only":true,"demo_claim":"RAG eliminates hallucinations."}`.
4. In a reviewed browser session, check 390px and 1440px widths. Manual
   arbitrary-claim entry must be disabled, a clear demo notice must appear,
   and the deterministic demo action must work.
5. Verify the fixture's run page, persisted evidence and graph without
   sending any sensitive data, and confirm an unauthorized arbitrary claim is
   rejected before any persistence.
6. Confirm no provider credentials are enabled, no raw internal traces are
   rendered, and no user-identifying data appear in screenshots or logs.
7. Record test evidence, the deployment ID, relevant date, and rollback plan.

If a step fails, do **not** mark the hosted preview production-ready. Keep the
local Compose demonstration and its screenshot as the portfolio proof.

## Deferred by design

Multi-user identity provider login, BFF/session auth, user-owned run
permissions, distributed abuse prevention on trusted client identity, live
retrieval benchmarking and paid provider quotas are **out of scope** for the
fixture-only exhibit. They should become a separate product milestone only
after the user intentionally chooses a multi-user service.
