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

## Hosted-demo parity — blocked until hosting ownership is verified

As checked on **2026-10-23**, the public preview endpoints returned:

| URL | Read-only result | Interpretation |
| --- | --- | --- |
| `https://vericlaim-web.vercel.app/` | HTTP 200 | A page is served; version not established |
| `https://vericlaim-api.vercel.app/health` | HTTP 200 | An API instance responds; version not established |
| `https://vericlaim-api.vercel.app/api/v1/capabilities` | HTTP 404 | The current required demo-only contract is **not** served |

The checked GitHub repository has no GitHub deployment records or environments.
The connected Vercel account's current Git-linked projects do not include
`Praciller/vericlaim-ai`. This does **not** prove no Vercel deployment exists:
the hosted URLs are reachable, but their exact owner, commit, configuration,
and automatic-deploy behavior remain unverified.

**Release block:** Do not represent those hosted URLs as the latest security-
hardened build. Do not deploy to an unverified Vercel project, add credentials,
or force-promote an unrelated deployment. An authorized owner must first
identify the API and Web project/team IDs and their Git links, confirm the
production branch, inspect environment variable **names/configuration only**
(not secret values), and identify any automatic production deploy on `main`.

After the owner verifies the target, deploy both Web and API from the
**same reviewed commit**. Confirm the API is configured for demo-only mode,
live retrieval/providers are disabled, and CORS plus the Web API base URL
match the intended HTTPS origins. For hosted persisted runs, ensure the DB
supports non-ephemeral storage and configure retention. Check the hosting
platform's ongoing free-tier eligibility and usage limits before choosing any
managed database; do not assume promotional credits are permanent.

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
