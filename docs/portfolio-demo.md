# VeriClaim AI — Portfolio Demo Release Runbook

Last verified: **2026-10-23** (Asia/Bangkok). This document describes the
**actual deployed** Vercel projects, not merely a tested Preview.

## Intended product and production status

VeriClaim AI is an **anonymous, fixture-only portfolio demonstration**.
Visitors can verify only the synthetic claim `RAG eliminates hallucinations.`,
inspect evidence and provenance, and open a shareable run. It is **not** an
authenticated multi-user fact-checking service, factual assurance, or an
invitation to submit confidential material.

| Vercel production service | URL | Reviewed deployed source |
| --- | --- | --- |
| Web | https://vericlaim-web.vercel.app | `7b7e886` |
| API | https://vericlaim-api.vercel.app | `1c356e7` |

Both existing projects belong to the authenticated Vercel **Hobby** team
`pracillatlove-3370s-projects`. Both deployment aliases were verified
`READY` after release. Web and API were originally deployed manually and
the projects are **not** configured for automatic deployment from GitHub.
The API commit differs from Web only because of a reviewed **backend-only**
PostgreSQL startup hotfix; Web source was unchanged by that commit.

## Production behavior and environment safety

Verified API behavior:

- `GET /health`, `GET /ready`, and `GET /api/v1/capabilities`: HTTP 200;
  capabilities advertises `demo_only=true` and the canonical fixture.
- Anonymous `POST /api/v1/claims/verify` with any other claim: HTTP 403
  `DEMO_ONLY` before inference, retrieval or persistence.
- Built-in fixture: HTTP 200; saved Run, Evidence, and Evidence Graph
  endpoints return HTTP 200 for its generated artifact.
- CORS preflight for the exact HTTPS Web production origin returns 200 with
  the expected `Access-Control-Allow-Origin` header.
- Chromium E2E on 1440px desktop and 390px mobile: arbitrary text input
  disabled, demo notice visible, fixture runs, shareable Run page and
  Evidence Graph render, with **0 JavaScript page errors**.

Production API server-side environment is configured for
`APP_ENV=production`, `PUBLIC_DEMO_ONLY=true`,
`MOCK_PROVIDER_ENABLED=true`, `DEFAULT_PROVIDER=mock`,
`LIVE_RETRIEVAL_ENABLED=false`, and
`RUN_RETENTION_PURGE_ENABLED=false`. The existing Neon PostgreSQL
`DATABASE_URL` is held in server-only Vercel environment settings.
No credential values, database connection strings or user records are
documented or printed here. These settings must never be exposed through
`NEXT_PUBLIC_` variables, public issue comments or test artifacts.

**Migration preservation is temporary:** `RUN_RETENTION_PURGE_ENABLED=false`
prevents automatic SQL deletion during API startup, saves and expired-run
reads. The default in the repository remains `true`. Expired Run/Evidence
APIs still return 404 but expired payloads remain physically stored.
Do not leave this exemption indefinitely. Access to the database must be
restricted, and physical purging should resume only after an independently
restorable backup and explicit retention approval. Track completion in
[issue #29](https://github.com/Praciller/vericlaim-ai/issues/29)
(target review: 2026-10-30).

## Production PostgreSQL recovery evidence

An authenticated Neon/Vercel account inspection verified the production
Neon project `vericlaim-production` against Vercel Storage metadata.
A child of its production `main` was created as
`backup-pre-vericlaim-release-2026-10-23` with **Data and Schema** and
`Expires: Never`; its presence and expiry setting were rechecked.

An isolated 1-hour recovery-test child was created **from the backup**,
not production, with full data and schema. Six key tables appeared on the
child and the child was later confirmed auto-expired. Production `main` and
the original backup were not reset, restored, deleted or re-parented.

Pre-cutover read-only `SELECT COUNT(*)` checks reported:

| Table | Production main | Backup |
| --- | ---: | ---: |
| `verification_runs` | 10 | 10 |
| `claims` | 10 | 10 |
| `evidence` | 22 | 22 |
| `agent_runs` | 90 | 90 |
| `verdicts` | 10 | 10 |
| `sources` | 22 | 22 |

**COUNT PARITY: PASS**, at the inspection time. These counts do not prove
byte-for-byte equality or an independently restorable disaster-recovery
copy. A child branch remains dependent on the same Neon project/account.
The Neon Free plan offered approximately six hours of PITR at inspection
and no pre-existing scheduled snapshots. No **off-project encrypted
PostgreSQL export or isolated full-data restore** has yet been verified.
Do not treat the Backup Branch as protection against whole-project loss.

## Release verification and observed startup fix

All feature, security and documentation PRs were merged through protected
`main` after GitHub Backend/Frontend/Containers, CodeQL and Gitleaks checks.
The current Container CI also runs the real Compose demo smoke.

During the first upgraded API deployment, Vercel logged one PostgreSQL
cold-start DDL race involving the `rate_limit_buckets` type. The reviewed
[PR #28](https://github.com/Praciller/vericlaim-ai/pull/28)
serializes PostgreSQL schema inspection and creation under a
transaction-scoped advisory lock. It added a disposable PostgreSQL 16
CI service and a regression test of **10 concurrent initializers**.
Local PostgreSQL integration and all required GitHub PR checks passed.

API Hotfix `1c356e7` was then deployed without changing Web source.
Production Browser E2E was rerun at 1440px and 390px and passed again.
Vercel's grouped runtime error scan over the **five minutes after the
hotfix** reported **no runtime error groups** for API or Web. Historical
error groups from the earlier API deployment remain visible in longer
time windows and must not be mistaken for new hotfix errors.

## Local and CI reproduction

From the repository root, with Docker and Python available:

```bash
docker compose up -d --build --wait
python scripts/smoke_public_demo.py
docker compose down -v --remove-orphans
```

The smoke deliberately refuses remote URLs to avoid submitting writes
to a public deployment. It tests health/readiness, fixture-only capabilities,
403 denial of arbitrary input, saved artifacts, and the shareable SSR page.
Do not disable its loopback safety gate to test a hosted API. Production
acceptance must use a separately reviewed synthetic-only procedure.

## Controlled deployment and rollback

Before any further release:

1. Confirm the exact GitHub `main` SHA and all protected-branch checks.
2. Confirm the intended Vercel team/project IDs, production environment
   **names/targets**, and server-only database identity. Do not print values.
3. Preserve `PUBLIC_DEMO_ONLY=true`, mock-only inference, and the explicit
   temporary `RUN_RETENTION_PURGE_ENABLED=false` until issue #29 is complete.
4. Deploy API first from the reviewed SHA. Confirm `READY`, `/ready`,
   `/api/v1/capabilities`, exact-origin CORS, rejection of custom claims,
   and fixture/artifact endpoints before updating Web if Web code changed.
5. Verify the Web run page at 390px and 1440px, provider-only fixture use,
   and Vercel runtime errors. Record the two deployment IDs for rollback.
6. If validation fails, rollback only the affected Vercel deployment.
   Code rollback **cannot restore deleted or corrupted database rows**.
   Do not reset, restore or switch the Neon production branch reflexively.

Do not promote the ephemeral SQLite-backed **Preview** as a Production
deployment: Production must retain its existing Neon PostgreSQL connection.
No new paid services, free trials or promotional-credit dependencies
were created for this release. Provider usage dashboards can lag; Hobby
and Neon Free status alone do not guarantee a precise zero-billing total.

## Deferred work

Multi-user identity provider login, user-owned authorization and sessions,
off-project encrypted backup/restore automation, stronger user-level abuse
controls, live-provider benchmarks and React major-version migration are
separate milestones. For now keep the public exhibit fixture-only and
resolve the independent backup plus retention exception in issue #29.
