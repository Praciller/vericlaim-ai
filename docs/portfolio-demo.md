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

**COUNT PARITY: PASS**, at the pre-cutover inspection time. These counts
did not prove byte-for-byte identity. A Neon child branch remains dependent
on the same Neon account/project. The Neon Free plan offered approximately
six hours of PITR at inspection and no pre-existing scheduled snapshots.

### Independently encrypted local export and full restore (2026-10-23)

An additional **provider-independent** archive was produced from Production
Neon with PostgreSQL 18 `pg_dump -Fc` on a direct/unpooled connection,
using a repeatable-read **exported snapshot**. The source connection
was transferred directly from the authorized server-side Vercel setting
into a local one-shot process over standard input, **not** published in
shell command history, source files, logs, or the repository. The dump
was encrypted in memory with **AES-256-GCM** before being written to disk.
The 256-bit encryption key is separately protected by **Windows
CurrentUser DPAPI**.

The encrypted archive is stored **only on the authorized operator's
Windows laptop**, outside Neon. It is deliberately **not** attached to
this public GitHub repository or any CI artifact. The local operator's
backup directory contains a manifest and executable recovery runbook.
Encrypted archive size at test: **51,350 bytes**.

The archive was decrypted in memory and fully imported with `pg_restore`
into an isolated, disposable **PostgreSQL 18 Docker** database, with no
connection to the production branch for restoration. The test was repeated
offline, and all six source-snapshot versus restored table counts matched:

| Table | Live export snapshot | Independently restored |
| --- | ---: | ---: |
| `verification_runs` | 15 | 15 |
| `claims` | 15 | 15 |
| `evidence` | 32 | 32 |
| `agent_runs` | 130 | 130 |
| `verdicts` | 15 | 15 |
| `sources` | 32 | 32 |

**ISOLATED FULL RESTORE: PASS.** Explicit post-restore checks also found
**zero orphan rows** across `claims`, `evidence`, `agent_runs`, `verdicts`,
and `sources`. Source and restore counts come from the **same exported
snapshot**; these may be greater than the earlier 10-run pre-cutover
Neon-branch comparison because the Production fixture demo was exercised.
The temporary restore PostgreSQL container and its data were removed.

**Important remaining risk:** while the encrypted archive is outside
Neon, its DPAPI-wrapped recovery key is only usable with the original
Windows user profile; neither a portable recovery-key escrow nor an
off-device encrypted archive copy has been confirmed. Local disk loss,
machine failure or profile loss could render this backup inaccessible.
Follow [issue #29](https://github.com/Praciller/vericlaim-ai/issues/29)
for secure portable-key storage and an off-device copy. Also, the API
temporarily retains expired data at rest (`RUN_RETENTION_PURGE_ENABLED=false`).
Do **not** reactivate destructive 30-day cleanup until the operator
explicitly approves both the recovery arrangement and data deletion.

### Read-only retention-impact preflight

Before any future retention cleanup approval, perform an aggregate-only preflight
on a **disposable local restored database**, never directly against the public
Production endpoint. The `scripts/retention_impact.py` command refuses remote
PostgreSQL hosts, missing SQLite files, and nonlocal database types; it does not
initialize schemas or execute DML. Both PostgreSQL and SQLite connections
enforce read-only query mode during inspection.

Supply credentials for the **disposable local restore only**, never Neon
or Vercel. Prompt interactively rather than typing a password into a command
or shell history, and pass the URL only through a temporary environment
variable. Example for a local PostgreSQL copy listening on loopback:

```powershell
$localCredential = Get-Credential -Message 'Disposable local PostgreSQL only'
$localUser = [uri]::EscapeDataString($localCredential.UserName)
$localPassword = [uri]::EscapeDataString($localCredential.GetNetworkCredential().Password)
try {
  $env:RETENTION_PREFLIGHT_DATABASE_URL = "postgresql+psycopg://${localUser}:${localPassword}@127.0.0.1:5432/vericlaim_ci"
  uv run --extra dev python scripts/retention_impact.py --days 30
} finally {
  Remove-Item Env:RETENTION_PREFLIGHT_DATABASE_URL -ErrorAction SilentlyContinue
  Remove-Variable localPassword, localCredential -ErrorAction SilentlyContinue
}
```

The tool returns only table counts and a UTC cutoff; it never displays run
IDs, claim contents, evidence, credentials, or keys. The reported 30-day count
is an estimate from the chosen **restored snapshot**, not proof of the live
Production state or authorization to delete. Re-run the assessment as part of
a separate, explicitly approved cleanup plan; issue #29 remains the safety
gate. No current Production retention setting changes are authorized.

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
