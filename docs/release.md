# Release candidate checklist

The repository baseline intentionally keeps runtime and experimental output out
of Git. Track source, tests, manifests, CI, documentation, deterministic
fixtures, and compact reference summaries. Keep `.env`, databases, downloaded
SciFact data, provider caches, checkpoints, raw evaluation runs, `node_modules`,
and `.next` local-only.

## Local RC smoke

Run the committed-source checks first:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
Set-Location apps/web
npm ci
npm run lint
npm run typecheck
npm run build
Set-Location ..\..
```

Build and start the production-shaped local stack with deterministic mock
inference:

```powershell
docker compose build
docker compose up -d
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
Invoke-RestMethod -Method Post http://localhost:8000/api/v1/claims/verify -ContentType 'application/json' -Body '{"claim":"RAG eliminates hallucinations"}'
docker compose down
```

The API image uses the frozen `uv.lock`; the web image uses `npm ci` and a
Next.js standalone runtime. The compose stack has no provider API keys and no
paid dependency. `/health` is liveness, `/ready` is database plus enabled
provider readiness, and the verification smoke must preserve the run/evidence
and evidence-graph endpoints.

The API also applies hard per-request bounds: 2,000 claim characters, 8 atomic
claims, 16 retrieval queries, 32 evidence candidates, and 8 provider-call
attempts. Verification has a 30-second cooperative request deadline and returns
a sanitized `504 REQUEST_TIMEOUT` when exceeded. Provider/retrieval timeouts, one
same-provider retry, bounded excerpts, and the absence of user-supplied URL
fetching are verified by tests.

Do not call this a public deployment until the compose build, both probes, the
offline verification, and the browser/API integration have been rerun from a
committed release candidate. Tag `v0.1.0` only after that evidence exists.
