# VeriClaim AI

VeriClaim AI is an evidence-driven claim verification MVP for AI, machine learning, software engineering, and scientific-computing claims. It is deliberately not a generic fact-check chatbot: agents have bounded responsibilities, retrieval is source-adapter based, and a verdict is valid only when it can be traced to stored evidence and provenance.

## Owner-only real-claim mode (prepared, not yet deployed)

The Web includes a server-side FastAPI gateway and owner-only Basic auth
for a future private live-claim release. This never puts the FastAPI bearer
token or AI provider keys in the browser. Anonymous portfolio demo mode remains
the **current live Production** until separately configured owner credentials,
free-tier provider access and a bounded end-to-end live smoke have all passed.
Follow [the real-claim Production cutover guide](docs/live-production.md);
portable disaster-recovery escrow is tracked separately and is not a blocker
for a non-destructive private release. A read-only hosted release preflight
(`python scripts/preflight_hosted_release.py --expect demo` on the reviewed
Web/API pair) checks deployment mode and anonymous access protections without
posting a claim, calling an LLM, or modifying the database. Owner-authenticated
inference/persistence still requires a separate gated live smoke.

## Portfolio demo (deterministic fixture only)

**Production verified (2026-10-23):** The hosted portfolio demo now serves
reviewed commits: **Web `7b7e886`**, **API `1c356e7`** (a backend-only
PostgreSQL startup fix). Both existing Vercel Hobby projects are `READY`,
and desktop/mobile browser E2E confirmed the fixture-only flow, persisted
Run/Evidence Graph and no JavaScript errors. API Health/Ready, CORS,
anonymous fixture acceptance and HTTP 403 for arbitrary claims all passed.

The API has `RUN_RETENTION_PURGE_ENABLED=false` as an **explicit, temporary
Production data-preservation exemption**: expired run links still return
404, but legacy rows remain in storage until an independently restorable
backup and retention cleanup are approved. Neon production's non-expiring
backup branch passed six-table row-count parity before cutover. Review
[the outstanding backup/retention issue](https://github.com/Praciller/vericlaim-ai/issues/29)
before enabling physical purging. Do not submit sensitive claims; this
is not a multi-user service.

See [Portfolio demo release checklist](docs/portfolio-demo.md).


- Web: https://vericlaim-web.vercel.app
- API: https://vericlaim-api.vercel.app

Use **Try deterministic demo** to verify `RAG eliminates hallucinations.` with the checked-in fixture path. The **repository's Compose demo configuration** explicitly disables live retrieval/providers so this flow does not consume live-provider quota. Do not infer deployed environment settings from source configuration until deployment parity is verified. After a run completes, **Open shareable run artifact** opens `/runs/<run_id>` and reloads the persisted verdict, evidence, source provenance, conditions, limitations, and evidence graph without rerunning verification or retrieval.

The public artifact is intentionally sanitized: provider usage, agent trace details, assessment rationale, credentials, prompts, raw provider responses, and internal errors are not rendered. The demo is bounded evidence for this fixture workflow, not a claim of universal factual correctness.

![Hosted deterministic run artifact](docs/assets/public-demo/run-artifact.png)

The local Compose stack can be checked end-to-end without any external provider
or paid service:

```powershell
docker compose up -d --build --wait
python scripts/smoke_public_demo.py
docker compose down -v --remove-orphans
```

CI runs the same fixture-only smoke against running Docker containers in
addition to unit tests, static checks and Docker image builds.

## What is implemented

- FastAPI API with stable verification run IDs; local development defaults to SQLite, and PostgreSQL is supported for durable hosted persistence. The Production database is verified as the linked Neon PostgreSQL project; credentials remain server-side.
- LangGraph workflow: analyze → decompose → plan → support/counter research → audit → judge → critic → deterministic validation.
- Strong-quantifier detection, compound-claim decomposition, Thai layout-only normalization, and structured Pydantic v2 models.
- Offline deterministic fixture provider/source for reproducible development and tests.
- Six provider adapters: Cerebras, Groq, Gemini, OpenRouter, OKMD, and ThaiLLM. Providers require both an explicit `*_ENABLED=true` flag and a key; missing or disabled providers degrade safely to the bounded fallback path.
- Provider responses normalize configured/actual model IDs, finish reason, latency, usage, quota metadata, and categorized failures. Cerebras remains representable as configured-but-disabled, and OpenRouter is restricted to free routes.
- OpenAlex and Crossref normalized source adapters with explicit HTTP timeouts and metadata-versus-abstract evidence levels. arXiv is an interface placeholder until its response parsing contract is added.
- Minimal Next.js UI with Overview-style result, evidence, sources, conditions, limitations, and run metadata.
- Server-side request bounds for claim size, atomic claims, retrieval queries, evidence candidates, and provider calls.
- A bounded SciFact evaluation path with a local, hash-validated manifest,
  deterministic closed-corpus BM25 retrieval, offline fixture execution, and
  persisted run artifacts. A deterministic regression gate re-runs the offline
  fixture benchmark in CI and fails on any claim, evidence, or call-bound
  metric drop against the checked-in baseline.

Supported verdicts are `SUPPORTED`, `REFUTED`, `MIXED`, `INSUFFICIENT_EVIDENCE`, and `NON_VERIFIABLE`. Confidence means confidence in the verdict given the evidence retrieved by this run, not the probability that the claim is objectively true. Production confidence is currently a deterministic evidence-rule heuristic (including the explicit `MIXED` heuristic value 0.65); it is not calibrated probability.

Current evidence boundaries:

- **IMPLEMENTED / REPRODUCIBLE:** deterministic/offline fixture mode; bounded
  SciFact evaluation; manifest and hash verification; offline architecture
  comparison; provider-isolation tooling; evidence provenance; deterministic
  verdict validation; and persisted run/evidence trace.
- **PARTIAL / LIMITED:** small bounded benchmark profiles; live-provider
  availability and quota constraints; incomplete paired live-isolation results;
  uncalibrated heuristic confidence where applicable; and source-adapter
  limitations.
- **NOT ESTABLISHED:** universal factual correctness; a production SLA;
  benchmark superiority from incomplete live runs; calibrated probability of
  truth; unrestricted web fact checking; autonomous browsing; or
  production-grade multi-tenant security.

## Architecture

```mermaid
flowchart LR
  UI[Next.js UI] --> API[FastAPI]
  API --> Graph[LangGraph workflow]
  Graph --> Analyze[Claim analysis/decomposition]
  Graph --> Plan[Support + counter query planning]
  Plan --> Sources[Source adapters]
  Sources --> Audit[Evidence audit]
  Audit --> Judge[Judge + independent critic]
  Judge --> Validate[Deterministic validator]
  Validate --> DB[(SQLite relational store)]
  Graph --> Router[Provider router]
  Router --> Providers[Mock / Cerebras / Groq / Gemini / OpenRouter]
```

Domain logic depends on normalized source/provider interfaces, so PostgreSQL or additional adapters can be introduced without changing verdict logic. The MVP intentionally has no Redis, queue, vector database, graph database, or autonomous browsing.

See [docs/architecture.md](docs/architecture.md) for responsibilities and persistence boundaries.

## Setup

Python 3.12+ and Node.js 20.9+ are required.

```powershell
uv venv
uv pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn vericlaim.api:app --reload
```

In another terminal:

```powershell
Set-Location apps/web
npm install
npm run dev
```

Open `http://localhost:3000`. The API is at `http://localhost:8000`.

## Evaluation

The repository includes a reproducible, bounded SciFact evaluation path under `evals/scifact`. The checked-in `evals/fixtures/scifact/` snapshot is a small deterministic fixture with a hash-validated manifest for offline CI; `scripts/prepare_scifact.py` can prepare the official AllenAI snapshot under ignored `evals/data/scifact/` for an explicitly reviewed run. The protocol uses deterministic closed-corpus BM25 retrieval and four primary architectures: `A_SINGLE_LLM`, `B_RETRIEVAL_JUDGE`, `C_SUPPORT_COUNTER`, and `D_FULL_VERICLAIM`. A follow-up isolation family (`D1_AUDITOR`, `D2_CRITIC`, `D3_AUDITOR_CRITIC`, and `D4_CONDITIONAL_CRITIC`) separates assurance mechanisms without adding them to the default critical path. Live evaluation uses fixed Groq `openai/gpt-oss-20b` and Gemini `gemini-flash-lite-latest` configurations when explicitly requested; the isolation runner can use one explicitly selected available provider for all stages; OpenRouter is excluded. No incomplete live run is promoted as a benchmark conclusion.

Prepare and validate the pinned local snapshot:

```powershell
python scripts/prepare_scifact.py --root evals/data/scifact --download
python scripts/eval_scifact.py --architecture all --profile smoke --dry-run
python scripts/eval_scifact.py --architecture all --profile smoke --offline
python scripts/eval_scifact.py --architecture isolation --profile smoke --offline
```

Use `--offline` for deterministic provider fixtures. For live free-tier runs, inspect the dry-run call estimate first, then run the same bounded profile with the authorized fixed providers. Results are written under `evals/results/<benchmark_run_id>/` with manifest, metrics, predictions, cache/usage, and error-analysis artifacts. Telemetry distinguishes logical stage invocations, actual provider calls, cache hits, current-run tokens, and provider latency. The default profile is a 10-example smoke; the 50-example pilot is quota-gated. `API_COST_USD=N/A` is intentional because no paid cost is inferred.

The isolated live-5 runner is separately hard-gated at 100 provider calls with a derived unknown-quota token ceiling. Review `python scripts/eval_agent_isolation.py --profile live-5 --dry-run` before invoking it. All stages must use one explicitly selected fixed provider/model; OKMD can be selected explicitly as `deepseek-v4-flash`, and its preflight requires two identical probes with matching configured/actual model and known quota metadata. The preflight reports cache hits/misses, reserves possible critic/recheck branches, and requires at least 10% call/token headroom. Provider rate limits, quota errors, incomplete responses, model substitutions, or incomplete pairs stop the run and are reported as `BLOCKED`; no provider fallback or automatic live-10 run is performed.

For unstable free providers, use the resumable windowed live-5 runner. Calibrate a transport-only reliability envelope on non-target claims, then run one fixed-provider/model benchmark in bounded windows:

```powershell
python scripts/eval_agent_isolation_resumable.py --provider okmd --calibrate-only --reliability-profile evals/results/scifact-reliability-live-5-okmd.json
python scripts/eval_agent_isolation_resumable.py --provider okmd --reliability-profile evals/results/scifact-reliability-live-5-okmd.json --window-max-calls 8
python scripts/eval_agent_isolation_resumable.py --provider okmd --resume --checkpoint evals/results/scifact-live-5-resumable-<run>/checkpoint.json --reliability-profile evals/results/scifact-reliability-live-5-okmd.json --window-max-calls 8
```

The checkpoint persists successful predictions, exact structured stage-cache provenance, authoritative provider-attempt telemetry, locked actual model, and per-window hashes/counters. Resume replays cache hits and calls only the first uncached stage; it never changes provider/model or falls back. Three consecutive low-progress failure windows produce `PROVIDER_UNSUITABLE_FOR_BENCHMARK`. Quality comparison is valid only after `25/25` paired predictions with zero model drift/substitution.

See [docs/evaluation.md](docs/evaluation.md) for dataset versioning, label mapping, evidence/abstention/calibration metrics, ablation interpretation, and limitations. Small smoke runs report `CALIBRATION_SAMPLE_TOO_SMALL` rather than calling heuristic confidence calibrated.

## Providers and free-tier routing

Copy `.env.example` to `.env`. `MOCK_PROVIDER_ENABLED=true` is the safe default. Only add provider keys you are authorized to use; `.env` remains ignored and local-only.
Provider keys are loaded as redacted `SecretStr` values and are unwrapped only at
the outbound provider boundary. Do not log `Settings`, serialize raw settings,
or place replacement credentials in Git. Rotate and revoke any key exposed during
development before using a preview or production environment.

The default specialization is deterministic rules first, Groq for claim/decomposition/classification assistance, Gemini for audit/judgment, OKMD for critique, ThaiLLM for Thai semantic review, and OpenRouter only as a last-resort free runtime fallback. `openrouter/free` is non-deterministic and is excluded from reproducible routing unless explicitly allowed. Cerebras is disabled by default because free inference must be re-probed before enabling.

OKMD quota metadata is recorded when the API returns `model_quota.daily_quota_tokens`, `daily_usage_tokens`, and `daily_remaining_tokens`. An embedded daily-budget error is classified as quota exhaustion and falls back without retrying the same provider. ThaiLLM uses HTTPS only and strips a complete leading `<think>...</think>` block with at most one bounded retry for an incomplete response. A word containing `think` without those tags is preserved.

Recommended non-secret models are documented in `.env.example`; enable flags remain false there.

Historical local smoke snapshot (2026-08-25): Gemini, Groq, OpenRouter free, and ThaiLLM returned usable content; Cerebras model access was visible but inference remained disabled; OKMD returned an API-level permission failure in that snapshot. Re-run the explicit provider preflight before treating any provider as healthy.

## API

```text
GET  /health
GET  /ready
POST /api/v1/claims/verify       {"claim":"RAG eliminates hallucinations."}
GET  /api/v1/runs/{run_id}
GET  /api/v1/runs/{run_id}/evidence
GET  /api/v1/runs/{run_id}/evidence-graph
GET  /api/v1/providers/status
```

`/health` is a liveness probe and does not call a provider. `/ready` checks the database connection and requires at least one explicitly enabled provider (the deterministic mock counts for local/demo deployments); it returns HTTP 503 with sanitized check names when the service is not ready. Set `CORS_ALLOWED_ORIGINS` to a comma-separated allowlist when deploying the web client.

The verification endpoint completes synchronously for the MVP. Offline mode uses the deterministic fixture source. The explicit live E2E boundary sets `LIVE_RETRIEVAL_ENABLED=true` and uses OpenAlex/Crossref; metadata-only records are persisted as sources but cannot become evidence. Each run stores claims, atomic claims, queries, sources, evidence, assessments, verdict details, agent runs, provider usage, and fallback breadcrumbs. Every final evidence reference is validated against the same run's stored evidence.

The result `issue_code` is an operational signal separate from the verdict. It can report `PROVIDER_UNAVAILABLE`, `QUOTA_EXHAUSTED`, `PROVIDER_RATE_LIMIT`, `PROVIDER_TIMEOUT`, `PROVIDER_AUTHENTICATION`, `PROVIDER_RESPONSE_INVALID`, `RETRIEVAL_UNAVAILABLE`, or `REQUEST_LIMIT_EXCEEDED`. A degraded result remains inspectable, but the UI explicitly tells the user to inspect the run trace before relying on provider-assisted output. `INSUFFICIENT_EVIDENCE` remains a verdict label, not a provider failure.

The API enforces conservative per-request bounds: a 2,000-character claim, at most 8 atomic claims, 16 retrieval queries, 32 evidence candidates, and 8 provider-call attempts. Verification has a 30-second cooperative request deadline and returns a sanitized `504 REQUEST_TIMEOUT` when exceeded. Provider and retrieval adapters also use explicit timeouts, retries are bounded to one same-provider retry, excerpts are capped, and the MVP does not fetch user-supplied URLs.

The evidence graph is a read-only projection of the stored run: claim → atomic claim → evidence → source. It performs no new retrieval or provider call, exposes source provenance and evidence level, and keeps evidence excerpts inspectable in the UI. This makes the local fixture demo useful even when external providers are unavailable.

## Testing and CI

```powershell
pytest
ruff check .
ruff format --check .
mypy src
git diff --check
python scripts/eval_regression_gate.py
Set-Location apps/web
npm install
npm run lint
npm run typecheck
npm run build
```

Normal tests make no external LLM or retrieval calls, even when a developer's ignored `.env` contains enabled providers. Real-provider tests require `RUN_LIVE_PROVIDER_TESTS=true` and are never required in CI. Run a minimal provider pass with `python scripts/smoke_providers.py --provider all`; it prints only provider/model/status/usage metadata. Run one live persisted verification with `python scripts/smoke_verification.py --preset english` or `--preset thai`; it prints counts, safe source domains/DOIs, provider/model usage, and fallback breadcrumbs, never response bodies or credentials. For deployment smoke checks, start the API and request both `/health` and `/ready`; neither endpoint performs inference. The checked-in SciFact fixture and hash-validated manifest support reproducible offline evaluation. Official dataset preparation and live-provider evaluation are explicit opt-ins, and live results are not comparable or promotable unless the fixed-provider, complete-pair requirements in [docs/evaluation.md](docs/evaluation.md) are satisfied.

## Private API deployment guard

- For `APP_ENV=production` or `staging`, the API **refuses to start** unless
  `PUBLIC_DEMO_ONLY=true` or a strong server-side `API_ACCESS_TOKEN` (32+
  characters) has been configured. A production typo or missing token therefore
  cannot silently expose arbitrary claim submission and stored artifacts.
- When `PUBLIC_DEMO_ONLY=false` and `API_ACCESS_TOKEN` is set, all verify,
  run, evidence and evidence-graph endpoints require `Authorization: Bearer`
  with that token. Unauthorized requests receive sanitized 401 responses.
  Generate a token using `openssl rand -hex 32`, keep it in a server-only
  secret manager, use HTTPS, and never expose it through `NEXT_PUBLIC_`
  variables, client-side JavaScript, URLs or browser storage.
- Anonymous fixture demos remain available without a token. Demo-only mode
  refuses to start if live retrieval or non-mock providers are enabled,
  avoiding accidental external calls or provider cost.
- This shared Bearer token is suitable only for **trusted internal automation**.
  It provides no individual user identities, role permissions, per-user run
  ownership, token rotation protocol or browser sign-in. The existing Web
  interface intentionally does **not** carry the token; use a trusted
  server/CLI client for private mode until a real identity-provider login
  and authorized BFF/session-based web flow have been implemented.
- Local development (`APP_ENV=development`) without a token retains the
  original no-login workflow. Never expose that mode publicly.

## Public demo safety and evidence interpretation

- The checked-in `compose.yaml` enables `PUBLIC_DEMO_ONLY=true` on the API.
  This **anonymous portfolio/demo** mode accepts only the built-in synthetic
  claim (`RAG eliminates hallucinations.`) and returns HTTP 403 `DEMO_ONLY`
  for any other claim, before retrieval, inference or persistence.
  `GET /api/v1/capabilities` advertises this restriction so the web UI
  disables arbitrary text entry while keeping the fixture demo working.
  Non-fixture records left from prior operation return 404 from the run,
  evidence and graph endpoints while demo-only mode is active.
  Set `PUBLIC_DEMO_ONLY=false` for a trusted local/internal installation;
  doing so enables arbitrary claim entry and exposes existing unexpired
  shareable artifacts. Do not accept sensitive data without real login,
  access checks and edge abuse controls. Demo-only is **not authentication**.

- `POST /api/v1/claims/verify` has a configurable peer-IP quota
  (`MAX_VERIFY_REQUESTS_PER_MINUTE=30` by default). Development uses
  `RATE_LIMIT_BACKEND=memory`: per-process sliding windows, not shared.
  For multiple API instances, set `RATE_LIMIT_BACKEND=database` and a
  randomly generated 32+ character `RATE_LIMIT_PEER_SECRET` **only on the
  server** (for example, generate a secret using `openssl rand -hex 32`).
  Database mode uses an atomic SQL upsert into the shared PostgreSQL database
  with one-minute fixed windows, fails closed with 503 when unavailable,
  returns 429/Retry-After when exhausted, stores only a keyed HMAC of each
  direct connection peer, and removes stale buckets opportunistically.
  SQLite file storage works on one host, not across separate machines.
  All replicas need the same secret, database and synchronized clocks.
  Behind a reverse proxy the observed peer may be the proxy, so deploy a
  trusted edge identity/quota mechanism. Do not trust user-supplied forwarding
  headers. Neither limiter substitutes for user authentication and authorization.
- Persisted run records are retained for up to `RUN_RETENTION_DAYS` days
  (default **30**, configurable from 1–365). After expiry the public run,
  evidence, and graph endpoints return 404. Expired runs and all corresponding
  claims, query plans, evidence, assessments, agent traces, and provider usage
  are deleted transactionally on application startup, before new saves, and
  when an expired ID is requested. Cleanup is **opportunistic**, not a strict
  timed deletion SLA; a dedicated scheduler or provider storage lifecycle is
  needed where expiry must be enforced promptly at rest. Reducing retention
  also removes historical demo links. Existing records are subject to this
  policy at the next app startup.
  **Temporary migration preservation:** set `RUN_RETENTION_PURGE_ENABLED=false`
  *server-side* to suspend physical deletion on startup, save and expired
  reads, while still returning 404 for run/evidence/graph after the 30-day
  cutoff. This is a temporary recovery/cutover safety measure, **not** a
  compliant at-rest deletion policy: expired source claims and traces remain
  stored and inaccessible via public endpoints, but still require restricted
  database access and a documented cleanup deadline. It must not be left
  disabled indefinitely. The default is `true`, so the existing retention
  behavior and tests remain unchanged.
- Persisted run URLs are **publicly readable to anyone with the run ID**.
  `GET /api/v1/runs/{run_id}` and its evidence endpoint serve narrow public
  fields without provider usage, internal agent traces, or assessment rationales.
  However they still expose the original user-entered claim and evidence
  excerpts. Do not submit private/confidential information in a public demo;
  production multi-user use requires access controls and data retention/deletion.
  CORS is not authentication.
- External evidence requires complete claim-aware stance classification;
  missing/malformed results or unassessable excerpts remain NEUTRAL rather than
  being treated as support/counter based on search direction. Strong absolute
  language receives a conservative scope veto. These controls reduce unsupported
  verdicts but **do not prove factual correctness**.
- Offline fixture judgments are illustrative and must not be treated as results
  from real publications. Evidence quality numbers are uncalibrated placeholders,
  not objective source-reliability scores.

## Limitations and roadmap

The checked-in fixture and small smoke/pilot profiles are bounded evaluation evidence, not universal benchmark claims. The offline reference summary records the deterministic configuration only; its small population reports `CALIBRATION_SAMPLE_TOO_SMALL`, and it does not establish live paired architecture superiority, production promotion, or calibrated confidence. Live paired isolation remains constrained by provider availability and quota; incomplete runs are diagnostic/`BLOCKED`, not superiority evidence. The workflow uses deterministic local domain logic to guarantee reproducibility; provider outputs are bounded structured candidates/advisories and cannot invent evidence or override deterministic validation. Retrieval adapters are bounded and not a general web browser. Next steps are richer evidence extraction, cache instrumentation, complete arXiv parsing, async job state if real retrieval latency requires it, browser-level UI verification, and expanded benchmark profiles.
