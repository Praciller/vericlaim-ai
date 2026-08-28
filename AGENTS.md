# Coding guidance for VeriClaim AI

## Boundaries

- Keep domain models and verdict rules independent from FastAPI, SQLAlchemy, providers, and source adapters.
- Keep support and counter evidence separate. Never let a researcher silently make the final judgment.
- Preserve original Thai text and only normalize whitespace unless a semantic-preserving transformation is tested.
- Keep provider routing bounded. No uncontrolled agent conversations, autonomous browsing, code execution, queues, or background workers in the MVP.
- A provider may be constructed only when its explicit `*_ENABLED` flag and API key are both present. Keep Cerebras disabled unless a current free-inference probe authorizes it, and allow only free OpenRouter routes.
- OKMD and ThaiLLM are supported adapters; ThaiLLM must use HTTPS and any reasoning-block retry is bounded to one attempt.
- External retrieval is opt-in via `LIVE_RETRIEVAL_ENABLED=true`; metadata-only records may be persisted for traceability but cannot become evidence. Persist actual provider models and safe fallback categories, never raw prompts or hidden reasoning.

## Validation commands

```powershell
uv pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy src
git diff --check
Set-Location apps/web; npm install; npm run lint; npm run typecheck; npm run build
```

The local environment must use Python 3.12+ for the supported runtime. Tests must mock external APIs and providers; real-provider checks are opt-in with `RUN_LIVE_PROVIDER_TESTS=true`. Use `python scripts/smoke_providers.py --provider all` for minimal live checks; never print response bodies or credentials.

## Security rules

- Never commit `.env`, API keys, cookies, or provider payloads containing secrets.
- Use explicit outbound HTTP timeouts and bounded fallback behavior.
- Return sanitized provider errors; do not expose keys, raw vendor payloads, or unnecessary prompt content.
- Do not add arbitrary URL fetching without an SSRF-safe allowlist and validation.
- Do not execute user-provided code.

## Evidence and evaluation rules

- A verdict may cite only evidence stored in the same run; deterministic validation must remain enabled.
- Do not fabricate sources, excerpts, citations, benchmark numbers, or deployment results.
- Fixture evidence is not external research and must remain labeled as such.
- New benchmark adapters must record dataset provenance, version/checksum, population, and reproducible artifacts.

## Change discipline

Preserve unrelated dirty work. Review `git diff --check`, run the relevant tests, and document any unverified provider, deployment, or production gate explicitly.
