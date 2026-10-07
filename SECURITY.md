# Security policy

## Report a vulnerability

Report security issues privately through GitHub's security reporting features when available. Do not include live credentials in a public issue, pull request, test fixture, screenshot, or log.

## Handle exposed credentials

Treat any credential committed to a public repository as exposed, even if the commit is later deleted or rewritten.

1. Revoke or rotate the credential at the provider first.
2. Replace it through the local or hosting platform secret manager.
3. Confirm that the old credential is disabled and review provider usage for unexpected activity.
4. Update application configuration and run the relevant offline tests before enabling external providers again.
5. Rewrite Git history only when cleanup is useful. History cleanup does not replace revocation or rotation.

Use a separate credential for each project, provider, and environment when the provider supports that separation. Prefer names that make the scope clear, such as `VERICLAIM_GEMINI_DEV` or `VERICLAIM_GROQ_PROD`.

## Keep secrets out of public artifacts

Never put provider credentials in:

- `NEXT_PUBLIC_*` or `VITE_*` variables.
- Browser bundles or client-side code.
- Application logs, exception text, or tracing payloads.
- Documentation, examples, fixtures, or test snapshots.
- Screenshots or recorded demos.
- Committed `.env` files.

Keep committed environment files limited to placeholders such as `.env.example`. Runtime secrets belong in a local secret manager or the hosting platform's encrypted secret store.

## AI and agent environments

Give Codex and other coding agents only the minimum credentials required for the task. Prefer non-production credentials with narrow provider permissions and quotas. Do not copy production master credentials into an agent environment for source review, tests, or other work that does not need live provider access.

VeriClaim AI keeps external providers disabled by default. Tests must use dummy credentials and mocked provider calls unless an operator explicitly opts into a documented live-provider check.
