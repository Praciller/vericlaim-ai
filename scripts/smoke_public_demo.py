#!/usr/bin/env python3
"""Smoke the real local Compose stack without keys or paid-model calls.

For safety, this script deliberately refuses all non-loopback URLs; never use
it against someone else's public API or a production database.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

FIXTURE = "RAG eliminates hallucinations."
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class RefuseRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        # Do not send a second HTTP request outside the loopback origin.
        return None


LOCAL_HTTP = build_opener(RefuseRedirect)


def loopback_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("Smoke tests accept only http://localhost/loopback base URLs")
    return value.rstrip("/")


def request(
    base: str,
    path: str,
    *,
    method: str = "GET",
    claim: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    data = None
    request_headers = {"Accept": "application/json"}
    request_headers.update(headers or {})
    if claim is not None:
        data = json.dumps({"claim": claim}).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    url = base + path
    req = Request(url, data=data, headers=request_headers, method=method)
    try:
        with LOCAL_HTTP.open(req, timeout=12) as response:
            status = response.status
            content = response.read(2_000_000).decode("utf-8")
    except HTTPError as exc:
        status = exc.code
        content = exc.read(100_000).decode("utf-8")
    try:
        body: dict[str, Any] | str = json.loads(content)
    except json.JSONDecodeError:
        body = content
    return status, body


def wait_ready(base: str, path: str, *, timeout_seconds: float = 90) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            status, _ = request(base, path)
            if status == 200:
                return
        except (OSError, URLError):
            pass
        if time.monotonic() >= deadline:
            raise AssertionError(f"Local service failed readiness check: {path}")
        time.sleep(1.0)


def run_smoke(api: str, web: str) -> None:
    wait_ready(api, "/ready")
    wait_ready(web, "/")
    status, caps = request(api, "/api/v1/capabilities")
    assert status == 200 and isinstance(caps, dict), "Capabilities endpoint unavailable"
    assert caps == {"demo_only": True, "demo_claim": FIXTURE}, (
        "Anonymous demo must explicitly enforce the fixture-only policy"
    )

    status, denied = request(
        api,
        "/api/v1/claims/verify",
        method="POST",
        claim="Confidential fictional project timeline",
        headers={"X-Forwarded-For": "203.0.113.20"},
    )
    assert status == 403 and isinstance(denied, dict), "Custom claims must be rejected"
    assert denied.get("detail", {}).get("code") == "DEMO_ONLY", "Expected DEMO_ONLY"

    status, result = request(api, "/api/v1/claims/verify", method="POST", claim=FIXTURE)
    assert status == 200 and isinstance(result, dict), "Fixture verification failed"
    run_id = result.get("run_id")
    assert isinstance(run_id, str) and run_id.startswith("run-"), "Missing run ID"
    assert result.get("original_claim") == FIXTURE, "Unexpected claim persisted"
    assert result.get("atomic_claims"), "No atomic claim in demo"

    for suffix in ("", "/evidence", "/evidence-graph"):
        status, artifact = request(api, f"/api/v1/runs/{run_id}{suffix}")
        assert status == 200 and isinstance(artifact, dict), (
            f"Cannot fetch persisted run artifact: {suffix}"
        )
        assert artifact.get("run_id") == run_id, "Wrong run artifact returned"

    status, html = request(web, f"/runs/{run_id}")
    assert status == 200 and isinstance(html, str), "Shareable run page not served"

    status, missing = request(api, "/api/v1/runs/run-no-such-identifier")
    assert status == 404 and isinstance(missing, dict), "Unknown run should return 404"

    print("PUBLIC_DEMO_SMOKE=PASS")
    print(
        "COVERAGE=readiness,capabilities,custom-claim-denial,fixture-verification,"
        "persisted-run,evidence,graph,share-link,missing-run"
    )
    # Never print run content, secrets, prompts or provider errors.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", type=loopback_url, default="http://127.0.0.1:8000")
    parser.add_argument("--web", type=loopback_url, default="http://127.0.0.1:3000")
    args = parser.parse_args()
    try:
        run_smoke(args.api, args.web)
    except (AssertionError, OSError, URLError, ValueError) as exc:
        print(f"PUBLIC_DEMO_SMOKE=FAIL ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
