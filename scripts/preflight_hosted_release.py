#!/usr/bin/env python3
"""Read-only, fail-closed hosted release preflight for VeriClaim AI.

No secrets, authentication, claims, provider calls, writes, or redirects.
Only checks the repository owner's known vericlaim-*.vercel.app endpoints.
Authenticated owner flows require a *separate* manual, bounded smoke gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

PRODUCTION_API = "https://vericlaim-api.vercel.app"
PRODUCTION_WEB = "https://vericlaim-web.vercel.app"
PREVIEW_HOST = re.compile(r"^vericlaim-[a-z0-9-]+-pracillatlove-3370s-projects[.]vercel[.]app$")
MAX_BYTES = 262_144
Mode = Literal["demo", "private"]


@dataclass(frozen=True)
class Reply:
    status: int
    body: Any
    headers: Mapping[str, str]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool


class DenyRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


OPENER = build_opener(DenyRedirects)


def validate_url(value: str, kind: Literal["api", "web"]) -> str:
    """Accept only HTTPS base origins owned by this deployment, never credentials."""
    parsed = urlsplit(value)
    expected = PRODUCTION_API if kind == "api" else PRODUCTION_WEB
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or value.rstrip("/") != f"{parsed.scheme}://{parsed.netloc}"
        or (value.rstrip("/") != expected and not PREVIEW_HOST.fullmatch(parsed.hostname))
    ):
        raise ValueError("Only official VeriClaim HTTPS Vercel origins are permitted")
    return value.rstrip("/")


def http_get(base: str, path: str) -> Reply:
    req = Request(
        base + path,
        method="GET",
        headers={"Accept": "application/json", "User-Agent": "VeriClaim-ReadOnly-Preflight/1"},
    )
    try:
        with OPENER.open(req, timeout=15) as response:
            status = response.status
            headers = dict(response.headers)
            content = response.read(MAX_BYTES + 1)
    except HTTPError as exc:
        status = exc.code
        headers = dict(exc.headers)
        content = exc.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("Response exceeded preflight size bound")
    try:
        body: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        body = None
    return Reply(status=status, body=body, headers=headers)


def run_checks(
    api: str,
    web: str,
    mode: Mode,
    *,
    fetch: Callable[[str, str], Reply] = http_get,
) -> list[Check]:
    """Probe safe GET endpoints only. No provider inference or database writes."""
    checks: list[Check] = []

    def check(name: str, ok: bool) -> None:
        checks.append(Check(name=name, passed=ok))

    def read(base: str, path: str) -> Reply:
        try:
            return fetch(base, path)
        except (OSError, URLError, ValueError, TimeoutError):
            return Reply(status=0, body=None, headers={})

    health = read(api, "/health")
    check(
        "api_health",
        health.status == 200
        and isinstance(health.body, dict)
        and health.body.get("status") == "ok",
    )

    ready = read(api, "/ready")
    check(
        "api_database_ready",
        ready.status == 200
        and isinstance(ready.body, dict)
        and ready.body.get("database") == "ok"
        and ready.body.get("status") == "ready",
    )

    capabilities = read(api, "/api/v1/capabilities")
    expected_demo = mode == "demo"
    check(
        "capabilities_mode",
        capabilities.status == 200
        and isinstance(capabilities.body, dict)
        and capabilities.body.get("demo_only") is expected_demo,
    )

    provider = read(api, "/api/v1/providers/status")
    entries = provider.body if isinstance(provider.body, list) else []
    mock = next((p for p in entries if isinstance(p, dict) and p.get("name") == "mock"), None)
    gemini = next((p for p in entries if isinstance(p, dict) and p.get("name") == "gemini"), None)
    check("provider_status", provider.status == 200 and mock is not None and gemini is not None)
    if mode == "demo":
        check(
            "demo_mock_only",
            bool(mock and mock.get("enabled") is True)
            and bool(gemini and gemini.get("enabled") is False),
        )
    else:
        check(
            "real_gemini_enabled",
            bool(gemini and gemini.get("enabled") is True and gemini.get("configured") is True),
        )
        check("mock_disabled", bool(mock and mock.get("enabled") is False))

    root = read(web, "/")
    proxy = read(web, "/api/backend/api/v1/capabilities")
    if mode == "demo":
        check("demo_web_home", root.status == 200)
        check(
            "demo_web_proxy",
            proxy.status == 200
            and isinstance(proxy.body, dict)
            and proxy.body.get("demo_only") is True,
        )
    else:
        # Distinguish our Basic challenge from unrelated Vercel deployment protection.
        for name, response in (("owner_home_denied", root), ("owner_proxy_denied", proxy)):
            challenge = response.headers.get("Www-Authenticate", "")
            if not challenge:
                challenge = next(
                    (v for k, v in response.headers.items() if k.lower() == "www-authenticate"), ""
                )
            check(
                name,
                response.status == 401 and challenge.startswith('Basic realm="VeriClaim AI Owner"'),
            )
        denied = read(api, "/api/v1/runs/run-preflight-unauthorized-00000000")
        check("direct_api_denied", denied.status == 401)

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=PRODUCTION_API)
    parser.add_argument("--web", default=PRODUCTION_WEB)
    parser.add_argument("--expect", choices=("demo", "private"), default="demo")
    args = parser.parse_args()
    try:
        api = validate_url(args.api, "api")
        web = validate_url(args.web, "web")
    except ValueError:
        print("HOSTED_RELEASE_PREFLIGHT=FAIL (untrusted or invalid host)", file=sys.stderr)
        return 2

    checks = run_checks(api, web, args.expect)
    for item in checks:
        print(f"{item.name}={'PASS' if item.passed else 'FAIL'}")
    passed = all(item.passed for item in checks)
    print(f"HOSTED_RELEASE_PREFLIGHT={'PASS' if passed else 'FAIL'}")
    print("READ_ONLY=YES;AUTHENTICATED_INFERENCE_TESTED=NO")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
