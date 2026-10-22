"""The Compose smoke must not accidentally send writes to hosted services."""

import pytest
from scripts.smoke_public_demo import RefuseRedirect, loopback_url


@pytest.mark.parametrize(
    "base",
    [
        "https://vericlaim-api.vercel.app",
        "http://example.com",
        "http://127.0.0.1.evil.example:8000",
        "http://localhost:8000/redirect",
        "http://user:password@localhost:8000",
        "http://localhost:8000/?token=secret",
    ],
)
def test_demo_smoke_rejects_non_loopback_or_unsafe_base_urls(base: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        loopback_url(base)


@pytest.mark.parametrize(
    "base",
    ["http://localhost:3000", "http://127.0.0.1:8000/", "http://[::1]:8000"],
)
def test_demo_smoke_allows_loopback_base_urls(base: str) -> None:
    assert loopback_url(base).startswith("http://")


def test_demo_smoke_never_follows_redirects() -> None:
    handler = RefuseRedirect()
    assert (
        handler.redirect_request(
            None, None, 302, "redirect", None, "https://vericlaim-api.vercel.app"
        )
        is None
    )
