"""Vercel's FastAPI entrypoint for the providerless preview API."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vericlaim.api import app  # noqa: E402

__all__ = ["app"]
