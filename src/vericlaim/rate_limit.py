"""Bounded per-process rate limiting for a single-worker public demo.

This does not replace edge-level rate limits or user authentication in a
distributed production deployment. Client identity uses the connection peer,
never an untrusted X-Forwarded-For header.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from threading import Lock


class InMemoryRateLimiter:
    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 10000,
    ) -> None:
        if max_requests < 1 or window_seconds <= 0 or max_clients < 1:
            raise ValueError("invalid rate limiter configuration")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clock = clock
        self.max_clients = max_clients
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, peer: str) -> tuple[bool, int]:
        now = self.clock()
        with self._lock:
            # Prune expired clients when capacity is reached to bound memory.
            if len(self._hits) >= self.max_clients and peer not in self._hits:
                expired = [
                    name
                    for name, hits in self._hits.items()
                    if not hits or hits[-1] <= now - self.window_seconds
                ]
                for name in expired:
                    del self._hits[name]
                if len(self._hits) >= self.max_clients:
                    return False, math.ceil(self.window_seconds)
            hits = self._hits.setdefault(peer, deque())
            while hits and hits[0] <= now - self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False, max(1, math.ceil(hits[0] + self.window_seconds - now))
            hits.append(now)
            return True, 0
