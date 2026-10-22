"""Process-local and shared SQL quota guards for claim verification.

The direct connection peer is deliberately used, never an untrusted forwarded
header. Shared SQL mode requires a keyed identity digest and shared database.
Neither mode substitutes for real user authentication or trusted edge controls.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from collections import deque
from collections.abc import Callable
from threading import Lock

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine


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


class DatabaseRateLimiter:
    """Atomic shared fixed-window limiter for PostgreSQL or shared-file SQLite.

    All replicas must use the same database, server clock discipline, and HMAC
    secret. SQL counter mutations are atomic; the secret itself is never stored.
    """

    def __init__(
        self,
        engine: Engine,
        *,
        max_requests: int,
        peer_secret: str,
        window_seconds: int = 60,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_requests < 1 or window_seconds < 1:
            raise ValueError("invalid rate limiter configuration")
        if len(peer_secret) < 32:
            raise ValueError("shared peer identity secret must be 32+ characters")
        if engine.dialect.name not in ("sqlite", "postgresql"):
            raise ValueError("unsupported shared rate limiter database dialect")
        self.engine = engine
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clock = clock
        self._key = peer_secret.encode("utf-8")
        self._last_cleanup_bucket: int | None = None
        self._cleanup_lock = Lock()

    def allow(self, peer: str) -> tuple[bool, int]:
        from .db import RateLimitBucketRow

        now = self.clock()
        window = int(math.floor(now / self.window_seconds))
        retry = max(1, math.ceil((window + 1) * self.window_seconds - now))
        digest = hmac.new(self._key, peer.encode("utf-8"), hashlib.sha256).hexdigest()

        insert_fn = sqlite_insert if self.engine.dialect.name == "sqlite" else postgresql_insert
        statement = insert_fn(RateLimitBucketRow).values(
            peer_hash=digest, window_index=window, hits=1
        )
        upsert = statement.on_conflict_do_update(
            index_elements=[
                RateLimitBucketRow.peer_hash,
                RateLimitBucketRow.window_index,
            ],
            set_={"hits": RateLimitBucketRow.hits + 1},
            where=RateLimitBucketRow.hits < self.max_requests,
        ).returning(RateLimitBucketRow.hits)
        with self.engine.begin() as connection:
            accepted = connection.execute(upsert).scalar_one_or_none() is not None

        # Bound retained bucket rows to approximately three windows. Cleanup
        # takes a separate transaction so a failed cleanup never alters the
        # decision for the current request.
        if self._last_cleanup_bucket != window:
            with self._cleanup_lock:
                if self._last_cleanup_bucket != window:
                    with self.engine.begin() as connection:
                        connection.execute(
                            delete(RateLimitBucketRow).where(
                                RateLimitBucketRow.window_index < window - 2
                            )
                        )
                    self._last_cleanup_bucket = window
        return accepted, (0 if accepted else retry)
