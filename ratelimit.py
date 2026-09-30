"""Small in-memory, thread-safe, fixed-window rate limiter for a single-process Flask app.

No Redis, no external service - this is a take-home-scale limiter (10 requests/minute/IP for
POST /demo). An injectable clock makes it testable without real sleeps. Stale per-key entries are
swept on every call so memory doesn't grow unboundedly as distinct IPs come and go.
"""

import threading
import time
from typing import Callable


class RateLimiter:
    """Fixed `window_seconds` rolling window per key, allowing at most `limit` calls in that window."""

    def __init__(self, limit: int, window_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit
        self._window_seconds = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        """True if `key` may make another request right now; also records the hit if allowed."""
        now = self._clock()
        cutoff = now - self._window_seconds
        with self._lock:
            timestamps = [t for t in self._hits.get(key, []) if t > cutoff]
            if len(timestamps) >= self._limit:
                self._hits[key] = timestamps
                return False
            timestamps.append(now)
            self._hits[key] = timestamps
            self._sweep_stale_keys(cutoff)
            return True

    def _sweep_stale_keys(self, cutoff: float) -> None:
        """Drop keys with no hits inside the current window - bounds memory across many distinct IPs."""
        stale = [key for key, timestamps in self._hits.items() if not timestamps or timestamps[-1] <= cutoff]
        for key in stale:
            del self._hits[key]
