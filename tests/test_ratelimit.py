"""Tests for ratelimit.RateLimiter - a fake injectable clock, no real sleeps."""

from ratelimit import RateLimiter


class _FakeClock:
    """Controllable monotonic clock for deterministic window tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_first_ten_requests_from_same_ip_allowed():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    for _ in range(10):
        assert limiter.allow("1.2.3.4") is True


def test_eleventh_request_within_window_blocked():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    for _ in range(10):
        limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4") is False


def test_separate_ip_has_independent_limit():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    for _ in range(10):
        limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4") is False
    assert limiter.allow("5.6.7.8") is True  # different key, untouched budget


def test_window_reset_allows_new_requests():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    for _ in range(10):
        limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4") is False

    clock.advance(61.0)  # past the 60s window
    assert limiter.allow("1.2.3.4") is True


def test_rolling_window_partial_expiry():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    for _ in range(5):
        limiter.allow("1.2.3.4")
    clock.advance(61.0)  # those 5 expire
    for _ in range(5):
        limiter.allow("1.2.3.4")
    # only 5 hits remain within the window - 5 more should be allowed before blocking
    for _ in range(5):
        assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False


def test_stale_entries_are_swept_and_do_not_grow_unbounded():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    for i in range(500):
        limiter.allow(f"10.0.0.{i}")
    clock.advance(61.0)
    limiter.allow("new-key")  # triggers a sweep as a side effect of any call
    # every one of the 500 earlier keys is now outside the window and must be gone
    assert len(limiter._hits) == 1
    assert "new-key" in limiter._hits


def test_limit_and_window_are_configurable():
    clock = _FakeClock()
    limiter = RateLimiter(limit=2, window_seconds=10.0, clock=clock)
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is False
