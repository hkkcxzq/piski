"""Weight-based token bucket shared by exchange clients."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class WeightRateLimiter:
    """Token bucket refilled continuously at ``capacity`` weight per ``period`` seconds.

    ``acquire(w)`` blocks until ``w`` tokens are available. ``charge(w)`` debits
    weight that is only known after a response arrives (e.g. per-item weight);
    the balance may go negative, which delays the next ``acquire``.
    """

    def __init__(
        self,
        capacity: float,
        period: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if capacity <= 0 or period <= 0:
            raise ValueError("capacity and period must be positive")
        self._capacity = float(capacity)
        self._rate = float(capacity) / float(period)
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(capacity)
        self._updated = clock()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> float:
        return self._capacity

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._updated = now

    def acquire(self, weight: float) -> None:
        if weight <= 0:
            return
        if weight > self._capacity:
            raise ValueError(f"weight {weight} exceeds bucket capacity {self._capacity}")
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= weight:
                    self._tokens -= weight
                    return
                wait = (weight - self._tokens) / self._rate
            self._sleep(wait)

    def charge(self, weight: float) -> None:
        if weight <= 0:
            return
        with self._lock:
            self._refill()
            self._tokens -= weight
