import pytest

from quant.app.logging import REDACTED, redact_secrets
from quant.core.ratelimit import WeightRateLimiter


def test_redacts_secret_keys_recursively() -> None:
    event = {"event": "x", "api_key": "abc", "nested": {"api_secret": "s", "ok": 1}, "signature": "sig"}
    out = redact_secrets(None, "info", event)
    assert out["api_key"] == REDACTED
    assert out["signature"] == REDACTED
    assert out["nested"] == {"api_secret": REDACTED, "ok": 1}
    assert out["event"] == "x"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def test_bucket_blocks_until_refilled() -> None:
    clock = FakeClock()
    lim = WeightRateLimiter(60, period=60, clock=clock, sleep=clock.sleep)
    lim.acquire(60)  # drains the bucket
    lim.acquire(30)  # needs 30 tokens at 1 token/s
    assert sum(clock.sleeps) == pytest.approx(30.0)


def test_post_hoc_charge_delays_next_request() -> None:
    clock = FakeClock()
    lim = WeightRateLimiter(100, period=100, clock=clock, sleep=clock.sleep)
    lim.acquire(10)
    lim.charge(140)  # balance -50
    lim.acquire(10)  # needs 60 tokens
    assert sum(clock.sleeps) == pytest.approx(60.0)


def test_weight_above_capacity_is_an_error() -> None:
    with pytest.raises(ValueError, match="capacity"):
        WeightRateLimiter(10).acquire(11)
