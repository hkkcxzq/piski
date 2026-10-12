import json
from typing import Any

import httpx
import pytest

from quant.core.ratelimit import WeightRateLimiter
from quant.exchanges.hyperliquid.client import HyperliquidError, HyperliquidInfoClient, parse_leaderboard_row


def _client(handler: Any) -> HyperliquidInfoClient:
    return HyperliquidInfoClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        limiter=WeightRateLimiter(1_000_000),
        sleep=lambda _s: None,
        backoff_base=0.0,
    )


def _fill(tid: int, t: int) -> dict[str, Any]:
    return {
        "coin": "BTC",
        "time": t,
        "px": "100",
        "sz": "1",
        "side": "B",
        "startPosition": "0",
        "closedPnl": "0",
        "fee": "0.01",
        "crossed": True,
        "dir": "Open Long",
        "tid": tid,
        "oid": tid,
        "hash": "0x0",
    }


def test_fill_pagination_overlaps_on_shared_timestamp_and_dedupes() -> None:
    # 5 fills, two share t=3000; the server pages 3 at a time, ascending from startTime.
    all_fills = [_fill(1, 1000), _fill(2, 2000), _fill(3, 3000), _fill(4, 3000), _fill(5, 4000)]
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["type"] == "userFillsByTime"
        start = body["startTime"]
        calls.append(start)
        page = [f for f in all_fills if f["time"] >= start][:3]
        return httpx.Response(200, json=page)

    fills = _client(handler).user_fills("0xabc", 0)
    assert [f["tid"] for f in fills] == [1, 2, 3, 4, 5]
    assert calls[0] == 0
    assert calls[1] == 3000  # restarted at the last seen timestamp (inclusive), not +1ms


def test_retries_on_429_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=[])

    assert _client(handler).portfolio("0xabc") == []
    assert attempts["n"] == 3


def test_non_retryable_error_raises_immediately() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(422, text="bad request")

    with pytest.raises(HyperliquidError, match="422"):
        _client(handler).portfolio("0xabc")
    assert attempts["n"] == 1


def test_retries_exhausted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(HyperliquidError, match="retries exhausted"):
        _client(handler).portfolio("0xabc")


def test_candles_pagination_advances_by_interval() -> None:
    hour = 3_600_000
    candles = [
        {"t": i * hour, "T": (i + 1) * hour - 1, "c": "1", "h": "1", "l": "1", "o": "1", "v": "1"} for i in range(7)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        req = json.loads(request.content)["req"]
        page = [c for c in candles if req["startTime"] <= c["t"] <= req["endTime"]][:3]
        return httpx.Response(200, json=page)

    out = _client(handler).candles("BTC", "1h", 0, 6 * hour)
    assert [c["t"] for c in out] == [i * hour for i in range(7)]


def test_leaderboard_parsing() -> None:
    row = {
        "ethAddress": "0xABC",
        "accountValue": "1234.5",
        "displayName": None,
        "windowPerformances": [
            ["day", {"pnl": "1", "roi": "0.1", "vlm": "10"}],
            ["allTime", {"pnl": "100", "roi": "2", "vlm": "5000"}],
        ],
    }
    out = parse_leaderboard_row(row)
    assert out["address"] == "0xabc"
    assert out["account_value"] == 1234.5
    assert out["allTime_pnl"] == 100.0
    assert out["day_vlm"] == 10.0


def test_leaderboard_bad_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": []})

    with pytest.raises(HyperliquidError, match="leaderboardRows"):
        _client(handler).leaderboard()
