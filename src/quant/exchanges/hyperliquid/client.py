"""Read-only client for Hyperliquid public data (info endpoint + leaderboard).

Only public, documented-by-usage endpoints are used: every position and fill on
Hyperliquid is public on-chain data, so this is a legal and complete source for
studying other traders. No authentication, no order placement.

Limits worth knowing (they shape the research, see docs/DECISIONS.md):
* ``userFillsByTime`` returns at most 2000 fills per call and only the 10 000 most
  recent fills of an address are reachable at all;
* ``candleSnapshot`` returns at most 5000 candles per request;
* request weight is limited per IP (1200/min); list endpoints cost extra weight per
  returned item, which we charge after the response.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any

import httpx

from quant.app.logging import get_logger
from quant.core.ratelimit import WeightRateLimiter

log = get_logger(__name__)

JSON = Any

FILLS_PAGE_LIMIT = 2000
MAX_REACHABLE_FILLS = 10_000
CANDLES_PAGE_LIMIT = 5000

_BASE_WEIGHT = 20
_LIGHT_WEIGHT = 2
_LIGHT_TYPES = frozenset({"clearinghouseState", "l2Book", "allMids", "orderStatus", "spotClearinghouseState"})
# extra weight: +1 per N returned items
_ITEM_WEIGHT_DIVISOR = {
    "userFillsByTime": 20,
    "userFills": 20,
    "userFunding": 20,
    "fundingHistory": 20,
    "candleSnapshot": 60,
}

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


class HyperliquidError(RuntimeError):
    """Non-retryable API error or retries exhausted."""


def _is_retryable_status(code: int) -> bool:
    return code == 429 or code >= 500


def fill_key(raw: Mapping[str, Any]) -> Hashable:
    """Unique identity of a fill. ``tid`` is unique per fill; fall back to a composite key."""
    tid = raw.get("tid")
    if tid is not None:
        return ("tid", int(tid))
    return ("cmp", raw.get("hash"), raw.get("oid"), raw.get("time"), raw.get("px"), raw.get("sz"), raw.get("side"))


def funding_key(raw: Mapping[str, Any]) -> Hashable:
    delta = raw.get("delta") or {}
    return (raw.get("time"), delta.get("coin"), raw.get("hash"))


class HyperliquidInfoClient:
    def __init__(
        self,
        api_url: str = "https://api.hyperliquid.xyz",
        stats_url: str = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard",
        *,
        http: httpx.Client | None = None,
        limiter: WeightRateLimiter | None = None,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._info_url = api_url.rstrip("/") + "/info"
        self._stats_url = stats_url
        self._http = http or httpx.Client(timeout=timeout, headers={"User-Agent": "quant-research/0.1"})
        self._owns_http = http is None
        self._limiter = limiter or WeightRateLimiter(800)
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._rng = rng or random.Random()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> HyperliquidInfoClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ transport
    def _request(self, method: str, url: str, payload: Mapping[str, Any] | None, weight: int) -> JSON:
        attempt = 0
        while True:
            self._limiter.acquire(weight)
            try:
                resp = self._http.post(url, json=payload) if method == "POST" else self._http.get(url)
            except httpx.TransportError as exc:
                error: str = f"transport error: {exc!r}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise HyperliquidError(f"invalid JSON from {url}: {exc}") from exc
                if not _is_retryable_status(resp.status_code):
                    raise HyperliquidError(f"HTTP {resp.status_code} from {url}: {resp.text[:300]}")
                error = f"HTTP {resp.status_code}"
            attempt += 1
            if attempt > self._max_retries:
                raise HyperliquidError(f"{url}: retries exhausted ({error})")
            delay = self._backoff_base * (2 ** (attempt - 1)) * (1.0 + self._rng.random())
            log.warning("hyperliquid_retry", url=url, attempt=attempt, error=error, delay=round(delay, 2))
            self._sleep(delay)

    def info(self, payload: Mapping[str, Any]) -> JSON:
        req_type = str(payload.get("type"))
        weight = _LIGHT_WEIGHT if req_type in _LIGHT_TYPES else _BASE_WEIGHT
        data = self._request("POST", self._info_url, payload, weight)
        divisor = _ITEM_WEIGHT_DIVISOR.get(req_type)
        if divisor and isinstance(data, list):
            self._limiter.charge(len(data) // divisor)
        return data

    # ------------------------------------------------------------------ endpoints
    def leaderboard(self) -> list[dict[str, Any]]:
        data = self._request("GET", self._stats_url, None, _BASE_WEIGHT)
        rows = data.get("leaderboardRows") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise HyperliquidError("unexpected leaderboard payload: missing 'leaderboardRows'")
        return rows

    def portfolio(self, user: str) -> list[Any]:
        data = self.info({"type": "portfolio", "user": user})
        if not isinstance(data, list):
            raise HyperliquidError(f"unexpected portfolio payload for {user}")
        return data

    def clearinghouse_state(self, user: str) -> dict[str, Any]:
        data = self.info({"type": "clearinghouseState", "user": user})
        if not isinstance(data, dict):
            raise HyperliquidError(f"unexpected clearinghouseState payload for {user}")
        return data

    def user_fills(self, user: str, start_ms: int, end_ms: int | None = None) -> list[dict[str, Any]]:
        """All reachable fills in ``[start_ms, end_ms]``, oldest first, de-duplicated."""
        return self._paginate(
            {"type": "userFillsByTime", "user": user, "aggregateByTime": False},
            start_ms,
            end_ms,
            time_key=lambda r: int(r["time"]),
            key=fill_key,
        )

    def user_funding(self, user: str, start_ms: int, end_ms: int | None = None) -> list[dict[str, Any]]:
        return self._paginate(
            {"type": "userFunding", "user": user},
            start_ms,
            end_ms,
            time_key=lambda r: int(r["time"]),
            key=funding_key,
        )

    def funding_history(self, coin: str, start_ms: int, end_ms: int | None = None) -> list[dict[str, Any]]:
        return self._paginate(
            {"type": "fundingHistory", "coin": coin},
            start_ms,
            end_ms,
            time_key=lambda r: int(r["time"]),
            key=lambda r: (r.get("coin"), r.get("time")),
        )

    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        if interval not in INTERVAL_MS:
            raise ValueError(f"unsupported interval {interval}")
        out: dict[int, dict[str, Any]] = {}
        cursor = start_ms
        while cursor <= end_ms:
            page = self.info(
                {
                    "type": "candleSnapshot",
                    "req": {"coin": coin, "interval": interval, "startTime": cursor, "endTime": end_ms},
                }
            )
            if not isinstance(page, list) or not page:
                break
            before = len(out)
            for row in page:
                out[int(row["t"])] = row
            last_open = max(int(r["t"]) for r in page)
            if len(out) == before:
                break
            cursor = last_open + INTERVAL_MS[interval]
        return [out[k] for k in sorted(out)]

    # ------------------------------------------------------------------ helpers
    def _paginate(
        self,
        base: Mapping[str, Any],
        start_ms: int,
        end_ms: int | None,
        *,
        time_key: Callable[[Mapping[str, Any]], int],
        key: Callable[[Mapping[str, Any]], Hashable],
    ) -> list[dict[str, Any]]:
        """Time-cursor pagination.

        The next page starts at the *last seen timestamp* (inclusive), not +1 ms:
        several records can share a millisecond and a page may end in the middle of
        them. Overlap is removed by ``key``. Stops when a page brings nothing new.
        """
        seen: dict[Hashable, dict[str, Any]] = {}
        cursor = start_ms
        while True:
            payload = dict(base, startTime=cursor)
            if end_ms is not None:
                payload["endTime"] = end_ms
            page = self.info(payload)
            if not isinstance(page, list):
                raise HyperliquidError(f"unexpected {base.get('type')} payload: {type(page).__name__}")
            new = 0
            for row in page:
                k = key(row)
                if k not in seen:
                    seen[k] = row
                    new += 1
            if not page or new == 0:
                break
            last = max(time_key(r) for r in page)
            if end_ms is not None and last >= end_ms:
                break
            cursor = last
        return sorted(seen.values(), key=lambda r: (time_key(r), str(key(r))))


def parse_leaderboard_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a leaderboard row into ``{address, account_value, <window>_{pnl,roi,vlm}}``."""
    out: dict[str, Any] = {
        "address": str(row["ethAddress"]).lower(),
        "account_value": float(row.get("accountValue") or 0.0),
        "display_name": row.get("displayName"),
    }
    windows: Iterable[Sequence[Any]] = row.get("windowPerformances") or []
    for item in windows:
        name, perf = item[0], item[1]
        for field in ("pnl", "roi", "vlm"):
            out[f"{name}_{field}"] = float(perf.get(field) or 0.0)
    return out
