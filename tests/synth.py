"""Deterministic synthetic market and trader data for tests (no network)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

HOUR = 3_600_000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % HOUR)  # aligned to the hour


def ou_candles(
    coin: str,
    n: int,
    seed: int,
    theta: float = 0.08,
    sigma: float = 0.01,
    start_ms: int = T0,
    interval_ms: int = HOUR,
    px0: float = 100.0,
) -> list[dict[str, Any]]:
    """Mean-reverting (Ornstein–Uhlenbeck) log-price candles in Hyperliquid raw format."""
    rng = random.Random(seed)
    x = math.log(px0)
    mu = x
    rows = []
    for i in range(n):
        o = x
        x = x + theta * (mu - x) + sigma * rng.gauss(0.0, 1.0)
        c = x
        hi = max(o, c) + abs(rng.gauss(0.0, sigma / 4))
        lo = min(o, c) - abs(rng.gauss(0.0, sigma / 4))
        t = start_ms + i * interval_ms
        rows.append(
            {
                "t": t,
                "T": t + interval_ms - 1,
                "s": coin,
                "i": "1h",
                "o": f"{math.exp(o):.6f}",
                "c": f"{math.exp(c):.6f}",
                "h": f"{math.exp(hi):.6f}",
                "l": f"{math.exp(lo):.6f}",
                "v": f"{1000 + rng.random() * 100:.3f}",
                "n": 100,
            }
        )
    return rows


def funding_rows(coin: str, n: int, seed: int, start_ms: int = T0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [
        {"coin": coin, "time": start_ms + i * HOUR, "fundingRate": f"{rng.gauss(1e-5, 1e-5):.8f}", "premium": "0"}
        for i in range(n)
    ]


@dataclass
class _Pos:
    size: Decimal = Decimal(0)


class FillFactory:
    """Builds consistent raw fills (startPosition chained, closedPnl computed)."""

    def __init__(self, fee_rate: str = "0.00035") -> None:
        self.tid = 0
        self.fee_rate = Decimal(fee_rate)
        self.pos: dict[str, Decimal] = {}
        self.entry: dict[str, Decimal] = {}

    def fill(
        self, coin: str, t: int, px: str | Decimal, signed_sz: str | Decimal, crossed: bool = True
    ) -> dict[str, Any]:
        px_d, sz_d = Decimal(str(px)), Decimal(str(signed_sz))
        start = self.pos.get(coin, Decimal(0))
        after = start + sz_d
        closed = Decimal(0)
        if start != 0 and (after == 0 or (after > 0) != (start > 0) or abs(after) < abs(start)):
            qty = min(abs(sz_d), abs(start))
            closed = (px_d - self.entry[coin]) * qty * (1 if start > 0 else -1)
        if after != 0 and (start == 0 or (after > 0) != (start > 0)):
            self.entry[coin] = px_d
        elif start != 0 and abs(after) > abs(start):
            self.entry[coin] = (self.entry[coin] * abs(start) + px_d * abs(sz_d)) / abs(after)
        self.pos[coin] = after
        self.tid += 1
        direction = (
            ("Open Long" if sz_d > 0 else "Open Short")
            if start == 0
            else ("Close Long" if start > 0 else "Close Short")
        )
        return {
            "coin": coin,
            "time": t,
            "px": str(px_d),
            "sz": str(abs(sz_d)),
            "side": "B" if sz_d > 0 else "A",
            "startPosition": str(start),
            "closedPnl": str(closed),
            "fee": str((px_d * abs(sz_d) * self.fee_rate).quantize(Decimal("0.000001"))),
            "crossed": crossed,
            "dir": direction,
            "tid": self.tid,
            "oid": self.tid,
            "hash": f"0x{self.tid:x}",
        }


def rule_trader_fills(
    candles: list[dict[str, Any]],
    coin: str,
    rule: str,
    seed: int,
    hold: int = 3,
    size: str = "1",
    threshold: float = 1.0,
    max_trades: int = 400,
) -> list[dict[str, Any]]:
    """Trader that enters at the close of hour i based on a rule, exits ``hold`` hours later.

    rule = "mean_reversion": trade against the last 4h move when |z| > threshold;
    rule = "random": random entries and sides.
    """
    rng = random.Random(seed)
    closes = [float(c["c"]) for c in candles]
    logret = [0.0] + [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    ff = FillFactory()
    fills: list[dict[str, Any]] = []
    i = 200
    while i < len(candles) - hold - 1 and len(fills) < 2 * max_trades:
        window = logret[i - 167 : i + 1]
        mean = sum(window) / len(window)
        sd = math.sqrt(sum((x - mean) ** 2 for x in window) / (len(window) - 1))
        z4 = math.log(closes[i] / closes[i - 4]) / (sd * 2)
        side = 0
        if rule == "mean_reversion" and abs(z4) > threshold:
            side = -1 if z4 > 0 else 1
        elif rule == "random" and rng.random() < 0.3:
            side = rng.choice((-1, 1))
        if side == 0:
            i += 1
            continue
        t_entry = int(candles[i]["t"]) + HOUR + rng.randint(1_000, 60_000)  # just after the candle closes
        t_exit = int(candles[i + hold]["t"]) + HOUR + rng.randint(1_000, 60_000)
        qty = Decimal(size) * side
        fills.append(ff.fill(coin, t_entry, candles[i]["c"], qty))
        fills.append(ff.fill(coin, t_exit, candles[i + hold]["c"], -qty))
        i += hold + 1
    return fills


def portfolio_from_fills(fills: list[dict[str, Any]], start_equity: float = 100_000.0) -> list[Any]:
    """Daily account-value / cumulative-PnL history consistent with the fills."""
    if not fills:
        return []
    events = sorted((int(f["time"]), float(f["closedPnl"]) - float(f["fee"])) for f in fills)
    day = 24 * HOUR
    t = events[0][0] - events[0][0] % day
    end = events[-1][0] + day
    cum, k = 0.0, 0
    av_hist, pnl_hist = [], []
    while t <= end:
        while k < len(events) and events[k][0] <= t:
            cum += events[k][1]
            k += 1
        av_hist.append([t, f"{start_equity + cum:.2f}"])
        pnl_hist.append([t, f"{cum:.2f}"])
        t += day
    return [["perpAllTime", {"accountValueHistory": av_hist, "pnlHistory": pnl_hist, "vlm": "0"}]]
