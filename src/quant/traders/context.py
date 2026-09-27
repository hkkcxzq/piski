"""Market context at a point in time, computed strictly from data available *before* it.

Look-ahead protection:
* a candle is usable at time ``t`` only if it has fully closed: ``open + interval <= t``;
* funding is the last settled rate with ``time <= t``;
* every lookback window is checked for gaps — if candles are missing, the feature is
  ``None`` rather than silently computed over a longer real-time span.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

DIRECTIONAL_FEATURES = (
    "z_ret_5m",
    "z_ret_15m",
    "z_ret_1h",
    "z_ret_4h",
    "z_ret_24h",
    "range_pos_24h",
    "funding",
    "funding_z",
)
NONDIRECTIONAL_FEATURES = ("vol_ratio_24h_7d", "volume_ratio_4h")
ALL_FEATURES = DIRECTIONAL_FEATURES + NONDIRECTIONAL_FEATURES


@dataclass(frozen=True, slots=True)
class CandleSeries:
    interval_ms: int
    open_ms: IntArray
    close: FloatArray
    high: FloatArray
    low: FloatArray
    volume: FloatArray

    @classmethod
    def from_raw(cls, rows: Sequence[Mapping[str, Any]], interval_ms: int) -> CandleSeries:
        by_open = {int(r["t"]): r for r in rows}
        keys = sorted(by_open)

        def get(name: str) -> FloatArray:
            return np.array([float(by_open[k][name]) for k in keys], dtype=np.float64)

        return cls(interval_ms, np.array(keys, dtype=np.int64), get("c"), get("h"), get("l"), get("v"))

    def __len__(self) -> int:
        return int(self.open_ms.size)

    def last_closed_index(self, t_ms: int) -> int:
        """Index of the last candle with ``open + interval <= t_ms``; -1 if none."""
        return int(np.searchsorted(self.open_ms + self.interval_ms, t_ms, side="right")) - 1

    def contiguous(self, start: int, end: int) -> bool:
        """True if candles ``start..end`` (inclusive) have no gaps."""
        if start < 0 or end >= len(self) or start > end:
            return False
        return int(self.open_ms[end] - self.open_ms[start]) == (end - start) * self.interval_ms

    def truncated(self, t_ms: int) -> CandleSeries:
        """Copy with only candles closed by ``t_ms`` (used by look-ahead tests)."""
        n = self.last_closed_index(t_ms) + 1
        return CandleSeries(
            self.interval_ms, self.open_ms[:n], self.close[:n], self.high[:n], self.low[:n], self.volume[:n]
        )


@dataclass(frozen=True, slots=True)
class FundingSeries:
    time_ms: IntArray
    rate: FloatArray

    @classmethod
    def from_raw(cls, rows: Sequence[Mapping[str, Any]]) -> FundingSeries:
        by_t = {int(r["time"]): float(r["fundingRate"]) for r in rows}
        keys = sorted(by_t)
        return cls(np.array(keys, dtype=np.int64), np.array([by_t[k] for k in keys], dtype=np.float64))

    def last_index(self, t_ms: int) -> int:
        return int(np.searchsorted(self.time_ms, t_ms, side="right")) - 1


@dataclass(slots=True)
class MarketContext:
    candles_1h: dict[str, CandleSeries] = field(default_factory=dict)
    candles_5m: dict[str, CandleSeries] = field(default_factory=dict)
    funding: dict[str, FundingSeries] = field(default_factory=dict)

    def coins(self) -> set[str]:
        return set(self.candles_1h)


def _log_ret(c: FloatArray, i: int, k: int) -> float:
    return math.log(c[i] / c[i - k])


def _sigma(c: FloatArray, i: int, n: int) -> float | None:
    rets = np.diff(np.log(c[i - n : i + 1]))
    sd = float(np.std(rets, ddof=1)) if rets.size >= 2 else 0.0
    return sd if sd > 0 and math.isfinite(sd) else None


def features_at(ctx: MarketContext, coin: str, t_ms: int) -> dict[str, float | None]:
    out: dict[str, float | None] = dict.fromkeys(ALL_FEATURES)
    h1 = ctx.candles_1h.get(coin)
    if h1 is not None:
        i = h1.last_closed_index(t_ms)
        if i >= 168 and h1.contiguous(i - 168, i):
            c = h1.close
            s7 = _sigma(c, i, 168)
            s1 = _sigma(c, i, 24)
            if s7 is not None:
                out["z_ret_1h"] = _log_ret(c, i, 1) / s7
                out["z_ret_4h"] = _log_ret(c, i, 4) / (s7 * 2.0)
                out["z_ret_24h"] = _log_ret(c, i, 24) / (s7 * math.sqrt(24))
                if s1 is not None:
                    out["vol_ratio_24h_7d"] = s1 / s7
            hi = float(h1.high[i - 23 : i + 1].max())
            lo = float(h1.low[i - 23 : i + 1].min())
            if hi > lo:
                out["range_pos_24h"] = (float(c[i]) - lo) / (hi - lo) - 0.5  # centred: +0.5 at high
            week_vol = float(h1.volume[i - 167 : i + 1].sum())
            if week_vol > 0:
                out["volume_ratio_4h"] = float(h1.volume[i - 3 : i + 1].sum()) / (week_vol / 42.0)
    m5 = ctx.candles_5m.get(coin)
    if m5 is not None:
        j = m5.last_closed_index(t_ms)
        if j >= 288 and m5.contiguous(j - 288, j):
            s = _sigma(m5.close, j, 288)
            if s is not None:
                out["z_ret_5m"] = _log_ret(m5.close, j, 1) / s
                out["z_ret_15m"] = _log_ret(m5.close, j, 3) / (s * math.sqrt(3))
    fs = ctx.funding.get(coin)
    if fs is not None:
        k = fs.last_index(t_ms)
        if k >= 0:
            out["funding"] = float(fs.rate[k])
            if k >= 168:
                window = fs.rate[k - 168 : k]
                sd = float(np.std(window, ddof=1))
                if sd > 0:
                    out["funding_z"] = (float(fs.rate[k]) - float(window.mean())) / sd
    return out
