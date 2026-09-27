"""Performance and significance statistics shared by trader research and backtests.

All functions are pure and operate on plain sequences; they return ``None`` when a
statistic is undefined for the input (too few observations, zero variance) instead of
producing misleading infinities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

MS_PER_YEAR = 365.25 * 24 * 3600 * 1000


def _arr(values: Sequence[float] | FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def sharpe(returns: Sequence[float] | FloatArray, periods_per_year: float) -> float | None:
    r = _arr(returns)
    if r.size < 2:
        return None
    sd = float(np.std(r, ddof=1))
    if sd == 0 or not math.isfinite(sd):
        return None
    return float(np.mean(r)) / sd * math.sqrt(periods_per_year)


def sortino(returns: Sequence[float] | FloatArray, periods_per_year: float) -> float | None:
    r = _arr(returns)
    if r.size < 2:
        return None
    downside = np.minimum(r, 0.0)
    dd = math.sqrt(float(np.mean(downside**2)))
    if dd == 0:
        return None
    return float(np.mean(r)) / dd * math.sqrt(periods_per_year)


def max_drawdown_from_returns(returns: Sequence[float] | FloatArray) -> float:
    """Maximum peak-to-trough decline of the compounded equity index, as a positive fraction."""
    r = _arr(returns)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    equity = np.concatenate(([1.0], equity))
    peaks = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peaks))


def max_drawdown_abs(cumulative: Sequence[float] | FloatArray) -> float:
    """Largest drop of a cumulative PnL curve from its running peak, in currency units."""
    c = _arr(cumulative)
    if c.size == 0:
        return 0.0
    c = np.concatenate(([0.0], c))
    return float(np.max(np.maximum.accumulate(c) - c))


def profit_factor(pnls: Sequence[float] | FloatArray) -> float | None:
    p = _arr(pnls)
    gains = float(p[p > 0].sum())
    losses = float(-p[p < 0].sum())
    if losses == 0:
        return None
    return gains / losses


def t_statistic(values: Sequence[float] | FloatArray) -> float | None:
    v = _arr(values)
    if v.size < 2:
        return None
    sd = float(np.std(v, ddof=1))
    if sd == 0:
        return None
    return float(np.mean(v)) / (sd / math.sqrt(v.size))


def longest_losing_streak(pnls: Sequence[float] | FloatArray) -> int:
    best = cur = 0
    for x in _arr(pnls):
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return best


def annualization_factor(timestamps_ms: Sequence[int]) -> float | None:
    """Periods per year implied by the median spacing of a (possibly irregular) time series."""
    if len(timestamps_ms) < 2:
        return None
    gaps = np.diff(np.asarray(timestamps_ms, dtype=np.float64))
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return None
    return MS_PER_YEAR / float(np.median(gaps))


def benjamini_hochberg(p_values: Sequence[float], q: float) -> list[bool]:
    """Benjamini–Hochberg FDR control. Returns a rejection flag per p-value (input order)."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    threshold_rank = 0
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= q * rank / m:
            threshold_rank = rank
    rejected = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= threshold_rank:
            rejected[idx] = True
    return rejected


def percentile_ranks(values: Sequence[float | None]) -> list[float | None]:
    """Average-rank percentile in [0, 1] among non-None values; None stays None."""
    present = [(v, i) for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    out: list[float | None] = [None] * len(values)
    n = len(present)
    if n == 0:
        return out
    if n == 1:
        out[present[0][1]] = 0.5
        return out
    present.sort()
    i = 0
    while i < n:
        j = i
        while j + 1 < n and present[j + 1][0] == present[i][0]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[present[k][1]] = rank / (n - 1)
        i = j + 1
    return out
