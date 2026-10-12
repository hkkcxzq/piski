"""Experiment-005: multi-day time-series momentum (swing). See docs/research/preregistration-005.md."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from functools import partial

import numpy as np

from quant.backtest.engine import Signals
from quant.data.bars import Bars
from quant.data.macro_calendar import fomc_times_ms, overlaps_event
from quant.strategies.scalping import Variant, _finish, _s
from quant.strategies.trend import BLACKOUT_AFTER_MS, BLACKOUT_BEFORE_MS

DAY_MS = 86_400_000


def daily_vol(b: Bars, days: int = 30) -> np.ndarray:
    """Std of (overlapping) 1-day log returns over the trailing ``days`` (causal)."""
    per_day = DAY_MS // b.interval_ms
    logc = _s(np.log(b.c))
    r1d = logc - logc.shift(per_day)
    return r1d.rolling(days * per_day, min_periods=days * per_day // 2).std().to_numpy()


def swing_tsmom(b: Bars, lookback_d: int, hold_d: int, k: float, long_only: bool) -> Signals:
    per_day = DAY_MS // b.interval_ms
    logc = _s(np.log(b.c))
    ret = (logc - logc.shift(lookback_d * per_day)).to_numpy()
    side = np.where(np.isfinite(ret), np.sign(ret), 0).astype(int)
    if long_only:
        side = np.where(side > 0, 1, 0)
    stop = k * daily_vol(b) * b.c
    sig = _finish(b, side, stop, np.nan, hold_d * per_day, False, b.c)
    # entry-only FOMC blackout, exactly like the live engine (PAUSE_NEW inside the window)
    blocked = overlaps_event(b.t, b.interval_ms, 0, fomc_times_ms(), BLACKOUT_BEFORE_MS, BLACKOUT_AFTER_MS)
    sig.side[blocked] = 0
    return sig


def grid005() -> Iterator[Variant]:
    """The pre-registered 24 variants of experiment 005, in a fixed order."""
    for lb, hold, k, lo in itertools.product((7, 14, 28), (7, 14), (2.0, 3.0), (False, True)):
        yield Variant(
            "H-015-SWING-TSMOM",
            240,
            {"lookback_d": lb, "hold_d": hold, "k": k, "long_only": lo},
            partial(swing_tsmom, lookback_d=lb, hold_d=hold, k=k, long_only=lo),
        )
