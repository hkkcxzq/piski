"""Experiment-003: time-series momentum held up to one day, optional FOMC blackout."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from functools import partial

import numpy as np

from quant.backtest.engine import Signals
from quant.data.bars import Bars
from quant.data.macro_calendar import fomc_times_ms, overlaps_event
from quant.strategies.intraday import breakout_trend
from quant.strategies.scalping import Variant, _finish, _s, atr

HOUR_MS = 3_600_000
BLACKOUT_BEFORE_MS = 12 * HOUR_MS
BLACKOUT_AFTER_MS = 2 * HOUR_MS


def _apply_blackout(b: Bars, sig: Signals) -> Signals:
    hold = int(sig.max_hold.max()) if sig.max_hold.size else 0
    blocked = overlaps_event(b.t, b.interval_ms, hold, fomc_times_ms(), BLACKOUT_BEFORE_MS, BLACKOUT_AFTER_MS)
    sig.side[blocked] = 0
    return sig


def tsmom(b: Bars, lookback_h: int, hold_h: int, k_stop: float, fomc: bool) -> Signals:
    """Go with the sign of the return over the last ``lookback_h`` hours; exit after ``hold_h``
    hours or at a k·ATR stop. No take-profit: trends are allowed to run until the time exit."""
    per_h = HOUR_MS // b.interval_ms
    lb = lookback_h * per_h
    logc = _s(np.log(b.c))
    ret = (logc - logc.shift(lb)).to_numpy()  # NaN until enough history (also when lb >= len)
    side = np.where(np.isfinite(ret), np.sign(ret), 0).astype(int)
    stop = k_stop * atr(b, 14)
    sig = _finish(b, side, stop, np.nan, hold_h * per_h, False, b.c)
    return _apply_blackout(b, sig) if fomc else sig


def breakout_trend_fomc(b: Bars, n: int, k_stop: float, r: float) -> Signals:
    return _apply_blackout(b, breakout_trend(b, n, k_stop, r))


def grid003() -> Iterator[Variant]:
    """The pre-registered 25 variants of experiment 003, in a fixed order."""
    for lb, hold, k, fomc in itertools.product((24, 72, 168), (8, 24), (2.0, 4.0), (False, True)):
        yield Variant(
            "H-009-TSMOM",
            60,
            {"lookback_h": lb, "hold_h": hold, "k_stop": k, "fomc": fomc},
            partial(tsmom, lookback_h=lb, hold_h=hold, k_stop=k, fomc=fomc),
        )
    yield Variant(
        "H-010-BRK-TREND-FOMC",
        240,
        {"N": 24, "k_stop": 2.0, "R": 2.0, "fomc": True},
        partial(breakout_trend_fomc, n=24, k_stop=2.0, r=2.0),
    )
