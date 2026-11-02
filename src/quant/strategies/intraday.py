"""Experiment-002 intraday hypotheses (15m–4h). See docs/research/preregistration-002.md."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from functools import partial

import numpy as np

from quant.backtest.engine import Signals
from quant.data.bars import Bars
from quant.strategies.scalping import Variant, _finish, _s, atr, ema, fvg, squeeze

HOLD_HOURS = 24  # every intraday position is closed within a day


def _hold(tf_min: int) -> int:
    return HOLD_HOURS * 60 // tf_min


def breakout_trend(b: Bars, n: int, k_stop: float, r: float) -> Signals:
    """Close beyond the prior N-bar high/low in the direction of the EMA-200 trend; stop k·ATR."""
    hi = _s(b.h).rolling(n).max().shift(1).to_numpy()
    lo = _s(b.l).rolling(n).min().shift(1).to_numpy()
    e = ema(b.c, 200)
    side = np.where((b.c > hi) & (b.c > e), 1, np.where((b.c < lo) & (b.c < e), -1, 0))
    stop = k_stop * atr(b, 14)
    return _finish(b, side, stop, r, _hold(b.interval_ms // 60_000), False, b.c)


def band_reversion(b: Bars, k: float, r: float, trend: bool) -> Signals:
    """Resting limit order k·ATR away from EMA-20 (fade the stretch), stop 1.5 ATR beyond the limit.

    Maker entry by design: the idea is to be paid the spread/fees instead of paying them.
    With ``trend`` only fades that go *with* the EMA-200 direction are placed (buy dips in uptrends).
    """
    a = atr(b, 14)
    e = ema(b.c, 20)
    buy_px = e - k * a
    sell_px = e + k * a
    long_ok = np.isfinite(buy_px) & (b.c > buy_px)
    short_ok = np.isfinite(sell_px) & (b.c < sell_px)
    if trend:
        slow = ema(b.c, 200)
        long_ok &= b.c > slow
        short_ok &= b.c < slow
    # one side per bar: the band nearer to the current price
    choose_long = long_ok & (~short_ok | (b.c - buy_px < sell_px - b.c))
    choose_short = short_ok & ~choose_long
    side = np.where(choose_long, 1, np.where(choose_short, -1, 0))
    entry = np.where(choose_long, buy_px, np.where(choose_short, sell_px, np.nan))
    stop = 1.5 * a
    return _finish(b, side, stop, r, _hold(b.interval_ms // 60_000), False, np.where(side != 0, entry, b.c), entry, 1)


def grid002() -> Iterator[Variant]:
    """The pre-registered 64 variants of experiment 002, in a fixed order."""
    for tf, n, ratio, r in itertools.product((15, 60), (12, 24), (0.5, 0.7), (2.0, 3.0)):
        yield Variant(
            "H-005-SQZ-ID",
            tf,
            {"N": n, "ratio": ratio, "R": r},
            partial(squeeze, n=n, ratio=ratio, r=r, active=False, hold=_hold(tf)),
        )
    for tf, n, k, r in itertools.product((60, 240), (24, 72), (2.0, 3.0), (2.0, 4.0)):
        yield Variant("H-006-BRK-TREND", tf, {"N": n, "k_stop": k, "R": r}, partial(breakout_trend, n=n, k_stop=k, r=r))
    for tf, k, r, tr in itertools.product((15, 60), (2.0, 3.0), (1.0, 2.0), (False, True)):
        yield Variant("H-007-BAND-MR", tf, {"k": k, "R": r, "trend": tr}, partial(band_reversion, k=k, r=r, trend=tr))
    for tf, g, r, tr in itertools.product((60, 240), (0.3, 0.6), (2.0, 3.0), (False, True)):
        yield Variant(
            "H-008-FVG-ID",
            tf,
            {"min_gap": g, "R": r, "trend": tr},
            partial(fvg, min_gap=g, r=r, trend=tr, active=False, hold=_hold(tf)),
        )
