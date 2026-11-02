"""Experiment-004: positioning hypotheses (funding, open interest, long/short ratios).

Positioning series come from Binance (``quant.data.binance_derivs``) and are aligned to the
bar grid causally: the value used at bar ``i`` is the latest one published at or before
``close_i - lag`` (see ``align``). See docs/research/preregistration-004.md.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quant.backtest.engine import Signals
from quant.data.bars import Bars
from quant.data.binance_derivs import load_funding, load_metrics
from quant.strategies.scalping import Variant, _finish, _s, atr, ema
from quant.strategies.trend import _apply_blackout

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
METRICS_LAG_MS = 5 * 60_000
METRICS_STALE_MS = HOUR_MS
FUNDING_STALE_MS = 12 * HOUR_MS
PCT_WINDOW_DAYS = 90
Z_WINDOW_DAYS = 30


def align(src_t: np.ndarray, src_v: np.ndarray, bar_close: np.ndarray, lag_ms: int, stale_ms: int) -> np.ndarray:
    """Latest source value with time <= close - lag; NaN if none or older than ``stale_ms``."""
    cutoff = bar_close - lag_ms
    idx = np.searchsorted(src_t, cutoff, side="right") - 1
    ok = idx >= 0
    safe = np.maximum(idx, 0)
    out = np.where(ok, src_v[safe] if src_v.size else np.nan, np.nan)
    if src_t.size:
        out = np.where(ok & (cutoff - src_t[safe] <= stale_ms), out, np.nan)
    return out.astype(np.float64)


@dataclass(frozen=True, slots=True)
class Positioning:
    oi: np.ndarray
    top_ls: np.ndarray
    acc_ls: np.ndarray
    funding: np.ndarray


_CACHE: dict[tuple[str, str, int, int, int], Positioning] = {}


def positioning(root: Path, symbol: str, b: Bars) -> Positioning:
    key = (str(root), symbol, b.interval_ms, int(b.t[0]), len(b))
    if key in _CACHE:
        return _CACHE[key]
    start = dt.datetime.fromtimestamp(int(b.t[0]) / 1000, dt.UTC).date()
    end = dt.datetime.fromtimestamp(int(b.t[-1]) / 1000, dt.UTC).date()
    m = load_metrics(root, symbol, start, end)
    f = load_funding(root, symbol, start, end)
    close = b.t + b.interval_ms
    empty = np.empty(0)

    def mcol(name: str) -> np.ndarray:
        return align(m.get("t", empty.astype(np.int64)), m.get(name, empty), close, METRICS_LAG_MS, METRICS_STALE_MS)

    p = Positioning(
        mcol("oi"),
        mcol("top_pos_ls"),
        mcol("acc_ls"),
        align(f.get("t", empty.astype(np.int64)), f.get("rate", empty), close, 0, FUNDING_STALE_MS),
    )
    _CACHE[key] = p
    return p


def _bars(b: Bars, ms: int) -> int:
    return max(1, ms // b.interval_ms)


def rolling_pct(x: np.ndarray, window: int) -> np.ndarray:
    """Percentile rank of the current value within the trailing window (inclusive, causal)."""
    return _s(x).rolling(window, min_periods=window // 2).rank(pct=True).to_numpy()


def rolling_z(x: np.ndarray, window: int) -> np.ndarray:
    s = _s(x)
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return ((s - mu) / sd).to_numpy()


def _log_change(x: np.ndarray, lag: int) -> np.ndarray:
    lx = np.log(np.where(x > 0, x, np.nan))
    return (_s(lx) - _s(lx).shift(lag)).to_numpy()


def fund_fade(b: Bars, p: Positioning, q: float, hold_h: int, k: float, tp: float) -> Signals:
    pct = rolling_pct(p.funding, _bars(b, PCT_WINDOW_DAYS * DAY_MS))
    side = np.where(pct > q, -1, np.where(pct < 1 - q, 1, 0))
    side = np.where(np.isfinite(pct), side, 0)
    return _apply_blackout(b, _finish(b, side, k * atr(b, 14), tp, _bars(b, hold_h * HOUR_MS), False, b.c))


def oi_breakout(b: Bars, p: Positioning, n: int, pctl: float, trend: bool, hold_h: int) -> Signals:
    hi = _s(b.h).rolling(n).max().shift(1).to_numpy()
    lo = _s(b.l).rolling(n).min().shift(1).to_numpy()
    d_oi = _log_change(p.oi, _bars(b, DAY_MS))
    oi_up = rolling_pct(d_oi, _bars(b, PCT_WINDOW_DAYS * DAY_MS)) > pctl
    long_ = (b.c > hi) & oi_up
    short = (b.c < lo) & oi_up
    if trend:
        e = ema(b.c, 200)
        long_ &= b.c > e
        short &= b.c < e
    side = np.where(long_, 1, np.where(short, -1, 0))
    return _apply_blackout(b, _finish(b, side, 2.0 * atr(b, 14), 2.0, _bars(b, hold_h * HOUR_MS), False, b.c))


def flush_fade(b: Bars, p: Positioning, w_h: int, pct: float, hold_h: int, k: float) -> Signals:
    w = _bars(b, w_h * HOUR_MS)
    win = _bars(b, PCT_WINDOW_DAYS * DAY_MS)
    ret = _log_change(b.c, w)
    d_oi = _log_change(p.oi, w)
    big_move = rolling_pct(np.abs(ret), win) > pct
    deleverage = rolling_pct(d_oi, win) < 1 - pct
    hit = big_move & deleverage
    side = np.where(hit & (ret < 0), 1, np.where(hit & (ret > 0), -1, 0))
    return _apply_blackout(b, _finish(b, side, k * atr(b, 14), 2.0, _bars(b, hold_h * HOUR_MS), False, b.c))


def top_vs_crowd(b: Bars, p: Positioning, thr: float, hold_h: int, k: float, tp: float) -> Signals:
    win = _bars(b, Z_WINDOW_DAYS * DAY_MS)
    top = np.log(np.where(p.top_ls > 0, p.top_ls, np.nan))
    acc = np.log(np.where(p.acc_ls > 0, p.acc_ls, np.nan))
    d = rolling_z(top, win) - rolling_z(acc, win)
    side = np.where(d > thr, 1, np.where(d < -thr, -1, 0))
    side = np.where(np.isfinite(d), side, 0)
    return _apply_blackout(b, _finish(b, side, k * atr(b, 14), tp, _bars(b, hold_h * HOUR_MS), False, b.c))


def _with_positioning(fn: Callable[..., Signals], root: Path, **params: object) -> Callable[[Bars, str], Signals]:
    def build(b: Bars, symbol: str) -> Signals:
        return fn(b, positioning(root, symbol, b), **params)

    return build


def grid004(root: Path) -> Iterator[Variant]:
    """The pre-registered 64 variants of experiment 004, in a fixed order."""
    nan = float("nan")
    for q, hold, k, tp in itertools.product((0.90, 0.97), (24, 72), (2.0, 3.0), (nan, 2.0)):
        yield Variant(
            "H-011-FUND-FADE",
            240,
            {"q": q, "hold_h": hold, "k": k, "tp": tp},
            symbol_aware=True,
            build=_with_positioning(fund_fade, root, q=q, hold_h=hold, k=k, tp=tp),
        )
    for n, pctl, trend, hold in itertools.product((12, 24), (0.5, 0.8), (False, True), (24, 72)):
        yield Variant(
            "H-012-OI-BREAKOUT",
            240,
            {"N": n, "p": pctl, "trend": trend, "hold_h": hold},
            symbol_aware=True,
            build=_with_positioning(oi_breakout, root, n=n, pctl=pctl, trend=trend, hold_h=hold),
        )
    for w, pct, hold, k in itertools.product((4, 24), (0.90, 0.97), (8, 24), (2.0, 3.0)):
        yield Variant(
            "H-013-FLUSH-FADE",
            60,
            {"W_h": w, "pct": pct, "hold_h": hold, "k": k},
            symbol_aware=True,
            build=_with_positioning(flush_fade, root, w_h=w, pct=pct, hold_h=hold, k=k),
        )
    for thr, hold, k, tp in itertools.product((1.0, 2.0), (24, 72), (2.0, 3.0), (nan, 2.0)):
        yield Variant(
            "H-014-TOP-VS-CROWD",
            240,
            {"thr": thr, "hold_h": hold, "k": k, "tp": tp},
            symbol_aware=True,
            build=_with_positioning(top_vs_crowd, root, thr=thr, hold_h=hold, k=k, tp=tp),
        )
