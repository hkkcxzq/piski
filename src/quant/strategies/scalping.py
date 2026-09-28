"""Experiment-001 scalping hypotheses as causal signal generators.

Every value used for a decision at bar ``i`` is computed from bars ``<= i`` (pandas rolling
windows end at the current row). The engine then acts from bar ``i + 1``.
See docs/research/preregistration-001.md for the hypotheses and grids.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from quant.backtest.engine import Signals
from quant.data.bars import Bars

MIN_STOP_FRAC = 0.0025  # stops closer than 0.25 % are skipped: costs would eat > half the risk
MINUTES_30D = 30 * 1440


def _s(x: np.ndarray) -> pd.Series:
    return pd.Series(x)


def atr(b: Bars, n: int) -> np.ndarray:
    prev_c = np.r_[np.nan, b.c[:-1]]
    tr = np.nanmax(np.vstack([b.h - b.l, np.abs(b.h - prev_c), np.abs(b.l - prev_c)]), axis=0)
    return _s(tr).rolling(n, min_periods=n).mean().to_numpy()


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return _s(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


_ACTIVE_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def active_mask(b: Bars) -> np.ndarray:
    """Realized vol over the last hour above its own 30-day median (both causal). Cached per series."""
    key = (id(b.c), len(b), b.interval_ms)
    cached = _ACTIVE_CACHE.get(key)
    if cached is not None:
        return cached
    _ACTIVE_CACHE[key] = mask = _active_mask(b)
    return mask


def _active_mask(b: Bars) -> np.ndarray:
    per_hour = max(1, 3_600_000 // b.interval_ms)
    per_30d = max(per_hour, MINUTES_30D * 60_000 // b.interval_ms)
    r = np.r_[0.0, np.diff(np.log(b.c))]
    rv = _s(r).rolling(per_hour, min_periods=per_hour).std()
    med = rv.rolling(per_30d, min_periods=per_30d // 2).median()
    return (rv > med).to_numpy()


def _finish(
    b: Bars,
    side: np.ndarray,
    stop: np.ndarray,
    tp_r: float,
    hold: int,
    active: bool,
    ref_px: np.ndarray,
    entry_px: np.ndarray | None = None,
    expiry: int = 0,
) -> Signals:
    n = len(b)
    ok = (side != 0) & np.isfinite(stop) & (stop / ref_px >= MIN_STOP_FRAC) & b.valid
    if active:
        ok &= active_mask(b)
    sig = Signals.empty(n)
    sig.side[ok] = side[ok].astype(np.int8)
    sig.stop_dist[ok] = stop[ok]
    sig.tp_dist[ok] = tp_r * stop[ok]
    sig.max_hold[ok] = hold
    if entry_px is not None:
        sig.entry_px[ok] = entry_px[ok]
        sig.expiry[ok] = expiry
    return sig


def fvg(b: Bars, min_gap: float, r: float, trend: bool, active: bool) -> Signals:
    a = atr(b, 14)
    h2 = np.r_[np.nan, np.nan, b.h[:-2]]
    l2 = np.r_[np.nan, np.nan, b.l[:-2]]
    v3 = b.valid & np.r_[False, b.valid[:-1]] & np.r_[False, False, b.valid[:-2]]
    bull = v3 & (b.l - h2 >= min_gap * a)
    bear = v3 & (l2 - b.h >= min_gap * a)
    if trend:
        e = ema(b.c, 50)
        slope = e - np.r_[np.full(10, np.nan), e[:-10]]
        bull &= slope > 0
        bear &= slope < 0
    both = bull & bear
    bull &= ~both
    bear &= ~both
    side = np.where(bull, 1, np.where(bear, -1, 0))
    entry = np.where(bull, b.l, np.where(bear, b.h, np.nan))  # retrace to the near edge of the gap
    far = np.where(bull, h2 - 0.1 * a, np.where(bear, l2 + 0.1 * a, np.nan))
    stop = np.abs(entry - far)
    return _finish(b, side, stop, r, 48, active, np.where(np.isfinite(entry), entry, b.c), entry, 12)


def exhaustion(b: Bars, thr: float, r: float, hold: int, active: bool) -> Signals:
    lr = np.r_[0.0, np.diff(np.log(b.c))]
    sigma = _s(lr).rolling(1440, min_periods=720).std().to_numpy()
    move = _s(np.log(b.c)).diff(3).to_numpy()
    z = move / (sigma * np.sqrt(3))
    vol3 = _s(b.v).rolling(3).sum().to_numpy()
    base = _s(b.v).rolling(1440, min_periods=720).mean().to_numpy() * 3
    hit = (np.abs(z) > thr) & (vol3 > 3 * base)
    side = np.where(hit, -np.sign(z), 0).astype(int)
    a = atr(b, 60)
    hi3 = _s(b.h).rolling(3).max().to_numpy()
    lo3 = _s(b.l).rolling(3).min().to_numpy()
    stop = np.where(side < 0, hi3 + 0.25 * a - b.c, np.where(side > 0, b.c - (lo3 - 0.25 * a), np.nan))
    return _finish(b, side, stop, r, hold, active, b.c)


def squeeze(b: Bars, n: int, ratio: float, r: float, active: bool) -> Signals:
    hi = _s(b.h).rolling(n).max().shift(1).to_numpy()
    lo = _s(b.l).rolling(n).min().shift(1).to_numpy()
    width = hi - lo
    week = max(n, 7 * 86_400_000 // b.interval_ms)
    rel = width / _s(width).rolling(week, min_periods=week // 2).mean().to_numpy()
    vol_ok = b.v > 1.5 * _s(b.v).rolling(288, min_periods=144).mean().to_numpy()
    compressed = rel < ratio
    long_ = compressed & vol_ok & (b.c > hi)
    short = compressed & vol_ok & (b.c < lo)
    side = np.where(long_, 1, np.where(short, -1, 0))
    mid = (hi + lo) / 2
    stop = np.where(side != 0, np.abs(b.c - mid), np.nan)
    return _finish(b, side, stop, r, 48, active, b.c)


def flow(b: Bars, k: int, imb: float, mode: str, active: bool) -> Signals:
    net = _s(b.bv - b.sv).rolling(k).sum().to_numpy()
    tot = _s(b.v).rolling(k).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = net / tot
    base = _s(b.v).rolling(1440, min_periods=720).mean().to_numpy() * k
    hit = (np.abs(ratio) > imb) & (tot > 2 * base)
    direction = np.sign(ratio) if mode == "cont" else -np.sign(ratio)
    side = np.where(hit, direction, 0).astype(int)
    stop = 1.5 * atr(b, 60)
    return _finish(b, side, stop, 1.5, 30, active, b.c)


@dataclass(frozen=True, slots=True)
class Variant:
    hypothesis: str
    timeframe_min: int
    params: dict[str, Any]
    build: Callable[[Bars], Signals]

    @property
    def key(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.hypothesis}[{self.timeframe_min}m;{p}]"


def grid() -> Iterator[Variant]:
    """The pre-registered 80 variants, in a fixed order."""
    for tf, g, r, tr, act in itertools.product((5, 15), (0.3, 0.6), (1.5, 2.5), (False, True), (False, True)):
        yield Variant(
            "H-001-FVG",
            tf,
            {"min_gap": g, "R": r, "trend": tr, "active": act},
            partial(fvg, min_gap=g, r=r, trend=tr, active=act),
        )
    for thr, r, hold, act in itertools.product((3.0, 4.5), (1.0, 2.0), (15, 60), (False, True)):
        yield Variant(
            "H-002-EXH",
            1,
            {"thr": thr, "R": r, "hold": hold, "active": act},
            partial(exhaustion, thr=thr, r=r, hold=hold, active=act),
        )
    for n, ratio, r, act in itertools.product((12, 36), (0.5, 0.7), (1.5, 3.0), (False, True)):
        yield Variant(
            "H-003-SQZ",
            5,
            {"N": n, "ratio": ratio, "R": r, "active": act},
            partial(squeeze, n=n, ratio=ratio, r=r, active=act),
        )
    for k, imb, mode, act in itertools.product((5, 15), (0.3, 0.5), ("cont", "rev"), (False, True)):
        yield Variant(
            "H-004-FLOW",
            1,
            {"k": k, "imb": imb, "mode": mode, "active": act},
            partial(flow, k=k, imb=imb, mode=mode, active=act),
        )
