"""Load 1-minute bars onto a regular time grid and resample to coarser timeframes.

Empty minutes (no trades, or whole missing days) are kept as explicit rows with zero
volume, the previous close as price and ``valid = False``, so time-based lookbacks never
silently span more real time than intended. Strategies must not trade on invalid bars.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from quant.data.bybit_archive import MINUTE_MS, date_range, day_path

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Bars:
    interval_ms: int
    t: I64  # bar open time, ms UTC
    o: F64
    h: F64
    l: F64
    c: F64
    v: F64
    bv: F64  # aggressive buy volume
    sv: F64  # aggressive sell volume
    n: F64
    valid: B  # bar contains at least one trade

    def __len__(self) -> int:
        return int(self.t.size)

    def slice(self, start: int, end: int) -> Bars:
        s = slice(start, end)
        return Bars(
            self.interval_ms,
            self.t[s],
            self.o[s],
            self.h[s],
            self.l[s],
            self.c[s],
            self.v[s],
            self.bv[s],
            self.sv[s],
            self.n[s],
            self.valid[s],
        )

    def index_of(self, t_ms: int) -> int:
        """First bar with open time >= ``t_ms``."""
        return int(np.searchsorted(self.t, t_ms, side="left"))


def load_minutes(root: Path, symbol: str, start: dt.date, end: dt.date) -> Bars:
    t0 = int(dt.datetime.combine(start, dt.time(), tzinfo=dt.UTC).timestamp() * 1000)
    days = date_range(start, end)
    n = len(days) * 1440
    grid = t0 + np.arange(n, dtype=np.int64) * MINUTE_MS
    cols = {k: np.zeros(n) for k in ("o", "h", "l", "c", "v", "bv", "sv", "n")}
    valid = np.zeros(n, dtype=bool)
    for day in days:
        path = day_path(root, symbol, day)
        if not path.exists():
            continue
        with np.load(path) as z:
            idx = ((z["t"] - t0) // MINUTE_MS).astype(np.int64)
            for k, arr in cols.items():
                arr[idx] = z[k]
            valid[idx] = True
    # forward-fill prices over empty minutes (volume stays 0, valid stays False)
    last = np.where(valid, np.arange(n), -1)
    np.maximum.accumulate(last, out=last)
    has_prev = last >= 0
    c = np.where(has_prev, cols["c"][np.maximum(last, 0)], np.nan)
    for k in ("o", "h", "l"):
        cols[k] = np.where(valid, cols[k], c)
    return Bars(
        MINUTE_MS, grid, cols["o"], cols["h"], cols["l"], c, cols["v"], cols["bv"], cols["sv"], cols["n"], valid
    )


def resample(b: Bars, factor: int) -> Bars:
    """Aggregate ``factor`` consecutive bars (grid-aligned). A coarse bar is valid only if
    every constituent bar existed on the grid and at least one had trades."""
    m = len(b) // factor
    if m == 0:
        raise ValueError("not enough bars to resample")
    k = m * factor

    def blocks(x: np.ndarray) -> np.ndarray:
        return x[:k].reshape(m, factor)

    return Bars(
        b.interval_ms * factor,
        blocks(b.t)[:, 0].copy(),
        blocks(b.o)[:, 0].copy(),
        blocks(b.h).max(axis=1),
        blocks(b.l).min(axis=1),
        blocks(b.c)[:, -1].copy(),
        blocks(b.v).sum(axis=1),
        blocks(b.bv).sum(axis=1),
        blocks(b.sv).sum(axis=1),
        blocks(b.n).sum(axis=1),
        blocks(b.valid).any(axis=1) & ~np.isnan(blocks(b.c)).any(axis=1),
    )
