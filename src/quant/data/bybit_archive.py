"""Bybit public trade archive → 1-minute bars with order-flow columns.

Source: ``https://public.bybit.com/trading/<SYMBOL>/<SYMBOL><YYYY-MM-DD>.csv.gz`` — every
USDT-perpetual trade with the aggressor side. This host serves the full tick history and,
unlike the trading API, is reachable from the cloud environment (see docs/DECISIONS.md).

Each day is downloaded, aggregated and discarded; only the bars are kept:
``data/bars/bybit/<SYMBOL>/1m/<YYYY-MM-DD>.npz``. Minutes without trades are absent in a
file; the loader puts bars on a regular grid and marks empty minutes explicitly.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import tempfile
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from quant.app.logging import get_logger

log = get_logger(__name__)

ARCHIVE_URL = "https://public.bybit.com/trading/{symbol}/{symbol}{day}.csv.gz"
BAR_FIELDS = ("t", "o", "h", "l", "c", "v", "bv", "sv", "n", "vwap", "maxq")
MINUTE_MS = 60_000


class ArchiveError(RuntimeError):
    pass


def day_path(root: Path, symbol: str, day: dt.date) -> Path:
    return root / "bars" / "bybit" / symbol / "1m" / f"{day.isoformat()}.npz"


def aggregate_trades(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Trades (timestamp[s], side, size, price) → per-minute OHLCV + buy/sell volume.

    ``bv``/``sv`` are aggressive (taker) buy/sell volumes in base units, ``n`` the trade
    count, ``maxq`` the largest single trade — the raw material for order-flow features.
    """
    if df.empty:
        return {k: np.empty(0) for k in BAR_FIELDS}
    ts_ms = np.round(df["timestamp"].to_numpy(dtype=np.float64) * 1000.0).astype(np.int64)
    order = np.argsort(ts_ms, kind="stable")
    ts_ms = ts_ms[order]
    price = df["price"].to_numpy(dtype=np.float64)[order]
    size = df["size"].to_numpy(dtype=np.float64)[order]
    is_buy = df["side"].to_numpy()[order] == "Buy"
    minute = ts_ms // MINUTE_MS
    starts = np.flatnonzero(np.r_[True, minute[1:] != minute[:-1]])
    ends = np.r_[starts[1:], minute.size]
    buy_sz = np.where(is_buy, size, 0.0)
    notional = price * size
    return {
        "t": minute[starts] * MINUTE_MS,
        "o": price[starts],
        "h": np.maximum.reduceat(price, starts),
        "l": np.minimum.reduceat(price, starts),
        "c": price[ends - 1],
        "v": np.add.reduceat(size, starts),
        "bv": np.add.reduceat(buy_sz, starts),
        "sv": np.add.reduceat(size - buy_sz, starts),
        "n": (ends - starts).astype(np.float64),
        "vwap": np.add.reduceat(notional, starts) / np.add.reduceat(size, starts),
        "maxq": np.maximum.reduceat(size, starts),
    }


def parse_csv(raw_gz: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw_gz), compression="gzip", usecols=["timestamp", "side", "size", "price"])
    if not {"timestamp", "side", "size", "price"} <= set(df.columns):
        raise ArchiveError("unexpected archive columns")
    return df


def _atomic_save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)  # type: ignore[arg-type]
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class DayResult:
    day: dt.date
    status: str  # "ok" | "exists" | "missing" | "error"
    bars: int = 0
    detail: str = ""


def fetch_day(root: Path, symbol: str, day: dt.date, timeout: float = 120.0) -> DayResult:
    path = day_path(root, symbol, day)
    if path.exists():
        return DayResult(day, "exists")
    url = ARCHIVE_URL.format(symbol=symbol, day=day.isoformat())
    try:
        resp = httpx.get(url, timeout=timeout)
        if resp.status_code == 404:
            return DayResult(day, "missing")
        resp.raise_for_status()
        bars = aggregate_trades(parse_csv(resp.content))
    except (httpx.HTTPError, ArchiveError, ValueError, OSError, EOFError) as exc:
        return DayResult(day, "error", detail=repr(exc)[:200])
    day_start = int(dt.datetime.combine(day, dt.time(), tzinfo=dt.UTC).timestamp() * 1000)
    if bars["t"].size and (bars["t"][0] < day_start or bars["t"][-1] >= day_start + 86_400_000):
        return DayResult(day, "error", detail="trades outside the file's UTC day")
    _atomic_save(path, bars)
    return DayResult(day, "ok", int(bars["t"].size))


def date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def download(
    root: Path,
    symbol: str,
    days: Iterable[dt.date],
    workers: int = 4,
    progress: Callable[[DayResult], None] | None = None,
) -> list[DayResult]:
    """Download and aggregate many days in parallel processes (parsing is CPU-bound)."""
    results: list[DayResult] = []
    todo = [d for d in days if not day_path(root, symbol, d).exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_day, root, symbol, d) for d in todo]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if progress:
                progress(res)
    return sorted(results, key=lambda r: r.day)
