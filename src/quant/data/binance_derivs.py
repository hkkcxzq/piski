"""Binance USDⓈ-M futures positioning history from the public archive ``data.binance.vision``.

* ``metrics`` (daily files, 5-minute rows): open interest, top-trader and all-account long/short
  ratios, taker buy/sell volume ratio;
* ``fundingRate`` (monthly files): every funding settlement.

Binance is the largest perpetual venue, so its positioning is used as a market-wide signal; orders
are still placed on Bybit. Files are converted to ``.npz`` and the zips discarded:
``data/derivs/binance/<SYMBOL>/metrics/<YYYY-MM-DD>.npz`` and ``.../funding/<YYYY-MM>.npz``.

Timing: a metrics row stamped ``create_time`` describes the state *at* that time; research code
must only use it for decisions made after it (the loaders keep the timestamp, nothing is shifted).
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from quant.data.bybit_archive import DayResult, _atomic_save

BASE = "https://data.binance.vision/data/futures/um"
METRICS_URL = BASE + "/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"
FUNDING_URL = BASE + "/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"

METRICS_COLUMNS = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
# short names used in the npz files
METRICS_FIELDS = {
    "sum_open_interest": "oi",
    "sum_open_interest_value": "oi_value",
    "count_toptrader_long_short_ratio": "top_acc_ls",
    "sum_toptrader_long_short_ratio": "top_pos_ls",
    "count_long_short_ratio": "acc_ls",
    "sum_taker_long_short_vol_ratio": "taker_ls",
}
FUNDING_COLUMNS = ("calc_time", "funding_interval_hours", "last_funding_rate")


def _csv_from_zip(raw: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one csv in zip, got {names}")
        return zf.read(names[0])


def _read_csv(csv: bytes, columns: tuple[str, ...]) -> pd.DataFrame:
    """Some archive files have a header row and some do not."""
    first = csv.split(b"\n", 1)[0].decode("utf-8", "replace")
    has_header = first.split(",")[0].strip() == columns[0]
    df = pd.read_csv(io.BytesIO(csv), header=0 if has_header else None)
    if not has_header:
        if df.shape[1] != len(columns):
            raise ValueError(f"expected {len(columns)} columns, got {df.shape[1]}")
        df.columns = list(columns)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    return df


def parse_metrics(raw_zip: bytes) -> dict[str, np.ndarray]:
    df = _read_csv(_csv_from_zip(raw_zip), METRICS_COLUMNS)
    ts = pd.to_datetime(df["create_time"], utc=True)
    # resolution-independent (pandas may store s, ms, us or ns)
    t = ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(dtype=np.int64)
    out = {"t": t}
    for col, name in METRICS_FIELDS.items():
        out[name] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    order = np.argsort(out["t"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def parse_funding(raw_zip: bytes) -> dict[str, np.ndarray]:
    df = _read_csv(_csv_from_zip(raw_zip), FUNDING_COLUMNS)
    t = pd.to_numeric(df["calc_time"]).to_numpy(dtype=np.int64)
    rate = pd.to_numeric(df["last_funding_rate"], errors="coerce").to_numpy(dtype=np.float64)
    order = np.argsort(t, kind="stable")
    return {"t": t[order], "rate": rate[order]}


def metrics_path(root: Path, symbol: str, day: dt.date) -> Path:
    return root / "derivs" / "binance" / symbol / "metrics" / f"{day.isoformat()}.npz"


def funding_path(root: Path, symbol: str, month: str) -> Path:
    return root / "derivs" / "binance" / symbol / "funding" / f"{month}.npz"


def _fetch(url: str, path: Path, parse: Callable[[bytes], dict[str, np.ndarray]], key: dt.date) -> DayResult:
    if path.exists():
        return DayResult(key, "exists")
    try:
        resp = httpx.get(url, timeout=60.0, follow_redirects=True)
        if resp.status_code == 404:
            return DayResult(key, "missing")
        resp.raise_for_status()
        arrays = parse(resp.content)
    except (httpx.HTTPError, ValueError, OSError, zipfile.BadZipFile) as exc:
        return DayResult(key, "error", detail=repr(exc)[:200])
    _atomic_save(path, arrays)
    return DayResult(key, "ok", int(arrays["t"].size))


def fetch_metrics_day(root: Path, symbol: str, day: dt.date) -> DayResult:
    url = METRICS_URL.format(symbol=symbol, day=day.isoformat())
    return _fetch(url, metrics_path(root, symbol, day), parse_metrics, day)


def fetch_funding_month(root: Path, symbol: str, month: dt.date) -> DayResult:
    m = month.strftime("%Y-%m")
    return _fetch(FUNDING_URL.format(symbol=symbol, month=m), funding_path(root, symbol, m), parse_funding, month)


def months(start: dt.date, end: dt.date) -> list[dt.date]:
    out, cur = [], start.replace(day=1)
    while cur <= end:
        out.append(cur)
        cur = (cur + dt.timedelta(days=32)).replace(day=1)
    return out


def download(
    root: Path,
    symbol: str,
    days: Iterable[dt.date],
    workers: int = 8,
    progress: Callable[[DayResult], None] | None = None,
) -> list[DayResult]:
    """Metrics for every day plus funding for every month touched (I/O-bound → threads)."""
    days = list(days)
    jobs: list[tuple[Callable[[Path, str, dt.date], DayResult], dt.date]] = [(fetch_metrics_day, d) for d in days]
    if days:
        jobs += [(fetch_funding_month, m) for m in months(min(days), max(days))]
    results: list[DayResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, root, symbol, d) for fn, d in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if progress:
                progress(res)
    return sorted(results, key=lambda r: r.day)


def _load(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    parts = []
    for p in paths:
        if p.exists():
            with np.load(p) as z:
                parts.append({k: z[k] for k in z.files})
    if not parts:
        return {}
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(out["t"], kind="stable")
    t = out["t"][order]
    keep = np.r_[True, t[1:] != t[:-1]]  # files can overlap at month/day boundaries
    return {k: v[order][keep] for k, v in out.items()}


def load_metrics(root: Path, symbol: str, start: dt.date, end: dt.date) -> dict[str, np.ndarray]:
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    return _load(metrics_path(root, symbol, d) for d in days)


def load_funding(root: Path, symbol: str, start: dt.date, end: dt.date) -> dict[str, np.ndarray]:
    return _load(funding_path(root, symbol, m.strftime("%Y-%m")) for m in months(start, end))
