"""Per-trader performance metrics.

Two independent views are computed and kept separate:

* **trade view** — from reconstructed round trips (only the fill window the venue lets
  us see: at most the 10 000 most recent fills);
* **account view** — from the venue's account-value / cumulative-PnL history, which
  covers the full account life and is not distorted by deposits and withdrawals
  (returns use PnL deltas, not account-value deltas).
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np

from quant.stats.metrics import (
    MS_PER_YEAR,
    annualization_factor,
    longest_losing_streak,
    max_drawdown_abs,
    max_drawdown_from_returns,
    profit_factor,
    sharpe,
    sortino,
    t_statistic,
)
from quant.traders.models import Side
from quant.traders.reconstruct import ReconstructionResult

MS_PER_DAY = 86_400_000
MIN_RETURN_BASE_USD = 100.0  # ignore periods where the account was (nearly) empty


class Status(StrEnum):
    OK = "ok"
    INSUFFICIENT_DATA = "insufficient_data"


class Style(StrEnum):
    HFT = "hft"  # median hold < 1 min
    SCALPER = "scalper"  # < 30 min
    INTRADAY = "intraday"  # < 24 h
    SWING = "swing"


@dataclass(frozen=True, slots=True)
class AsofSeries:
    """Step function: value at t = last observation with time <= t."""

    times: tuple[int, ...]
    values: tuple[float, ...]

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[int, float]]) -> AsofSeries:
        merged = dict(pairs)
        keys = sorted(merged)
        return cls(tuple(keys), tuple(merged[k] for k in keys))

    def value_at(self, t_ms: int) -> float | None:
        idx = bisect.bisect_right(self.times, t_ms) - 1
        return None if idx < 0 else self.values[idx]


@dataclass(frozen=True, slots=True)
class PortfolioHistory:
    times: tuple[int, ...]
    account_value: tuple[float, ...]
    cum_pnl: tuple[float, ...]


def parse_portfolio(raw: Sequence[Any], prefer: Sequence[str] = ("perpAllTime", "allTime")) -> PortfolioHistory | None:
    """Parse the ``portfolio`` info response: ``[[window, {accountValueHistory, pnlHistory}], ...]``."""
    windows = {item[0]: item[1] for item in raw if isinstance(item, (list, tuple)) and len(item) == 2}
    for name in prefer:
        data = windows.get(name)
        if not isinstance(data, Mapping):
            continue
        av = {int(t): float(v) for t, v in data.get("accountValueHistory") or []}
        pnl = {int(t): float(v) for t, v in data.get("pnlHistory") or []}
        common = sorted(set(av) & set(pnl))
        if len(common) < 2:
            continue
        return PortfolioHistory(tuple(common), tuple(av[t] for t in common), tuple(pnl[t] for t in common))
    return None


@dataclass(slots=True)
class TraderMetrics:
    address: str
    cohort: str
    status: Status = Status.INSUFFICIENT_DATA
    style: Style | None = None
    flags: list[str] = field(default_factory=list)
    # trade view
    n_fills: int = 0
    n_trades: int = 0
    n_excluded_trades: int = 0
    first_trade_ms: int | None = None
    last_trade_ms: int | None = None
    active_days: float = 0.0
    trades_per_day: float | None = None
    win_rate: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None
    payoff_ratio: float | None = None
    profit_factor: float | None = None
    expectancy_usd: float | None = None
    expectancy_bps: float | None = None
    t_stat_bps: float | None = None
    net_pnl_trades: float = 0.0
    fees_total: float = 0.0
    funding_total: float = 0.0
    fees_to_gross_pnl: float | None = None
    median_hold_min: float | None = None
    mean_hold_min: float | None = None
    long_share: float | None = None
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    maker_fraction: float | None = None
    adds_against_share: float | None = None
    top_trade_share: float | None = None
    largest_loss: float | None = None
    longest_losing_streak: int = 0
    liquidations: int = 0
    positive_month_share: float | None = None
    n_months: int = 0
    leverage_median: float | None = None
    leverage_p95: float | None = None
    coins: dict[str, int] = field(default_factory=dict)
    entry_hour_hist: list[int] = field(default_factory=lambda: [0] * 24)
    pnl_crosscheck: float | None = None
    # account view
    account_value: float | None = None
    acct_span_days: float | None = None
    acct_total_pnl: float | None = None
    acct_return_total: float | None = None
    acct_cagr: float | None = None
    acct_sharpe: float | None = None
    acct_sortino: float | None = None
    acct_max_dd: float | None = None
    acct_max_dd_usd: float | None = None
    acct_calmar: float | None = None
    beta_btc: float | None = None
    corr_btc: float | None = None
    alpha_annual: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["style"] = self.style.value if self.style else None
        return d


def _style(median_hold_min: float) -> Style:
    if median_hold_min < 1:
        return Style.HFT
    if median_hold_min < 30:
        return Style.SCALPER
    if median_hold_min < 24 * 60:
        return Style.INTRADAY
    return Style.SWING


def compute_trade_metrics(m: TraderMetrics, rec: ReconstructionResult, account: AsofSeries | None) -> None:
    trips = rec.complete_trips
    m.n_fills = rec.n_fills
    m.n_trades = len(trips)
    m.n_excluded_trades = len(rec.trips) - len(trips)
    m.pnl_crosscheck = rec.pnl_crosscheck()
    if not trips:
        return
    pnls = np.array([float(t.net_pnl) for t in trips])
    bps = np.array([t.return_bps for t in trips])
    holds = np.array([t.holding_ms or 0 for t in trips], dtype=np.float64) / 60_000.0
    m.first_trade_ms = min(t.open_time_ms for t in trips)
    m.last_trade_ms = max(t.close_time_ms or t.open_time_ms for t in trips)
    m.active_days = max((m.last_trade_ms - m.first_trade_ms) / MS_PER_DAY, 1e-9)
    m.trades_per_day = m.n_trades / max(m.active_days, 1.0)
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    m.win_rate = float((pnls > 0).mean())
    m.avg_win = float(wins.mean()) if wins.size else None
    m.avg_loss = float(losses.mean()) if losses.size else None
    if m.avg_win is not None and m.avg_loss:
        m.payoff_ratio = m.avg_win / abs(m.avg_loss)
    m.profit_factor = profit_factor(pnls)
    m.expectancy_usd = float(pnls.mean())
    m.expectancy_bps = float(bps.mean())
    m.t_stat_bps = t_statistic(bps)
    m.net_pnl_trades = float(pnls.sum())
    m.fees_total = float(sum(t.fees for t in trips))
    m.funding_total = float(sum(t.funding for t in trips))
    gross_wins = float(sum(max(float(t.gross_pnl), 0.0) for t in trips))
    m.fees_to_gross_pnl = m.fees_total / gross_wins if gross_wins > 0 else None
    m.median_hold_min = float(np.median(holds))
    m.mean_hold_min = float(holds.mean())
    longs = [t for t in trips if t.side is Side.LONG]
    m.long_share = len(longs) / len(trips)
    m.long_pnl = float(sum(t.net_pnl for t in longs))
    m.short_pnl = m.net_pnl_trades - m.long_pnl
    total_notional = sum(float(t.total_notional) for t in trips)
    m.maker_fraction = (sum(float(t.maker_notional) for t in trips) / total_notional) if total_notional else None
    with_adds = [t for t in trips if t.n_adds > 0]
    m.adds_against_share = sum(1 for t in trips if t.adds_against > 0) / len(trips) if with_adds else 0.0
    if m.net_pnl_trades > 0 and wins.size:
        m.top_trade_share = float(wins.max()) / m.net_pnl_trades
    m.largest_loss = float(losses.min()) if losses.size else None
    m.longest_losing_streak = longest_losing_streak(pnls)
    m.liquidations = sum(1 for t in rec.trips if t.liquidated)
    m.coins = dict(Counter(t.coin for t in trips).most_common(10))
    for t in trips:
        m.entry_hour_hist[datetime.fromtimestamp(t.open_time_ms / 1000, tz=UTC).hour] += 1

    monthly: dict[str, list[float]] = defaultdict(list)
    for t, p in zip(trips, pnls, strict=True):
        close = t.close_time_ms or t.open_time_ms
        monthly[datetime.fromtimestamp(close / 1000, tz=UTC).strftime("%Y-%m")].append(float(p))
    months = [sum(v) for v in monthly.values() if len(v) >= 5]
    m.n_months = len(months)
    if months:
        m.positive_month_share = sum(1 for x in months if x > 0) / len(months)

    if account is not None:
        levs = []
        for t in trips:
            av = account.value_at(t.open_time_ms)
            if av and av > MIN_RETURN_BASE_USD:
                levs.append(float(t.peak_notional) / av)
        if levs:
            m.leverage_median = float(np.median(levs))
            m.leverage_p95 = float(np.percentile(levs, 95))
    m.style = _style(m.median_hold_min)


def compute_account_metrics(m: TraderMetrics, hist: PortfolioHistory, btc: AsofSeries | None) -> None:
    times = np.asarray(hist.times, dtype=np.int64)
    av = np.asarray(hist.account_value, dtype=np.float64)
    pnl = np.asarray(hist.cum_pnl, dtype=np.float64)
    m.account_value = float(av[-1])
    m.acct_span_days = float(times[-1] - times[0]) / MS_PER_DAY
    m.acct_total_pnl = float(pnl[-1] - pnl[0])
    m.acct_max_dd_usd = max_drawdown_abs(pnl - pnl[0])

    base = av[:-1]
    valid = base > MIN_RETURN_BASE_USD
    rets = np.where(valid, np.diff(pnl) / np.where(valid, base, 1.0), 0.0)
    rets = np.clip(rets, -0.9999, None)
    ret_times = times[1:][valid]
    rets = rets[valid]
    if rets.size < 2:
        return
    ppy = annualization_factor(list(times))
    if ppy is None:
        return
    growth = float(np.exp(np.log1p(rets).sum()))
    m.acct_return_total = growth - 1.0
    years = (float(times[-1] - times[0])) / MS_PER_YEAR
    if years > 0 and growth > 0:
        m.acct_cagr = growth ** (1.0 / years) - 1.0
    m.acct_sharpe = sharpe(rets, ppy)
    m.acct_sortino = sortino(rets, ppy)
    m.acct_max_dd = max_drawdown_from_returns(rets)
    if m.acct_cagr is not None and m.acct_max_dd and m.acct_max_dd > 0:
        m.acct_calmar = m.acct_cagr / m.acct_max_dd

    if btc is None:
        return
    prev_times = times[:-1][valid]
    pairs = []
    for t0, t1, r in zip(prev_times, ret_times, rets, strict=True):
        p0, p1 = btc.value_at(int(t0)), btc.value_at(int(t1))
        if p0 and p1:
            pairs.append((r, p1 / p0 - 1.0))
    if len(pairs) < 10:
        return
    arr = np.asarray(pairs)
    x, y = arr[:, 1], arr[:, 0]
    var_x = float(np.var(x, ddof=1))
    if var_x == 0 or float(np.std(y, ddof=1)) == 0:
        return
    beta = float(np.cov(x, y, ddof=1)[0, 1]) / var_x
    m.beta_btc = beta
    m.corr_btc = float(np.corrcoef(x, y)[0, 1])
    m.alpha_annual = (float(y.mean()) - beta * float(x.mean())) * ppy


def finalize(m: TraderMetrics, min_trades: int, min_history_days: float) -> None:
    """Assign status and behavioural flags. Flags describe; scoring decides."""
    flags: list[str] = []
    if m.maker_fraction is not None and m.maker_fraction >= 0.8 and (m.trades_per_day or 0) >= 50:
        flags.append("market_maker_like")
    if m.median_hold_min is not None and m.median_hold_min < 1 and (m.trades_per_day or 0) >= 200:
        flags.append("hft_like")
    if (m.adds_against_share or 0) > 0.25:
        flags.append("martingale_like")
    if (m.top_trade_share or 0) > 0.5:
        flags.append("concentrated_pnl")
    if m.liquidations > 0:
        flags.append("liquidated")
    if m.pnl_crosscheck is not None and m.pnl_crosscheck > 0.05:
        flags.append("pnl_crosscheck_failed")
    if m.leverage_p95 is not None and m.leverage_p95 > 20:
        flags.append("high_leverage")
    m.flags = flags
    enough = m.n_trades >= min_trades and m.active_days >= min_history_days
    m.status = Status.OK if enough and math.isfinite(m.net_pnl_trades) else Status.INSUFFICIENT_DATA
