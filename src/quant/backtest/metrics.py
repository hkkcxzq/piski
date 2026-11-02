"""Trade-list statistics with risk-based position sizing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from quant.backtest.engine import Trades
from quant.stats.metrics import max_drawdown_from_returns, profit_factor, sharpe, t_statistic

DAY_MS = 86_400_000


def equity_returns(trades: Trades, risk: float = 0.0025, max_leverage: float = 5.0) -> npt.NDArray[np.float64]:
    """Per-trade return on equity: risk a fixed fraction to the stop, leverage capped."""
    leverage = np.minimum(risk / trades.stop_frac, max_leverage)
    return leverage * trades.net


@dataclass(slots=True)
class Summary:
    n: int = 0
    trades_per_day: float = 0.0
    win_rate: float | None = None
    mean_net_bps: float | None = None
    t_stat: float | None = None
    profit_factor: float | None = None
    mean_r: float | None = None
    total_return: float | None = None
    max_dd: float | None = None
    daily_sharpe: float | None = None
    avg_hold_bars: float | None = None
    fee_drag_bps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(
    trades: Trades,
    bar_t: npt.NDArray[np.int64],
    span_days: float,
    round_trip_cost_bps: float,
    risk: float = 0.0025,
    max_leverage: float = 5.0,
) -> Summary:
    s = Summary(n=len(trades))
    if len(trades) == 0:
        return s
    net = trades.net
    s.trades_per_day = len(trades) / max(span_days, 1e-9)
    s.win_rate = float((net > 0).mean())
    s.mean_net_bps = float(net.mean() * 1e4)
    s.t_stat = t_statistic(net)
    s.profit_factor = profit_factor(net)
    s.mean_r = float(trades.r_multiple.mean())
    eq = equity_returns(trades, risk, max_leverage)
    s.total_return = float(np.prod(1.0 + eq) - 1.0)
    s.max_dd = max_drawdown_from_returns(eq)
    days = bar_t[trades.exit_i] // DAY_MS
    uniq, inv = np.unique(days, return_inverse=True)
    daily = np.zeros(round(span_days) + 1)
    per_day = np.zeros(uniq.size)
    np.add.at(per_day, inv, np.log1p(eq))
    daily[: uniq.size] = per_day  # only the distribution matters for Sharpe with zero-return days added
    s.daily_sharpe = sharpe(np.expm1(daily), 365.0)
    s.avg_hold_bars = float((trades.exit_i - trades.entry_i).mean())
    s.fee_drag_bps = round_trip_cost_bps
    return s
