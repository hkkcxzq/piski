"""Risk rules for live/demo trading. Pure functions: easy to test, impossible to bypass.

* position size comes from the stop: risk a fixed fraction of equity to the stop,
  leverage capped (the owner's rule "long stop → lower leverage");
* loss limits (day / week / peak drawdown) halt trading until a manual reset;
* the news/calendar risk level can reduce size, block entries or force flat.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from quant.app.config import RiskLimits
from quant.exchanges.bybit.client import Instrument
from quant.news.policy import RiskLevel, calendar_signal, load_state

NEWS_STATE_MAX_AGE_MS = 15 * 60_000  # a stale news state (monitor stopped) is ignored, calendar still applies


def position_qty(
    equity: Decimal,
    entry: Decimal,
    stop_dist: Decimal,
    inst: Instrument,
    limits: RiskLimits,
    risk_mult: Decimal = Decimal(1),
) -> Decimal:
    """Quantity risking ``max_position_risk × risk_mult`` of equity to the stop, leverage ≤ max_leverage.
    Returns 0 when the resulting size is below the exchange minimum (skip the trade, never round up)."""
    if equity <= 0 or entry <= 0 or stop_dist <= 0:
        return Decimal(0)
    risk_cash = equity * Decimal(str(limits.max_position_risk)) * risk_mult
    qty = risk_cash / stop_dist
    max_lev = min(Decimal(str(limits.max_leverage)), inst.max_leverage)
    qty = min(qty, equity * max_lev / entry)
    qty = inst.round_qty(qty)
    return qty if qty >= inst.min_qty else Decimal(0)


@dataclass(slots=True)
class LossCheck:
    halt: bool
    reason: str = ""


def check_losses(
    equity: Decimal, day_start: Decimal, week_start: Decimal, peak: Decimal, limits: RiskLimits
) -> LossCheck:
    def loss(ref: Decimal) -> float:
        return float((ref - equity) / ref) if ref > 0 else 0.0

    if loss(day_start) >= limits.max_daily_loss:
        return LossCheck(True, f"daily loss {loss(day_start):.2%} >= {limits.max_daily_loss:.2%}")
    if loss(week_start) >= limits.max_weekly_loss:
        return LossCheck(True, f"weekly loss {loss(week_start):.2%} >= {limits.max_weekly_loss:.2%}")
    if loss(peak) >= limits.max_drawdown:
        return LossCheck(True, f"drawdown {loss(peak):.2%} >= {limits.max_drawdown:.2%}")
    return LossCheck(False)


def period_keys(now_ms: int) -> tuple[str, str]:
    d = dt.datetime.fromtimestamp(now_ms / 1000, tz=dt.UTC).date()
    iso = d.isocalendar()
    return d.isoformat(), f"{iso.year}-W{iso.week:02d}"


def effective_risk_level(news_state_path: Path, now_ms: int) -> tuple[RiskLevel, str]:
    """Most severe of: fresh news monitor state, FOMC calendar window (always applied)."""
    level, reason = RiskLevel.NORMAL, ""
    cal = calendar_signal(now_ms)
    if cal is not None:
        level, reason = cal.level, cal.reason
    try:
        st = load_state(news_state_path)
    except (OSError, ValueError, KeyError):
        return level, reason
    if st.updated_ms and now_ms - st.updated_ms <= NEWS_STATE_MAX_AGE_MS and st.until_ms > now_ms and st.level > level:
        level, reason = st.level, "; ".join(st.reasons)
    return level, reason


def risk_multiplier(level: RiskLevel) -> Decimal:
    return Decimal("0.5") if level is RiskLevel.CAUTION else Decimal(1)
