"""Strategies available to the live/demo engine (same code as in the backtests)."""

from __future__ import annotations

from functools import partial

from quant.live.engine import LiveStrategy
from quant.strategies.trend import breakout_trend_fomc

# Technical demo run only: the closest near-miss of experiments 002/003 (≈ zero edge in backtests).
# Its purpose is to exercise orders, stops, restarts, news pauses and alerts — not to make money.
TECH_DEMO = LiveStrategy("brk4h-fomc", 240, partial(breakout_trend_fomc, n=24, k_stop=2.0, r=2.0))

STRATEGIES = {TECH_DEMO.id: TECH_DEMO}
