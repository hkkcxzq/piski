"""Strategies available to the live/demo engine (same code as in the backtests)."""

from __future__ import annotations

from functools import partial

from quant.live.engine import LiveStrategy
from quant.strategies.swing import swing_tsmom
from quant.strategies.trend import breakout_trend_fomc

# Technical demo run only: the closest near-miss of experiments 002/003 (≈ zero edge in backtests).
# Its purpose is to exercise orders, stops, restarts, news pauses and alerts — not to make money.
TECH_DEMO = LiveStrategy("brk4h-fomc", 240, partial(breakout_trend_fomc, n=24, k_stop=2.0, r=2.0))

# Multi-day momentum: passed DEV → VALIDATION → HOLDOUT (experiments 005/006, docs/DECISIONS.md D-015).
# Needs ~60 days of 4h history; the engine loads 1000 bars (~166 days).
SWING = LiveStrategy("swing-mom", 240, partial(swing_tsmom, lookback_d=28, hold_d=14, k=3.0, long_only=False))

STRATEGIES = {TECH_DEMO.id: TECH_DEMO, SWING.id: SWING}
