"""Experiment 006: open HOLDOUT once for the swing-momentum candidate (preregistration-006)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quant.data.bars import load_minutes, resample
from quant.research.experiment001 import DEFAULT_COSTS, END, START, SYMBOLS, run_split
from quant.stats.metrics import profit_factor, t_statistic
from quant.strategies.swing import grid005

CANDIDATE = {"lookback_d": 28, "hold_d": 14, "k": 3.0, "long_only": False}
RISK = 0.0025


def _equity_dd(r_multiples: np.ndarray) -> float:
    eq = np.cumprod(1 + RISK * r_multiples)
    return float(1 - (eq / np.maximum.accumulate(eq)).min()) if eq.size else 0.0


def run(root: Path) -> dict[str, Any]:
    frames = {s: resample(load_minutes(root, s, START, END), 240) for s in SYMBOLS}
    variants = [v for v in grid005() if not v.params["long_only"]]
    cand = next(v for v in variants if v.params == CANDIDATE)
    per_symbol: dict[str, Any] = {}
    nets, stressed, rs = [], [], []
    for s in SYMBOLS:
        b = frames[s]
        sig = cand.build(b)
        t, summ = run_split(b, sig, "holdout", DEFAULT_COSTS)
        ts, _ = run_split(b, sig, "holdout", DEFAULT_COSTS.stressed(1.5))
        per_symbol[s] = summ.to_dict()
        nets.append(t.net)
        stressed.append(ts.net)
        rs.append(t.r_multiple)
    net, st = np.concatenate(nets), np.concatenate(stressed)
    # drawdown of each symbol's equity curve at 0.25 % risk per trade; the worst one is reported
    dd = max(_equity_dd(r) for r in rs)
    pf = profit_factor(net) or 0.0
    passed = bool(
        net.size > 0
        and net.mean() > 0
        and pf >= 1.1
        and st.mean() > 0
        and any((per_symbol[s]["mean_net_bps"] or -1) >= 0 for s in SYMBOLS)
        and dd < 0.05
    )
    family: list[dict[str, Any]] = []
    for v in variants:
        parts = [run_split(frames[s], v.build(frames[s]), "holdout", DEFAULT_COSTS)[0].net for s in SYMBOLS]
        x = np.concatenate(parts)
        family.append({"key": v.key, "n": int(x.size), "mean_bps": float(x.mean() * 1e4) if x.size else None})
    means = [float(f["mean_bps"]) for f in family if f["mean_bps"] is not None]
    return {
        "candidate": cand.key,
        "status": "VALIDATION" if passed else "REJECTED",
        "n": int(net.size),
        "mean_net_bps": float(net.mean() * 1e4) if net.size else None,
        "stressed_mean_bps": float(st.mean() * 1e4) if st.size else None,
        "profit_factor": pf,
        "t": t_statistic(net),
        "win_rate": float((net > 0).mean()) if net.size else None,
        "max_dd_per_symbol": dd,
        "per_symbol": per_symbol,
        "family": family,
        "family_mean_bps": float(np.mean(means)) if means else None,
        "family_positive": sum(1 for m in means if m > 0),
    }
