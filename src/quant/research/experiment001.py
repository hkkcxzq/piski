"""Run pre-registered experiment 001 (docs/research/preregistration-001.md).

Signals are computed once on the full causal series; each variant is then simulated
separately on DEV, VALIDATION and HOLDOUT index ranges. HOLDOUT results are computed
only for variants that pass VALIDATION.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quant.backtest.engine import Costs, Signals, Trades, simulate
from quant.backtest.metrics import Summary, summarize
from quant.data.bars import Bars, load_minutes, resample
from quant.stats.metrics import profit_factor, t_statistic
from quant.strategies.scalping import Variant, grid

SYMBOLS = ("BTCUSDT", "ETHUSDT")
START, END = dt.date(2021, 1, 1), dt.date(2026, 9, 26)
SPLITS = {
    "dev": (dt.date(2021, 1, 1), dt.date(2024, 7, 1)),
    "validation": (dt.date(2024, 7, 1), dt.date(2025, 7, 1)),
    "holdout": (dt.date(2025, 7, 1), dt.date(2026, 9, 27)),
}
MIN_DEV_TRADES = 200
DEFAULT_COSTS = Costs()
MIN_VAL_TRADES = 50


def _ms(d: dt.date) -> int:
    return int(dt.datetime.combine(d, dt.time(), tzinfo=dt.UTC).timestamp() * 1000)


def _slice_signals(sig: Signals, a: int, b: int) -> Signals:
    return Signals(
        sig.side[a:b], sig.entry_px[a:b], sig.expiry[a:b], sig.stop_dist[a:b], sig.tp_dist[a:b], sig.max_hold[a:b]
    )


def _concat(trades: list[Trades]) -> Trades:
    return Trades(*(np.concatenate([getattr(t, f) for t in trades]) for f in Trades.__slots__))


@dataclass(slots=True)
class VariantResult:
    key: str
    hypothesis: str
    per_split: dict[str, dict[str, Summary]] = field(default_factory=dict)  # split -> symbol|"all" -> summary
    pooled_net: dict[str, np.ndarray] = field(default_factory=dict)


def run_split(bars: Bars, sig: Signals, split: str, costs: Costs) -> tuple[Trades, Summary]:
    a_d, b_d = SPLITS[split]
    a, b = bars.index_of(_ms(a_d)), bars.index_of(_ms(b_d))
    part = bars.slice(a, b)
    t = simulate(part.o, part.h, part.l, part.c, part.valid, _slice_signals(sig, a, b), costs)
    rt_bps = (costs.taker_fee * 2 + costs.slippage * 2) * 1e4
    return t, summarize(t, part.t, (b_d - a_d).days, rt_bps)


def run(
    root: Path,
    costs: Costs = DEFAULT_COSTS,
    progress: Any = None,
    variants: list[Variant] | None = None,
    min_dev_trades: int = MIN_DEV_TRADES,
    min_val_trades: int = MIN_VAL_TRADES,
) -> dict[str, Any]:
    minutes = {s: load_minutes(root, s, START, END) for s in SYMBOLS}
    frames: dict[tuple[str, int], Bars] = {}
    for s, m in minutes.items():
        frames[(s, 1)] = m
        for tf in sorted({v.timeframe_min for v in (variants or grid())} - {1}):
            frames[(s, tf)] = resample(m, tf)

    variants = list(variants) if variants is not None else list(grid())
    results: list[VariantResult] = []
    signals_cache: dict[tuple[str, str], Signals] = {}
    for n, var in enumerate(variants, 1):
        res = VariantResult(var.key, var.hypothesis)
        for split in ("dev", "validation"):
            pooled: list[np.ndarray] = []
            res.per_split[split] = {}
            for s in SYMBOLS:
                bars = frames[(s, var.timeframe_min)]
                sig = signals_cache.get((s, var.key))
                if sig is None:
                    sig = signals_cache[(s, var.key)] = var.build(bars, s) if var.symbol_aware else var.build(bars)
                t, summ = run_split(bars, sig, split, costs)
                res.per_split[split][s] = summ
                pooled.append(t.net)
            res.pooled_net[split] = np.concatenate(pooled)
        results.append(res)
        if progress:
            progress(n, len(variants), var.key)

    # ---- selection on DEV (one variant per hypothesis), VALIDATION gate, HOLDOUT once
    by_h: dict[str, list[VariantResult]] = defaultdict(list)
    for r in results:
        by_h[r.hypothesis].append(r)
    decisions: dict[str, dict[str, Any]] = {}
    var_by_key = {v.key: v for v in variants}
    for h, rs in by_h.items():
        eligible = [r for r in rs if r.pooled_net["dev"].size >= min_dev_trades and r.pooled_net["dev"].mean() > 0]
        if not eligible:
            decisions[h] = {"status": "REJECTED", "stage": "dev", "reason": "no variant with positive net on DEV"}
            continue
        best = max(eligible, key=lambda r: t_statistic(r.pooled_net["dev"]) or -1e9)
        val = best.pooled_net["validation"]
        val_ok = (
            val.size >= min_val_trades
            and val.mean() > 0
            and (t_statistic(val) or 0) >= 2.0
            and (profit_factor(val) or 0) >= 1.1
            and all((best.per_split["validation"][s].mean_net_bps or -1) > 0 for s in SYMBOLS)
        )
        decision: dict[str, Any] = {
            "selected": best.key,
            "status": "VALIDATION" if val_ok else "REJECTED",
            "stage": "validation",
        }
        if val_ok:
            var = var_by_key[best.key]
            ho: dict[str, Any] = {}
            nets, stressed = [], []
            for s in SYMBOLS:
                bars = frames[(s, var.timeframe_min)]
                sig = signals_cache[(s, var.key)]
                t, summ = run_split(bars, sig, "holdout", costs)
                ts, _ = run_split(bars, sig, "holdout", costs.stressed(1.5))
                ho[s] = summ.to_dict()
                nets.append(t.net)
                stressed.append(ts.net)
            net = np.concatenate(nets)
            st = np.concatenate(stressed)
            ho_ok = net.size > 0 and net.mean() > 0 and (profit_factor(net) or 0) >= 1.05 and st.mean() > 0
            decision.update(
                stage="holdout",
                status="VALIDATION" if ho_ok else "REJECTED",
                holdout=ho,
                holdout_mean_net_bps=float(net.mean() * 1e4) if net.size else None,
                holdout_stressed_mean_bps=float(st.mean() * 1e4) if st.size else None,
            )
        decisions[h] = decision

    table = []
    for r in results:
        row: dict[str, Any] = {"key": r.key, "hypothesis": r.hypothesis}
        for split in ("dev", "validation"):
            pn = r.pooled_net[split]
            row[f"{split}_n"] = int(pn.size)
            row[f"{split}_mean_bps"] = float(pn.mean() * 1e4) if pn.size else None
            row[f"{split}_t"] = t_statistic(pn)
            row[f"{split}_pf"] = profit_factor(pn)
            for s in SYMBOLS:
                row[f"{split}_{s}"] = r.per_split[split][s].to_dict()
        table.append(row)
    return {
        "trials": len(variants),
        "decisions": decisions,
        "variants": table,
        "costs": {"taker": costs.taker_fee, "maker": costs.maker_fee, "slippage": costs.slippage},
    }


def render(result: dict[str, Any], number: str = "001") -> str:
    def f(x: Any, fmt: str = ".2f") -> str:
        return "—" if x is None else format(x, fmt)

    lines = [
        f"# Эксперимент {number} — результаты",
        "",
        f"План: `docs/research/preregistration-{number}.md` (записан до запуска). "
        f"Попыток (вариантов): **{result['trials']}**. Издержки: taker {result['costs']['taker']:.4%}, "
        f"maker {result['costs']['maker']:.4%}, проскальзывание {result['costs']['slippage']:.4%}.",
        "",
        "## Решения по гипотезам",
        "",
        "| Гипотеза | Итог | Этап | Выбранный вариант | Holdout, bps/сделка | Holdout при издержках ×1.5 |",
        "|---|---|---|---|---|---|",
    ]
    for h, d in sorted(result["decisions"].items()):
        lines.append(
            f"| {h} | **{d['status']}** | {d['stage']} | `{d.get('selected', '—')}` | "
            f"{f(d.get('holdout_mean_net_bps'), '.1f')} | {f(d.get('holdout_stressed_mean_bps'), '.1f')} |"
        )
    lines += [
        "",
        "## Все варианты (DEV и VALIDATION, BTC+ETH вместе)",
        "",
        "Средняя чистая сделка в bps (после комиссий и проскальзывания), t-статистика, profit factor, число сделок.",
        "",
        "| Вариант | DEV bps | DEV t | DEV PF | DEV n | VAL bps | VAL t | VAL PF | VAL n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(result["variants"], key=lambda r: (r["hypothesis"], -(r["dev_t"] or -99))):
        lines.append(
            f"| `{r['key']}` | {f(r['dev_mean_bps'], '.1f')} | {f(r['dev_t'], '.1f')} | {f(r['dev_pf'])} | "
            f"{r['dev_n']} | {f(r['validation_mean_bps'], '.1f')} | {f(r['validation_t'], '.1f')} | "
            f"{f(r['validation_pf'])} | {r['validation_n']} |"
        )
    return "\n".join(lines) + "\n"
