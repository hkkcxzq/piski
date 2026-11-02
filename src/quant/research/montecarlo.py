"""Monte Carlo of the live swing strategy: what a normal run of luck looks like.

The strategy's backtest trades (BTC+ETH, 2021-01 … 2026-09, all splits) are merged in entry order
and resampled with a **block bootstrap** (consecutive blocks of trades), which keeps the clustering
of losses in choppy markets and the overlap of BTC/ETH positions that an i.i.d. shuffle would hide.
Results are in R (multiples of the risk per trade) and in account % at a given risk per trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from quant.backtest.engine import Costs, simulate
from quant.data.bars import load_minutes, resample
from quant.research.experiment001 import END, START, SYMBOLS
from quant.strategies.swing import swing_tsmom

F64 = npt.NDArray[np.float64]
CANDIDATE = {"lookback_d": 28, "hold_d": 14, "k": 3.0, "long_only": False}  # = live "swing-mom"
HORIZON_MONTHS = {"1 месяц": 1, "3 месяца": 3, "6 месяцев": 6, "1 год": 12}
BLOCK = 4
N_SIMS = 20_000
RISK = 0.0025


@dataclass(frozen=True, slots=True)
class TradeLog:
    entry_ms: npt.NDArray[np.int64]
    r: F64


def backtest_trades(root: Path) -> TradeLog:
    ts, rs = [], []
    for s in SYMBOLS:
        b = resample(load_minutes(root, s, START, END), 240)
        sig = swing_tsmom(b, lookback_d=28, hold_d=14, k=3.0, long_only=False)
        t = simulate(b.o, b.h, b.l, b.c, b.valid, sig, Costs())
        ts.append(b.t[t.entry_i])
        rs.append(t.r_multiple)
    entry = np.concatenate(ts)
    order = np.argsort(entry, kind="stable")
    return TradeLog(entry[order], np.concatenate(rs)[order])


def block_bootstrap(r: F64, n: int, sims: int, block: int, rng: np.random.Generator) -> F64:
    """``sims`` paths of ``n`` trades built from random consecutive blocks of the history."""
    blocks = -(-n // block)
    starts = rng.integers(0, r.size - block + 1, size=(sims, blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(sims, -1)[:, :n]
    return np.asarray(r[idx], dtype=np.float64)


def path_stats(paths: F64, risk: float) -> dict[str, F64]:
    equity = np.cumprod(1 + risk * paths, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((paths.shape[0], 1)), equity], axis=1), axis=1)[:, 1:]
    dd = 1 - (equity / peak).min(axis=1)
    losing = paths < 0
    streak = np.zeros(paths.shape[0], dtype=np.int64)
    cur = np.zeros(paths.shape[0], dtype=np.int64)
    for j in range(paths.shape[1]):
        cur = np.where(losing[:, j], cur + 1, 0)
        streak = np.maximum(streak, cur)
    return {
        "total_r": paths.sum(axis=1),
        "return": equity[:, -1] - 1,
        "max_dd": dd,
        "losing_streak": streak.astype(np.float64),
    }


def _pct(x: F64) -> dict[str, float]:
    return {f"p{q}": float(np.percentile(x, q)) for q in (5, 25, 50, 75, 95)}


def run(root: Path, seed: int = 7) -> dict[str, Any]:
    log = backtest_trades(root)
    rng = np.random.default_rng(seed)
    months = (END - START).days / 30.44
    per_month = log.r.size / months
    horizons: dict[str, Any] = {}
    for name, m in HORIZON_MONTHS.items():
        n = max(1, round(per_month * m))
        st = path_stats(block_bootstrap(log.r, n, N_SIMS, BLOCK, rng), RISK)
        horizons[name] = {
            "trades": n,
            "p_loss": float((st["return"] < 0).mean()),
            "return": _pct(st["return"]),
            "total_r": _pct(st["total_r"]),
            "max_dd": _pct(st["max_dd"]),
            "losing_streak": _pct(st["losing_streak"]),
        }
    # cumulative-R bands by number of trades, for comparing the live journal (quant live report)
    max_n = 60
    paths = block_bootstrap(log.r, max_n, N_SIMS, BLOCK, rng).cumsum(axis=1)
    bands = {q: np.percentile(paths, q, axis=0).tolist() for q in (5, 50, 95)}
    r = log.r
    return {
        "candidate": CANDIDATE,
        "risk_per_trade": RISK,
        "trades_per_month": per_month,
        "history": {
            "trades": int(r.size),
            "win_rate": float((r > 0).mean()),
            "mean_r": float(r.mean()),
            "median_r": float(np.median(r)),
            "best_r": float(r.max()),
            "worst_r": float(r.min()),
        },
        "block": BLOCK,
        "sims": N_SIMS,
        "horizons": horizons,
        "cum_r_bands": {"p5": bands[5], "p50": bands[50], "p95": bands[95]},
    }


def render(res: dict[str, Any]) -> str:
    h = res["history"]
    risk = res["risk_per_trade"]
    lines = [
        "# Monte Carlo стратегии swing-mom — чего ожидать на демо",
        "",
        f"Основа: {h['trades']} сделок бэктеста (BTC+ETH, 2021-01 … 2026-09, после комиссий и проскальзывания), "
        f"≈ {res['trades_per_month']:.1f} сделки в месяц.",
        f"Выигрышных {h['win_rate']:.0%}, средняя сделка {h['mean_r']:+.2f} R, медианная {h['median_r']:+.2f} R, "
        f"лучшая {h['best_r']:+.1f} R, худшая {h['worst_r']:+.1f} R.",
        f"Блочный бутстрэп (блоки по {res['block']} сделки подряд, {res['sims']:,} симуляций), риск "
        f"{risk:.2%} на сделку. R — результат в долях риска: −1 R = стоп.",
        "",
        "| Период (≈ сделок) | Шанс быть в минусе | Результат: плохой (5 %) / обычный (50 %) / хороший (95 %) | "
        "Просадка: обычная / плохая (95 %) | Убыточных сделок подряд: обычно / плохо (95 %) |",
        "|---|---|---|---|---|",
    ]
    for name, x in res["horizons"].items():
        ret, dd, ls = x["return"], x["max_dd"], x["losing_streak"]
        lines.append(
            f"| {name} ({x['trades']}) | {x['p_loss']:.0%} | "
            f"{ret['p5']:+.2%} / {ret['p50']:+.2%} / {ret['p95']:+.2%} | "
            f"{dd['p50']:.2%} / {dd['p95']:.2%} | {ls['p50']:.0f} / {ls['p95']:.0f} |"
        )
    lines += [
        "",
        "## Как читать",
        "- Это трендовая стратегия: выигрывает около половины сделок, медианная сделка почти ноль, прибыль "
        "дают редкие длинные тренды (до +6 R) при убытке не больше −1 R.",
        "- Минус за 1–3 месяца — обычное дело, а не поломка (см. «шанс быть в минусе»).",
        "- **Сигнал тревоги:** результат демо ниже нижней границы (5 %) или серия убытков длиннее плохой (95 %) — "
        "значит, рынок изменился или в исполнении ошибка; тогда разбираемся, а не «пересиживаем».",
        "- Оговорка: история включает периоды, на которых выбирались параметры, поэтому оценка скорее "
        "оптимистичная; реальность может оказаться ближе к нижней половине диапазона.",
        "- Цифры в % счёта масштабируются с риском: при 0.5 % на сделку — примерно вдвое больше и доход, и просадки.",
        "",
        "Сравнение живых сделок с этими границами: `quant live report`.",
    ]
    return "\n".join(lines) + "\n"
