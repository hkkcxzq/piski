"""Summary of the demo/live trade journal, compared with the strategy's Monte Carlo bands."""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from quant.live.state import LiveState


def _dec(x: Any) -> Decimal | None:
    try:
        return None if x in (None, "", "None") else Decimal(str(x))
    except InvalidOperation:
        return None


def closed_trades(journal: Path, strategy_id: str | None = None) -> list[dict[str, Any]]:
    """Closed trades with their result in USDT and in R (multiples of the risk at entry)."""
    if not journal.exists():
        return []
    out = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") != "close" or (strategy_id and row.get("strategy_id") != strategy_id):
            continue
        entry, stop, qty = _dec(row.get("entry")), _dec(row.get("stop")), _dec(row.get("quantity"))
        pnl = _dec(row.get("pnl"))
        exit_px = _dec(row.get("exit"))
        if pnl is None and None not in (entry, exit_px, qty):
            pnl = (exit_px - entry) * qty * int(row.get("side", 0))  # type: ignore[operator]
        risk = abs(entry - stop) * qty if None not in (entry, stop, qty) else None  # type: ignore[operator]
        r = float(pnl / risk) if pnl is not None and risk else None
        out.append({**row, "pnl_usdt": float(pnl) if pnl is not None else None, "r": r})
    return out


def _fmt_ts(ms: Any) -> str:
    try:
        return dt.datetime.fromtimestamp(int(ms) / 1000, dt.UTC).strftime("%d.%m %H:%M")
    except (TypeError, ValueError):
        return "?"


def build(state: LiveState, journal: Path, mc_path: Path, strategy_id: str) -> str:
    trades = closed_trades(journal, strategy_id)
    lines = [f"Стратегия {strategy_id}. Остановка: {state.halted or 'нет'}."]
    if state.open_trades:
        lines.append("Открыто:")
        for sym, t in state.open_trades.items():
            lines.append(
                f"  {sym} {'LONG' if t.side > 0 else 'SHORT'} {t.qty} @ {t.entry}, стоп {t.stop}, "
                f"открыта {_fmt_ts(t.opened_ms)} UTC, выход по времени {_fmt_ts(t.opened_ms + t.max_hold_ms)} UTC"
            )
    else:
        lines.append("Открытых позиций нет.")
    if not trades:
        lines.append("Закрытых сделок пока нет — сравнивать не с чем.")
        return "\n".join(lines)

    rs = [t["r"] for t in trades if t["r"] is not None]
    pnl = sum(t["pnl_usdt"] or 0.0 for t in trades)
    wins = sum(1 for t in trades if (t["pnl_usdt"] or 0) > 0)
    lines.append(f"Закрыто сделок: {len(trades)}, в плюсе: {wins} ({wins / len(trades):.0%}), итог {pnl:+,.2f} USDT.")
    for tr in trades[-5:]:
        r = "?" if tr["r"] is None else f"{tr['r']:+.2f} R"
        lines.append(
            f"  {_fmt_ts(tr.get('ts'))} {tr.get('symbol')} {'LONG' if int(tr.get('side', 0)) > 0 else 'SHORT'}: "
            f"{(tr['pnl_usdt'] or 0):+,.2f} USDT ({r}), выход: {tr.get('reason_for_exit')}"
        )
    if not rs:
        return "\n".join(lines)
    total_r = sum(rs)
    streak = cur = 0
    for r_ in rs:
        cur = cur + 1 if r_ < 0 else 0
        streak = max(streak, cur)
    lines.append(
        f"Сумма {total_r:+.2f} R, в среднем {total_r / len(rs):+.2f} R на сделку, худшая серия убытков: {streak}."
    )

    if mc_path.exists():
        mc = json.loads(mc_path.read_text(encoding="utf-8"))
        bands = mc["cum_r_bands"]
        n = min(len(rs), len(bands["p50"]))
        lo, mid, hi = bands["p5"][n - 1], bands["p50"][n - 1], bands["p95"][n - 1]
        lines.append(
            f"Бэктест для {n} сделок: обычно {mid:+.2f} R, нормальный диапазон {lo:+.2f} … {hi:+.2f} R "
            f"(5–95 % симуляций); средняя сделка в истории {mc['history']['mean_r']:+.2f} R."
        )
        bad_streak = max(x["losing_streak"]["p95"] for x in mc["horizons"].values())
        if total_r < lo or streak > bad_streak:
            lines.append("⚠️  Хуже нормального диапазона — нужно разобраться (рынок изменился или ошибка исполнения).")
        elif total_r > hi:
            lines.append("Лучше нормального диапазона — приятно, но это тоже повод проверить, не везение ли это.")
        else:
            lines.append("✅ В пределах ожидаемого по бэктесту.")
    return "\n".join(lines)
