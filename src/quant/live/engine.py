"""Live/demo trading engine: one cycle = reconcile → protect → risk checks → maybe enter.

Design rules (see docs/ARCHITECTURE.md §7, §10):
* the exchange is the source of truth; local state is reconciled against it every cycle;
* every position must carry an exchange-side stop — if it cannot be confirmed, the position
  is closed immediately;
* the intent is persisted *before* an order is sent, so a crash between "sent" and "saved"
  is recovered by reconciliation, never by sending a second order;
* signals come from the same strategy code as the backtest, on *closed* bars only;
* risk (loss limits, news/calendar level, kill switch) is checked by code the strategy cannot bypass.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

import numpy as np

from quant.app.config import RiskLimits
from quant.app.logging import get_logger
from quant.backtest.engine import Signals
from quant.data.bars import Bars
from quant.exchanges.bybit.client import BybitClient, BybitError, Instrument, Position
from quant.live.risk import check_losses, effective_risk_level, period_keys, position_qty, risk_multiplier
from quant.live.state import LiveState, OpenTrade, Store
from quant.news.policy import RiskLevel
from quant.strategies.scalping import atr

log = get_logger(__name__)

STOP_CONFIRM_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class LiveStrategy:
    id: str
    timeframe_min: int
    build: Callable[[Bars], Signals]


def bars_from_klines(rows: list[tuple[int, float, float, float, float, float]], interval_min: int, now_ms: int) -> Bars:
    """Closed bars only: the still-forming bar is dropped (no peeking at an unfinished candle)."""
    step = interval_min * 60_000
    closed = [r for r in rows if r[0] + step <= now_ms]
    a = np.asarray(closed, dtype=np.float64).reshape(-1, 6)
    n = a.shape[0]
    zeros = np.zeros(n)
    return Bars(
        step,
        a[:, 0].astype(np.int64),
        a[:, 1].copy(),
        a[:, 2].copy(),
        a[:, 3].copy(),
        a[:, 4].copy(),
        a[:, 5].copy(),
        zeros,
        zeros.copy(),
        np.ones(n),
        np.ones(n, dtype=bool),
    )


class LiveEngine:
    def __init__(
        self,
        trade: BybitClient,
        market: BybitClient,
        strategy: LiveStrategy,
        symbols: tuple[str, ...],
        limits: RiskLimits,
        store: Store,
        news_state_path: Path,
        *,
        clock: Callable[[], int] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self.trade = trade
        self.market = market
        self.strategy = strategy
        self.symbols = symbols
        self.limits = limits
        self.store = store
        self.news_state_path = news_state_path
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.sleep = sleep
        self.notify = notify or (lambda _msg: None)
        self._instruments: dict[str, Instrument] = {}

    # ------------------------------------------------------------------ helpers
    def _inst(self, symbol: str) -> Instrument:
        if symbol not in self._instruments:
            self._instruments[symbol] = self.market.instrument(symbol)
        return self._instruments[symbol]

    def _event(self, kind: str, **fields: object) -> None:
        row = {"ts": self.clock(), "event": kind, **fields}
        self.store.event(row)
        log.info("live_" + kind, **{k: str(v) for k, v in fields.items()})

    def _alert(self, text: str) -> None:
        self._event("alert", text=text)
        self.notify(text)

    def _close(self, st: LiveState, symbol: str, reason: str) -> None:
        try:
            closed = self.trade.close_position(symbol, st.next_id(f"x{symbol[:3]}"))
        except BybitError as exc:
            self._alert(f"{symbol}: FAILED to close ({reason}): {exc}")
            return
        if closed:
            self._event("close_sent", symbol=symbol, reason=reason)
        rec = st.open_trades.get(symbol)
        if rec is not None:
            self._finish(st, rec, reason)

    def _finish(self, st: LiveState, rec: OpenTrade, reason: str) -> None:
        pnl = None
        exit_px = None
        try:
            rows = [
                r
                for r in self.trade.closed_pnl(rec.symbol, rec.opened_ms)
                if int(r.get("createdTime", r.get("updatedTime", 0)) or 0) >= rec.opened_ms
            ]
            if rows:
                pnl = str(sum(Decimal(str(r.get("closedPnl", "0"))) for r in rows))
                exit_px = str(rows[0].get("avgExitPrice"))
        except BybitError as exc:
            self._event("closed_pnl_unavailable", symbol=rec.symbol, error=str(exc))
        self.store.journal(
            {
                "ts": self.clock(),
                "type": "close",
                "strategy_id": rec.strategy_id,
                "signal_id": rec.signal_id,
                "symbol": rec.symbol,
                "side": rec.side,
                "entry": rec.entry,
                "stop": rec.stop,
                "take_profit": rec.take_profit,
                "quantity": rec.qty,
                "leverage": rec.leverage,
                "exit": exit_px,
                "pnl": pnl,
                "fees_funding": "included in exchange closedPnl",
                "reason_for_entry": rec.reason_for_entry,
                "reason_for_exit": reason,
                "opened_ms": rec.opened_ms,
            }
        )
        st.open_trades.pop(rec.symbol, None)

    def flatten_all(self, st: LiveState, reason: str) -> None:
        for symbol in self.symbols:
            try:
                pos = self.trade.position(symbol)
            except BybitError as exc:
                self._alert(f"{symbol}: cannot read position while flattening: {exc}")
                continue
            if pos.size != 0 or symbol in st.open_trades:
                self._close(st, symbol, reason)

    # ------------------------------------------------------------------ cycle
    def run_cycle(self) -> LiveState:
        now = self.clock()
        st = self.store.load()
        try:
            if self.store.kill_path.exists():
                if not st.halted:
                    st.halted = "kill switch file"
                    self._alert("KILL switch: flattening and halting")
                self.flatten_all(st, "kill switch")
                return st
            equity = self.trade.equity()
            self._update_baselines(st, equity, now)
            if not st.halted:
                loss = check_losses(
                    equity,
                    Decimal(st.day_start_equity),
                    Decimal(st.week_start_equity),
                    Decimal(st.peak_equity),
                    self.limits,
                )
                if loss.halt:
                    st.halted = loss.reason
                    self._alert(f"HALT: {loss.reason}. Flattening; manual reset required.")
                    self.flatten_all(st, loss.reason)
            level, why = effective_risk_level(self.news_state_path, now)
            if level is RiskLevel.FLATTEN:
                if st.open_trades:
                    self._alert(f"News FLATTEN: {why}")
                self.flatten_all(st, f"news: {why}")
            for symbol in self.symbols:
                self._manage(st, symbol, now)
            if not st.halted and level < RiskLevel.PAUSE_NEW:
                for symbol in self.symbols:
                    self._maybe_enter(st, symbol, now, equity, level)
            elif level >= RiskLevel.PAUSE_NEW:
                self._event("entries_paused", risk=level.name, reason=why)
            log.info(
                "live_heartbeat",
                equity=str(equity),
                risk=level.name,
                open=",".join(st.open_trades) or "-",
                halted=st.halted or "no",
            )
        finally:
            self.store.save(st)
        return st

    def _update_baselines(self, st: LiveState, equity: Decimal, now: int) -> None:
        day, week = period_keys(now)
        if st.day_key != day:
            st.day_key, st.day_start_equity = day, str(equity)
        if st.week_key != week:
            st.week_key, st.week_start_equity = week, str(equity)
        st.peak_equity = str(max(Decimal(st.peak_equity), equity))

    def _manage(self, st: LiveState, symbol: str, now: int) -> None:
        try:
            pos = self.trade.position(symbol)
        except BybitError as exc:
            self._alert(f"{symbol}: cannot read position: {exc}")
            return
        rec = st.open_trades.get(symbol)
        if pos.size == 0:
            if rec is not None and now - rec.opened_ms > 60_000:
                self._finish(st, rec, "exchange exit (stop / take-profit / liquidation)")
            return
        if rec is None:
            rec = self._adopt(st, symbol, pos, now)
            if rec is None:
                return
        if pos.stop_loss is None:
            self._ensure_stop(st, symbol, rec)
            return
        if now - rec.opened_ms >= rec.max_hold_ms:
            self._close(st, symbol, "time exit")

    def _adopt(self, st: LiveState, symbol: str, pos: Position, now: int) -> OpenTrade | None:
        side = 1 if pos.size > 0 else -1
        stop = pos.stop_loss
        if stop is None:
            try:
                bars = bars_from_klines(
                    self.market.klines(symbol, self.strategy.timeframe_min, 100), self.strategy.timeframe_min, now
                )
                a = Decimal(str(atr(bars, 14)[-1]))
                stop = self._inst(symbol).round_price(pos.avg_price - side * 2 * a)
            except (BybitError, IndexError, ValueError) as exc:
                self._alert(f"{symbol}: unknown position without stop and no ATR ({exc}); closing")
                self._close(st, symbol, "adopted without stop")
                return None
        rec = OpenTrade(
            symbol,
            "adopted",
            "-",
            "-",
            side,
            str(abs(pos.size)),
            str(pos.avg_price),
            str(stop),
            None if pos.take_profit is None else str(pos.take_profit),
            "?",
            now,
            24 * 3_600_000,
            "position found on exchange without local record",
            adopted=True,
        )
        st.open_trades[symbol] = rec
        self._alert(f"{symbol}: adopted unknown position {pos.size} @ {pos.avg_price}, stop {stop}")
        return rec

    def _ensure_stop(self, st: LiveState, symbol: str, rec: OpenTrade) -> bool:
        for attempt in range(STOP_CONFIRM_ATTEMPTS):
            try:
                pos = self.trade.position(symbol)
                if pos.size == 0:
                    return True
                if pos.stop_loss is not None:
                    return True
                self.trade.set_trading_stop(
                    symbol, Decimal(rec.stop), None if rec.take_profit is None else Decimal(rec.take_profit)
                )
            except BybitError as exc:
                self._event("stop_attempt_failed", symbol=symbol, attempt=attempt, error=str(exc))
            self.sleep(1.0)
        try:
            if self.trade.position(symbol).stop_loss is not None:
                return True
        except BybitError:
            pass
        self._alert(f"{symbol}: protective stop NOT confirmed — closing position")
        self._close(st, symbol, "no protective stop")
        return False

    def _maybe_enter(self, st: LiveState, symbol: str, now: int, equity: Decimal, level: RiskLevel) -> None:
        if symbol in st.open_trades:
            return
        tf = self.strategy.timeframe_min
        try:
            bars = bars_from_klines(self.market.klines(symbol, tf, 1000), tf, now)
        except BybitError as exc:
            self._event("market_data_unavailable", symbol=symbol, error=str(exc))
            return
        if len(bars) < 250:
            self._event("not_enough_bars", symbol=symbol, bars=len(bars))
            return
        last = int(bars.t[-1])
        if st.last_bar_ms.get(symbol) == last:
            return  # this bar was already evaluated
        st.last_bar_ms[symbol] = last
        sig = self.strategy.build(bars)
        i = len(bars) - 1
        side = int(sig.side[i])
        if side == 0:
            return
        inst = self._inst(symbol)
        ref = Decimal(str(bars.c[i]))
        stop_dist = Decimal(str(sig.stop_dist[i]))
        qty = position_qty(equity, ref, stop_dist, inst, self.limits, risk_multiplier(level))
        if qty == 0:
            self._event("skip_size_below_minimum", symbol=symbol, equity=equity, stop_dist=stop_dist)
            return
        stop_px = inst.round_price(ref - side * stop_dist)
        tp_dist = sig.tp_dist[i]
        tp_px = None if not np.isfinite(tp_dist) else inst.round_price(ref + side * Decimal(str(tp_dist)))
        lev = max(Decimal(1), (qty * ref / equity).to_integral_value(rounding=ROUND_CEILING))
        signal_id = st.next_id(f"s{symbol[:3]}")
        link_id = f"{self.strategy.id[:12]}-{signal_id}"
        rec = OpenTrade(
            symbol,
            self.strategy.id,
            signal_id,
            link_id,
            side,
            str(qty),
            str(ref),
            str(stop_px),
            None if tp_px is None else str(tp_px),
            str(lev),
            now,
            int(sig.max_hold[i]) * tf * 60_000,
            f"signal on bar {last} ({self.strategy.id})",
        )
        st.open_trades[symbol] = rec
        self.store.save(st)  # persist the intent before sending the order
        try:
            self.trade.set_leverage(symbol, lev)
            self.trade.market_order(
                symbol, "Buy" if side > 0 else "Sell", qty, link_id, stop_loss=stop_px, take_profit=tp_px
            )
        except BybitError as exc:
            self._event("order_error", symbol=symbol, error=str(exc))
            try:
                if self.trade.position(symbol).size == 0:
                    st.open_trades.pop(symbol, None)  # nothing happened on the exchange
                    return
            except BybitError:
                return  # unknown: the next cycle reconciles
        if not self._ensure_stop(st, symbol, rec):
            return
        try:
            pos = self.trade.position(symbol)
        except BybitError:
            return
        if pos.size == 0:
            st.open_trades.pop(symbol, None)
            self._event("order_not_filled", symbol=symbol)
            return
        rec.entry, rec.qty = str(pos.avg_price), str(abs(pos.size))
        self.store.journal({"ts": now, "type": "open", **{k: getattr(rec, k) for k in rec.__slots__}})
        self.notify(
            f"{symbol}: {'LONG' if side > 0 else 'SHORT'} {rec.qty} @ {rec.entry}, stop {rec.stop}, "
            f"tp {rec.take_profit}, lev {lev}x ({self.strategy.id})"
        )

    def run_forever(self, interval_s: float = 60.0) -> None:
        failures = 0
        while True:
            try:
                self.run_cycle()
                failures = 0
            except Exception as exc:  # the loop must survive; state + exchange stops keep positions safe
                failures += 1
                log.error("live_cycle_failed", error=repr(exc), failures=failures)
                if failures in (3, 10, 30):
                    self.notify(f"Engine cycle failing ({failures}x): {exc!r}")
            self.sleep(interval_s)
