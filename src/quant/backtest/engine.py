"""Bar-level single-position backtest engine with conservative fill rules.

Signals are produced at the *close* of bar ``i`` using data up to and including bar ``i``;
the engine can act on them from bar ``i + 1`` only (no look-ahead by construction).

Fill rules (all deliberately pessimistic):
* market entry: open of bar ``i + 1`` plus slippage, taker fee;
* limit entry: fills only if price trades *through* the limit (``low < px`` for a buy),
  at the limit or at a better open, maker fee; expires after ``expiry`` bars;
* stop: if the bar opens beyond the stop → exit at the open (gap), otherwise at the stop;
  slippage + taker fee;
* take profit: resting limit, needs a trade *through* the level, maker fee;
* stop and take profit touched in the same bar → the stop is assumed to come first;
* no take profit on the entry bar (the order of events inside the bar is unknown);
* time exit at the close of the bar where the holding limit is reached, taker fee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

EXIT_STOP, EXIT_TP, EXIT_TIME, EXIT_END = 1, 2, 3, 4


@dataclass(frozen=True, slots=True)
class Costs:
    taker_fee: float = 0.00055  # Bybit USDT perpetual, base tier (verify /v5/account/fee-rate)
    maker_fee: float = 0.00020
    slippage: float = 0.0001  # 1 bp on every market-type execution

    def stressed(self, factor: float) -> Costs:
        return Costs(self.taker_fee * factor, self.maker_fee * factor, self.slippage * factor)


@dataclass(slots=True)
class Signals:
    """Per-bar intents. ``side`` 0 = no signal. Distances are in price units."""

    side: npt.NDArray[np.int8]
    entry_px: F64  # NaN = market entry at next open
    expiry: npt.NDArray[np.int64]  # bars a limit order stays active
    stop_dist: F64
    tp_dist: F64  # NaN = no take profit
    max_hold: npt.NDArray[np.int64]  # bars

    @classmethod
    def empty(cls, n: int) -> Signals:
        return cls(
            np.zeros(n, np.int8),
            np.full(n, np.nan),
            np.zeros(n, np.int64),
            np.full(n, np.nan),
            np.full(n, np.nan),
            np.zeros(n, np.int64),
        )


@nb.njit(cache=True)
def _simulate(  # type: ignore[no-untyped-def]
    o,
    h,
    l,
    c,
    valid,
    side,
    entry_px,
    expiry,
    stop_dist,
    tp_dist,
    max_hold,
    taker,
    maker,
    slip,
):  # pragma: no cover - compiled
    n = o.size
    cap = n // 2 + 1
    t_entry_i = np.empty(cap, np.int64)
    t_exit_i = np.empty(cap, np.int64)
    t_side = np.empty(cap, np.int8)
    t_entry = np.empty(cap)
    t_exit = np.empty(cap)
    t_stop_frac = np.empty(cap)
    t_net = np.empty(cap)
    t_reason = np.empty(cap, np.int8)
    k = 0

    state = 0  # 0 flat, 1 pending limit, 2 in position
    s = 0
    px = 0.0
    stop = 0.0
    tp = np.nan
    fee_in = 0.0
    entry_bar = 0
    expire_bar = 0
    hold = 0
    sd = 0.0
    td = np.nan
    i = 0
    while i < n:
        if state == 1:
            if i > expire_bar:
                state = 0
            elif valid[i] and ((s > 0 and l[i] < px) or (s < 0 and h[i] > px)):
                fill = min(px, o[i]) if s > 0 else max(px, o[i])
                px = fill
                stop = px - s * sd
                tp = px + s * td if not np.isnan(td) else np.nan
                fee_in = maker
                entry_bar = i
                state = 2
                # stop inside the fill bar (pessimistic)
                if (s > 0 and l[i] <= stop) or (s < 0 and h[i] >= stop):
                    ex = stop * (1.0 - s * slip)
                    t_entry_i[k] = entry_bar
                    t_exit_i[k] = i
                    t_side[k] = s
                    t_entry[k] = px
                    t_exit[k] = ex
                    t_stop_frac[k] = sd / px
                    t_net[k] = s * (ex - px) / px - fee_in - taker
                    t_reason[k] = 1
                    k += 1
                    state = 0
                i += 1
                continue
        if state == 2 and i > entry_bar:
            ex = np.nan
            reason = 0
            fee_out = taker
            if valid[i]:
                if (s > 0 and o[i] <= stop) or (s < 0 and o[i] >= stop):
                    ex = o[i] * (1.0 - s * slip)
                    reason = 1
                elif (s > 0 and l[i] <= stop) or (s < 0 and h[i] >= stop):
                    ex = stop * (1.0 - s * slip)
                    reason = 1
                elif not np.isnan(tp) and ((s > 0 and h[i] > tp) or (s < 0 and l[i] < tp)):
                    ex = tp
                    reason = 2
                    fee_out = maker
            if reason == 0 and i - entry_bar >= hold:
                ex = c[i] * (1.0 - s * slip)
                reason = 3
            if reason == 0 and i == n - 1:
                ex = c[i]
                reason = 4
            if reason != 0:
                t_entry_i[k] = entry_bar
                t_exit_i[k] = i
                t_side[k] = s
                t_entry[k] = px
                t_exit[k] = ex
                t_stop_frac[k] = sd / px
                t_net[k] = s * (ex - px) / px - fee_in - fee_out
                t_reason[k] = reason
                k += 1
                state = 0
        if state == 0 and side[i] != 0 and valid[i] and i + 1 < n and stop_dist[i] > 0:
            s = side[i]
            sd = stop_dist[i]
            td = tp_dist[i]
            hold = max_hold[i]
            if np.isnan(entry_px[i]):
                px = o[i + 1] * (1.0 + s * slip)
                stop = px - s * sd
                tp = px + s * td if not np.isnan(td) else np.nan
                fee_in = taker
                entry_bar = i + 1
                state = 2
                # stop inside the entry bar (pessimistic)
                j = i + 1
                if valid[j] and ((s > 0 and l[j] <= stop) or (s < 0 and h[j] >= stop)):
                    ex = stop * (1.0 - s * slip)
                    t_entry_i[k] = entry_bar
                    t_exit_i[k] = j
                    t_side[k] = s
                    t_entry[k] = px
                    t_exit[k] = ex
                    t_stop_frac[k] = sd / px
                    t_net[k] = s * (ex - px) / px - fee_in - taker
                    t_reason[k] = 1
                    k += 1
                    state = 0
                    i += 2
                    continue
            else:
                px = entry_px[i]
                expire_bar = i + expiry[i]
                state = 1
        i += 1
    return (t_entry_i[:k], t_exit_i[:k], t_side[:k], t_entry[:k], t_exit[:k], t_stop_frac[:k], t_net[:k], t_reason[:k])


@dataclass(frozen=True, slots=True)
class Trades:
    entry_i: npt.NDArray[np.int64]
    exit_i: npt.NDArray[np.int64]
    side: npt.NDArray[np.int8]
    entry: F64
    exit: F64
    stop_frac: F64  # stop distance / entry price
    net: F64  # net return on notional, after fees and slippage
    reason: npt.NDArray[np.int8]

    def __len__(self) -> int:
        return int(self.net.size)

    @property
    def r_multiple(self) -> F64:
        return self.net / self.stop_frac


def simulate(
    o: F64,
    h: F64,
    l: F64,
    c: F64,
    valid: npt.NDArray[np.bool_],
    sig: Signals,
    costs: Costs,
) -> Trades:
    out = _simulate(
        o,
        h,
        l,
        c,
        valid,
        sig.side,
        sig.entry_px,
        sig.expiry,
        sig.stop_dist,
        sig.tp_dist,
        sig.max_hold,
        costs.taker_fee,
        costs.maker_fee,
        costs.slippage,
    )
    return Trades(*out)
