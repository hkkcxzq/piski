"""Golden scenarios for the bar engine: every expected number is computed by hand."""

import numpy as np
import pytest

from quant.backtest.engine import EXIT_STOP, EXIT_TIME, EXIT_TP, Costs, Signals, simulate

NO_COST = Costs(0.0, 0.0, 0.0)


def bars(rows: list[tuple[float, float, float, float]]) -> tuple[np.ndarray, ...]:
    a = np.asarray(rows, dtype=np.float64)
    return a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(), a[:, 3].copy(), np.ones(len(rows), dtype=bool)


def signal_at(
    n: int,
    i: int,
    side: int,
    stop: float,
    tp: float = np.nan,
    hold: int = 100,
    entry_px: float = np.nan,
    expiry: int = 0,
) -> Signals:
    s = Signals.empty(n)
    s.side[i] = side
    s.stop_dist[i] = stop
    s.tp_dist[i] = tp
    s.max_hold[i] = hold
    s.entry_px[i] = entry_px
    s.expiry[i] = expiry
    return s


def test_market_long_hits_take_profit_next_bars() -> None:
    o, h, l, c, v = bars(
        [(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 102.5, 100.2, 102), (102, 102, 102, 102)]
    )
    t = simulate(o, h, l, c, v, signal_at(4, 0, 1, stop=1.0, tp=2.0), NO_COST)
    assert len(t) == 1
    assert t.entry[0] == 100 and t.exit[0] == 102 and t.reason[0] == EXIT_TP
    assert t.entry_i[0] == 1 and t.exit_i[0] == 2
    assert t.net[0] == pytest.approx(0.02)
    assert t.r_multiple[0] == pytest.approx(2.0)


def test_no_take_profit_on_entry_bar_but_stop_is_checked() -> None:
    # entry bar reaches the TP level but also the stop: the stop wins (pessimistic)
    o, h, l, c, v = bars([(100, 100, 100, 100), (100, 103, 98.9, 101), (101, 101, 101, 101)])
    t = simulate(o, h, l, c, v, signal_at(3, 0, 1, stop=1.0, tp=2.0), NO_COST)
    assert t.reason[0] == EXIT_STOP and t.exit[0] == 99 and t.exit_i[0] == 1


def test_stop_and_tp_in_same_bar_assumes_stop() -> None:
    o, h, l, c, v = bars([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (100, 103, 98, 100)])
    t = simulate(o, h, l, c, v, signal_at(3, 0, 1, stop=1.0, tp=2.0), NO_COST)
    assert t.reason[0] == EXIT_STOP and t.exit[0] == 99


def test_gap_through_stop_exits_at_open() -> None:
    o, h, l, c, v = bars([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (97, 97.5, 96, 97)])
    t = simulate(o, h, l, c, v, signal_at(3, 0, 1, stop=1.0), NO_COST)
    assert t.exit[0] == 97 and t.net[0] == pytest.approx(-0.03)


def test_short_time_exit_with_fees_and_slippage() -> None:
    costs = Costs(taker_fee=0.001, maker_fee=0.0, slippage=0.0005)
    o, h, l, c, v = bars([(100, 100, 100, 100), (100, 100.2, 99.8, 99.9), (99.9, 100, 99, 99), (99, 99, 99, 99)])
    t = simulate(o, h, l, c, v, signal_at(4, 0, -1, stop=2.0, hold=1), costs)
    entry = 100 * (1 - 0.0005)
    exit_ = 99 * (1 + 0.0005)
    assert t.reason[0] == EXIT_TIME and t.exit_i[0] == 2
    assert t.entry[0] == pytest.approx(entry)
    assert t.net[0] == pytest.approx(-(exit_ - entry) / entry - 0.002)


def test_limit_needs_trade_through_and_expires() -> None:
    o, h, l, c, v = bars([(100, 100, 100, 100), (100, 100, 99.0, 99.5), (99.5, 99.6, 98.9, 99.2), (99, 99, 99, 99)])
    # buy limit at 99: bar 1 only touches 99 (no fill), bar 2 trades through -> fill at 99
    t = simulate(o, h, l, c, v, signal_at(4, 0, 1, stop=5.0, hold=1, entry_px=99.0, expiry=5), NO_COST)
    assert len(t) == 1 and t.entry_i[0] == 2 and t.entry[0] == 99.0
    # same order expiring after 1 bar never fills
    t2 = simulate(o, h, l, c, v, signal_at(4, 0, 1, stop=5.0, hold=1, entry_px=99.0, expiry=1), NO_COST)
    assert len(t2) == 0


def test_limit_uses_maker_fee() -> None:
    costs = Costs(taker_fee=0.001, maker_fee=0.0002, slippage=0.0)
    o, h, l, c, v = bars([(100, 100, 100, 100), (100, 100, 98.5, 99), (99, 101.5, 99, 101), (101, 101, 101, 101)])
    t = simulate(o, h, l, c, v, signal_at(4, 0, 1, stop=1.0, tp=2.0, entry_px=99.0, expiry=3), costs)
    assert t.reason[0] == EXIT_TP
    assert t.net[0] == pytest.approx(2 / 99 - 0.0004)


def test_signal_on_invalid_bar_is_ignored_and_one_position_at_a_time() -> None:
    o, h, l, c, v = bars([(100, 100, 100, 100)] * 6)
    v[0] = False
    s = Signals.empty(6)
    s.side[:] = 1
    s.stop_dist[:] = 1.0
    s.max_hold[:] = 2
    t = simulate(o, h, l, c, v, s, NO_COST)
    # first valid signal at bar 1 -> enter 2, exit at 4 (time); next signal at 4 -> enter 5 and exit at end
    assert list(t.entry_i) == [2, 5]
    assert all(t.exit_i[:-1] < t.entry_i[1:])


def test_no_lookahead_future_bars_do_not_change_past_trades() -> None:
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 500)))
    o = np.r_[px[0], px[:-1]]
    h = np.maximum(o, px) * 1.001
    l = np.minimum(o, px) * 0.999
    v = np.ones(500, dtype=bool)
    s = Signals.empty(500)
    s.side[::17] = 1
    s.side[5::23] = -1
    s.stop_dist[:] = 0.3
    s.tp_dist[:] = 0.6
    s.max_hold[:] = 10
    full = simulate(o, h, l, px, v, s, Costs())
    cut = 300
    part = simulate(
        o[:cut],
        h[:cut],
        l[:cut],
        px[:cut],
        v[:cut],
        Signals(s.side[:cut], s.entry_px[:cut], s.expiry[:cut], s.stop_dist[:cut], s.tp_dist[:cut], s.max_hold[:cut]),
        Costs(),
    )
    closed_before = full.exit_i < cut - 1
    k = int(closed_before.sum())
    assert np.array_equal(full.entry_i[:k], part.entry_i[:k])
    assert np.allclose(full.net[:k], part.net[:k])
