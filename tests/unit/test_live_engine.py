import json
from decimal import Decimal
from pathlib import Path

import httpx
import numpy as np
import pytest

from fake_bybit import FakeBybit
from quant.app.config import RiskLimits
from quant.backtest.engine import Signals
from quant.data.bars import Bars
from quant.exchanges.bybit.client import BybitClient, Instrument, sign
from quant.live.engine import LiveEngine, LiveStrategy, bars_from_klines
from quant.live.risk import check_losses, position_qty
from quant.live.state import Store
from quant.news.policy import RiskLevel, RiskState, save_state

H4 = 4 * 3_600_000
T0 = 1_900_000_000_000 - (1_900_000_000_000 % H4)  # far from any FOMC date in the calendar
N_BARS = 300
NOW = T0 + N_BARS * H4 + 60_000  # one minute after the last bar closed


def klines(n: int = N_BARS, px: float = 100.0) -> list[list[str]]:
    rows = []
    for i in range(n + 1):  # +1: the still-open bar, which the engine must ignore
        t = T0 + i * H4
        rows.append([str(t), str(px), str(px + 1), str(px - 1), str(px), "10", "1000"])
    return rows[::-1]


def always(side: int, stop: float = 2.0, tp: float = 4.0, hold: int = 6):
    def build(b: Bars) -> Signals:
        s = Signals.empty(len(b))
        s.side[-1] = side
        s.stop_dist[-1] = stop
        s.tp_dist[-1] = tp
        s.max_hold[-1] = hold
        return s

    return build


def make(tmp_path: Path, fake: FakeBybit, side: int = 1, clock: list[int] | None = None) -> tuple[LiveEngine, Store]:
    clock = clock or [NOW]
    http = httpx.Client(transport=fake.transport())
    trade = BybitClient("https://demo", fake.key, fake.secret, http=http, clock=lambda: clock[0], sleep=lambda _s: None)
    market = BybitClient("https://main", http=http, sleep=lambda _s: None)
    store = Store(tmp_path)
    notes: list[str] = []
    eng = LiveEngine(
        trade,
        market,
        LiveStrategy("test", 240, always(side)),
        ("BTCUSDT",),
        RiskLimits(),
        store,
        tmp_path / "news" / "risk_state.json",
        clock=lambda: clock[0],
        sleep=lambda _s: None,
        notify=notes.append,
    )
    eng.notes = notes  # type: ignore[attr-defined]
    return eng, store


INST = Instrument("BTCUSDT", Decimal("0.1"), Decimal("0.001"), Decimal("0.001"), Decimal("100"))


def test_signature_matches_v5_spec() -> None:
    # HMAC-SHA256(secret, timestamp + key + recv_window + payload), hex
    import hashlib  # noqa: PLC0415
    import hmac  # noqa: PLC0415

    expected = hmac.new(b"sec", b"1700000000000key5000category=linear", hashlib.sha256).hexdigest()
    assert sign("sec", "1700000000000", "key", "5000", "category=linear") == expected


def test_position_size_risks_fixed_fraction_and_caps_leverage() -> None:
    lim = RiskLimits()  # 0.25 % risk, leverage <= 5
    qty = position_qty(Decimal(10000), Decimal(100), Decimal(2), INST, lim)
    assert qty == Decimal("12.5")  # 25 $ risk / 2 $ stop
    tight = position_qty(Decimal(10000), Decimal(100), Decimal("0.01"), INST, lim)
    assert tight == Decimal("500")  # capped by 5x leverage: 50 000 $ / 100
    assert position_qty(Decimal(10), Decimal(100000), Decimal(2000), INST, lim) == 0  # below min qty -> skip


def test_loss_limits() -> None:
    lim = RiskLimits()
    assert not check_losses(Decimal(9850), Decimal(10000), Decimal(10000), Decimal(10000), lim).halt
    assert check_losses(Decimal(9800), Decimal(10000), Decimal(10000), Decimal(10000), lim).halt
    assert "drawdown" in check_losses(Decimal(8999), Decimal(9000), Decimal(9000), Decimal(10000), lim).reason


def test_bars_exclude_the_open_candle() -> None:
    rows = sorted((int(r[0]), *map(float, r[1:6])) for r in klines())
    b = bars_from_klines(rows, 240, NOW)
    assert len(b) == N_BARS and b.t[-1] == T0 + (N_BARS - 1) * H4


def test_entry_with_exchange_stop_journal_and_no_duplicate(tmp_path: Path) -> None:
    fake = FakeBybit(klines())
    eng, store = make(tmp_path, fake)
    st = eng.run_cycle()
    assert "BTCUSDT" in st.open_trades
    (order,) = fake.orders
    assert order["side"] == "Buy" and order["qty"] == "12.500" and order["stopLoss"] == "98.0"
    assert order["takeProfit"] == "104.0" and order["orderLinkId"].startswith("test-")
    assert fake.leverage["BTCUSDT"] == "1"
    opened = [json.loads(x) for x in store.journal_path.read_text().splitlines()]
    assert opened[0]["type"] == "open" and opened[0]["strategy_id"] == "test"
    eng.run_cycle()  # same bar again: no second order
    assert len(fake.orders) == 1


def test_missing_stop_is_repaired_or_position_closed(tmp_path: Path) -> None:
    fake = FakeBybit(klines())
    fake.drop_stops = True
    eng, _ = make(tmp_path, fake)
    eng.run_cycle()
    assert fake.positions["BTCUSDT"]["stopLoss"] == "98.0"  # repaired via trading-stop
    fake2 = FakeBybit(klines())
    fake2.drop_stops = True
    fake2.reject_trading_stop = True
    eng2, _ = make(tmp_path / "b", fake2)
    st = eng2.run_cycle()
    assert "BTCUSDT" not in fake2.positions and not st.open_trades  # closed: no unprotected positions
    assert any("NOT confirmed" in n for n in eng2.notes)  # type: ignore[attr-defined]


def test_exchange_stop_out_is_journaled(tmp_path: Path) -> None:
    fake = FakeBybit(klines())
    clock = [NOW]
    eng, store = make(tmp_path, fake, clock=clock)
    eng.run_cycle()
    fake.stop_out("BTCUSDT")
    clock[0] += 5 * 60_000
    st = eng.run_cycle()
    assert not st.open_trades
    rows = [json.loads(x) for x in store.journal_path.read_text().splitlines()]
    assert rows[-1]["type"] == "close" and rows[-1]["pnl"] == "-25"


def test_time_exit(tmp_path: Path) -> None:
    fake = FakeBybit(klines())
    clock = [NOW]
    eng, store = make(tmp_path, fake, clock=clock)
    eng.run_cycle()
    clock[0] += 6 * H4 + 1
    eng.run_cycle()
    assert [o["reduceOnly"] for o in fake.orders] == [False, True, False]  # time exit, then a fresh signal
    rows = [json.loads(x) for x in store.journal_path.read_text().splitlines()]
    assert [r["type"] for r in rows] == ["open", "close", "open"] and rows[1]["reason_for_exit"] == "time exit"


def test_restart_adopts_unknown_position_and_keeps_its_stop(tmp_path: Path) -> None:
    fake = FakeBybit(klines())
    fake.positions["BTCUSDT"] = {
        "symbol": "BTCUSDT",
        "size": "1",
        "side": "Buy",
        "avgPrice": "100",
        "stopLoss": "",
        "takeProfit": "",
    }
    eng, _ = make(tmp_path, fake, side=0)
    st = eng.run_cycle()
    assert st.open_trades["BTCUSDT"].adopted
    assert fake.positions["BTCUSDT"]["stopLoss"] != ""  # a protective stop was placed


def test_daily_loss_halts_and_flattens(tmp_path: Path) -> None:
    fake = FakeBybit(klines())
    clock = [NOW]
    eng, _ = make(tmp_path, fake, clock=clock)
    eng.run_cycle()
    fake.equity = Decimal("9790")  # -2.1 % today
    clock[0] += 60_000
    st = eng.run_cycle()
    assert st.halted.startswith("daily loss") and "BTCUSDT" not in fake.positions
    n = len(fake.orders)
    clock[0] += H4
    eng.run_cycle()
    assert len(fake.orders) == n  # halted: no new entries until manual reset


def test_news_pause_blocks_entries_and_flatten_closes(tmp_path: Path) -> None:
    fake = FakeBybit(klines())
    eng, _ = make(tmp_path, fake)
    save_state(
        tmp_path / "news" / "risk_state.json",
        RiskState(RiskLevel.PAUSE_NEW, NOW + H4, ["high/exchange: test"], [], NOW),
    )
    eng.run_cycle()
    assert not fake.orders
    save_state(tmp_path / "news" / "risk_state.json", RiskState(RiskLevel.NORMAL, 0, [], [], NOW))
    fake.klines = klines(N_BARS + 1)
    eng.clock = lambda: NOW + H4  # type: ignore[method-assign]
    eng.run_cycle()
    assert "BTCUSDT" in fake.positions
    save_state(
        tmp_path / "news" / "risk_state.json",
        RiskState(RiskLevel.FLATTEN, NOW + 2 * H4, ["critical/exchange: hack"], [], NOW + H4),
    )
    eng.run_cycle()
    assert "BTCUSDT" not in fake.positions


def test_kill_switch(tmp_path: Path) -> None:
    fake = FakeBybit(klines())
    eng, store = make(tmp_path, fake)
    eng.run_cycle()
    store.kill_path.write_text("x")
    st = eng.run_cycle()
    assert st.halted == "kill switch file" and "BTCUSDT" not in fake.positions


def test_rejected_order_leaves_no_phantom_trade(tmp_path: Path) -> None:
    fake = FakeBybit(klines())
    fake.fail_orders = True
    eng, _ = make(tmp_path, fake)
    st = eng.run_cycle()
    assert not st.open_trades
    # the signal is not lost: once the exchange accepts orders again, the same bar is retried
    fake.fail_orders = False
    st = eng.run_cycle()
    assert "BTCUSDT" in st.open_trades and len(fake.orders) == 1


def test_no_market_data_request_until_a_new_bar_closes(tmp_path: Path) -> None:
    fake = FakeBybit(klines())
    clock = [NOW]
    calls: list[str] = []
    inner = fake.handler

    def counting(req):  # type: ignore[no-untyped-def]
        calls.append(req.url.path)
        return inner(req)

    fake.handler = counting  # type: ignore[method-assign]
    eng, _ = make(tmp_path, fake, side=0, clock=clock)
    eng.run_cycle()
    clock[0] += 60_000  # one minute later, same bar
    eng.run_cycle()
    assert calls.count("/v5/market/kline") == 1


@pytest.mark.parametrize("side", [1, -1])
def test_short_and_long_prices(tmp_path: Path, side: int) -> None:
    fake = FakeBybit(klines())
    eng, _ = make(tmp_path, fake, side=side)
    eng.run_cycle()
    o = fake.orders[0]
    stop, tp = float(o["stopLoss"]), float(o["takeProfit"])
    assert (stop < 100 < tp) if side > 0 else (tp < 100 < stop)
    assert np.isclose(abs(100 - stop), 2.0)


def test_swing_strategy_signals_on_engine_history() -> None:
    from quant.live.strategies import STRATEGIES  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n = 999  # the engine loads 1000 klines, the last one still open
    px = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, n + 1)))  # uptrend with noise
    rows = [(T0 + i * H4, px[i], px[i] * 1.004, px[i] * 0.996, px[i], 10.0) for i in range(n + 1)]
    bars = bars_from_klines(rows, 240, T0 + n * H4 + 60_000)
    swing = STRATEGIES["swing-mom"]
    sig = swing.build(bars)
    assert sig.side[-1] == 1  # 28-day return is positive
    assert np.isfinite(sig.stop_dist[-1]) and sig.stop_dist[-1] > 0.01 * bars.c[-1]  # multi-day stop, not intraday
    assert sig.max_hold[-1] == 14 * 6  # 14 days of 4h bars
