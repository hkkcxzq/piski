from decimal import Decimal

import pytest

from quant.traders.models import Fill, FillParseError, Side
from quant.traders.reconstruct import assign_funding, reconstruct
from synth import FillFactory


def test_open_add_reduce_close_single_episode() -> None:
    ff = FillFactory(fee_rate="0.001")
    fills = [
        ff.fill("BTC", 1, "100", "1"),
        ff.fill("BTC", 2, "90", "1"),  # add while under water -> adds_against
        ff.fill("BTC", 3, "110", "-1"),
        ff.fill("BTC", 4, "120", "-1"),
    ]
    rec = reconstruct("0xa", fills)
    assert len(rec.trips) == 1 and not rec.open_trips
    t = rec.trips[0]
    assert t.complete and t.side is Side.LONG
    assert t.entry_vwap == Decimal(95)
    assert t.exit_vwap == Decimal(115)
    assert t.max_abs_position == Decimal(2)
    assert t.gross_pnl == Decimal(40)  # (110-95) + (120-95)
    assert t.price_pnl == t.gross_pnl
    assert t.fees == Decimal("0.42")  # 0.1 + 0.09 + 0.11 + 0.12
    assert t.n_adds == 1 and t.adds_against == 1
    assert t.holding_ms == 3
    assert rec.pnl_crosscheck() == 0.0


def test_flip_splits_fill_between_episodes() -> None:
    ff = FillFactory(fee_rate="0.001")
    fills = [ff.fill("ETH", 1, "100", "2"), ff.fill("ETH", 2, "110", "-5"), ff.fill("ETH", 3, "100", "3")]
    rec = reconstruct("0xa", fills)
    assert [t.side for t in rec.trips] == [Side.LONG, Side.SHORT]
    long_t, short_t = rec.trips
    assert long_t.exit_qty == 2 and long_t.gross_pnl == Decimal(20)
    # 2/5 of the flip fill's fee belongs to the closing leg, 3/5 to the new short
    assert long_t.fees == Decimal("0.2") + Decimal("0.55") * Decimal(2) / Decimal(5)
    assert short_t.entry_qty == 3 and short_t.entry_vwap == Decimal(110)
    assert short_t.gross_pnl == Decimal(30)
    total_fees = sum(Decimal(f["fee"]) for f in fills)
    assert sum(t.fees for t in rec.trips) == total_fees


def test_position_existing_before_window_is_not_complete() -> None:
    fills = [
        {
            "coin": "BTC",
            "time": 1,
            "px": "100",
            "sz": "1",
            "side": "A",
            "startPosition": "1",
            "closedPnl": "5",
            "fee": "0",
            "crossed": True,
            "dir": "Close Long",
            "tid": 1,
        }
    ]
    rec = reconstruct("0xa", fills)
    assert len(rec.trips) == 1
    assert not rec.trips[0].opened_from_flat
    assert rec.complete_trips == []


def test_gap_marks_episode_unclean() -> None:
    ff = FillFactory()
    fills = [ff.fill("BTC", 1, "100", "1")]
    # a fill is missing: venue says position was 3 before this sell
    fills.append(
        {
            "coin": "BTC",
            "time": 5,
            "px": "101",
            "sz": "3",
            "side": "A",
            "startPosition": "3",
            "closedPnl": "3",
            "fee": "0",
            "crossed": True,
            "dir": "Close Long",
            "tid": 99,
        }
    )
    rec = reconstruct("0xa", fills)
    assert rec.n_gaps == 1
    assert len(rec.trips) == 1 and not rec.trips[0].clean
    assert rec.complete_trips == []


def test_spot_and_malformed_fills_are_skipped() -> None:
    ff = FillFactory()
    good = ff.fill("BTC", 1, "100", "1")
    spot = dict(good, coin="@107")
    bad = dict(good, px="abc", tid=1000)
    rec = reconstruct("0xa", [good, spot, bad])
    assert rec.n_skipped_non_perp == 1
    assert rec.n_parse_errors == 1
    assert len(rec.open_trips) == 1


def test_maker_fraction_by_notional() -> None:
    ff = FillFactory()
    fills = [ff.fill("BTC", 1, "100", "1", crossed=False), ff.fill("BTC", 2, "100", "-1", crossed=True)]
    t = reconstruct("0xa", fills).trips[0]
    assert t.maker_fraction == pytest.approx(0.5)


def test_funding_assigned_to_open_episode() -> None:
    ff = FillFactory()
    fills = [
        ff.fill("BTC", 1000, "100", "1"),
        ff.fill("BTC", 5000, "100", "-1"),
        ff.fill("BTC", 9000, "100", "1"),
        ff.fill("BTC", 12000, "100", "-1"),
    ]
    rec = reconstruct("0xa", fills)
    events = [
        {"time": 3000, "delta": {"coin": "BTC", "usdc": "-1.5"}},
        {"time": 10000, "delta": {"coin": "BTC", "usdc": "0.5"}},
        {"time": 7000, "delta": {"coin": "BTC", "usdc": "9"}},  # flat at that time
        {"time": 3000, "delta": {"coin": "ETH", "usdc": "9"}},
    ]
    unmatched = assign_funding(rec.trips, events)
    assert unmatched == 2
    assert [t.funding for t in rec.trips] == [Decimal("-1.5"), Decimal("0.5")]
    assert rec.trips[0].net_pnl == Decimal("-1.5") - rec.trips[0].fees


def test_fill_parse_validation() -> None:
    base = {"coin": "BTC", "time": 1, "px": "1", "sz": "1", "side": "B", "startPosition": "0"}
    assert Fill.from_raw(base).fee == 0
    with pytest.raises(FillParseError):
        Fill.from_raw(dict(base, side="X"))
    with pytest.raises(FillParseError):
        Fill.from_raw(dict(base, sz="0"))
    with pytest.raises(FillParseError):
        Fill.from_raw({k: v for k, v in base.items() if k != "startPosition"})
    liq = Fill.from_raw(dict(base, liquidation={"method": "market"}))
    assert liq.liquidation
