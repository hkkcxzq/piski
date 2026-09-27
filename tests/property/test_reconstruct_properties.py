from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from quant.traders.reconstruct import reconstruct
from synth import FillFactory

EPS = Decimal("1e-18")  # Decimal context precision is 28 digits; VWAP division may round


def close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= EPS


sizes = st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0)
prices = st.integers(min_value=50, max_value=150)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(sizes, prices), min_size=1, max_size=40))
def test_accounting_is_conserved(steps: list[tuple[int, int]]) -> None:
    ff = FillFactory(fee_rate="0.001")
    fills = [ff.fill("BTC", i + 1, str(px), str(sz)) for i, (sz, px) in enumerate(steps)]
    rec = reconstruct("0xa", fills)
    episodes = rec.trips + rec.open_trips

    # fees and realized PnL are neither lost nor duplicated
    assert close(sum((t.fees for t in episodes), Decimal(0)), sum(Decimal(f["fee"]) for f in fills))
    assert close(sum((t.gross_pnl for t in episodes), Decimal(0)), sum(Decimal(f["closedPnl"]) for f in fills))

    # every closed episode is flat: opened quantity == closed quantity
    for t in rec.trips:
        assert t.entry_qty == t.exit_qty
        assert t.complete
        assert close(t.price_pnl, t.gross_pnl)

    # the final open episode matches the final position
    final = ff.pos["BTC"]
    if final == 0:
        assert not rec.open_trips
    else:
        (open_t,) = rec.open_trips
        assert open_t.entry_qty - open_t.exit_qty == abs(final)
        assert open_t.side.sign == (1 if final > 0 else -1)

    # episodes never overlap in time and are ordered
    closes = [t.close_time_ms or 0 for t in rec.trips]
    opens = [t.open_time_ms for t in rec.trips]
    assert all(c <= o for c, o in zip(closes, opens[1:], strict=False))
