from decimal import Decimal
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from quant.traders.reconstruct import ReconstructionResult, reconstruct
from synth import FillFactory

EPS = Decimal("1e-18")  # Decimal context precision is 28 digits; VWAP division may round

sizes = st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0)
prices = st.integers(min_value=50, max_value=150)
step_lists = st.lists(st.tuples(sizes, prices), min_size=1, max_size=40)


def close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= EPS


def _check_conservation(ff: FillFactory, fills: list[dict[str, Any]], rec: ReconstructionResult) -> None:
    episodes = rec.trips + rec.open_trips
    # fees and realized PnL are neither lost nor duplicated
    assert close(sum((t.fees for t in episodes), Decimal(0)), sum(Decimal(f["fee"]) for f in fills))
    assert close(sum((t.gross_pnl for t in episodes), Decimal(0)), sum(Decimal(f["closedPnl"]) for f in fills))
    assert close(sum((r.gross_pnl for r in rec.realizations), Decimal(0)), sum(Decimal(f["closedPnl"]) for f in fills))
    assert close(sum((e.notional for e in rec.entries), Decimal(0)), sum((t.entry_cost for t in episodes), Decimal(0)))
    # every closed episode is flat and fully observed
    for t in rec.trips:
        assert t.entry_qty == t.exit_qty
        assert t.complete
    # the final open episode matches the final position
    final = ff.pos["BTC"]
    if final == 0:
        assert not rec.open_trips
    else:
        (open_t,) = rec.open_trips
        assert open_t.entry_qty - open_t.exit_qty == abs(final)
        assert open_t.side.sign == (1 if final > 0 else -1)
    assert rec.n_gaps == 0


@settings(max_examples=200, deadline=None)
@given(step_lists)
def test_accounting_with_unique_timestamps(steps: list[tuple[int, int]]) -> None:
    ff = FillFactory(fee_rate="0.001")
    fills = [ff.fill("BTC", i + 1, str(px), str(sz)) for i, (sz, px) in enumerate(steps)]
    rec = reconstruct("0xa", fills)
    _check_conservation(ff, fills, rec)
    for t in rec.trips:
        assert close(t.price_pnl, t.gross_pnl)
    closes = [t.close_time_ms or 0 for t in rec.trips]
    opens = [t.open_time_ms for t in rec.trips]
    assert all(c <= o for c, o in zip(closes, opens[1:], strict=False))


@settings(max_examples=200, deadline=None)
@given(step_lists)
def test_accounting_with_scrambled_same_millisecond_fills(steps: list[tuple[int, int]]) -> None:
    # groups of 3 consecutive fills share a timestamp (like one order sweeping the book)
    # and arrive in reversed order within that millisecond. The position chain must be
    # recovered without false gaps; totals must be conserved. (Which entry a same-ms
    # close belongs to can be ambiguous, so per-trip VWAP PnL is not asserted here.)
    ff = FillFactory(fee_rate="0.001")
    fills = [ff.fill("BTC", 1 + i // 3, str(px), str(sz)) for i, (sz, px) in enumerate(steps)]
    fills = [f for chunk in range(0, len(fills), 3) for f in reversed(fills[chunk : chunk + 3])]
    rec = reconstruct("0xa", fills)
    _check_conservation(ff, fills, rec)
