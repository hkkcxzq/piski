"""Reconstruct position episodes (round trips) from raw fills.

The venue reports ``startPosition`` with every fill, so the position path is fully
determined by the data. We never trust our own running sum blindly: if a fill's
``startPosition`` disagrees with the tracked position (missing fills, ADL, transfers),
the affected episode is marked unclean and excluded from statistics instead of being
silently "repaired".
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from quant.traders.models import Fill, FillParseError, RoundTrip, Side, is_perp_coin

ZERO = Decimal(0)


def _sign(x: Decimal) -> int:
    return (x > 0) - (x < 0)


@dataclass(slots=True)
class ReconstructionResult:
    trips: list[RoundTrip] = field(default_factory=list)  # closed episodes (clean or not)
    open_trips: list[RoundTrip] = field(default_factory=list)  # still open at end of data
    n_fills: int = 0
    n_skipped_non_perp: int = 0
    n_parse_errors: int = 0
    n_gaps: int = 0
    first_fill_ms: int | None = None
    last_fill_ms: int | None = None

    @property
    def complete_trips(self) -> list[RoundTrip]:
        return [t for t in self.trips if t.complete]

    def pnl_crosscheck(self) -> float | None:
        """Relative gap between venue closedPnl and VWAP-recomputed PnL over complete trips."""
        trips = self.complete_trips
        venue = sum((t.gross_pnl for t in trips), ZERO)
        recomputed = sum((t.price_pnl for t in trips), ZERO)
        scale = sum((abs(t.gross_pnl) for t in trips), ZERO)
        if scale == 0:
            return None
        return float(abs(venue - recomputed) / scale)


class _CoinBook:
    __slots__ = ("coin", "current", "position", "trader")

    def __init__(self, trader: str, coin: str) -> None:
        self.trader = trader
        self.coin = coin
        self.position: Decimal | None = None
        self.current: RoundTrip | None = None


def _new_trip(book: _CoinBook, side: Side, fill: Fill, from_flat: bool) -> RoundTrip:
    return RoundTrip(
        trader=book.trader,
        coin=book.coin,
        side=side,
        open_time_ms=fill.time_ms,
        opened_from_flat=from_flat,
        first_fill_px=fill.px,
    )


def _book_part(trip: RoundTrip, fill: Fill, qty: Decimal, opening: bool) -> None:
    share = qty / fill.sz
    notional = fill.px * qty
    trip.fees += fill.fee * share
    trip.total_notional += notional
    if not fill.crossed:
        trip.maker_notional += notional
    if opening:
        trip.entry_qty += qty
        trip.entry_cost += notional
    else:
        trip.exit_qty += qty
        trip.exit_cost += notional
        trip.gross_pnl += fill.closed_pnl  # realized PnL belongs entirely to the closing leg


def reconstruct(trader: str, raw_fills: Iterable[Mapping[str, Any]]) -> ReconstructionResult:
    result = ReconstructionResult()
    fills: list[Fill] = []
    for raw in raw_fills:
        try:
            fill = Fill.from_raw(dict(raw))
        except FillParseError:
            result.n_parse_errors += 1
            continue
        if not is_perp_coin(fill.coin):
            result.n_skipped_non_perp += 1
            continue
        fills.append(fill)
    fills.sort(key=lambda f: (f.time_ms, f.tid if f.tid is not None else -1))
    result.n_fills = len(fills)
    if fills:
        result.first_fill_ms, result.last_fill_ms = fills[0].time_ms, fills[-1].time_ms

    books: dict[str, _CoinBook] = {}
    for fill in fills:
        book = books.get(fill.coin)
        if book is None:
            book = books[fill.coin] = _CoinBook(trader, fill.coin)
        _apply(book, fill, result)

    for book in books.values():
        if book.current is not None:
            result.open_trips.append(book.current)
    result.trips.sort(key=lambda t: (t.open_time_ms, t.coin))
    return result


def _close(book: _CoinBook, time_ms: int, result: ReconstructionResult) -> None:
    if book.current is None:
        raise RuntimeError(f"{book.coin}: close without an open episode")
    book.current.close_time_ms = time_ms
    result.trips.append(book.current)
    book.current = None


def _apply(book: _CoinBook, fill: Fill, result: ReconstructionResult) -> None:
    if book.position is None:
        # First observation of this coin: an existing position predates our data.
        book.position = fill.start_position
        if fill.start_position != 0:
            book.current = _new_trip(book, Side.LONG if fill.start_position > 0 else Side.SHORT, fill, False)
            book.current.max_abs_position = abs(fill.start_position)
    elif fill.start_position != book.position:
        result.n_gaps += 1
        if book.current is not None:
            book.current.clean = False
            if _sign(fill.start_position) != book.current.side.sign:
                _close(book, fill.time_ms, result)
        book.position = fill.start_position
        if fill.start_position != 0 and book.current is None:
            book.current = _new_trip(book, Side.LONG if fill.start_position > 0 else Side.SHORT, fill, False)
            book.current.clean = False

    before = book.position
    after = before + fill.signed_sz
    closing_qty = ZERO
    opening_qty = ZERO
    if before == 0 or (_sign(after) == _sign(before) and abs(after) > abs(before)):
        opening_qty = fill.sz
    elif after == 0 or (_sign(after) == _sign(before) and abs(after) < abs(before)):
        closing_qty = fill.sz
    else:  # flip through zero
        closing_qty = abs(before)
        opening_qty = abs(after)

    if closing_qty:
        trip = book.current
        if trip is None:
            raise RuntimeError(f"{book.coin}: closing fill without an open episode")
        trip.n_fills += 1
        trip.liquidated = trip.liquidated or fill.liquidation
        _book_part(trip, fill, closing_qty, opening=False)
        if after == 0 or _sign(after) != _sign(before):
            _close(book, fill.time_ms, result)

    if opening_qty:
        if book.current is None:
            book.current = _new_trip(book, Side.LONG if after > 0 else Side.SHORT, fill, True)
        else:
            trip = book.current
            trip.n_adds += 1
            if trip.entry_qty and (fill.px - trip.entry_vwap) * trip.side.sign < 0:
                trip.adds_against += 1
        book.current.n_fills += 1
        _book_part(book.current, fill, opening_qty, opening=True)
        book.current.max_abs_position = max(book.current.max_abs_position, abs(after))

    book.position = after


def assign_funding(trips: list[RoundTrip], funding_events: Iterable[Mapping[str, Any]]) -> int:
    """Attach per-user funding payments to the episode that was open at payment time.

    Returns the number of events that could not be matched (e.g. positions opened
    before our fill window). Events are ``{"time": ms, "delta": {"coin", "usdc"}}``.
    """
    by_coin: dict[str, list[RoundTrip]] = defaultdict(list)
    for trip in trips:
        by_coin[trip.coin].append(trip)
    starts: dict[str, list[int]] = {}
    for coin, coin_trips in by_coin.items():
        coin_trips.sort(key=lambda t: t.open_time_ms)
        starts[coin] = [t.open_time_ms for t in coin_trips]
    unmatched = 0
    for event in funding_events:
        delta = event.get("delta") or {}
        coin = str(delta.get("coin", ""))
        t = int(event["time"])
        candidates = by_coin.get(coin)
        if not candidates:
            unmatched += 1
            continue
        idx = bisect.bisect_right(starts[coin], t) - 1
        if idx < 0:
            unmatched += 1
            continue
        trip = candidates[idx]
        if trip.close_time_ms is not None and t > trip.close_time_ms:
            unmatched += 1
            continue
        trip.funding += Decimal(str(delta.get("usdc", "0")))
    return unmatched
