"""Reconstruct trading decisions and position episodes from raw fills.

The venue reports ``startPosition`` with every fill, so the position path is fully
determined by the data. Two consequences:

* **Ordering.** Many fills share one millisecond (one taker order sweeping several
  levels) and their trade ids are not chronological. Within a millisecond we order
  fills by chaining ``startPosition`` → ``startPosition + size``.
* **Gaps.** If a fill's ``startPosition`` still disagrees with the tracked position
  (missing history, ADL, transfers), the affected episode is marked unclean and
  excluded from statistics instead of being silently "repaired".

Outputs, from the same pass:

* ``trips`` — flat-to-flat position episodes (round trips);
* ``entries`` — one event per order that opened/increased a position (a *decision*);
* ``realizations`` — one event per order that reduced/closed a position;
* inventory statistics for a Little's-law estimate of the holding time, which works
  for traders who scale in and out and rarely go flat.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from quant.traders.models import EntryEvent, Fill, FillParseError, Realization, RoundTrip, Side, is_perp_coin

ZERO = Decimal(0)


def _sign(x: Decimal) -> int:
    return (x > 0) - (x < 0)


@dataclass(slots=True)
class ReconstructionResult:
    trips: list[RoundTrip] = field(default_factory=list)  # closed episodes (clean or not)
    open_trips: list[RoundTrip] = field(default_factory=list)  # still open at end of data
    entries: list[EntryEvent] = field(default_factory=list)
    realizations: list[Realization] = field(default_factory=list)
    n_fills: int = 0
    n_skipped_non_perp: int = 0
    n_parse_errors: int = 0
    n_gaps: int = 0
    first_fill_ms: int | None = None
    last_fill_ms: int | None = None
    total_notional: Decimal = ZERO
    maker_notional: Decimal = ZERO
    total_fees: Decimal = ZERO
    # Little's law inputs per coin: ∫|position|dt (units·ms) and closed units
    inventory_area: dict[str, Decimal] = field(default_factory=dict)
    closed_units: dict[str, Decimal] = field(default_factory=dict)
    closed_notional: dict[str, Decimal] = field(default_factory=dict)

    @property
    def complete_trips(self) -> list[RoundTrip]:
        return [t for t in self.trips if t.complete]

    @property
    def maker_fraction(self) -> float | None:
        return float(self.maker_notional / self.total_notional) if self.total_notional else None

    @property
    def avg_fee_rate(self) -> float:
        return float(self.total_fees / self.total_notional) if self.total_notional else 0.0

    def littles_hold_ms(self) -> float | None:
        """Average holding time W = L / λ per coin, weighted by closed notional."""
        num = 0.0
        den = 0.0
        for coin, units in self.closed_units.items():
            if units <= 0:
                continue
            w = float(self.inventory_area.get(coin, ZERO) / units)
            weight = float(self.closed_notional.get(coin, ZERO))
            num += w * weight
            den += weight
        return num / den if den > 0 else None

    def pnl_crosscheck(self) -> float | None:
        """Relative gap between venue closedPnl and VWAP-recomputed PnL over complete trips."""
        trips = self.complete_trips
        venue = sum((t.gross_pnl for t in trips), ZERO)
        recomputed = sum((t.price_pnl for t in trips), ZERO)
        scale = sum((abs(t.gross_pnl) for t in trips), ZERO)
        if scale == 0:
            return None
        return float(abs(venue - recomputed) / scale)


MAX_CHAIN_SEARCH = 400  # larger same-millisecond groups fall back to greedy chaining


def _find_chain(fills: Sequence[Fill], start: Decimal, budget: int = 20_000) -> list[Fill] | None:
    """Depth-first search for an order that uses every fill and keeps positions chained."""
    by_start: dict[Decimal, list[int]] = defaultdict(list)
    for i, f in enumerate(fills):
        by_start[f.start_position].append(i)
    used = [False] * len(fills)
    path: list[int] = []
    steps = 0

    def dfs(pos: Decimal) -> bool:
        nonlocal steps
        if len(path) == len(fills):
            return True
        steps += 1
        if steps > budget:
            return False
        for i in by_start.get(pos, ()):
            if not used[i]:
                used[i] = True
                path.append(i)
                if dfs(fills[i].end_position):
                    return True
                used[i] = False
                path.pop()
        return False

    return [fills[i] for i in path] if dfs(start) else None


def _greedy_chain(group: Sequence[Fill], position: Decimal | None) -> list[Fill]:
    remaining = list(group)
    out: list[Fill] = []
    pos = position
    while remaining:
        idx = next((i for i, f in enumerate(remaining) if pos is not None and f.start_position == pos), None)
        if idx is None:
            ends = {f.end_position for f in remaining}
            idx = next((i for i, f in enumerate(remaining) if f.start_position not in ends), 0)
        fill = remaining.pop(idx)
        out.append(fill)
        pos = fill.end_position
    return out


def order_same_time(group: Sequence[Fill], position: Decimal | None) -> list[Fill]:
    """Order fills sharing one timestamp so that their positions chain.

    Tries, in order: a full chain starting at the current ``position``; a chain starting
    at a *head* (a fill whose start is not the end of another fill — a real gap); when
    the position is unknown, a chain starting flat. Falls back to greedy chaining.
    """
    if len(group) < 2:
        return list(group)
    if len(group) <= MAX_CHAIN_SEARCH:
        ends = {f.end_position for f in group}
        starts: list[Decimal] = []
        if position is not None:
            starts.append(position)
        elif any(f.start_position == 0 for f in group):
            starts.append(Decimal(0))
        starts += [f.start_position for f in group if f.start_position not in ends]
        starts += [f.start_position for f in group]
        tried: set[Decimal] = set()
        for start in starts:
            if start in tried:
                continue
            tried.add(start)
            chain = _find_chain(group, start)
            if chain is not None:
                return chain
    return _greedy_chain(group, position)


class _CoinBook:
    __slots__ = ("coin", "current", "entry_idx", "last_time", "position", "real_idx", "trader")

    def __init__(self, trader: str, coin: str) -> None:
        self.trader = trader
        self.coin = coin
        self.position: Decimal | None = None
        self.current: RoundTrip | None = None
        self.last_time: int | None = None
        self.entry_idx: dict[int, EntryEvent] = {}
        self.real_idx: dict[int, Realization] = {}


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


def _parse(raw_fills: Iterable[Mapping[str, Any]], result: ReconstructionResult) -> dict[str, list[Fill]]:
    by_coin: dict[str, list[Fill]] = defaultdict(list)
    for raw in raw_fills:
        try:
            fill = Fill.from_raw(dict(raw))
        except FillParseError:
            result.n_parse_errors += 1
            continue
        if not is_perp_coin(fill.coin):
            result.n_skipped_non_perp += 1
            continue
        by_coin[fill.coin].append(fill)
    return by_coin


def reconstruct(trader: str, raw_fills: Iterable[Mapping[str, Any]]) -> ReconstructionResult:
    result = ReconstructionResult()
    by_coin = _parse(raw_fills, result)
    for coin in sorted(by_coin):
        fills = by_coin[coin]
        fills.sort(key=lambda f: f.time_ms)
        book = _CoinBook(trader, coin)
        i = 0
        while i < len(fills):
            j = i
            while j + 1 < len(fills) and fills[j + 1].time_ms == fills[i].time_ms:
                j += 1
            for fill in order_same_time(fills[i : j + 1], book.position):
                _apply(book, fill, result)
            i = j + 1
        if book.current is not None:
            result.open_trips.append(book.current)
        result.entries.extend(book.entry_idx.values())
        result.realizations.extend(book.real_idx.values())
        result.n_fills += len(fills)
        if fills:
            first, last = fills[0].time_ms, fills[-1].time_ms
            result.first_fill_ms = first if result.first_fill_ms is None else min(result.first_fill_ms, first)
            result.last_fill_ms = last if result.last_fill_ms is None else max(result.last_fill_ms, last)
    result.trips.sort(key=lambda t: (t.open_time_ms, t.coin))
    result.entries.sort(key=lambda e: (e.time_ms, e.coin))
    result.realizations.sort(key=lambda r: (r.time_ms, r.coin))
    return result


def _close(book: _CoinBook, time_ms: int, result: ReconstructionResult) -> None:
    if book.current is None:
        raise RuntimeError(f"{book.coin}: close without an open episode")
    book.current.close_time_ms = time_ms
    result.trips.append(book.current)
    book.current = None


def _order_key(fill: Fill) -> int:
    if fill.oid is not None:
        return fill.oid
    return -(fill.tid or 0) - 1


def _apply(book: _CoinBook, fill: Fill, result: ReconstructionResult) -> None:
    coin = book.coin
    # inventory integral for Little's law (uses the position held since the previous fill)
    if book.last_time is not None and book.position is not None:
        dt = Decimal(fill.time_ms - book.last_time)
        result.inventory_area[coin] = result.inventory_area.get(coin, ZERO) + abs(book.position) * dt
    book.last_time = fill.time_ms

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

    result.total_notional += fill.notional
    result.total_fees += fill.fee
    if not fill.crossed:
        result.maker_notional += fill.notional

    if closing_qty:
        share = closing_qty / fill.sz
        notional = fill.px * closing_qty
        result.closed_units[coin] = result.closed_units.get(coin, ZERO) + closing_qty
        result.closed_notional[coin] = result.closed_notional.get(coin, ZERO) + notional
        key = _order_key(fill)
        real = book.real_idx.get(key)
        if real is None:
            real = book.real_idx[key] = Realization(coin, fill.time_ms, _sign(before), ZERO, ZERO, ZERO)
        real.closed_notional += notional
        real.gross_pnl += fill.closed_pnl
        real.fees += fill.fee * share

        trip = book.current
        if trip is None:
            raise RuntimeError(f"{coin}: closing fill without an open episode")
        trip.n_fills += 1
        trip.liquidated = trip.liquidated or fill.liquidation
        _book_part(trip, fill, closing_qty, opening=False)
        if after == 0 or _sign(after) != _sign(before):
            _close(book, fill.time_ms, result)

    if opening_qty:
        key = _order_key(fill)
        entry = book.entry_idx.get(key)
        if entry is None:
            entry = book.entry_idx[key] = EntryEvent(coin, fill.time_ms, _sign(after), ZERO)
        entry.notional += fill.px * opening_qty

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
