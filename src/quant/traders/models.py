"""Domain models for trader research."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any


class FillParseError(ValueError):
    pass


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1


def _dec(raw: Any, name: str) -> Decimal:
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, TypeError) as exc:
        raise FillParseError(f"field {name!r} is not a number: {raw!r}") from exc
    if not value.is_finite():
        raise FillParseError(f"field {name!r} is not finite: {raw!r}")
    return value


def is_perp_coin(coin: str) -> bool:
    """Spot markets on Hyperliquid look like ``@107`` or ``PURR/USDC``."""
    return not coin.startswith("@") and "/" not in coin


@dataclass(frozen=True, slots=True)
class Fill:
    coin: str
    time_ms: int
    px: Decimal
    sz: Decimal  # always positive
    is_buy: bool
    start_position: Decimal  # signed position before this fill
    closed_pnl: Decimal  # realized PnL reported by the venue, excluding fees
    fee: Decimal  # positive = paid, negative = rebate
    crossed: bool  # True = taker
    direction: str
    tid: int | None
    oid: int | None
    liquidation: bool

    @property
    def signed_sz(self) -> Decimal:
        return self.sz if self.is_buy else -self.sz

    @property
    def end_position(self) -> Decimal:
        return self.start_position + self.signed_sz

    @property
    def notional(self) -> Decimal:
        return self.px * self.sz

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> Fill:
        try:
            side = raw["side"]
            if side not in ("B", "A"):
                raise FillParseError(f"unknown side {side!r}")
            sz = _dec(raw["sz"], "sz")
            if sz <= 0:
                raise FillParseError(f"non-positive size {sz}")
            fee = _dec(raw.get("fee", "0"), "fee") + _dec(raw.get("builderFee", "0"), "builderFee")
            tid = raw.get("tid")
            oid = raw.get("oid")
            return cls(
                coin=str(raw["coin"]),
                time_ms=int(raw["time"]),
                px=_dec(raw["px"], "px"),
                sz=sz,
                is_buy=side == "B",
                start_position=_dec(raw["startPosition"], "startPosition"),
                closed_pnl=_dec(raw.get("closedPnl", "0"), "closedPnl"),
                fee=fee,
                crossed=bool(raw.get("crossed", True)),
                direction=str(raw.get("dir", "")),
                tid=int(tid) if tid is not None else None,
                oid=int(oid) if oid is not None else None,
                liquidation=bool(raw.get("liquidation")) or "liquidat" in str(raw.get("dir", "")).lower(),
            )
        except KeyError as exc:
            raise FillParseError(f"missing field {exc}") from exc


@dataclass(slots=True)
class EntryEvent:
    """One decision to open or increase a position: all opening fills of one order."""

    coin: str
    time_ms: int
    sign: int  # +1 long, -1 short
    notional: Decimal


@dataclass(slots=True)
class Realization:
    """One decision to reduce or close a position: all closing fills of one order."""

    coin: str
    time_ms: int
    position_sign: int  # sign of the position being reduced
    closed_notional: Decimal
    gross_pnl: Decimal
    fees: Decimal


@dataclass(slots=True)
class RoundTrip:
    """One position episode on one coin: from flat (or first observation) back to flat."""

    trader: str
    coin: str
    side: Side
    open_time_ms: int
    close_time_ms: int | None = None
    opened_from_flat: bool = True  # False = position existed before our data window
    clean: bool = True  # False = data gap detected inside the episode
    entry_qty: Decimal = Decimal(0)
    entry_cost: Decimal = Decimal(0)  # Σ px·qty of opening fills
    exit_qty: Decimal = Decimal(0)
    exit_cost: Decimal = Decimal(0)
    max_abs_position: Decimal = Decimal(0)
    gross_pnl: Decimal = Decimal(0)  # Σ closedPnl
    fees: Decimal = Decimal(0)
    funding: Decimal = Decimal(0)  # signed: positive = received
    n_fills: int = 0
    n_adds: int = 0
    adds_against: int = 0  # adds while the position was under water
    maker_notional: Decimal = Decimal(0)
    total_notional: Decimal = Decimal(0)
    liquidated: bool = False
    first_fill_px: Decimal = Decimal(0)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def closed(self) -> bool:
        return self.close_time_ms is not None

    @property
    def complete(self) -> bool:
        """Usable for statistics: fully observed from flat to flat without gaps."""
        return self.closed and self.opened_from_flat and self.clean

    @property
    def entry_vwap(self) -> Decimal:
        return self.entry_cost / self.entry_qty if self.entry_qty else Decimal(0)

    @property
    def exit_vwap(self) -> Decimal:
        return self.exit_cost / self.exit_qty if self.exit_qty else Decimal(0)

    @property
    def net_pnl(self) -> Decimal:
        return self.gross_pnl - self.fees + self.funding

    @property
    def peak_notional(self) -> Decimal:
        return self.max_abs_position * self.entry_vwap

    @property
    def return_bps(self) -> float:
        """Net PnL relative to peak notional, in basis points (size-independent)."""
        notional = self.peak_notional
        return float(self.net_pnl / notional * 10_000) if notional else 0.0

    @property
    def price_pnl(self) -> Decimal:
        """PnL recomputed from VWAPs; used to cross-check the venue's closedPnl."""
        return (self.exit_vwap - self.entry_vwap) * self.exit_qty * self.side.sign

    @property
    def holding_ms(self) -> int | None:
        return None if self.close_time_ms is None else self.close_time_ms - self.open_time_ms

    @property
    def maker_fraction(self) -> float:
        return float(self.maker_notional / self.total_notional) if self.total_notional else 0.0
