"""Minimal, strict Bybit V5 REST client for USDT-perpetual (``category=linear``) one-way mode.

Signing (V5, HMAC-SHA256): ``sign = HMAC(secret, timestamp + api_key + recv_window + payload)``
where ``payload`` is the exact query string for GET and the exact JSON body for POST.

Only what the trading engine needs is implemented. Every response is checked: ``retCode != 0``
raises :class:`BybitError`. GET requests are retried on network errors / 5xx; order-creating
POSTs are **never** retried blindly — the engine reconciles with the exchange instead, so a
timeout can never create a duplicate order.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any
from urllib.parse import urlencode

import httpx

MAINNET = "https://api.bybit.com"
DEMO = "https://api-demo.bybit.com"  # demo trading account (virtual funds, real market prices)
CATEGORY = "linear"
LEVERAGE_NOT_MODIFIED = 110043


class BybitError(RuntimeError):
    def __init__(self, ret_code: int, msg: str, path: str) -> None:
        super().__init__(f"{path}: retCode={ret_code} {msg}")
        self.ret_code = ret_code


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    tick: Decimal
    qty_step: Decimal
    min_qty: Decimal
    max_leverage: Decimal

    def round_qty(self, qty: Decimal) -> Decimal:
        return (qty / self.qty_step).to_integral_value(rounding=ROUND_DOWN) * self.qty_step

    def round_price(self, px: Decimal) -> Decimal:
        return (px / self.tick).to_integral_value() * self.tick


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    size: Decimal  # signed: >0 long, <0 short, 0 flat
    avg_price: Decimal
    stop_loss: Decimal | None
    take_profit: Decimal | None


def sign(secret: str, timestamp: str, api_key: str, recv_window: str, payload: str) -> str:
    msg = f"{timestamp}{api_key}{recv_window}{payload}"
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def _dec_or_none(v: Any) -> Decimal | None:
    if v in (None, "", "0", "0.0", 0):
        return None
    return Decimal(str(v))


class BybitClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        api_secret: str = "",
        *,
        http: httpx.Client | None = None,
        recv_window: int = 5000,
        clock: Callable[[], int] | None = None,
        max_get_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base = base_url.rstrip("/")
        self._key = api_key
        self._secret = api_secret
        self._http = http or httpx.Client(timeout=15.0)
        self._recv = str(recv_window)
        self._clock = clock or (lambda: int(time.time() * 1000))
        self._retries = max_get_retries
        self._sleep = sleep

    # ------------------------------------------------------------------ transport
    def _headers(self, payload: str) -> dict[str, str]:
        ts = str(self._clock())
        return {
            "X-BAPI-API-KEY": self._key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": self._recv,
            "X-BAPI-SIGN": sign(self._secret, ts, self._key, self._recv, payload),
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _unwrap(resp: httpx.Response, path: str) -> dict[str, Any]:
        if resp.status_code >= 400:
            raise BybitError(-resp.status_code, resp.text[:200], path)
        data = resp.json()
        if data.get("retCode") != 0:
            raise BybitError(int(data.get("retCode", -1)), str(data.get("retMsg")), path)
        result = data.get("result") or {}
        if not isinstance(result, dict):
            raise BybitError(-1, "unexpected result type", path)
        return result

    def get(self, path: str, params: Mapping[str, Any], auth: bool = False) -> dict[str, Any]:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        attempt = 0
        while True:
            try:
                headers = self._headers(query) if auth else {}
                resp = self._http.get(f"{self.base}{path}?{query}", headers=headers)
                if resp.status_code < 500:
                    return self._unwrap(resp, path)
                err: Exception = BybitError(-resp.status_code, "server error", path)
            except httpx.TransportError as exc:
                err = exc
            attempt += 1
            if attempt > self._retries:
                raise err if isinstance(err, BybitError) else BybitError(-1, repr(err), path)
            self._sleep(0.5 * 2**attempt)

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.dumps({k: v for k, v in body.items() if v is not None}, separators=(",", ":"))
        try:
            resp = self._http.post(f"{self.base}{path}", content=payload, headers=self._headers(payload))
        except httpx.TransportError as exc:
            raise BybitError(-1, f"transport error (state unknown, reconcile): {exc!r}", path) from exc
        return self._unwrap(resp, path)

    # ------------------------------------------------------------------ market data
    def instrument(self, symbol: str) -> Instrument:
        r = self.get("/v5/market/instruments-info", {"category": CATEGORY, "symbol": symbol})
        item = (r.get("list") or [None])[0]
        if not item:
            raise BybitError(-1, f"unknown symbol {symbol}", "/v5/market/instruments-info")
        lot, price, lev = item["lotSizeFilter"], item["priceFilter"], item["leverageFilter"]
        return Instrument(
            symbol,
            Decimal(price["tickSize"]),
            Decimal(lot["qtyStep"]),
            Decimal(lot["minOrderQty"]),
            Decimal(lev["maxLeverage"]),
        )

    def klines(
        self, symbol: str, interval_min: int, limit: int = 1000
    ) -> list[tuple[int, float, float, float, float, float]]:
        """Ascending (start_ms, o, h, l, c, volume). The last row may be the still-open bar."""
        r = self.get(
            "/v5/market/kline", {"category": CATEGORY, "symbol": symbol, "interval": str(interval_min), "limit": limit}
        )
        rows = [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])) for x in r.get("list", [])]
        return sorted(rows)

    def ticker(self, symbol: str) -> dict[str, Any]:
        r = self.get("/v5/market/tickers", {"category": CATEGORY, "symbol": symbol})
        return dict((r.get("list") or [{}])[0])

    # ------------------------------------------------------------------ account
    def equity(self) -> Decimal:
        r = self.get("/v5/account/wallet-balance", {"accountType": "UNIFIED"}, auth=True)
        acct = (r.get("list") or [{}])[0]
        return Decimal(str(acct.get("totalEquity") or "0"))

    def position(self, symbol: str) -> Position:
        r = self.get("/v5/position/list", {"category": CATEGORY, "symbol": symbol}, auth=True)
        for p in r.get("list", []):
            size = Decimal(str(p.get("size") or "0"))
            if size == 0:
                continue
            signed = size if p.get("side") == "Buy" else -size
            return Position(
                symbol,
                signed,
                Decimal(str(p.get("avgPrice") or "0")),
                _dec_or_none(p.get("stopLoss")),
                _dec_or_none(p.get("takeProfit")),
            )
        return Position(symbol, Decimal(0), Decimal(0), None, None)

    def closed_pnl(self, symbol: str, start_ms: int) -> list[dict[str, Any]]:
        r = self.get(
            "/v5/position/closed-pnl", {"category": CATEGORY, "symbol": symbol, "startTime": start_ms}, auth=True
        )
        return list(r.get("list", []))

    # ------------------------------------------------------------------ trading
    def set_leverage(self, symbol: str, leverage: Decimal) -> None:
        lev = str(leverage.normalize())
        try:
            self.post(
                "/v5/position/set-leverage",
                {"category": CATEGORY, "symbol": symbol, "buyLeverage": lev, "sellLeverage": lev},
            )
        except BybitError as exc:
            if exc.ret_code != LEVERAGE_NOT_MODIFIED:
                raise

    def market_order(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
        link_id: str,
        *,
        reduce_only: bool = False,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
    ) -> str:
        if side not in ("Buy", "Sell"):
            raise ValueError(side)
        body: dict[str, Any] = {
            "category": CATEGORY,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "orderLinkId": link_id,
            "positionIdx": 0,
            "reduceOnly": reduce_only,
        }
        if stop_loss is not None:
            body.update(stopLoss=str(stop_loss), slTriggerBy="MarkPrice", tpslMode="Full")
        if take_profit is not None:
            body.update(takeProfit=str(take_profit), tpTriggerBy="MarkPrice", tpslMode="Full")
        r = self.post("/v5/order/create", body)
        return str(r.get("orderId", ""))

    def set_trading_stop(self, symbol: str, stop_loss: Decimal, take_profit: Decimal | None = None) -> None:
        body: dict[str, Any] = {
            "category": CATEGORY,
            "symbol": symbol,
            "positionIdx": 0,
            "tpslMode": "Full",
            "stopLoss": str(stop_loss),
            "slTriggerBy": "MarkPrice",
        }
        if take_profit is not None:
            body.update(takeProfit=str(take_profit), tpTriggerBy="MarkPrice")
        self.post("/v5/position/trading-stop", body)

    def close_position(self, symbol: str, link_id: str) -> bool:
        pos = self.position(symbol)
        if pos.size == 0:
            return False
        side = "Sell" if pos.size > 0 else "Buy"
        self.market_order(symbol, side, abs(pos.size), link_id, reduce_only=True)
        return True
