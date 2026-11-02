"""A stateful in-memory imitation of the Bybit V5 endpoints the engine uses."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from quant.exchanges.bybit.client import sign


class FakeBybit:
    def __init__(self, klines: list[list[str]], equity: str = "10000", key: str = "k", secret: str = "s") -> None:  # noqa: S107 - test double
        self.klines = klines  # Bybit order: newest first, [start, o, h, l, c, v, turnover]
        self.equity = Decimal(equity)
        self.key, self.secret = key, secret
        self.positions: dict[str, dict[str, Any]] = {}
        self.orders: list[dict[str, Any]] = []
        self.leverage: dict[str, str] = {}
        self.closed: list[dict[str, Any]] = []
        self.drop_stops = False  # simulate an exchange that ignores SL on order create
        self.reject_trading_stop = False
        self.fail_orders = False
        self.ticker_price: Decimal | None = None  # live price differing from the last bar close

    def last_price(self) -> Decimal:
        return Decimal(self.klines[0][4])

    def _ok(self, result: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json={"retCode": 0, "retMsg": "OK", "result": result, "time": 0})

    def _err(self, code: int, msg: str) -> httpx.Response:
        return httpx.Response(200, json={"retCode": code, "retMsg": msg, "result": {}, "time": 0})

    def _check_sig(self, req: httpx.Request, payload: str) -> None:
        h = req.headers
        expected = sign(self.secret, h["X-BAPI-TIMESTAMP"], self.key, h["X-BAPI-RECV-WINDOW"], payload)
        assert h["X-BAPI-SIGN"] == expected, "bad signature"

    def handler(self, req: httpx.Request) -> httpx.Response:
        url = urlparse(str(req.url))
        path = url.path
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if req.method == "GET":
            if "X-BAPI-SIGN" in req.headers:
                self._check_sig(req, url.query)
            if path == "/v5/market/instruments-info":
                return self._ok(
                    {
                        "list": [
                            {
                                "symbol": q["symbol"],
                                "priceFilter": {"tickSize": "0.1"},
                                "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001"},
                                "leverageFilter": {"maxLeverage": "100"},
                            }
                        ]
                    }
                )
            if path == "/v5/market/tickers":
                return self._ok(
                    {"list": [{"symbol": q["symbol"], "lastPrice": str(self.ticker_price or self.last_price())}]}
                )
            if path == "/v5/market/kline":
                return self._ok({"list": self.klines[: int(q.get("limit", 200))]})
            if path == "/v5/account/wallet-balance":
                return self._ok({"list": [{"totalEquity": str(self.equity)}]})
            if path == "/v5/position/list":
                p = self.positions.get(q["symbol"])
                return self._ok({"list": [p] if p else [{"symbol": q["symbol"], "size": "0", "side": ""}]})
            if path == "/v5/position/closed-pnl":
                return self._ok({"list": [c for c in self.closed if c["symbol"] == q["symbol"]]})
            return httpx.Response(404)
        body_text = req.content.decode()
        self._check_sig(req, body_text)
        body = json.loads(body_text)
        if path == "/v5/position/set-leverage":
            if self.leverage.get(body["symbol"]) == body["buyLeverage"]:
                return self._err(110043, "leverage not modified")
            self.leverage[body["symbol"]] = body["buyLeverage"]
            return self._ok({})
        if path == "/v5/order/create":
            if self.fail_orders:
                return self._err(10001, "rejected")
            self.orders.append(body)
            self._fill(body)
            return self._ok({"orderId": f"o{len(self.orders)}", "orderLinkId": body["orderLinkId"]})
        if path == "/v5/position/trading-stop":
            if self.reject_trading_stop:
                return self._err(10001, "trading stop rejected")
            p = self.positions.get(body["symbol"])
            if p:
                p["stopLoss"] = body["stopLoss"]
                p["takeProfit"] = body.get("takeProfit", p.get("takeProfit", ""))
            return self._ok({})
        return httpx.Response(404)

    def _fill(self, o: dict[str, Any]) -> None:
        sym = o["symbol"]
        qty = Decimal(o["qty"])
        signed = qty if o["side"] == "Buy" else -qty
        cur = self.positions.get(sym)
        cur_size = Decimal(0) if not cur else (Decimal(cur["size"]) * (1 if cur["side"] == "Buy" else -1))
        new = cur_size + signed
        if new == 0:
            self.positions.pop(sym, None)
            self.closed.append(
                {"symbol": sym, "closedPnl": "12.5", "avgExitPrice": str(self.last_price()), "createdTime": str(10**15)}
            )
            return
        self.positions[sym] = {
            "symbol": sym,
            "size": str(abs(new)),
            "side": "Buy" if new > 0 else "Sell",
            "avgPrice": str(self.last_price()),
            "stopLoss": "" if self.drop_stops else o.get("stopLoss", ""),
            "takeProfit": o.get("takeProfit", ""),
        }

    def stop_out(self, sym: str) -> None:
        """The exchange stop triggers while the engine is not looking."""
        self.positions.pop(sym, None)
        self.closed.append({"symbol": sym, "closedPnl": "-25", "avgExitPrice": "1", "createdTime": str(10**15)})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
