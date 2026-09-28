"""Record Bybit data that cannot be downloaded later: liquidations, open interest, funding.

* liquidations: public WebSocket topic ``allLiquidation.<SYMBOL>`` (every liquidation order);
* open interest, funding rate, mark/index price: REST ``/v5/market/tickers`` polled every minute.

Files: ``data/recorded/<kind>/<SYMBOL>/<YYYY-MM-DD>.jsonl`` with both the exchange timestamp and
the local receive time. Gaps (computer asleep, network down) are recorded in ``gaps.jsonl``
instead of being papered over.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from quant.app.logging import get_logger
from quant.exchanges.bybit.client import BybitClient, BybitError

log = get_logger(__name__)

PUBLIC_WS = "wss://stream.bybit.com/v5/public/linear"
TICKER_FIELDS = (
    "markPrice",
    "indexPrice",
    "lastPrice",
    "fundingRate",
    "nextFundingTime",
    "openInterest",
    "openInterestValue",
    "bid1Price",
    "ask1Price",
    "volume24h",
)
GAP_MS = 3 * 60_000


class RecordStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "recorded"

    def append(self, kind: str, symbol: str, ts_ms: int, row: dict[str, Any]) -> None:
        day = dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.UTC).date().isoformat()
        path = self.root / kind / symbol / f"{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    def gap(self, kind: str, start_ms: int, end_ms: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "gaps.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": kind, "from": start_ms, "to": end_ms}) + "\n")


def parse_liquidations(msg: dict[str, Any], recv_ms: int) -> list[tuple[str, dict[str, Any]]]:
    """``allLiquidation`` message → [(symbol, row)]. Side is the side of the liquidation *order*."""
    topic = str(msg.get("topic", ""))
    if not topic.startswith("allLiquidation."):
        return []
    out = []
    for d in msg.get("data") or []:
        out.append((str(d["s"]), {"t": int(d["T"]), "recv": recv_ms, "side": d["S"], "qty": d["v"], "px": d["p"]}))
    return out


def poll_tickers(client: BybitClient, symbols: Iterable[str], store: RecordStore, now_ms: int) -> int:
    n = 0
    for sym in symbols:
        try:
            t = client.ticker(sym)
        except BybitError as exc:
            log.warning("ticker_failed", symbol=sym, error=str(exc))
            continue
        row = {"recv": now_ms, **{k: t.get(k) for k in TICKER_FIELDS}}
        store.append("tickers", sym, now_ms, row)
        n += 1
    return n


class Recorder:
    def __init__(self, root: Path, symbols: tuple[str, ...], client: BybitClient | None = None) -> None:
        from quant.exchanges.bybit.client import MAINNET  # noqa: PLC0415

        self.store = RecordStore(root)
        self.symbols = symbols
        self.client = client or BybitClient(MAINNET)
        self._last_liq_msg = 0
        self._last_poll = 0

    async def _liquidations(self) -> None:
        import websockets  # noqa: PLC0415

        args = [f"allLiquidation.{s}" for s in self.symbols]
        while True:
            try:
                async with websockets.connect(PUBLIC_WS, ping_interval=None) as ws:
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))
                    log.info("liquidations_connected", topics=args)
                    last_ping = time.monotonic()
                    while True:
                        if time.monotonic() - last_ping > 20:
                            await ws.send(json.dumps({"op": "ping"}))
                            last_ping = time.monotonic()
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        except TimeoutError:
                            continue
                        now = int(time.time() * 1000)
                        if self._last_liq_msg and now - self._last_liq_msg > GAP_MS:
                            self.store.gap("liquidations_ws", self._last_liq_msg, now)
                        self._last_liq_msg = now
                        for sym, row in parse_liquidations(json.loads(raw), now):
                            self.store.append("liquidations", sym, row["t"], row)
            except Exception as exc:  # reconnect on anything; the gap is recorded on the next message
                log.warning("liquidations_reconnect", error=repr(exc))
                await asyncio.sleep(5)

    async def _poll(self, interval_s: float) -> None:
        while True:
            now = int(time.time() * 1000)
            if self._last_poll and now - self._last_poll > GAP_MS:
                self.store.gap("tickers", self._last_poll, now)
            self._last_poll = now
            await asyncio.to_thread(poll_tickers, self.client, self.symbols, self.store, now)
            await asyncio.sleep(interval_s)

    async def run(self, interval_s: float = 60.0) -> None:
        await asyncio.gather(self._liquidations(), self._poll(interval_s))
