import json
from pathlib import Path

import httpx

from quant.data.recorder import RecordStore, parse_liquidations, poll_tickers
from quant.exchanges.bybit.client import BybitClient


def test_parse_liquidations() -> None:
    msg = {
        "topic": "allLiquidation.BTCUSDT",
        "type": "snapshot",
        "ts": 1,
        "data": [{"T": 1700000000000, "s": "BTCUSDT", "S": "Sell", "v": "0.5", "p": "43000.1"}],
    }
    ((sym, row),) = parse_liquidations(msg, 1700000000500)
    assert sym == "BTCUSDT" and row == {
        "t": 1700000000000,
        "recv": 1700000000500,
        "side": "Sell",
        "qty": "0.5",
        "px": "43000.1",
    }
    assert parse_liquidations({"op": "pong"}, 0) == []


def test_poll_tickers_writes_daily_files(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [{"symbol": "BTCUSDT", "fundingRate": "0.0001", "openInterest": "50000", "markPrice": "1"}]
                },
            },
        )

    client = BybitClient("https://x", http=httpx.Client(transport=httpx.MockTransport(handler)))
    store = RecordStore(tmp_path)
    assert poll_tickers(client, ["BTCUSDT"], store, 1700000000000) == 1
    (f,) = (tmp_path / "recorded" / "tickers" / "BTCUSDT").glob("*.jsonl")
    row = json.loads(f.read_text())
    assert row["openInterest"] == "50000" and row["recv"] == 1700000000000
