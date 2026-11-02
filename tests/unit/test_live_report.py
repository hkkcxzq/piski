import json
from pathlib import Path

import numpy as np

from quant.live.report import build, closed_trades
from quant.live.state import LiveState
from quant.research.montecarlo import block_bootstrap, path_stats


def _close(pnl: str, side: int = 1) -> dict[str, object]:
    return {
        "ts": 1_790_000_000_000,
        "type": "close",
        "strategy_id": "swing-mom",
        "symbol": "BTCUSDT",
        "side": side,
        "entry": "100",
        "stop": "90",
        "quantity": "2",
        "exit": "105",
        "pnl": pnl,
        "reason_for_exit": "time",
    }


def test_r_multiple_from_journal(tmp_path: Path) -> None:
    j = tmp_path / "trades.jsonl"
    rows = [{"type": "open"}, _close("10"), _close("-20"), {**_close("5"), "strategy_id": "other"}]
    j.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    trades = closed_trades(j, "swing-mom")
    assert [t["r"] for t in trades] == [0.5, -1.0]  # risk = |100-90| * 2 = 20 USDT


def test_report_without_trades_and_with_bands(tmp_path: Path) -> None:
    j = tmp_path / "trades.jsonl"
    assert "пока нет" in build(LiveState(), j, tmp_path / "mc.json", "swing-mom")
    j.write_text(json.dumps(_close("-20")) + "\n")
    mc = {
        "history": {"mean_r": 0.25},
        "horizons": {"1": {"losing_streak": {"p95": 3}}},
        "cum_r_bands": {"p5": [-1.0, -2.0], "p50": [0.1, 0.2], "p95": [3.0, 4.0]},
    }
    (tmp_path / "mc.json").write_text(json.dumps(mc))
    out = build(LiveState(), j, tmp_path / "mc.json", "swing-mom")
    assert "-1.00 R" in out and "слишком мало" in out  # one stop is no evidence either way
    j.write_text("".join(json.dumps(_close("10")) + "\n" for _ in range(12)))
    mc["cum_r_bands"] = {"p5": [-3.0] * 12, "p50": [1.0] * 12, "p95": [9.0] * 12}
    (tmp_path / "mc.json").write_text(json.dumps(mc))
    assert "В пределах" in build(LiveState(), j, tmp_path / "mc.json", "swing-mom")


def test_pnl_update_fills_missing_close(tmp_path: Path) -> None:
    j = tmp_path / "trades.jsonl"
    rows = [
        {**_close("10"), "signal_id": "s1", "pnl": None, "exit": None},
        {"type": "pnl_update", "signal_id": "s1", "pnl": "-6", "exit": "97"},
    ]
    j.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    [t] = closed_trades(j, "swing-mom")
    assert t["pnl_usdt"] == -6.0 and t["r"] == -0.3


def test_block_bootstrap_shapes_and_stats() -> None:
    r = np.array([1.0, -1.0, 2.0, -1.0, 0.5, -1.0])
    paths = block_bootstrap(r, 5, 100, 2, np.random.default_rng(0))
    assert paths.shape == (100, 5) and set(np.unique(paths)) <= set(r)
    st = path_stats(np.array([[-1.0, -1.0, 2.0, -1.0]]), 0.01)
    assert st["losing_streak"][0] == 2 and st["total_r"][0] == -1.0
    assert 0.019 < st["max_dd"][0] < 0.03
