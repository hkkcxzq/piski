"""End-to-end: fake Hyperliquid -> collect -> analyze -> report, fully offline."""

from pathlib import Path
from typing import Any

from quant.app.config import Settings, TraderResearchConfig
from quant.traders.pipeline import analyze, collect, save_analysis
from quant.traders.report import render_report
from quant.traders.store import TraderStore
from synth import HOUR, T0, funding_rows, ou_candles, portfolio_from_fills, rule_trader_fills

N_HOURS = 2400
NOW = T0 + N_HOURS * HOUR


class FakeHyperliquid:
    def __init__(self) -> None:
        self.candle_rows = ou_candles("BTC", N_HOURS, seed=11)
        self.fills: dict[str, list[dict[str, Any]]] = {}
        rows = []
        for i in range(9):
            addr = f"0x{i:040x}"
            rule = "mean_reversion" if i < 6 else "random"
            self.fills[addr] = rule_trader_fills(self.candle_rows, "BTC", rule, seed=100 + i, threshold=1.0 + 0.05 * i)
            rows.append(
                {
                    "ethAddress": addr,
                    "accountValue": "100000",
                    "displayName": None,
                    "windowPerformances": [
                        ["month", {"pnl": "1", "roi": "0", "vlm": "5000000"}],
                        ["allTime", {"pnl": str(1000 - i), "roi": "0", "vlm": "9000000"}],
                    ],
                }
            )
        # an ineligible account (too small) must be ignored
        rows.append({"ethAddress": "0xsmall", "accountValue": "10", "windowPerformances": []})
        self.rows = rows
        self.calls: dict[str, int] = {}

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def leaderboard(self) -> list[dict[str, Any]]:
        self._count("leaderboard")
        return self.rows

    def portfolio(self, user: str) -> list[Any]:
        self._count("portfolio")
        return portfolio_from_fills(self.fills[user])

    def user_fills(
        self, user: str, start_ms: int, end_ms: int | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        self._count("user_fills")
        return [f for f in self.fills[user] if f["time"] >= start_ms and (end_ms is None or f["time"] <= end_ms)]

    def user_funding(self, user: str, start_ms: int, end_ms: int | None = None) -> list[dict[str, Any]]:
        return []

    def funding_history(self, coin: str, start_ms: int, end_ms: int | None = None) -> list[dict[str, Any]]:
        return funding_rows(coin, N_HOURS, seed=4) if coin == "BTC" else []

    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        return self.candle_rows if (coin == "BTC" and interval == "1h") else []


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        trader_research=TraderResearchConfig(
            top_by_all_time=20,
            top_by_month=0,
            control_sample=0,
            min_trades=30,
            min_history_days=10,
            context_coins=("BTC",),
            min_supporting_traders=2,
        ),
    )


def test_collect_analyze_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    fake = FakeHyperliquid()
    store = TraderStore(tmp_path)

    summary = collect(settings, fake, store, NOW)
    assert summary.candidates == 9
    assert summary.collected == 9 and not summary.failed
    assert "0xsmall" not in store.load_registry()

    # a second run within the refresh window does not re-download anything
    again = collect(settings, fake, store, NOW + 60_000)
    assert again.skipped_fresh == 9 and again.collected == 0

    result = analyze(settings, store, NOW)
    assert len(result.metrics) == 9
    assert all(m.status.value == "ok" for m in result.metrics)
    assert len(result.skilled) == 3
    ids = {h.id for h in result.hypotheses}
    assert "H-HL-z_ret_4h-neg" in ids, ids

    report = render_report(result)
    out = save_analysis(result, store, report)
    assert (out / "report.md").exists() and (out / "hypotheses.json").exists()
    assert "Mean reversion" in report
    assert "Ограничения и смещения" in report


def test_analysis_is_deterministic(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = TraderStore(tmp_path)
    collect(settings, FakeHyperliquid(), store, NOW)
    a = analyze(settings, store, NOW)
    b = analyze(settings, store, NOW)
    assert [(t.trader, t.feature, t.p_value) for t in a.tests] == [(t.trader, t.feature, t.p_value) for t in b.tests]
    assert [h.to_dict() for h in a.hypotheses] == [h.to_dict() for h in b.hypotheses]
