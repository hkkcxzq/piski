"""Trader research pipeline: collect raw data, then analyse it into metrics and hypotheses.

``collect`` needs network access to Hyperliquid; ``analyze`` works offline on stored
raw data and is fully deterministic for a given seed.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from quant.app.config import Settings
from quant.app.logging import get_logger
from quant.exchanges.hyperliquid.client import (
    CANDLES_PAGE_LIMIT,
    INTERVAL_MS,
    HyperliquidError,
    fill_key,
    funding_key,
    parse_leaderboard_row,
)
from quant.traders.context import CandleSeries, FundingSeries, MarketContext
from quant.traders.hypotheses import Hypothesis, build_hypotheses
from quant.traders.metrics import (
    AsofSeries,
    Status,
    TraderMetrics,
    compute_account_metrics,
    compute_trade_metrics,
    finalize,
    parse_portfolio,
)
from quant.traders.patterns import FeatureTest, PatternEvidence, TradeSample, aggregate, apply_fdr, evaluate_trader
from quant.traders.reconstruct import assign_funding, reconstruct
from quant.traders.scoring import QualityScore, score_population
from quant.traders.selection import select_candidates
from quant.traders.store import TraderStore, write_json

log = get_logger(__name__)

NOT_REPLICABLE = frozenset({"market_maker_like", "hft_like"})
BTC_DAILY_START_MS = 1_672_531_200_000  # 2023-01-01


class InfoSource(Protocol):
    def leaderboard(self) -> list[dict[str, Any]]: ...
    def portfolio(self, user: str) -> list[Any]: ...
    def user_fills(self, user: str, start_ms: int, end_ms: int | None = None) -> list[dict[str, Any]]: ...
    def user_funding(self, user: str, start_ms: int, end_ms: int | None = None) -> list[dict[str, Any]]: ...
    def funding_history(self, coin: str, start_ms: int, end_ms: int | None = None) -> list[dict[str, Any]]: ...
    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class CollectSummary:
    leaderboard_rows: int = 0
    candidates: int = 0
    collected: int = 0
    skipped_fresh: int = 0
    failed: list[str] = field(default_factory=list)
    new_fills: int = 0


def _time(r: dict[str, Any]) -> int:
    return int(r["time"])


def collect(
    settings: Settings,
    client: InfoSource,
    store: TraderStore,
    now_ms: int,
    *,
    max_traders: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> CollectSummary:
    cfg = settings.trader_research
    summary = CollectSummary()
    raw_rows = client.leaderboard()
    write_json(store.leaderboard_path(now_ms), raw_rows)
    rows = []
    for r in raw_rows:
        try:
            rows.append(parse_leaderboard_row(r))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    summary.leaderboard_rows = len(rows)
    candidates = select_candidates(rows, cfg, random.Random(cfg.random_seed))
    summary.candidates = len(candidates)

    registry = store.load_registry()
    for addr, cohort in candidates:
        entry = registry.setdefault(addr, {"first_seen_ms": now_ms, "cohort": cohort, "source": "hyperliquid"})
        entry["last_seen_on_leaderboard_ms"] = now_ms
    store.save_registry(registry)

    targets = sorted(registry)
    if max_traders is not None:
        targets = targets[:max_traders]
    refresh_ms = int(cfg.refresh_after_hours * 3_600_000)
    for n, addr in enumerate(targets, start=1):
        entry = registry[addr]
        if progress:
            progress(n, len(targets), addr)
        if now_ms - int(entry.get("last_collected_ms", 0)) < refresh_ms:
            summary.skipped_fresh += 1
            continue
        try:
            fills_path = store.fills_path(addr)
            known = store.load_list(fills_path)
            start = _time(known[-1]) if known else 0
            fills = client.user_fills(addr, start_ms=start)
            summary.new_fills += store.merge_records(fills_path, fills, fill_key, _time)
            all_fills = store.load_list(fills_path)
            if all_fills:
                fpath = store.funding_user_path(addr)
                known_f = store.load_list(fpath)
                f_start = _time(known_f[-1]) if known_f else _time(all_fills[0])
                store.merge_records(fpath, client.user_funding(addr, start_ms=f_start), funding_key, _time)
            write_json(store.portfolio_path(addr), client.portfolio(addr))
        except HyperliquidError as exc:
            log.warning("trader_collect_failed", address=addr, error=str(exc))
            entry["last_error"] = str(exc)
            summary.failed.append(addr)
            store.save_registry(registry)
            continue
        entry["last_collected_ms"] = now_ms
        entry.pop("last_error", None)
        summary.collected += 1
        store.save_registry(registry)  # persist after every trader: the run is resumable

    collect_market_context(settings, client, store, now_ms)
    return summary


def collect_market_context(settings: Settings, client: InfoSource, store: TraderStore, now_ms: int) -> None:
    coins = list(dict.fromkeys(("BTC", *settings.trader_research.context_coins)))
    for coin in coins:
        for interval in ("1h", "5m"):
            span = CANDLES_PAGE_LIMIT * INTERVAL_MS[interval]
            rows = client.candles(coin, interval, now_ms - span, now_ms)
            store.merge_records(store.candles_path(coin, interval), rows, lambda r: int(r["t"]), lambda r: int(r["t"]))
        start = now_ms - CANDLES_PAGE_LIMIT * INTERVAL_MS["1h"] - 8 * 86_400_000
        store.merge_records(
            store.funding_path(coin), client.funding_history(coin, start), lambda r: int(r["time"]), _time
        )
    rows = client.candles("BTC", "1d", BTC_DAILY_START_MS, now_ms)
    store.merge_records(store.candles_path("BTC", "1d"), rows, lambda r: int(r["t"]), lambda r: int(r["t"]))


# ---------------------------------------------------------------------- analysis
@dataclass(slots=True)
class AnalysisResult:
    run_id: str
    metrics: list[TraderMetrics]
    scores: list[QualityScore]
    tests: list[FeatureTest]
    evidence: list[PatternEvidence]
    hypotheses: list[Hypothesis]
    skilled: list[str]
    context_coins: list[str]
    output_dir: Path | None = None


def load_market_context(settings: Settings, store: TraderStore) -> tuple[MarketContext, AsofSeries | None]:
    ctx = MarketContext()
    for coin in settings.trader_research.context_coins:
        h1 = store.load_list(store.candles_path(coin, "1h"))
        if h1:
            ctx.candles_1h[coin] = CandleSeries.from_raw(h1, INTERVAL_MS["1h"])
        m5 = store.load_list(store.candles_path(coin, "5m"))
        if m5:
            ctx.candles_5m[coin] = CandleSeries.from_raw(m5, INTERVAL_MS["5m"])
        fr = store.load_list(store.funding_path(coin))
        if fr:
            ctx.funding[coin] = FundingSeries.from_raw(fr)
    closes: list[tuple[int, float]] = []
    for interval in ("1d", "1h"):
        for r in store.load_list(store.candles_path("BTC", interval)):
            closes.append((int(r["t"]) + INTERVAL_MS[interval], float(r["c"])))
    return ctx, (AsofSeries.from_pairs(closes) if closes else None)


def analyze(settings: Settings, store: TraderStore, now_ms: int) -> AnalysisResult:
    cfg = settings.trader_research
    registry = store.load_registry()
    ctx, btc = load_market_context(settings, store)
    metrics: list[TraderMetrics] = []
    trips_by: dict[str, list[Any]] = {}
    samples: dict[str, TradeSample] = {}
    for addr in sorted(registry):
        entry = registry[addr]
        rec = reconstruct(addr, store.load_list(store.fills_path(addr)))
        assign_funding(rec.trips + rec.open_trips, store.load_list(store.funding_user_path(addr)))
        m = TraderMetrics(address=addr, cohort=str(entry.get("cohort", "unknown")))
        hist = None
        if store.portfolio_path(addr).exists():
            hist = parse_portfolio(store.load_list(store.portfolio_path(addr)))
        account = AsofSeries(hist.times, hist.account_value) if hist else None
        compute_trade_metrics(m, rec, account)
        if hist is not None:
            compute_account_metrics(m, hist, btc)
        finalize(m, cfg.min_trades, cfg.min_history_days)
        metrics.append(m)
        complete = rec.complete_trips
        trips_by[addr] = complete
        samples[addr] = TradeSample(
            hold_min=[(t.holding_ms or 0) / 60_000 for t in complete],
            return_bps=[t.return_bps for t in complete],
        )

    scores = score_population(metrics)
    score_by = {s.address: s for s in scores}
    replicable = [m for m in metrics if m.status is Status.OK and not (set(m.flags) & NOT_REPLICABLE)]
    ranked = sorted(
        (m for m in replicable if score_by[m.address].score is not None),
        key=lambda m: -(score_by[m.address].score or 0.0),
    )
    n_skilled = max(len(ranked) // 3, min(len(ranked), 3))
    skilled = {m.address for m in ranked[:n_skilled]}

    rng = random.Random(cfg.random_seed)
    tests: list[FeatureTest] = []
    for m in replicable:
        tests.extend(evaluate_trader(m.address, trips_by[m.address], ctx, rng))
    apply_fdr(tests, cfg.fdr_q)
    evidence = aggregate(tests, skilled, samples, cfg.min_supporting_traders)
    hypotheses = build_hypotheses(evidence)
    run_id = datetime.fromtimestamp(now_ms / 1000, tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return AnalysisResult(run_id, metrics, scores, tests, evidence, hypotheses, sorted(skilled), sorted(ctx.coins()))


def save_analysis(result: AnalysisResult, store: TraderStore, report_md: str) -> Path:
    out = store.research / result.run_id
    write_json(out / "metrics.json", [m.to_dict() for m in result.metrics])
    write_json(
        out / "scores.json",
        [
            {
                "address": s.address,
                "score": s.score,
                "rank": s.rank,
                "components": s.components,
                "penalty": s.penalty,
                "penalty_reasons": s.penalty_reasons,
            }
            for s in result.scores
        ],
    )
    write_json(
        out / "feature_tests.json",
        [
            {
                "trader": t.trader,
                "feature": t.feature,
                "n_entries": t.n_entries,
                "n_baseline": t.n_baseline,
                "effect": t.effect,
                "p_value": t.p_value,
                "outcome_corr": t.outcome_corr,
                "outcome_p": t.outcome_p,
                "significant": t.significant,
            }
            for t in result.tests
        ],
    )
    write_json(out / "hypotheses.json", [h.to_dict() for h in result.hypotheses])
    (out / "report.md").write_text(report_md, encoding="utf-8")
    result.output_dir = out
    return out
