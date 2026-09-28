"""Command-line entry point: ``quant <command>``."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import time
from pathlib import Path

from quant.app.config import ConfigError, Settings, load_settings
from quant.app.logging import configure_logging, get_logger
from quant.core.ratelimit import WeightRateLimiter
from quant.data.bybit_archive import DayResult, date_range, download
from quant.exchanges.hyperliquid.client import HyperliquidError, HyperliquidInfoClient
from quant.traders.pipeline import analyze, collect, save_analysis
from quant.traders.report import render_report
from quant.traders.store import TraderStore

log = get_logger("quant.cli")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _client(settings: Settings) -> HyperliquidInfoClient:
    hl = settings.hyperliquid
    return HyperliquidInfoClient(
        hl.api_url,
        hl.stats_url,
        limiter=WeightRateLimiter(hl.weight_budget_per_minute),
        timeout=hl.timeout_seconds,
        max_retries=hl.max_retries,
    )


def cmd_traders_collect(settings: Settings, args: argparse.Namespace) -> int:
    store = TraderStore(settings.data_dir)

    def progress(n: int, total: int, addr: str) -> None:
        print(f"[{n}/{total}] {addr}", file=sys.stderr, flush=True)  # noqa: T201

    with _client(settings) as client:
        summary = collect(settings, client, store, _now_ms(), max_traders=args.max_traders, progress=progress)
    log.info(
        "collect_done",
        **{
            k: getattr(summary, k)
            for k in ("leaderboard_rows", "candidates", "collected", "skipped_fresh", "new_fills")
        },
        failed=len(summary.failed),
    )
    return 0 if not summary.failed or summary.collected else 1


def cmd_traders_analyze(settings: Settings, args: argparse.Namespace) -> int:
    store = TraderStore(settings.data_dir)
    result = analyze(settings, store, _now_ms())
    report = render_report(result)
    out = save_analysis(result, store, report)
    if args.publish:
        target = Path(args.publish)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "report.md", target / f"traders-{result.run_id}.md")
        shutil.copyfile(out / "hypotheses.json", target / f"hypotheses-{result.run_id}.json")
    log.info("analyze_done", output=str(out), traders=len(result.metrics), hypotheses=len(result.hypotheses))
    print(out / "report.md")  # noqa: T201
    return 0


def cmd_data_bybit(settings: Settings, args: argparse.Namespace) -> int:
    days = date_range(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))
    done = {"n": 0}

    def progress(r: DayResult) -> None:
        done["n"] += 1
        if r.status != "ok" or done["n"] % 25 == 0:
            print(f"[{done['n']}/{len(days)}] {r.day} {r.status} {r.detail}", file=sys.stderr, flush=True)  # noqa: T201

    results = download(settings.data_dir, args.symbol, days, workers=args.workers, progress=progress)
    bad = [r for r in results if r.status in ("error", "missing")]
    log.info(
        "bybit_archive_done",
        symbol=args.symbol,
        fetched=sum(r.status == "ok" for r in results),
        missing=sum(r.status == "missing" for r in results),
        errors=sum(r.status == "error" for r in results),
    )
    return 1 if any(r.status == "error" for r in bad) else 0


def cmd_research_exp001(settings: Settings, args: argparse.Namespace) -> int:
    from quant.research import experiment001  # heavy imports (numba) only when needed  # noqa: PLC0415
    from quant.traders.store import write_json  # noqa: PLC0415

    def progress(n: int, total: int, key: str) -> None:
        print(f"[{n}/{total}] {key}", file=sys.stderr, flush=True)  # noqa: T201

    result = experiment001.run(settings.data_dir, progress=progress)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "experiment-001.json", result)
    (out / "experiment-001.md").write_text(experiment001.render(result), encoding="utf-8")
    print(out / "experiment-001.md")  # noqa: T201
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant", description="Systematic trading research system")
    parser.add_argument("--config", action="append", type=Path, help="extra YAML config (repeatable)")
    sub = parser.add_subparsers(dest="command", required=True)
    traders = sub.add_parser("traders", help="public trader research (Hyperliquid)")
    tsub = traders.add_subparsers(dest="action", required=True)
    c = tsub.add_parser("collect", help="download leaderboard, fills, portfolios and market context")
    c.add_argument("--max-traders", type=int, default=None, help="limit for a quick trial run")
    c.set_defaults(func=cmd_traders_collect)
    a = tsub.add_parser("analyze", help="compute metrics, rating, patterns and hypotheses offline")
    a.add_argument("--publish", default=None, help="also copy report + hypotheses into this directory")
    a.set_defaults(func=cmd_traders_analyze)
    data = sub.add_parser("data", help="market data")
    dsub = data.add_subparsers(dest="action", required=True)
    b = dsub.add_parser("bybit-archive", help="download Bybit public trade archive as 1m bars")
    b.add_argument("--symbol", default="BTCUSDT")
    b.add_argument("--start", required=True, help="YYYY-MM-DD")
    b.add_argument("--end", required=True, help="YYYY-MM-DD")
    b.add_argument("--workers", type=int, default=4)
    b.set_defaults(func=cmd_data_bybit)
    research = sub.add_parser("research", help="pre-registered experiments")
    rsub = research.add_subparsers(dest="action", required=True)
    e1 = rsub.add_parser("exp001", help="experiment 001: first scalping hypotheses on Bybit")
    e1.add_argument("--out", default="docs/research")
    e1.set_defaults(func=cmd_research_exp001)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = None
        if args.config:
            paths = [Path("config/default.yaml"), *args.config]
        settings = load_settings(paths)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    configure_logging(settings.log_level, settings.log_json)
    try:
        return int(args.func(settings, args))
    except HyperliquidError as exc:
        log.error("hyperliquid_unavailable", error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
