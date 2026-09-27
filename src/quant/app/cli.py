"""Command-line entry point: ``quant <command>``."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from quant.app.config import ConfigError, Settings, load_settings
from quant.app.logging import configure_logging, get_logger
from quant.core.ratelimit import WeightRateLimiter
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
