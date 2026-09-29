"""Command-line entry point: ``quant <command>``."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import time
from collections.abc import Callable
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


def cmd_data_binance(settings: Settings, args: argparse.Namespace) -> int:
    from quant.data import binance_derivs  # noqa: PLC0415

    days = date_range(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))
    done = {"n": 0}

    def progress(r: DayResult) -> None:
        done["n"] += 1
        if r.status not in ("ok", "exists") or done["n"] % 100 == 0:
            print(f"[{done['n']}] {r.day} {r.status} {r.detail}", file=sys.stderr, flush=True)  # noqa: T201

    results = binance_derivs.download(settings.data_dir, args.symbol, days, workers=args.workers, progress=progress)
    log.info(
        "binance_derivs_done",
        symbol=args.symbol,
        fetched=sum(r.status == "ok" for r in results),
        missing=sum(r.status == "missing" for r in results),
        errors=sum(r.status == "error" for r in results),
    )
    return 1 if any(r.status == "error" for r in results) else 0


def cmd_research_exp001(settings: Settings, args: argparse.Namespace) -> int:
    from quant.research import experiment001  # heavy imports (numba) only when needed  # noqa: PLC0415
    from quant.traders.store import write_json  # noqa: PLC0415

    def progress(n: int, total: int, key: str) -> None:
        print(f"[{n}/{total}] {key}", file=sys.stderr, flush=True)  # noqa: T201

    number = args.experiment
    variants = None
    if number == "002":
        from quant.strategies.intraday import grid002  # noqa: PLC0415

        variants = list(grid002())
    elif number == "003":
        from quant.strategies.trend import grid003  # noqa: PLC0415

        variants = list(grid003())
    min_dev, min_val = experiment001.MIN_DEV_TRADES, experiment001.MIN_VAL_TRADES
    if number == "004":
        from quant.strategies.positioning import grid004  # noqa: PLC0415

        variants = list(grid004(settings.data_dir))
        min_dev, min_val = 60, 30  # preregistration-004: slow 4h signals are rare
    elif number == "005":
        from quant.strategies.swing import grid005  # noqa: PLC0415

        variants = list(grid005())
        min_dev, min_val = 60, 20  # preregistration-005: multi-day holds, few trades
    result = experiment001.run(
        settings.data_dir, progress=progress, variants=variants, min_dev_trades=min_dev, min_val_trades=min_val
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"experiment-{number}.json", result)
    (out / f"experiment-{number}.md").write_text(experiment001.render(result, number), encoding="utf-8")
    print(out / f"experiment-{number}.md")  # noqa: T201
    return 0


def _telegram_notifier(settings: Settings) -> Callable[[str], None] | None:
    import os  # noqa: PLC0415

    chat_id = os.environ.get("QUANT_TELEGRAM_CHAT_ID")
    if settings.telegram_bot_token is None or not chat_id:
        return None
    token = settings.telegram_bot_token.get_secret_value()

    def send(text: str) -> None:
        import httpx  # noqa: PLC0415

        try:
            httpx.post(
                f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=10
            )
        except httpx.HTTPError as exc:
            log.warning("telegram_failed", error=type(exc).__name__)

    return send


def cmd_news(settings: Settings, args: argparse.Namespace) -> int:
    from quant.news.classifier import Classifier, anthropic_transport, llm_classifier  # noqa: PLC0415
    from quant.news.keywords import keyword_classify  # noqa: PLC0415
    from quant.news.monitor import NewsMonitor  # noqa: PLC0415
    from quant.news.policy import load_state  # noqa: PLC0415

    if args.action == "status":
        state = load_state(settings.data_dir / "news" / "risk_state.json")
        print(state.level.name, *state.reasons, sep="\n")  # noqa: T201
        return 0
    use_claude = args.classifier == "claude" or (args.classifier == "auto" and os.environ.get("ANTHROPIC_API_KEY"))
    classifier: Classifier = (
        llm_classifier(anthropic_transport(args.model, args.effort)) if use_claude else keyword_classify
    )
    print(f"news classifier: {'claude' if use_claude else 'keywords (free)'}", file=sys.stderr)  # noqa: T201
    monitor = NewsMonitor(settings.data_dir, classifier, notify=_telegram_notifier(settings))
    if args.action == "once":
        res = monitor.run_once()
        print(f"fetched={res.fetched} new={res.new} classified={res.classified} level={res.state.level.name}")  # noqa: T201
        for src, err in res.feed_errors.items():
            print(f"feed error {src}: {err}", file=sys.stderr)  # noqa: T201
        return 0
    monitor.watch(args.interval)
    return 0


def cmd_live(settings: Settings, args: argparse.Namespace) -> int:
    from quant.app.config import TradingMode  # noqa: PLC0415
    from quant.exchanges.bybit.client import DEMO, MAINNET, BybitClient  # noqa: PLC0415
    from quant.live.engine import LiveEngine  # noqa: PLC0415
    from quant.live.state import Store  # noqa: PLC0415
    from quant.live.strategies import STRATEGIES  # noqa: PLC0415

    store = Store(settings.data_dir)
    if args.action == "status":
        st = store.load()
        print(f"halted: {st.halted or 'no'}")  # noqa: T201
        for sym, t in st.open_trades.items():
            print(f"{sym}: side={t.side} qty={t.qty} entry={t.entry} stop={t.stop} tp={t.take_profit}")  # noqa: T201
        return 0
    if args.action == "kill":
        store.kill_path.parent.mkdir(parents=True, exist_ok=True)
        store.kill_path.write_text("kill", encoding="utf-8")
        print("KILL file created: the running engine will flatten and halt on its next cycle")  # noqa: T201
        return 0
    if args.action == "reset":
        st = store.load()
        st.halted = ""
        store.kill_path.unlink(missing_ok=True)
        store.save(st)
        print("halt cleared")  # noqa: T201
        return 0
    if settings.mode is TradingMode.DEMO:
        base = DEMO
    elif settings.mode is TradingMode.LIVE:  # only reachable with live_enabled=true (config guard)
        base = MAINNET
    else:
        print("set QUANT_MODE=demo (live trading is locked)", file=sys.stderr)  # noqa: T201
        return 2
    if settings.bybit_api_key is None or settings.bybit_api_secret is None:
        print("QUANT_BYBIT_API_KEY / QUANT_BYBIT_API_SECRET are not set", file=sys.stderr)  # noqa: T201
        return 2
    trade = BybitClient(base, settings.bybit_api_key.get_secret_value(), settings.bybit_api_secret.get_secret_value())
    market = BybitClient(MAINNET)
    engine = LiveEngine(
        trade,
        market,
        STRATEGIES[args.strategy],
        settings.universe.trade,
        settings.risk,
        store,
        settings.data_dir / "news" / "risk_state.json",
        notify=_telegram_notifier(settings),
    )
    if args.action == "once":
        st = engine.run_cycle()
        print(f"halted={st.halted or 'no'} open={list(st.open_trades)}")  # noqa: T201
        return 0
    engine.run_forever(args.interval)
    return 0


def cmd_record(settings: Settings, args: argparse.Namespace) -> int:
    import asyncio  # noqa: PLC0415

    from quant.data.recorder import Recorder  # noqa: PLC0415

    asyncio.run(Recorder(settings.data_dir, settings.universe.record).run(args.interval))
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
    bn = dsub.add_parser("binance-derivs", help="download Binance futures OI / long-short / funding history")
    bn.add_argument("--symbol", default="BTCUSDT")
    bn.add_argument("--start", required=True, help="YYYY-MM-DD")
    bn.add_argument("--end", required=True, help="YYYY-MM-DD")
    bn.add_argument("--workers", type=int, default=8)
    bn.set_defaults(func=cmd_data_binance)
    research = sub.add_parser("research", help="pre-registered experiments")
    rsub = research.add_subparsers(dest="action", required=True)
    for number, help_ in (
        ("001", "first scalping hypotheses"),
        ("002", "intraday hypotheses (15m-4h)"),
        ("003", "intraday trend + FOMC blackout"),
        ("004", "positioning: funding, open interest, long/short"),
        ("005", "multi-day momentum (swing)"),
    ):
        e = rsub.add_parser(f"exp{number}", help=f"experiment {number}: {help_}")
        e.add_argument("--out", default="docs/research")
        e.set_defaults(func=cmd_research_exp001, experiment=number)
    news = sub.add_parser("news", help="news risk monitor (runs on the owner's computer)")
    nsub = news.add_subparsers(dest="action", required=True)
    for name, help_ in (("once", "one fetch/classify cycle"), ("watch", "run continuously"), ("status", "print risk")):
        n = nsub.add_parser(name, help=help_)
        n.add_argument("--interval", type=float, default=180.0, help="seconds between cycles (watch)")
        n.add_argument("--model", default="claude-opus-5")
        n.add_argument("--effort", default="low", choices=["low", "medium", "high"])
        n.add_argument(
            "--classifier",
            default="auto",
            choices=["auto", "claude", "keywords"],
            help="auto: Claude if ANTHROPIC_API_KEY is set, otherwise free keyword rules",
        )
        n.set_defaults(func=cmd_news)
    rec = sub.add_parser("record", help="record liquidations, open interest and funding (owner's computer)")
    rec.add_argument("--interval", type=float, default=60.0)
    rec.set_defaults(func=cmd_record)
    live = sub.add_parser("live", help="demo/live trading engine (runs on the owner's computer)")
    lsub = live.add_subparsers(dest="action", required=True)
    for name, help_ in (
        ("once", "one cycle"),
        ("run", "run continuously"),
        ("status", "open trades and halt state"),
        ("kill", "flatten everything and halt"),
        ("reset", "clear a halt / kill switch"),
    ):
        lp = lsub.add_parser(name, help=help_)
        lp.add_argument("--strategy", default="brk4h-fomc")
        lp.add_argument("--interval", type=float, default=60.0)
        lp.set_defaults(func=cmd_live)
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
