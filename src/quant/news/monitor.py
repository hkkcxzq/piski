"""News monitor loop: fetch → de-duplicate → classify → update risk state → log.

Outputs (under ``data/news/``):
* ``risk_state.json`` — the current risk level the trading engine must respect;
* ``assessments.jsonl`` — every classified headline with timestamps (for forward evaluation:
  did the market actually move after "high" news?);
* ``seen.json`` — ids already processed.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from quant.app.logging import get_logger
from quant.news.classifier import ClassifierError, Transport, classify
from quant.news.feeds import DEFAULT_FEEDS, Headline, fetch_all
from quant.news.policy import RiskState, load_state, save_state, signal_from_assessment, update_state

log = get_logger(__name__)

MAX_AGE_MS = 6 * 3_600_000  # ignore items older than 6 h (feeds often repeat old stories)
BATCH = 25
SEEN_LIMIT = 20_000


@dataclass(slots=True)
class CycleResult:
    fetched: int = 0
    new: int = 0
    classified: int = 0
    feed_errors: dict[str, str] = field(default_factory=dict)
    classifier_error: str | None = None
    state: RiskState = field(default_factory=RiskState)


class NewsMonitor:
    def __init__(
        self,
        data_dir: Path,
        transport: Transport,
        http: httpx.Client | None = None,
        feeds: dict[str, str] | None = None,
        clock: Callable[[], int] | None = None,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self.dir = data_dir / "news"
        self.transport = transport
        self.http = http or httpx.Client(timeout=20.0, follow_redirects=True)
        self.feeds = feeds or DEFAULT_FEEDS
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.notify = notify
        self.state_path = self.dir / "risk_state.json"
        self.log_path = self.dir / "assessments.jsonl"
        self.seen_path = self.dir / "seen.json"

    def _load_seen(self) -> list[str]:
        if not self.seen_path.exists():
            return []
        return list(json.loads(self.seen_path.read_text(encoding="utf-8")))

    def _save_seen(self, seen: list[str]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.seen_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(seen[-SEEN_LIMIT:]), encoding="utf-8")
        tmp.replace(self.seen_path)

    def run_once(self) -> CycleResult:
        now = self.clock()
        res = CycleResult()
        headlines, res.feed_errors = fetch_all(self.feeds, self.http)
        res.fetched = len(headlines)
        seen = self._load_seen()
        seen_set = set(seen)
        fresh: list[Headline] = []
        for h in headlines:
            if h.id in seen_set:
                continue
            seen_set.add(h.id)
            seen.append(h.id)
            if h.published_ms is not None and now - h.published_ms > MAX_AGE_MS:
                continue  # old story: remember it, but do not spend a model call on it
            fresh.append(h)
        res.new = len(fresh)
        signals = []
        now_iso = dt.datetime.fromtimestamp(now / 1000, tz=dt.UTC).isoformat(timespec="seconds")
        by_id = {h.id: h for h in fresh}
        classified_ids: set[str] = set()
        for i in range(0, len(fresh), BATCH):
            chunk = fresh[i : i + BATCH]
            try:
                assessments = classify(chunk, self.transport, now_iso)
            except ClassifierError as exc:
                res.classifier_error = str(exc)
                log.warning("news_classifier_failed", error=str(exc))
                # unclassified items are not marked seen, so they are retried next cycle
                for h in chunk:
                    seen_set.discard(h.id)
                    seen.remove(h.id)
                continue
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                for a in assessments:
                    h = by_id[a.id]
                    classified_ids.add(a.id)
                    fh.write(
                        json.dumps({"ts": now, **h.to_dict(), **a.model_dump(mode="json")}, ensure_ascii=False) + "\n"
                    )
                    sig = signal_from_assessment(a, h.title, now)
                    if sig is not None:
                        signals.append(sig)
        res.classified = len(classified_ids)
        previous = load_state(self.state_path)
        res.state = update_state(previous, signals, now)
        save_state(self.state_path, res.state)
        self._save_seen(seen)
        if self.notify and res.state.level != previous.level:
            self.notify(f"Risk level: {previous.level.name} -> {res.state.level.name}. " + "; ".join(res.state.reasons))
        log.info(
            "news_cycle",
            fetched=res.fetched,
            new=res.new,
            classified=res.classified,
            level=res.state.level.name,
            feed_errors=len(res.feed_errors),
        )
        return res

    def watch(self, interval_s: float, stop: Callable[[], bool] = lambda: False) -> None:
        while not stop():
            try:
                self.run_once()
            except Exception as exc:  # the loop must survive anything; the state file keeps the last level
                log.error("news_cycle_crashed", error=repr(exc))
            time.sleep(interval_s)
