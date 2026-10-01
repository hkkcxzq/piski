"""Deterministic mapping from news assessments and the macro calendar to a risk state.

The language model only *describes* news. What the trading engine is allowed to do is decided
here, by fixed and testable rules:

* the direction of news is never used to open positions (it is only logged for later,
  forward-only evaluation);
* the most severe active signal wins; every signal expires on its own;
* FLATTEN needs a critical, confident, new piece of information.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

from quant.data.macro_calendar import fomc_times_ms
from quant.news.classifier import Assessment, Severity

MINUTE_MS = 60_000


class RiskLevel(IntEnum):
    NORMAL = 0
    CAUTION = 1  # halve new position risk
    PAUSE_NEW = 2  # no new entries; manage existing positions
    FLATTEN = 3  # close everything and stay flat until expiry


# (level, minutes the signal stays active)
RULES: dict[Severity, tuple[RiskLevel, int]] = {
    Severity.NONE: (RiskLevel.NORMAL, 0),
    Severity.LOW: (RiskLevel.NORMAL, 0),
    Severity.MEDIUM: (RiskLevel.CAUTION, 60),
    Severity.HIGH: (RiskLevel.PAUSE_NEW, 180),
    Severity.CRITICAL: (RiskLevel.FLATTEN, 360),
}
MIN_CONFIDENCE = 0.6
FLATTEN_MIN_CONFIDENCE = 0.75
FOMC_BEFORE_MS = 12 * 60 * MINUTE_MS
FOMC_AFTER_MS = 2 * 60 * MINUTE_MS


@dataclass(slots=True)
class Signal:
    level: RiskLevel
    until_ms: int
    reason: str


@dataclass(slots=True)
class RiskState:
    level: RiskLevel = RiskLevel.NORMAL
    until_ms: int = 0
    reasons: list[str] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    updated_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.name
        d["signals"] = [{"level": s.level.name, "until_ms": s.until_ms, "reason": s.reason} for s in self.signals]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RiskState:
        sigs = [Signal(RiskLevel[s["level"]], int(s["until_ms"]), str(s["reason"])) for s in d.get("signals", [])]
        return cls(
            RiskLevel[d.get("level", "NORMAL")],
            int(d.get("until_ms", 0)),
            list(d.get("reasons", [])),
            sigs,
            int(d.get("updated_ms", 0)),
        )


def signal_from_assessment(a: Assessment, title: str, now_ms: int) -> Signal | None:
    level, minutes = RULES[a.severity]
    if level is RiskLevel.NORMAL or a.confidence < MIN_CONFIDENCE:
        return None
    if level is RiskLevel.FLATTEN and (a.confidence < FLATTEN_MIN_CONFIDENCE or not a.new_information):
        level, minutes = RULES[Severity.HIGH]  # downgrade: not certain enough to close everything
    if not a.new_information and level > RiskLevel.CAUTION:
        level, minutes = RULES[Severity.MEDIUM]
    return Signal(level, now_ms + minutes * MINUTE_MS, f"{a.severity.value}/{a.category}: {title[:120]}")


def calendar_signal(now_ms: int, events_ms: np.ndarray | None = None) -> Signal | None:
    events = fomc_times_ms() if events_ms is None else events_ms
    for ev in events:
        if ev - FOMC_BEFORE_MS <= now_ms <= ev + FOMC_AFTER_MS:
            return Signal(RiskLevel.PAUSE_NEW, int(ev + FOMC_AFTER_MS), "FOMC decision window")
    return None


def update_state(
    state: RiskState, new_signals: Iterable[Signal], now_ms: int, events_ms: np.ndarray | None = None
) -> RiskState:
    active = [s for s in state.signals if s.until_ms > now_ms]
    active.extend(new_signals)
    cal = calendar_signal(now_ms, events_ms)
    if cal is not None and not any(s.reason == cal.reason and s.until_ms == cal.until_ms for s in active):
        active.append(cal)
    if not active:
        return RiskState(updated_ms=now_ms)
    top = max(s.level for s in active)
    winners = [s for s in active if s.level == top]
    return RiskState(top, max(s.until_ms for s in winners), [s.reason for s in winners], active, now_ms)


def save_state(path: Path, state: RiskState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_state(path: Path) -> RiskState:
    if not path.exists():
        return RiskState()
    return RiskState.from_dict(json.loads(path.read_text(encoding="utf-8")))
