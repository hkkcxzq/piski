"""Persistent engine state (survives restarts) and the append-only trade journal."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OpenTrade:
    symbol: str
    strategy_id: str
    signal_id: str
    link_id: str
    side: int  # +1 long, -1 short
    qty: str
    entry: str
    stop: str
    take_profit: str | None
    leverage: str
    opened_ms: int
    max_hold_ms: int
    reason_for_entry: str
    adopted: bool = False  # found on the exchange without a record (e.g. after a crash)


@dataclass(slots=True)
class LiveState:
    open_trades: dict[str, OpenTrade] = field(default_factory=dict)
    last_bar_ms: dict[str, int] = field(default_factory=dict)
    day_key: str = ""
    week_key: str = ""
    day_start_equity: str = "0"
    week_start_equity: str = "0"
    peak_equity: str = "0"
    halted: str = ""  # non-empty = trading halted until manual reset
    seq: int = 0

    def next_id(self, prefix: str) -> str:
        self.seq += 1
        return f"{prefix}-{self.seq:06d}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["open_trades"] = {k: asdict(v) for k, v in self.open_trades.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LiveState:
        st = cls(**{k: v for k, v in d.items() if k != "open_trades"})
        st.open_trades = {k: OpenTrade(**v) for k, v in d.get("open_trades", {}).items()}
        return st


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Store:
    def __init__(self, root: Path) -> None:
        self.dir = root / "live"
        self.state_path = self.dir / "state.json"
        self.journal_path = self.dir / "trades.jsonl"
        self.events_path = self.dir / "events.jsonl"
        self.kill_path = self.dir / "KILL"

    def load(self) -> LiveState:
        if not self.state_path.exists():
            return LiveState()
        return LiveState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))

    def save(self, st: LiveState) -> None:
        _atomic_write(self.state_path, json.dumps(st.to_dict(), ensure_ascii=False, indent=2))

    def _append(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def journal(self, row: dict[str, Any]) -> None:
        self._append(self.journal_path, row)

    def event(self, row: dict[str, Any]) -> None:
        self._append(self.events_path, row)
