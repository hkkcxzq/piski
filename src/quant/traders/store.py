"""File storage for raw trader-research data.

Raw API payloads are stored as gzip JSON, exactly as received, so any analysis can
be re-run deterministically. Writes are atomic (temp file + rename). Fills are merged
by unique key, so repeated collection only ever adds data.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import Any


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, obj: Any, *, compress: bool | None = None) -> None:
    compress = path.suffix == ".gz" if compress is None else compress
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":") if compress else None,
        indent=None if compress else 2,
        sort_keys=not compress,
    ).encode("utf-8")
    _atomic_write_bytes(path, gzip.compress(payload, mtime=0) if compress else payload)


def read_json(path: Path) -> Any:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


class TraderStore:
    def __init__(self, data_dir: Path) -> None:
        self.raw = data_dir / "raw" / "hyperliquid"
        self.research = data_dir / "research" / "traders"

    # ---------------------------------------------------------------- paths
    def leaderboard_path(self, ts_ms: int) -> Path:
        return self.raw / "leaderboard" / f"{ts_ms}.json.gz"

    def fills_path(self, address: str) -> Path:
        return self.raw / "fills" / f"{address}.json.gz"

    def funding_user_path(self, address: str) -> Path:
        return self.raw / "user_funding" / f"{address}.json.gz"

    def portfolio_path(self, address: str) -> Path:
        return self.raw / "portfolio" / f"{address}.json.gz"

    def candles_path(self, coin: str, interval: str) -> Path:
        return self.raw / "candles" / f"{coin}_{interval}.json.gz"

    def funding_path(self, coin: str) -> Path:
        return self.raw / "funding" / f"{coin}.json.gz"

    @property
    def registry_path(self) -> Path:
        return self.research / "registry.json"

    # ---------------------------------------------------------------- registry
    def load_registry(self) -> dict[str, dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        data = read_json(self.registry_path)
        if not isinstance(data, dict):
            raise ValueError("corrupt trader registry")
        return data

    def save_registry(self, registry: dict[str, dict[str, Any]]) -> None:
        write_json(self.registry_path, registry)

    # ---------------------------------------------------------------- merges
    def merge_records(
        self,
        path: Path,
        new: Iterable[dict[str, Any]],
        key: Callable[[dict[str, Any]], Hashable],
        sort_key: Callable[[dict[str, Any]], Any],
    ) -> int:
        """Merge ``new`` into the list stored at ``path``; returns the number of added records."""
        existing: list[dict[str, Any]] = read_json(path) if path.exists() else []
        index = {key(r): r for r in existing}
        added = 0
        for r in new:
            k = key(r)
            if k not in index:
                index[k] = r
                added += 1
        if added or not path.exists():
            write_json(path, sorted(index.values(), key=sort_key))
        return added

    def load_list(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a list")
        return data
