"""Choose which accounts to study.

Leaderboards are survivorship-biased by construction. To make that bias measurable
rather than invisible, we study two cohorts:

* ``top_all_time`` / ``top_month`` — what everyone looks at;
* ``control`` — a random sample of *active, sufficiently large* accounts that are not
  at the top. Patterns that are equally common in the control group are habits, not
  skill.

Once an address enters the registry it stays there forever (with ``first_seen``),
even if it disappears from the leaderboard: later analyses can then evaluate traders
forward-only, on data recorded after they were selected.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from quant.app.config import TraderResearchConfig


def eligible(row: dict[str, Any], cfg: TraderResearchConfig) -> bool:
    return bool(
        row.get("account_value", 0.0) >= cfg.min_account_value
        and row.get("allTime_vlm", 0.0) >= cfg.min_all_time_volume
        and row.get("month_vlm", 0.0) > 0.0
    )


def select_candidates(
    rows: Sequence[dict[str, Any]],
    cfg: TraderResearchConfig,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Return ``(address, cohort)`` pairs; cohorts are disjoint, order is deterministic."""
    pool = [r for r in rows if eligible(r, cfg)]
    chosen: dict[str, str] = {}
    by_all = sorted(pool, key=lambda r: (-r.get("allTime_pnl", 0.0), r["address"]))
    for r in by_all[: cfg.top_by_all_time]:
        chosen.setdefault(r["address"], "top_all_time")
    by_month = sorted(pool, key=lambda r: (-r.get("month_pnl", 0.0), r["address"]))
    for r in by_month[: cfg.top_by_month]:
        chosen.setdefault(r["address"], "top_month")
    rest = sorted(r["address"] for r in pool if r["address"] not in chosen)
    for addr in rng.sample(rest, k=min(cfg.control_sample, len(rest))):
        chosen[addr] = "control"
    return sorted(chosen.items(), key=lambda kv: (kv[1], kv[0]))
