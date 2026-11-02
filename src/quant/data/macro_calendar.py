"""Scheduled US macro events with a known release time.

Only events whose dates are fixed in advance are used, so filtering on them is causal:
at any moment the next event time was public knowledge.

FOMC: decision/statement at 14:00 America/New_York on the second day of each scheduled
meeting. Dates 2021–2022 from the Federal Reserve calendar; 2023–2025 verified against
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (Sep 2025); 2026 schedule verified Sep 2026.
CPI / NFP are not included yet: their historical schedule must come from a verified source
(BLS), which the research environment cannot reach — see docs/research/preregistration-003.md.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

NY = ZoneInfo("America/New_York")

FOMC_DECISION_DAYS: tuple[str, ...] = (
    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",
    "2022-01-26",
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",
    "2023-02-01",
    "2023-03-22",
    "2023-05-03",
    "2023-06-14",
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
)


def fomc_times_ms() -> npt.NDArray[np.int64]:
    out = []
    for day in FOMC_DECISION_DAYS:
        local = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(14, 0), tzinfo=NY)
        out.append(int(local.timestamp() * 1000))
    return np.array(sorted(out), dtype=np.int64)


def overlaps_event(
    bar_t: npt.NDArray[np.int64],
    interval_ms: int,
    hold_bars: int,
    events_ms: npt.NDArray[np.int64],
    before_ms: int,
    after_ms: int,
) -> npt.NDArray[np.bool_]:
    """True where a position opened at the next bar and held ``hold_bars`` would be open
    inside any window ``[event - before, event + after]``."""
    start = bar_t + interval_ms
    end = start + hold_bars * interval_ms
    # next event whose window ends after the position starts
    idx = np.searchsorted(events_ms + after_ms, start, side="right")
    has = idx < events_ms.size
    ev = np.where(has, events_ms[np.minimum(idx, events_ms.size - 1)], np.iinfo(np.int64).max // 2)
    return has & (ev - before_ms <= end)  # touching the window counts as overlap (conservative)
