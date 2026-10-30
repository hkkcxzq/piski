import datetime as dt

import numpy as np

from quant.data.macro_calendar import FOMC_DECISION_DAYS, fomc_times_ms, overlaps_event

HOUR = 3_600_000


def test_fomc_times_are_14_new_york_in_utc() -> None:
    t = fomc_times_ms()
    assert t.size == len(FOMC_DECISION_DAYS) == len(set(FOMC_DECISION_DAYS))
    first = dt.datetime.fromtimestamp(t[0] / 1000, tz=dt.UTC)
    assert (first.date(), first.hour) == (dt.date(2021, 1, 27), 19)  # EST: UTC-5
    summer = dt.datetime.fromtimestamp(t[FOMC_DECISION_DAYS.index("2024-07-31")] / 1000, tz=dt.UTC)
    assert summer.hour == 18  # EDT: UTC-4
    assert np.all(np.diff(t) > 0)


def test_overlap_window() -> None:
    ev = np.array([100 * HOUR], dtype=np.int64)
    bars = np.arange(0, 120, dtype=np.int64) * HOUR
    blocked = overlaps_event(bars, HOUR, 4, ev, 12 * HOUR, 2 * HOUR)
    # position [t+1h, t+5h] must not intersect [88h, 102h]
    assert not blocked[82]  # 83..87
    assert blocked[83]  # 84..88 touches the window start
    assert blocked[100]  # 101..105 starts inside the window
    assert not blocked[101]  # 102..106 starts at the window end
