import datetime as dt
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from quant.data.bars import Bars
from quant.data.binance_derivs import load_metrics, months, parse_funding, parse_metrics
from quant.strategies.positioning import (
    Positioning,
    align,
    flush_fade,
    fund_fade,
    grid004,
    oi_breakout,
    top_vs_crowd,
)

HOUR = 3_600_000


def _zip(csv: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", csv)
    return buf.getvalue()


METRICS_ROW = "2023-01-01 00:05:00,BTCUSDT,100.5,1650000.0,1.1,1.2,0.9,1.05"
HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
    "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio"
)


@pytest.mark.parametrize("header", [True, False])
def test_parse_metrics_with_and_without_header(header: bool) -> None:
    csv = (HEADER + "\n" if header else "") + METRICS_ROW + "\n" + METRICS_ROW.replace("00:05", "00:00") + "\n"
    m = parse_metrics(_zip(csv))
    expected = int(dt.datetime(2023, 1, 1, 0, 0, tzinfo=dt.UTC).timestamp() * 1000)
    assert m["t"].tolist() == [expected, expected + 300_000]  # milliseconds, sorted
    assert m["oi"][0] == 100.5 and m["top_pos_ls"][0] == 1.2 and m["acc_ls"][0] == 0.9


def test_parse_funding() -> None:
    f = parse_funding(
        _zip("calc_time,funding_interval_hours,last_funding_rate\n1700000028800,8,-0.0001\n1700000000000,8,0.0002\n")
    )
    assert f["t"].tolist() == [1700000000000, 1700000028800]
    assert f["rate"].tolist() == [0.0002, -0.0001]


def test_parse_rejects_bad_zip() -> None:
    with pytest.raises(ValueError, match="columns"):
        parse_metrics(_zip("a,b\n1,2\n"))


def test_months() -> None:
    assert months(dt.date(2023, 11, 15), dt.date(2024, 2, 1)) == [
        dt.date(2023, 11, 1),
        dt.date(2023, 12, 1),
        dt.date(2024, 1, 1),
        dt.date(2024, 2, 1),
    ]


def test_load_metrics_missing_is_empty(tmp_path: Path) -> None:
    assert load_metrics(tmp_path, "BTCUSDT", dt.date(2023, 1, 1), dt.date(2023, 1, 2)) == {}


def test_align_is_causal_and_marks_stale() -> None:
    src_t = np.array([0, 10, 20, 30], dtype=np.int64)
    src_v = np.array([1.0, 2.0, 3.0, 4.0])
    close = np.array([5, 15, 25, 100], dtype=np.int64)
    out = align(src_t, src_v, close, lag_ms=5, stale_ms=20)
    # cutoff = close - 5 → 0, 10, 20, 95; the value at the cutoff itself is allowed
    assert out[:3].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(out[3])  # last value (t=30) is 65 ms old > stale 20
    assert np.isnan(align(src_t, src_v, np.array([3], dtype=np.int64), 5, 100))[0]  # nothing published yet


def _bars(n: int, seed: int, interval: int = 4 * HOUR) -> Bars:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * 1.003
    l = np.minimum(o, c) * 0.997
    v = np.ones(n)
    t0 = int(dt.datetime(2024, 1, 1, tzinfo=dt.UTC).timestamp() * 1000)
    return Bars(
        interval, t0 + np.arange(n, dtype=np.int64) * interval, o, h, l, c, v, v / 2, v / 2, v, np.ones(n, bool)
    )


def _pos(n: int, seed: int) -> Positioning:
    rng = np.random.default_rng(seed)
    return Positioning(
        oi=1000 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
        top_ls=np.exp(rng.normal(0.2, 0.2, n)),
        acc_ls=np.exp(rng.normal(0.3, 0.2, n)),
        funding=rng.normal(1e-4, 1e-4, n),
    )


STRATS = [
    lambda b, p: fund_fade(b, p, 0.9, 24, 2.0, float("nan")),
    lambda b, p: oi_breakout(b, p, 12, 0.5, False, 24),
    lambda b, p: flush_fade(b, p, 4, 0.9, 8, 2.0),
    lambda b, p: top_vs_crowd(b, p, 1.0, 24, 2.0, 2.0),
]


@pytest.mark.parametrize("strat", STRATS)
def test_signals_do_not_depend_on_future(strat) -> None:  # type: ignore[no-untyped-def]
    n, cut = 1500, 1100
    b, p = _bars(n, 1), _pos(n, 2)
    full = strat(b, p)
    # scramble everything after `cut`
    b2, p2 = _bars(n, 1), _pos(n, 2)
    rng = np.random.default_rng(9)
    for arr in (b2.c, b2.o, b2.h, b2.l, p2.oi, p2.top_ls, p2.acc_ls, p2.funding):
        arr[cut:] = arr[cut:] * rng.uniform(0.5, 1.5, n - cut)
    part = strat(b2, p2)
    assert np.array_equal(full.side[:cut], part.side[:cut])
    assert np.allclose(full.stop_dist[:cut], part.stop_dist[:cut], equal_nan=True)
    assert full.side.any()  # the synthetic data does produce signals


def test_grid004_has_64_unique_variants(tmp_path: Path) -> None:
    vs = list(grid004(tmp_path))
    assert len(vs) == 64 and len({v.key for v in vs}) == 64
    assert all(v.symbol_aware for v in vs)
    assert {v.hypothesis for v in vs} == {
        "H-011-FUND-FADE",
        "H-012-OI-BREAKOUT",
        "H-013-FLUSH-FADE",
        "H-014-TOP-VS-CROWD",
    }
