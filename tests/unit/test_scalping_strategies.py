import numpy as np
import pytest

from quant.data.bars import Bars
from quant.strategies.intraday import grid002
from quant.strategies.scalping import fvg, grid
from quant.strategies.trend import grid003

MIN = 60_000


def make_bars(c: np.ndarray, spread: float = 0.001, seed: int = 0) -> Bars:
    rng = np.random.default_rng(seed)
    n = c.size
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + spread * rng.random(n))
    low = np.minimum(o, c) * (1 - spread * rng.random(n))
    v = 1 + rng.random(n) * 10
    bv = v * rng.random(n)
    return Bars(MIN, np.arange(n, dtype=np.int64) * MIN, o, h, low, c, v, bv, v - bv, np.ones(n), np.ones(n, bool))


def test_fvg_detects_bullish_gap_and_places_limit_and_stop() -> None:
    n = 60
    c = np.full(n, 100.0)
    b = make_bars(c, spread=0.0)
    h, low = b.h.copy(), b.l.copy()
    # bars 40..42: h[40]=100.5, bar 41 impulsive, l[42]=102 -> gap 100.5..102
    h[40], low[40] = 100.5, 99.5
    h[41], low[41] = 102.5, 100.4
    h[42], low[42] = 103.0, 102.0
    b = Bars(MIN, b.t, b.o, h, low, b.c, b.v, b.bv, b.sv, b.n, b.valid)
    sig = fvg(b, min_gap=0.3, r=2.0, trend=False, active=False)
    assert sig.side[42] == 1
    assert sig.entry_px[42] == pytest.approx(102.0)
    assert sig.stop_dist[42] > 1.5  # below the far edge (100.5) minus a buffer
    assert sig.tp_dist[42] == pytest.approx(2.0 * sig.stop_dist[42])


@pytest.mark.parametrize("variant", [*grid(), *grid002(), *grid003()], ids=lambda v: v.key)
def test_signals_are_causal(variant) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(1)
    n = 6000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    b = make_bars(c)
    full = variant.build(b)
    cut = 4500
    part = variant.build(b.slice(0, cut))
    assert np.array_equal(full.side[:cut], part.side)
    assert np.allclose(full.stop_dist[:cut], part.stop_dist, equal_nan=True)
    assert np.allclose(full.entry_px[:cut], part.entry_px, equal_nan=True)
