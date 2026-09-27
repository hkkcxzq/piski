import random

import pytest

from quant.traders.context import ALL_FEATURES, CandleSeries, FundingSeries, MarketContext, features_at
from quant.traders.hypotheses import build_hypotheses
from quant.traders.patterns import FeatureTest, TradeSample, aggregate, apply_fdr, evaluate_trader
from quant.traders.reconstruct import reconstruct
from synth import HOUR, T0, funding_rows, ou_candles, rule_trader_fills


@pytest.fixture(scope="module")
def candles() -> list[dict[str, object]]:
    return ou_candles("BTC", 2400, seed=7)


@pytest.fixture(scope="module")
def ctx(candles: list[dict[str, object]]) -> MarketContext:
    c = MarketContext()
    c.candles_1h["BTC"] = CandleSeries.from_raw(candles, HOUR)
    c.funding["BTC"] = FundingSeries.from_raw(funding_rows("BTC", 2400, seed=3))
    return c


def test_candle_is_usable_only_after_close(ctx: MarketContext) -> None:
    s = ctx.candles_1h["BTC"]
    assert s.last_closed_index(T0 + HOUR - 1) == -1
    assert s.last_closed_index(T0 + HOUR) == 0
    assert s.last_closed_index(T0 + 5 * HOUR + 30 * 60_000) == 4


def test_features_have_no_lookahead(ctx: MarketContext) -> None:
    """Future invariance: truncating everything after t must not change features at t."""
    rng = random.Random(1)
    full = ctx.candles_1h["BTC"]
    for _ in range(50):
        t = T0 + rng.randint(170, 2399) * HOUR + rng.randint(0, HOUR - 1)
        trunc = MarketContext(candles_1h={"BTC": full.truncated(t)}, funding=ctx.funding)
        assert features_at(ctx, "BTC", t) == features_at(trunc, "BTC", t)


def test_gap_in_candles_disables_features(candles: list[dict[str, object]]) -> None:
    holed = [c for i, c in enumerate(candles) if i != 300]
    ctx = MarketContext(candles_1h={"BTC": CandleSeries.from_raw(holed, HOUR)})
    t_inside = T0 + 350 * HOUR + 1
    assert features_at(ctx, "BTC", t_inside)["z_ret_24h"] is None
    t_after = T0 + 600 * HOUR + 1
    assert features_at(ctx, "BTC", t_after)["z_ret_24h"] is not None


def test_features_keys_and_ranges(ctx: MarketContext) -> None:
    f = features_at(ctx, "BTC", T0 + 1000 * HOUR + 5)
    assert set(f) == set(ALL_FEATURES)
    assert f["range_pos_24h"] is not None and -0.5 <= f["range_pos_24h"] <= 0.5
    assert f["vol_ratio_24h_7d"] is not None and f["vol_ratio_24h_7d"] > 0
    assert f["z_ret_5m"] is None  # no 5m data supplied
    assert f["funding"] is not None


def test_mean_reversion_trader_is_detected_and_random_trader_is_not(
    candles: list[dict[str, object]],
    ctx: MarketContext,
) -> None:
    mr = reconstruct("0xmr", rule_trader_fills(candles, "BTC", "mean_reversion", seed=1, threshold=1.0))
    rnd = reconstruct("0xrnd", rule_trader_fills(candles, "BTC", "random", seed=2))
    tests = evaluate_trader("0xmr", mr.complete_trips, ctx, random.Random(5))
    tests += evaluate_trader("0xrnd", rnd.complete_trips, ctx, random.Random(6))
    apply_fdr(tests, 0.05)
    by = {(t.trader, t.feature): t for t in tests}
    z4 = by[("0xmr", "z_ret_4h")]
    assert z4.significant and z4.direction == -1 and z4.effect < -1.0
    assert not any(t.significant for t in tests if t.trader == "0xrnd")


def _ft(trader: str, feature: str, effect: float, sig: bool = True) -> FeatureTest:
    return FeatureTest(trader, feature, 100, 300, effect, 1e-6 if sig else 0.5, -0.2, 0.01, significant=sig)


def test_aggregate_requires_support_and_reports_lift() -> None:
    tests = [
        _ft("a", "z_ret_4h", -1.2),
        _ft("b", "z_ret_4h", -0.9),
        _ft("c", "z_ret_4h", -1.1),
        _ft("d", "z_ret_4h", -1.0),
        _ft("e", "z_ret_4h", 0.3, sig=False),
        _ft("f", "z_ret_4h", 0.1, sig=False),
        _ft("a", "volume_ratio_4h", 0.5),
        _ft("b", "volume_ratio_4h", -0.5),
    ]
    samples = {k: TradeSample(hold_min=[180.0], return_bps=[12.0]) for k in "abcdef"}
    ev = aggregate(tests, skilled={"a", "b", "c"}, samples=samples, min_support=3)
    assert len(ev) == 1
    e = ev[0]
    assert e.feature == "z_ret_4h" and e.direction == -1
    assert e.supporting == ["a", "b", "c"]
    assert e.prevalence_skilled == 1.0
    assert e.prevalence_rest == pytest.approx(1 / 3)
    assert e.median_hold_min == 180.0
    (h,) = build_hypotheses(ev)
    assert h.id == "H-HL-z_ret_4h-neg"
    assert h.status == "RESEARCH"
    assert "Mean reversion" in h.title
    assert h.evidence["supporting_traders"] == 3
