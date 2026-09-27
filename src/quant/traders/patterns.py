"""Find recurring entry conditions in traders' behaviour.

For every trader and every context feature we ask: *are the conditions at this
trader's entries different from the conditions at random moments?*

* Baseline = random timestamps inside the trader's own active span, on the same coin,
  paired with the **same side** as the real trade. Directional features are multiplied
  by the side sign, so a long-biased trader in a bull market does not look like a
  "momentum trader" just because of the drift — the baseline carries the same bias.
* Test: Mann–Whitney U (entries vs baseline), effect size = Cohen's d.
* Outcome link: Spearman correlation between the feature and the trade's net return —
  does the condition matter for *profitability*, or is it just a habit?
* Multiple testing: Benjamini–Hochberg across all (trader, feature) tests.
* A pattern counts only if several independent skilled traders show it with the same
  sign, and it is compared against its prevalence among the remaining traders.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from quant.stats.metrics import benjamini_hochberg
from quant.traders.context import DIRECTIONAL_FEATURES, MarketContext, features_at
from quant.traders.models import RoundTrip

MIN_OBS = 30


@dataclass(slots=True)
class FeatureTest:
    trader: str
    feature: str
    n_entries: int
    n_baseline: int
    effect: float  # Cohen's d, entries - baseline
    p_value: float
    outcome_corr: float | None  # Spearman(feature, return_bps) over the trader's trades
    outcome_p: float | None
    significant: bool = False

    @property
    def direction(self) -> int:
        return 1 if self.effect > 0 else -1


@dataclass(slots=True)
class TradeSample:
    """Per-trader values used for hypothesis drafting."""

    hold_min: list[float] = field(default_factory=list)
    return_bps: list[float] = field(default_factory=list)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    return 0.0 if pooled == 0 else (float(a.mean()) - float(b.mean())) / pooled


def evaluate_trader(
    trader: str,
    trips: Sequence[RoundTrip],
    ctx: MarketContext,
    rng: random.Random,
    baseline_per_trade: int = 3,
) -> list[FeatureTest]:
    usable = [t for t in trips if t.complete and t.coin in ctx.coins()]
    if len(usable) < MIN_OBS:
        return []
    lo = min(t.open_time_ms for t in usable)
    hi = max(t.open_time_ms for t in usable)
    actual: dict[str, list[tuple[float, float]]] = defaultdict(list)  # feature -> (value, return_bps)
    baseline: dict[str, list[float]] = defaultdict(list)
    for trip in usable:
        sign = trip.side.sign
        for name, value in features_at(ctx, trip.coin, trip.open_time_ms).items():
            if value is not None and math.isfinite(value):
                v = value * sign if name in DIRECTIONAL_FEATURES else value
                actual[name].append((v, trip.return_bps))
        for _ in range(baseline_per_trade):
            t_rand = rng.randint(lo, hi) if hi > lo else lo
            for name, value in features_at(ctx, trip.coin, t_rand).items():
                if value is not None and math.isfinite(value):
                    baseline[name].append(value * sign if name in DIRECTIONAL_FEATURES else value)
    out: list[FeatureTest] = []
    for name in sorted(actual):
        pairs = actual[name]
        base = np.asarray(baseline.get(name, []), dtype=np.float64)
        if len(pairs) < MIN_OBS or base.size < MIN_OBS:
            continue
        a = np.asarray([p[0] for p in pairs], dtype=np.float64)
        r = np.asarray([p[1] for p in pairs], dtype=np.float64)
        if np.ptp(a) == 0 and np.ptp(base) == 0:
            continue
        p_value = float(stats.mannwhitneyu(a, base, alternative="two-sided").pvalue)
        corr = corr_p = None
        if np.ptp(a) > 0 and np.ptp(r) > 0:
            res = stats.spearmanr(a, r)
            corr, corr_p = float(res.statistic), float(res.pvalue)
        out.append(FeatureTest(trader, name, a.size, base.size, _cohens_d(a, base), p_value, corr, corr_p))
    return out


def apply_fdr(tests: Sequence[FeatureTest], q: float) -> None:
    flags = benjamini_hochberg([t.p_value for t in tests], q)
    for t, f in zip(tests, flags, strict=True):
        t.significant = f


@dataclass(slots=True)
class PatternEvidence:
    feature: str
    direction: int
    supporting: list[str]
    n_tested_skilled: int
    prevalence_skilled: float
    prevalence_rest: float | None
    median_effect: float
    median_outcome_corr: float | None
    median_hold_min: float | None
    median_return_bps: float | None

    @property
    def lift(self) -> float | None:
        return None if self.prevalence_rest is None else self.prevalence_skilled - self.prevalence_rest


def aggregate(
    tests: Iterable[FeatureTest],
    skilled: set[str],
    samples: dict[str, TradeSample],
    min_support: int,
) -> list[PatternEvidence]:
    by_feature: dict[str, list[FeatureTest]] = defaultdict(list)
    for t in tests:
        by_feature[t.feature].append(t)
    out: list[PatternEvidence] = []
    for feature, ft in sorted(by_feature.items()):
        sk = [t for t in ft if t.trader in skilled]
        rest = [t for t in ft if t.trader not in skilled]
        for direction in (1, -1):
            sup = [t for t in sk if t.significant and t.direction == direction]
            if len(sup) < min_support:
                continue
            rest_sup = [t for t in rest if t.significant and t.direction == direction]
            corrs = [t.outcome_corr for t in sup if t.outcome_corr is not None]
            holds = [h for t in sup for h in samples.get(t.trader, TradeSample()).hold_min]
            rets = [x for t in sup for x in samples.get(t.trader, TradeSample()).return_bps]
            out.append(
                PatternEvidence(
                    feature=feature,
                    direction=direction,
                    supporting=sorted(t.trader for t in sup),
                    n_tested_skilled=len(sk),
                    prevalence_skilled=len(sup) / len(sk),
                    prevalence_rest=(len(rest_sup) / len(rest)) if rest else None,
                    median_effect=float(np.median([t.effect for t in sup])),
                    median_outcome_corr=float(np.median(corrs)) if corrs else None,
                    median_hold_min=float(np.median(holds)) if holds else None,
                    median_return_bps=float(np.median(rets)) if rets else None,
                )
            )
    out.sort(key=lambda e: (-(e.lift if e.lift is not None else e.prevalence_skilled), -len(e.supporting)))
    return out
