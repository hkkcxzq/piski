"""Find recurring entry conditions in traders' decisions.

Unit of analysis: an **entry decision** — an order that opened or increased a position
(see ``reconstruct``). For every trader and every context feature we ask: *are the
conditions at this trader's entries different from the conditions at random moments?*

* **No pseudo-replication.** Traders often scale in with many orders within minutes;
  those share almost identical context. Entries on the same coin and side within a
  cooldown window are collapsed into one decision before testing.
* **Baseline** = random timestamps inside the trader's own active span, on the same
  coin, paired with the **same side** as the real decision. Directional features are
  multiplied by the side sign, so a long-biased trader in a bull market does not look
  like a "momentum trader" just because of the drift — the baseline carries the same bias.
* **Test:** Mann–Whitney U (entries vs baseline); effect size = Cohen's d.
* **Outcome link:** Spearman correlation between the feature and the *signed forward
  return* after the decision — does the condition matter for the result, or is it a habit?
  The forward return is a label only; features never see data after the decision.
* **Multiple testing:** Benjamini–Hochberg across all (trader, feature) tests.
* A pattern counts only if several independent skilled traders show it with the same
  sign; its prevalence among the remaining traders is reported as a control.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from quant.stats.metrics import benjamini_hochberg
from quant.traders.context import DIRECTIONAL_FEATURES, MarketContext, features_at, forward_return
from quant.traders.models import EntryEvent

MIN_OBS = 30
DEFAULT_COOLDOWN_MS = 60 * 60_000
DEFAULT_HORIZON_MS = 60 * 60_000
DEFAULT_MAX_EVENTS = 1500


@dataclass(slots=True)
class FeatureTest:
    trader: str
    feature: str
    n_entries: int
    n_baseline: int
    effect: float  # Cohen's d, entries - baseline
    p_value: float
    outcome_corr: float | None  # Spearman(feature, signed forward return)
    outcome_p: float | None
    significant: bool = False

    @property
    def direction(self) -> int:
        return 1 if self.effect > 0 else -1


@dataclass(slots=True)
class TradeSample:
    """Per-trader values used when drafting hypotheses."""

    hold_min: float | None = None
    return_bps: float | None = None  # mean net return per trade unit


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    return 0.0 if pooled == 0 else (float(a.mean()) - float(b.mean())) / pooled


def thin_events(events: Sequence[EntryEvent], cooldown_ms: int) -> list[EntryEvent]:
    """Collapse same-coin, same-side entries closer than ``cooldown_ms`` to the previous kept one."""
    last_kept: dict[tuple[str, int], int] = {}
    out: list[EntryEvent] = []
    for e in sorted(events, key=lambda x: x.time_ms):
        key = (e.coin, e.sign)
        prev = last_kept.get(key)
        if prev is not None and e.time_ms - prev < cooldown_ms:
            continue
        last_kept[key] = e.time_ms
        out.append(e)
    return out


def evaluate_trader(
    trader: str,
    events: Sequence[EntryEvent],
    ctx: MarketContext,
    rng: random.Random,
    *,
    baseline_per_event: int = 3,
    cooldown_ms: int = DEFAULT_COOLDOWN_MS,
    horizon_ms: int = DEFAULT_HORIZON_MS,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> list[FeatureTest]:
    coins = ctx.coins()
    usable = thin_events([e for e in events if e.coin in coins], cooldown_ms)
    if len(usable) > max_events:
        usable = sorted(rng.sample(usable, max_events), key=lambda e: e.time_ms)
    if len(usable) < MIN_OBS:
        return []
    lo = min(e.time_ms for e in usable)
    hi = max(e.time_ms for e in usable)
    actual: dict[str, list[tuple[float, float | None]]] = defaultdict(list)  # feature -> (value, outcome)
    baseline: dict[str, list[float]] = defaultdict(list)
    for ev in usable:
        sign = ev.sign
        fwd = forward_return(ctx, ev.coin, ev.time_ms, horizon_ms)
        outcome = None if fwd is None else fwd * sign * 10_000
        for name, value in features_at(ctx, ev.coin, ev.time_ms).items():
            if value is not None and math.isfinite(value):
                actual[name].append((value * sign if name in DIRECTIONAL_FEATURES else value, outcome))
        for _ in range(baseline_per_event):
            t_rand = rng.randint(lo, hi) if hi > lo else lo
            for name, value in features_at(ctx, ev.coin, t_rand).items():
                if value is not None and math.isfinite(value):
                    baseline[name].append(value * sign if name in DIRECTIONAL_FEATURES else value)
    out: list[FeatureTest] = []
    for name in sorted(actual):
        pairs = actual[name]
        base = np.asarray(baseline.get(name, []), dtype=np.float64)
        if len(pairs) < MIN_OBS or base.size < MIN_OBS:
            continue
        a = np.asarray([p[0] for p in pairs], dtype=np.float64)
        if np.ptp(a) == 0 and np.ptp(base) == 0:
            continue
        p_value = float(stats.mannwhitneyu(a, base, alternative="two-sided").pvalue)
        corr = corr_p = None
        labelled = [(v, o) for v, o in pairs if o is not None]
        if len(labelled) >= MIN_OBS:
            x = np.asarray([v for v, _ in labelled], dtype=np.float64)
            y = np.asarray([o for _, o in labelled], dtype=np.float64)
            if np.ptp(x) > 0 and np.ptp(y) > 0:
                res = stats.spearmanr(x, y)
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
            holds = [s.hold_min for t in sup if (s := samples.get(t.trader)) and s.hold_min is not None]
            rets = [s.return_bps for t in sup if (s := samples.get(t.trader)) and s.return_bps is not None]
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
