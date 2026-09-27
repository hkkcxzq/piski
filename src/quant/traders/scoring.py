"""Quality rating of a trading history. ROI is deliberately *not* an input.

Each component is a percentile rank inside the population of traders with enough data,
so the score is relative ("better than X % of the observed population"), robust to
outliers, and comparable across components. Penalties are multiplicative and explicit.
A trader without enough data is *unrated*, not rated low.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from quant.stats.metrics import percentile_ranks
from quant.traders.metrics import Status, TraderMetrics

# weight, getter (higher = better)
COMPONENTS: dict[str, tuple[float, Callable[[TraderMetrics], float | None]]] = {
    "significance": (0.25, lambda m: m.t_stat_bps),
    "risk_adjusted": (0.20, lambda m: m.acct_sortino),
    "drawdown": (0.15, lambda m: None if m.acct_max_dd is None else -m.acct_max_dd),
    "stability": (0.15, lambda m: m.positive_month_share if m.n_months >= 3 else None),
    "alpha": (0.15, lambda m: m.alpha_annual),
    "sample_size": (0.10, lambda m: float(m.n_trades)),
}

PENALTIES: dict[str, float] = {
    "martingale_like": 0.5,
    "concentrated_pnl": 0.6,
    "liquidated": 0.7,
    "high_leverage": 0.7,
    "pnl_crosscheck_failed": 0.5,
}


@dataclass(slots=True)
class QualityScore:
    address: str
    score: float | None
    components: dict[str, float | None] = field(default_factory=dict)
    penalty: float = 1.0
    penalty_reasons: list[str] = field(default_factory=list)
    rank: int | None = None


def score_population(metrics: Sequence[TraderMetrics]) -> list[QualityScore]:
    rated = [m for m in metrics if m.status is Status.OK]
    ranks: dict[str, list[float | None]] = {
        name: percentile_ranks([getter(m) for m in rated]) for name, (_, getter) in COMPONENTS.items()
    }
    scores: dict[str, QualityScore] = {}
    for i, m in enumerate(rated):
        comps = {name: ranks[name][i] for name in COMPONENTS}
        total_w = sum(COMPONENTS[n][0] for n, v in comps.items() if v is not None)
        # require the core evidence to be present
        if comps["significance"] is None or total_w < 0.5:
            base = None
        else:
            base = sum(COMPONENTS[n][0] * v for n, v in comps.items() if v is not None) / total_w
        reasons = [f for f in m.flags if f in PENALTIES]
        penalty = 1.0
        for f in reasons:
            penalty *= PENALTIES[f]
        scores[m.address] = QualityScore(
            address=m.address,
            score=None if base is None else base * penalty,
            components=comps,
            penalty=penalty,
            penalty_reasons=reasons,
        )
    ordered = sorted((s for s in scores.values() if s.score is not None), key=lambda s: -(s.score or 0.0))
    for r, s in enumerate(ordered, start=1):
        s.rank = r
    out = [scores.get(m.address) or QualityScore(address=m.address, score=None) for m in metrics]
    return out
