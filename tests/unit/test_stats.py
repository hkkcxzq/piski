import math

import numpy as np
import pytest

from quant.stats.metrics import (
    annualization_factor,
    benjamini_hochberg,
    longest_losing_streak,
    max_drawdown_abs,
    max_drawdown_from_returns,
    percentile_ranks,
    profit_factor,
    sharpe,
    sortino,
    t_statistic,
)


def test_sharpe_matches_formula() -> None:
    r = [0.01, -0.005, 0.02, 0.0, 0.004]
    expected = np.mean(r) / np.std(r, ddof=1) * math.sqrt(365)
    assert sharpe(r, 365) == pytest.approx(expected)


def test_undefined_statistics_return_none() -> None:
    assert sharpe([0.01], 365) is None
    assert sharpe([0.01, 0.01], 365) is None
    assert sortino([0.01, 0.02], 365) is None
    assert profit_factor([1.0, 2.0]) is None
    assert t_statistic([1.0]) is None


def test_sortino_uses_downside_deviation() -> None:
    r = [0.02, -0.01, 0.03, -0.02]
    dd = math.sqrt((0.01**2 + 0.02**2) / 4)
    assert sortino(r, 1) == pytest.approx(np.mean(r) / dd)


def test_drawdowns() -> None:
    assert max_drawdown_from_returns([0.1, -0.5, 0.2]) == pytest.approx(0.5)
    assert max_drawdown_from_returns([-0.1]) == pytest.approx(0.1)
    assert max_drawdown_from_returns([]) == 0.0
    assert max_drawdown_abs([10, 5, 12, 2]) == pytest.approx(10)


def test_profit_factor_and_streak() -> None:
    assert profit_factor([3.0, -1.0, 1.0, -1.0]) == pytest.approx(2.0)
    assert longest_losing_streak([1, -1, -2, 3, -1, -1, -1, 2]) == 3


def test_annualization_from_median_spacing() -> None:
    day = 86_400_000
    assert annualization_factor([0, day, 2 * day, 10 * day]) == pytest.approx(365.25)


def test_benjamini_hochberg() -> None:
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert benjamini_hochberg(p, 0.05) == [True, True] + [False] * 8
    assert benjamini_hochberg([], 0.05) == []


def test_percentile_ranks_handle_ties_and_missing() -> None:
    assert percentile_ranks([3.0, None, 1.0, 3.0, 2.0]) == [
        pytest.approx(5 / 6),
        None,
        0.0,
        pytest.approx(5 / 6),
        pytest.approx(1 / 3),
    ]
    assert percentile_ranks([None]) == [None]
    assert percentile_ranks([7.0]) == [0.5]
