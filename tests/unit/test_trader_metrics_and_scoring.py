from typing import Any

import pytest

from quant.traders.metrics import (
    AsofSeries,
    Status,
    Style,
    TraderMetrics,
    compute_account_metrics,
    compute_trade_metrics,
    finalize,
    parse_portfolio,
)
from quant.traders.reconstruct import reconstruct
from quant.traders.scoring import score_population
from synth import HOUR, T0, FillFactory, portfolio_from_fills


def _round_trips(
    pnls: list[int], hold_ms: int = 10 * 60_000, start: int = T0, gap: int = 12 * HOUR, fee: str = "0"
) -> list[dict[str, Any]]:
    ff = FillFactory(fee_rate=fee)
    fills = []
    t = start
    for i, p in enumerate(pnls):
        side = 1 if i % 2 == 0 else -1
        fills.append(ff.fill("BTC", t, "100", str(side)))
        fills.append(ff.fill("BTC", t + hold_ms, str(100 + p * side), str(-side)))
        t += gap
    return fills


def _metrics(fills: list[dict[str, Any]], min_trades: int = 5, min_days: float = 1.0) -> TraderMetrics:
    rec = reconstruct("0xa", fills)
    m = TraderMetrics("0xa", "test")
    hist = parse_portfolio(portfolio_from_fills(fills))
    assert hist is not None
    compute_trade_metrics(m, rec, AsofSeries(hist.times, hist.account_value))
    compute_account_metrics(m, hist, None)
    finalize(m, min_trades, min_days)
    return m


def test_trade_metrics_values() -> None:
    m = _metrics(_round_trips([2, -1, 3, -1, 2, 2, -1, 1, 1, -2]))
    assert m.n_trades == 10
    assert m.win_rate == pytest.approx(0.6)
    assert m.profit_factor == pytest.approx(11 / 5)
    assert m.expectancy_usd == pytest.approx(0.6)
    assert m.avg_win == pytest.approx(11 / 6)
    assert m.avg_loss == pytest.approx(-5 / 4)
    assert m.longest_losing_streak == 1
    assert m.median_hold_min == pytest.approx(10)
    assert m.style is Style.SCALPER
    assert m.long_share == pytest.approx(0.5)
    assert m.status is Status.OK
    assert m.acct_total_pnl == pytest.approx(6.0)
    assert m.leverage_median == pytest.approx(100 / 100_000, rel=1e-3)


def test_insufficient_data_is_unrated_not_low_rated() -> None:
    m = _metrics(_round_trips([1, 1]), min_trades=5)
    assert m.status is Status.INSUFFICIENT_DATA
    (score,) = score_population([m])
    assert score.score is None and score.rank is None


def test_martingale_flag_and_penalty() -> None:
    ff = FillFactory()
    fills = []
    t = T0
    for _ in range(12):
        fills += [
            ff.fill("BTC", t, "100", "1"),
            ff.fill("BTC", t + 60_000, "95", "2"),
            ff.fill("BTC", t + 120_000, "101", "-3"),
        ]
        t += 6 * HOUR
    m = _metrics(fills)
    assert "martingale_like" in m.flags


def test_roi_is_not_a_scoring_input() -> None:
    steady = _metrics(_round_trips([1, 2, 1, 1, 2, 1, 1, 2, 1, -1] * 3))
    lucky = _metrics(_round_trips([-2] * 14 + [100] + [-2] * 15))  # one jackpot, otherwise losing
    lucky.address = "0xlucky"
    assert (lucky.acct_total_pnl or 0) > (steady.acct_total_pnl or 0)
    assert "concentrated_pnl" in lucky.flags
    s_steady, s_lucky = score_population([steady, lucky])
    assert s_steady.score is not None and s_lucky.score is not None
    assert s_steady.score > s_lucky.score
    assert s_steady.rank == 1


def test_parse_portfolio_prefers_perp_window_and_aligns_series() -> None:
    raw = [
        ["allTime", {"accountValueHistory": [[1, "10"], [2, "11"]], "pnlHistory": [[1, "0"], [2, "1"]]}],
        ["perpAllTime", {"accountValueHistory": [[1, "10"], [2, "12"], [3, "13"]], "pnlHistory": [[1, "0"], [3, "3"]]}],
    ]
    hist = parse_portfolio(raw)
    assert hist is not None
    assert hist.times == (1, 3)
    assert hist.account_value == (10.0, 13.0)
    assert parse_portfolio([]) is None


def test_account_returns_ignore_deposits() -> None:
    day = 86_400_000
    # account value jumps by a 1M deposit, PnL is flat -> no return, no drawdown
    raw = [
        [
            "perpAllTime",
            {
                "accountValueHistory": [[i * day, "1000" if i < 5 else "1001000"] for i in range(10)],
                "pnlHistory": [[i * day, "0"] for i in range(10)],
            },
        ]
    ]
    hist = parse_portfolio(raw)
    assert hist is not None
    m = TraderMetrics("0xa", "t")
    compute_account_metrics(m, hist, None)
    assert m.acct_return_total == pytest.approx(0.0)
    assert m.acct_max_dd == pytest.approx(0.0)


def test_beta_to_btc() -> None:
    day = 86_400_000
    btc = [100.0]
    for i in range(40):
        btc.append(btc[-1] * (1.01 if i % 3 else 0.985))
    av = 10_000.0
    pnl = [0.0]
    for i in range(1, 41):
        r = 2.0 * (btc[i] / btc[i - 1] - 1.0)
        pnl.append(pnl[-1] + r * (av + pnl[-1]))
    raw = [
        [
            "perpAllTime",
            {
                "accountValueHistory": [[i * day, str(av + p)] for i, p in enumerate(pnl)],
                "pnlHistory": [[i * day, str(p)] for i, p in enumerate(pnl)],
            },
        ]
    ]
    hist = parse_portfolio(raw)
    assert hist is not None
    m = TraderMetrics("0xa", "t")
    compute_account_metrics(m, hist, AsofSeries.from_pairs([(i * day, p) for i, p in enumerate(btc)]))
    assert m.beta_btc == pytest.approx(2.0, rel=1e-6)
    assert m.corr_btc == pytest.approx(1.0, rel=1e-6)
    assert m.alpha_annual == pytest.approx(0.0, abs=1e-6)


def test_deposit_and_loss_inside_one_period_is_not_below_minus_100_percent() -> None:
    day = 86_400_000
    # day 0: 10k; during day 1 the trader deposits 20k and loses 15k -> ends at 15k, pnl -15k
    raw = [
        [
            "perpAllTime",
            {
                "accountValueHistory": [[0, "10000"], [day, "15000"], [2 * day, "15000"]],
                "pnlHistory": [[0, "0"], [day, "-15000"], [2 * day, "-15000"]],
            },
        ]
    ]
    hist = parse_portfolio(raw)
    assert hist is not None
    m = TraderMetrics("0xa", "t")
    compute_account_metrics(m, hist, None)
    assert m.acct_max_dd == pytest.approx(0.5)  # lost 15k of 30k at risk
