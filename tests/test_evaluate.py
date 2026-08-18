import datetime as dt

import numpy as np
import polars as pl
import pytest

from etfs.evaluate import (RandomPolicy, Split, aggregate, all_cash, compare,
                           equal_weight, evaluate_policy, make_env, make_splits,
                           simulate_weights, summarise)
from etfs.game import TradingGame


def dates(n=1700, start=dt.date(2022, 1, 1)):
    return [start + dt.timedelta(days=i) for i in range(n)]


def prices(n=1700, k=5, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, k)), axis=0))


# -- splits -----------------------------------------------------------------

def test_splits_are_chronological_and_contiguous():
    s = make_splits(dates(1700))
    assert s["train"].hi + 1 == s["validate"].lo
    assert s["validate"].hi + 1 == s["test"].lo
    assert s["train"].end < s["validate"].start < s["validate"].end < s["test"].start


def test_splits_respect_the_configured_boundaries():
    s = make_splits(dates(1700))
    assert s["train"].end < dt.date(2024, 1, 1)
    assert dt.date(2024, 1, 1) <= s["validate"].start
    assert s["validate"].end < dt.date(2026, 1, 1)
    assert s["test"].start >= dt.date(2026, 1, 1)


def test_splits_cover_every_bar_exactly_once():
    s = make_splits(dates(1700))
    assert s["train"].lo == 0
    assert s["test"].hi == 1699
    assert sum(x.n_bars for x in s.values()) == 1700


def test_empty_split_is_an_error_not_a_silent_pass():
    with pytest.raises(ValueError, match="empty"):
        make_splits(dates(100, start=dt.date(2018, 1, 1)))   # never reaches 2024


# -- window enforcement: the point of the whole exercise --------------------

def test_env_cannot_trade_outside_its_window():
    p, d = prices(), dates()
    s = make_splits(d)
    env = make_env(p, p, s["validate"], lookback=20)
    assert env.first_step >= s["validate"].lo
    assert env.last_step <= s["validate"].hi


def test_windows_do_not_overlap_between_splits():
    p, d = prices(), dates()
    s = make_splits(d)
    tr = make_env(p, p, s["train"], lookback=20)
    va = make_env(p, p, s["validate"], lookback=20)
    assert tr.last_step < va.first_step, "train and validate share bars"


def test_too_short_a_window_is_rejected():
    p, d = prices(), dates()
    with pytest.raises(ValueError, match="window too short"):
        # the window ends before the lookback has even warmed up
        make_env(p, p, Split("tiny", 10, 20, d[10], d[20]), lookback=50)


# -- benchmark accounting ---------------------------------------------------

def test_all_cash_earns_and_costs_nothing():
    p, d = prices(), dates()
    r = all_cash(p, make_splits(d)["validate"])
    assert r.total_return == pytest.approx(0.0)
    assert r.sharpe == 0.0 and r.avg_turnover == 0.0 and r.avg_exposure == 0.0


def test_equal_weight_is_fully_invested_and_pays_cost_once():
    p, d = prices(), dates()
    r = equal_weight(p, make_splits(d)["validate"])
    assert r.avg_exposure == pytest.approx(1.0, abs=1e-9)
    # bought once at the start, then held: turnover concentrated in day one
    assert 0 < r.avg_turnover < 0.01


def test_rebalancing_costs_more_than_holding():
    p, d = prices(), dates()
    s = make_splits(d)["validate"]
    hold = equal_weight(p, s, transaction_cost=0.001)
    rebal = equal_weight(p, s, transaction_cost=0.001, rebalance=True)
    assert rebal.avg_turnover > hold.avg_turnover
    assert rebal.total_return < hold.total_return


def test_single_asset_benchmark_matches_its_own_open_to_open_return():
    p, d = prices()[:, :1], dates()
    s = Split("w", 100, 200, d[100], d[200])
    r = simulate_weights(p, np.array([1.0]), s, transaction_cost=0.0)
    # trades run lo..hi-1 and each settles one bar later, so the series ends at hi
    expected = p[200, 0] / p[100, 0] - 1.0
    assert r.total_return == pytest.approx(expected, rel=1e-9)


def test_rejects_leveraged_or_negative_benchmarks():
    p, d = prices(), dates()
    s = make_splits(d)["validate"]
    with pytest.raises(ValueError, match="sum to <= 1"):
        simulate_weights(p, np.full(5, 0.5), s)
    with pytest.raises(ValueError, match="non-negative"):
        simulate_weights(p, np.array([-0.1, 0.2, 0.1, 0.1, 0.1]), s)


# -- policy evaluation ------------------------------------------------------

def test_evaluate_policy_covers_the_window_and_is_reproducible():
    p, d = prices(), dates()
    s = make_splits(d)
    env = make_env(p, p, s["validate"], lookback=20, max_actions=10)
    env.split_name = "validate"
    a = evaluate_policy(env, RandomPolicy(env.n_etfs, seed=1), name="r")
    b = evaluate_policy(env, RandomPolicy(env.n_etfs, seed=1), name="r")
    assert a.n_days > 0
    assert a.sharpe == pytest.approx(b.sharpe)
    assert a.split == "validate"


def test_a_stopping_policy_earns_the_cash_return():
    """A policy that only ever STOPs holds nothing and must match all-cash."""
    class AlwaysStop:
        def select_action(self, obs, deterministic=True):
            return 2, -1, 0.0, 0.0

    p, d = prices(), dates()
    s = make_splits(d)
    env = make_env(p, p, s["validate"], lookback=20)
    env.split_name = "validate"
    r = evaluate_policy(env, AlwaysStop(), name="stop")
    assert r.total_return == pytest.approx(0.0)
    assert r.avg_exposure == pytest.approx(0.0)


# -- reporting --------------------------------------------------------------

def test_summarise_metrics_are_consistent():
    r = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
    rep = summarise("x", "test", r, [0.1] * 5, [1.0] * 5, capital=1000.0)
    assert rep.n_days == 5
    assert rep.total_return == pytest.approx(np.prod(1 + r) - 1)
    assert rep.final_nav == pytest.approx(1000.0 * np.prod(1 + r))
    assert rep.avg_exposure == pytest.approx(1.0)
    assert rep.max_drawdown <= 0.0


def test_compare_sorts_by_risk_adjusted_return():
    reps = [summarise(n, "t", r, [0], [1]) for n, r in [
        ("bad", np.array([0.01, -0.02, 0.01, -0.02])),
        ("good", np.array([0.01, 0.008, 0.011, 0.009])),
    ]]
    assert compare(reps)["name"][0] == "good"


def test_aggregate_reports_spread_across_seeds():
    reps = [summarise(f"s{i}", "validate", np.array([0.01 * (i + 1), -0.005]),
                      [0.1], [1.0]) for i in range(5)]
    agg = aggregate(reps, name="ppo")
    assert agg["seeds"][0] == 5
    assert agg["sharpe_std"][0] > 0, "a single number hides the seed spread"
    assert set(agg.columns) >= {"sharpe_mean", "sharpe_std", "ann_return_mean"}


def test_aggregate_handles_a_single_seed():
    rep = summarise("s", "t", np.array([0.01, 0.02]), [0.1], [1.0])
    assert aggregate([rep])["sharpe_std"][0] == 0.0



def test_a_split_never_touches_the_next_splits_bars():
    """Executing at bar t settles at open[t+1]. If the window ran to `hi`, the
    final trade would price off the first bar of the following split."""
    p, d = prices(), dates()
    s = make_splits(d)
    env = make_env(p, p, s["train"], lookback=20)
    assert env.last_step + 1 <= s["train"].hi
    assert env.last_step + 1 < s["validate"].lo + 1
