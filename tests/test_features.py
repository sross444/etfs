import datetime as dt

import numpy as np
import polars as pl
import pytest

from etfs.features import apply_standardisation, build_features, standardise
from etfs.game import STOP, TradingGame


def panel(n=400, k=4, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(k)]
    dates = [dt.datetime(2022, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    rows = []
    for j, t in enumerate(tickers):
        p = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        for i in range(n):
            rows.append((dates[i], t, p[i], p[i] * 1.01, p[i] * 0.99, p[i],
                         1e6 + i, "x", "g"))
    return pl.DataFrame(
        rows,
        schema=["dt", "ticker", "open", "high", "low", "close", "volume",
                "desc", "group"],
        orient="row",
    )


# -- layout -----------------------------------------------------------------

def test_feature_tensor_is_shaped_and_aligned():
    df = panel()
    F, tickers, dates, names = build_features(df, windows=[3, 7])
    assert F.shape == (len(dates), len(tickers), len(names))
    assert tickers == sorted(tickers)
    assert np.isfinite(F).all(), "nan or inf would poison the first gradient step"


def test_ticker_axis_matches_the_price_matrices():
    """If these disagree, every feature is attached to the wrong ETF."""
    df = panel()
    F, tickers, _, _ = build_features(df, windows=[3])
    env = TradingGame.from_panel(df, lookback=20)
    assert tickers == env.tickers


def test_elliott_features_can_be_included():
    df = panel()
    a, _, _, names_a = build_features(df, windows=[3], elliott=False)
    b, _, _, names_b = build_features(df, windows=[3], elliott=True)
    assert len(names_b) == len(names_a) + 10
    assert b.shape[2] == a.shape[2] + 10


# -- standardisation --------------------------------------------------------

def test_statistics_come_only_from_the_slice_given():
    """Fitting on the whole panel would leak validate and test into training."""
    F = np.random.default_rng(0).normal(5.0, 2.0, (200, 3, 4)).astype(np.float32)
    train = slice(0, 100)
    _, mean, std = standardise(F, train)
    np.testing.assert_allclose(mean.ravel(), F[train].mean(axis=(0, 1)), rtol=1e-5)
    np.testing.assert_allclose(std.ravel(), F[train].std(axis=(0, 1)), rtol=1e-5)


def test_later_data_does_not_move_the_statistics():
    F = np.random.default_rng(1).normal(0, 1, (200, 3, 4)).astype(np.float32)
    tampered = F.copy()
    tampered[100:] *= 50.0
    _, m1, s1 = standardise(F, slice(0, 100))
    _, m2, s2 = standardise(tampered, slice(0, 100))
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(s1, s2)


def test_train_window_is_centred_and_clipped():
    F = np.random.default_rng(2).normal(3.0, 4.0, (300, 5, 6)).astype(np.float32)
    out, _, _ = standardise(F, slice(0, 200), clip=5.0)
    assert abs(float(out[:200].mean())) < 0.05
    assert out.min() >= -5.0 and out.max() <= 5.0


def test_constant_feature_does_not_divide_by_zero():
    F = np.ones((50, 2, 3), dtype=np.float32)
    out, _, _ = standardise(F, slice(0, 25))
    assert np.isfinite(out).all()


def test_apply_standardisation_reuses_fitted_statistics():
    F = np.random.default_rng(3).normal(0, 1, (100, 2, 3)).astype(np.float32)
    a, mean, std = standardise(F, slice(0, 50))
    b = apply_standardisation(F, mean, std)
    np.testing.assert_allclose(a, b)


def test_empty_slice_is_rejected():
    with pytest.raises(ValueError, match="no rows"):
        standardise(np.zeros((10, 2, 3), dtype=np.float32), slice(5, 5))


# -- the information line, again --------------------------------------------

def test_observation_reads_the_previous_bar_never_the_current_one():
    """The decision for day t happens before day t exists, so the newest row it
    may read is t-1."""
    df = panel()
    F, _, _, _ = build_features(df, windows=[3])
    env = TradingGame.from_panel(df, features=F, lookback=20, reward_mode="return")
    obs, _ = env.reset()
    expected = F[env.t - 1, 0, :]
    np.testing.assert_allclose(obs[:F.shape[2]], expected, rtol=1e-5)


def test_tampering_with_the_present_and_future_cannot_move_the_observation():
    df = panel()
    F, _, _, _ = build_features(df, windows=[3])
    env = TradingGame.from_panel(df, features=F, lookback=20, reward_mode="return")
    obs1, _ = env.reset()

    tampered = F.copy()
    tampered[env.t:] = 999.0
    env2 = TradingGame.from_panel(df, features=tampered, lookback=20,
                                  reward_mode="return")
    obs2, _ = env2.reset()
    np.testing.assert_allclose(obs1, obs2, rtol=1e-6)


def test_working_weight_is_appended_per_etf():
    df = panel()
    F, _, _, _ = build_features(df, windows=[3])
    env = TradingGame.from_panel(df, features=F, lookback=20, reward_mode="return")
    assert env.obs_dim == env.n_etfs * (F.shape[2] + 1) + 4
    before, _ = env.reset()
    after, _, _, _ = env.step(0, 1)          # BUY etf 1
    assert not np.allclose(before, after)


def test_mismatched_feature_shape_is_rejected():
    df = panel()
    with pytest.raises(ValueError, match=r"\[T, n_etfs, k\]"):
        TradingGame.from_panel(df, features=np.zeros((10, 2, 3)), lookback=20)
