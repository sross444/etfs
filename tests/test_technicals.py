import datetime as dt

import numpy as np
import polars as pl
import pytest

from etfs.technicals import (
    _rolling_weighted_quantiles,
    _weights,
    efficiency_ratio,
    metric_columns,
    metrics,
    rsi,
    stochastic_oscillator,
    trend_strength,
    volume_quantile_ratios,
)


def bars(closes, ticker="A", highs=None, lows=None, volume=None, start=1):
    n = len(closes)
    highs = [c + 1 for c in closes] if highs is None else highs
    lows = [c - 1 for c in closes] if lows is None else lows
    volume = [1000.0] * n if volume is None else volume
    return pl.DataFrame({
        "ticker": [ticker] * n,
        "dt": [dt.datetime(2024, 1, 1) + dt.timedelta(days=i + start) for i in range(n)],
        "open": [float(c) for c in closes],
        "high": [float(h) for h in highs],
        "low": [float(lo) for lo in lows],
        "close": [float(c) for c in closes],
        "volume": [float(v) for v in volume],
    })


# --------------------------------------------------------------------------
# weights -- the decay must favour the most recent bar
# --------------------------------------------------------------------------

def test_weights_put_the_most_weight_on_the_newest_bar():
    w = _weights(4, 0.5)
    assert w[-1] > w[0]                      # newest > oldest
    assert np.isclose(w.sum(), 1.0)
    assert np.all(np.diff(w) > 0)            # monotonically increasing


def test_weights_are_flat_when_decay_is_one():
    assert np.allclose(_weights(5, 1.0), 0.2)


# --------------------------------------------------------------------------
# per-ticker isolation -- the bug that made the original panel-unsafe
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [trend_strength, efficiency_ratio, rsi])
def test_indicators_do_not_bleed_across_tickers(fn):
    a = bars([10, 11, 12, 13, 14, 15, 16, 17], ticker="A")
    b = bars([50, 40, 30, 20, 10, 5, 4, 3], ticker="B")
    name = fn(a, window=3).columns[-1]

    alone = fn(a, window=3)[name].to_list()
    together = (
        fn(pl.concat([a, b]), window=3).filter(pl.col("ticker") == "A")[name].to_list()
    )
    assert alone == together


def test_stochastic_does_not_bleed_across_tickers():
    a = bars([10, 11, 12, 13, 14], ticker="A")
    b = bars([99, 98, 97, 96, 95], ticker="B")
    alone = stochastic_oscillator(a, 3)["so_3"].to_list()
    together = (
        stochastic_oscillator(pl.concat([a, b]), 3)
        .filter(pl.col("ticker") == "A")["so_3"].to_list()
    )
    assert alone == together


def test_volume_quantiles_do_not_bleed_across_tickers():
    a = bars([10, 11, 12, 13, 14], ticker="A")
    b = bars([99, 98, 97, 96, 95], ticker="B")
    col = "rvwq_3_0.5"
    alone = volume_quantile_ratios(a, 3)[col].to_numpy()
    together = (
        volume_quantile_ratios(pl.concat([a, b]), 3)
        .filter(pl.col("ticker") == "A")[col].to_numpy()
    )
    np.testing.assert_allclose(alone, together, equal_nan=True)


# --------------------------------------------------------------------------
# indicator behaviour at the extremes
# --------------------------------------------------------------------------

def test_trend_strength_is_positive_in_a_clean_uptrend():
    df = trend_strength(bars(list(range(10, 30))), window=5)
    assert df["ts_5_0.9"].drop_nulls().tail(5).min() > 0.9


def test_trend_strength_is_negative_in_a_clean_downtrend():
    df = trend_strength(bars(list(range(30, 10, -1))), window=5)
    assert df["ts_5_0.9"].drop_nulls().tail(5).max() < -0.9


def test_trend_strength_stays_in_range_on_noise():
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 300))
    out = trend_strength(bars(closes), window=14)["ts_14_0.9"].drop_nulls()
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_efficiency_ratio_is_one_for_a_straight_line():
    out = efficiency_ratio(bars(list(range(10, 40))), window=5)["er_5_0.9"]
    assert np.isclose(out.drop_nulls().tail(5).min(), 1.0)


def test_efficiency_ratio_is_near_zero_for_perfect_chop():
    out = efficiency_ratio(bars([10, 11] * 15), window=4)["er_4_0.9"]
    assert abs(out.drop_nulls().tail(5)).max() < 0.35


def test_rsi_saturates_high_and_low():
    up = rsi(bars(list(range(10, 40))), window=5)["rsi_5_0.9"].drop_nulls()
    down = rsi(bars(list(range(40, 10, -1))), window=5)["rsi_5_0.9"].drop_nulls()
    assert np.isclose(up.tail(3).min(), 1.0)
    assert np.isclose(down.tail(3).max(), 0.0)


def test_stochastic_marks_top_and_bottom_of_range():
    closes = [10, 12, 14, 16, 18]
    out = stochastic_oscillator(bars(closes, highs=closes, lows=closes), 5)["so_5"]
    assert np.isclose(out[-1], 1.0)          # close at the top of the range
    out2 = stochastic_oscillator(
        bars(closes[::-1], highs=closes[::-1], lows=closes[::-1]), 5
    )["so_5"]
    assert np.isclose(out2[-1], 0.0)


def test_indicators_are_null_during_warmup_not_wrong():
    out = trend_strength(bars(list(range(10, 20))), window=5)["ts_5_0.9"]
    assert out.head(4).null_count() == 4
    assert out.tail(6).null_count() == 0


# --------------------------------------------------------------------------
# volume-weighted quantiles
# --------------------------------------------------------------------------

def test_rolling_weighted_quantiles_match_numpy():
    rng = np.random.default_rng(7)
    v = rng.normal(100, 5, 40)
    w = rng.integers(1, 500, 40).astype(float)
    got = _rolling_weighted_quantiles(v, w, 6, [0.1, 0.5, 0.9])
    for i in range(5, 40):
        sl = slice(i - 5, i + 1)
        for q in (0.1, 0.5, 0.9):
            ref = np.quantile(v[sl], q=q, weights=w[sl], method="inverted_cdf")
            assert np.isclose(got[q][i], ref)


def test_rolling_weighted_quantiles_are_null_during_warmup():
    got = _rolling_weighted_quantiles(np.arange(10.0), np.ones(10), 4, [0.5])
    assert np.isnan(got[0.5][:3]).all()
    assert not np.isnan(got[0.5][3:]).any()


def test_zero_volume_window_falls_back_to_unweighted():
    """A stretch of imputed bars has no volume; the quantile must not collapse."""
    v = np.array([10.0, 20.0, 30.0, 40.0])
    got = _rolling_weighted_quantiles(v, np.zeros(4), 4, [0.5])
    assert np.isclose(got[0.5][-1], np.quantile(v, 0.5, method="inverted_cdf"))


def test_volume_weighting_pulls_the_quantile_toward_heavy_bars():
    closes = [10.0, 20.0, 30.0]
    heavy_low = volume_quantile_ratios(
        bars(closes, volume=[1000.0, 1.0, 1.0]), 3, [0.5]
    )["rvwq_3_0.5"][-1]
    heavy_high = volume_quantile_ratios(
        bars(closes, volume=[1.0, 1.0, 1000.0]), 3, [0.5]
    )["rvwq_3_0.5"][-1]
    # close/quantile: a low-priced heavy bar gives a small denominator
    assert heavy_low > heavy_high


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def panel(n=80):
    rng = np.random.default_rng(3)
    out = []
    for t in ("A", "B"):
        closes = 100 + np.cumsum(rng.normal(0, 1, n))
        out.append(bars(closes, ticker=t, volume=rng.integers(1e5, 1e6, n).astype(float)))
    return pl.concat(out)


def test_metrics_adds_exactly_the_advertised_columns():
    df = panel()
    out = metrics(df, windows=[3, 7])
    added = [c for c in out.columns if c not in df.columns]
    assert added == metric_columns([3, 7])


def test_metrics_preserves_every_row():
    df = panel()
    assert metrics(df, windows=[3, 7]).height == df.height


def test_metrics_drop_warmup_removes_only_incomplete_rows():
    df = panel()
    out = metrics(df, windows=[3, 7], drop_warmup=True)
    assert out.height == df.height - 2 * 6      # 2 tickers x (7-1) warmup bars
    assert out.select(metric_columns([3, 7])).null_count().sum_horizontal().sum() == 0


def test_metrics_requires_a_ticker_column():
    with pytest.raises(ValueError, match="ticker"):
        metrics(panel().drop("ticker"), windows=[3])


def test_metrics_rejects_a_bad_imputed_mode():
    with pytest.raises(ValueError, match="on_imputed"):
        metrics(panel(), windows=[3], on_imputed="nonsense")


# --- imputed handling -----------------------------------------------------

def imputed_panel(n=60, gap=40):
    """Panel where bar `gap` (by position) of ticker A is a synthetic no-trade
    day: flat OHLC at the prior close, zero volume."""
    df = panel(n).sort(["ticker", "dt"])
    flag = (pl.col("ticker") == "A") & (pl.int_range(pl.len()).over("ticker") == gap)
    prior = pl.col("close").shift(1).over("ticker")
    return df.with_columns(
        flag.alias("imputed"),
        *[pl.when(flag).then(prior).otherwise(pl.col(c)).alias(c)
          for c in ("open", "high", "low", "close")],
        pl.when(flag).then(0.0).otherwise(pl.col("volume")).alias("volume"),
    )


def test_on_imputed_null_leaves_imputed_rows_empty():
    df = imputed_panel()
    out = metrics(df, windows=[3, 7], on_imputed="null")
    imp = out.filter(pl.col("imputed"))
    assert imp.height > 0
    assert imp.select(metric_columns([3, 7])).null_count().sum_horizontal().sum() == (
        imp.height * len(metric_columns([3, 7]))
    )


def test_on_imputed_skip_carries_the_last_real_value():
    df = imputed_panel()
    out = metrics(df, windows=[7], on_imputed="skip").sort(["ticker", "dt"])
    a = out.filter(pl.col("ticker") == "A")
    i = a["imputed"].to_list().index(True)
    names = metric_columns([7])
    # the synthetic bar inherits the previous real bar's indicators, verbatim
    assert a[names].row(i) == a[names].row(i - 1)
    # and it is not left null, unlike the warm-up rows
    assert a[names].slice(i, 1).null_count().sum_horizontal().sum() == 0


def test_on_imputed_skip_excludes_the_bar_from_the_window():
    """The whole point: a no-trade day must not drag the indicator."""
    df = imputed_panel()
    skip = metrics(df, windows=[7], on_imputed="skip").sort(["ticker", "dt"])
    inc = metrics(df, windows=[7], on_imputed="include").sort(["ticker", "dt"])
    a_skip = skip.filter(pl.col("ticker") == "A")
    a_inc = inc.filter(pl.col("ticker") == "A")
    i = a_skip["imputed"].to_list().index(True)
    # bars after the gap differ, because include() has the flat bar in-window
    after = [a_skip["ts_7_0.9"][j] != a_inc["ts_7_0.9"][j] for j in range(i + 1, i + 7)]
    assert any(after)


def test_on_imputed_does_not_affect_other_tickers():
    df = imputed_panel()
    skip = metrics(df, windows=[7], on_imputed="skip")
    inc = metrics(df, windows=[7], on_imputed="include")
    b_skip = skip.filter(pl.col("ticker") == "B")["ts_7_0.9"].to_list()
    b_inc = inc.filter(pl.col("ticker") == "B")["ts_7_0.9"].to_list()
    assert b_skip == b_inc


def test_metrics_without_imputed_column_still_works():
    df = panel().drop_nulls()
    assert "imputed" not in df.columns
    out = metrics(df, windows=[3], on_imputed="skip")
    assert out.height == df.height


# --------------------------------------------------------------------------
# true range -- pinned to Wilder's definition
# --------------------------------------------------------------------------

def test_true_range_equals_wilders_three_term_max():
    """max(H, Cprev) - min(L, Cprev) is algebraically Wilder's
    max(H-L, |H-Cprev|, |L-Cprev|). Pin it so nobody 'simplifies' it back."""
    from etfs.technicals import _true_range

    rng = np.random.default_rng(11)
    n = 5000
    low = rng.normal(100, 10, n)
    high = low + np.abs(rng.normal(0, 3, n))
    close = low + rng.random(n) * (high - low)

    df = pl.DataFrame({
        "ticker": ["A"] * n,
        "dt": [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)],
        "high": high, "low": low, "close": close,
    })
    got = df.with_columns(_true_range().alias("tr"))["tr"].to_numpy()[1:]

    prev = close[:-1]
    want = np.maximum.reduce([
        high[1:] - low[1:], np.abs(high[1:] - prev), np.abs(low[1:] - prev)
    ])
    np.testing.assert_allclose(got, want)


def test_true_range_captures_an_overnight_gap():
    """A gap must widen TR beyond the intraday range."""
    from etfs.technicals import _true_range

    df = pl.DataFrame({
        "ticker": ["A", "A"],
        "dt": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
        "high": [101.0, 120.0], "low": [99.0, 118.0], "close": [100.0, 119.0],
    })
    tr = df.with_columns(_true_range().alias("tr"))["tr"].to_list()
    assert tr[1] == 20.0                 # 120 - 100, not the 2.0 intraday range


def test_true_range_first_bar_is_the_intraday_range():
    """No previous close exists, so TR falls back to high - low."""
    from etfs.technicals import _true_range

    df = pl.DataFrame({
        "ticker": ["A"], "dt": [dt.datetime(2024, 1, 1)],
        "high": [101.0], "low": [99.0], "close": [100.0],
    })
    assert df.with_columns(_true_range().alias("tr"))["tr"][0] == 2.0
