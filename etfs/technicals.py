"""Technical indicators over the daily panel.

Every function takes the whole panel and computes per ticker via `.over("ticker")`,
returning the input frame plus one new column. Column names follow the original
research convention (`ts_28_0.9`, `so_14`, `rvwq_28_0.5`, ...) so downstream code
keeps working.

Ported from `etfs_old/technicals.py`. Four things changed, deliberately:

1. **Per-ticker windows.** The original shifted and rolled without grouping, so
   on a stacked panel every indicator bled across the boundary between one
   ticker and the next. It was only correct on a single ticker at a time.
2. **True range.** The original computed
   `max(high[t-1], close[t]) - min(low[t-1], low[t])`, mixing the previous bar's
   high with the current close. True range is measured against the previous
   *close*: `max(high[t], close[t-1]) - min(low[t], close[t-1])`.
3. **Directional movement was divided by true range twice** (`pos_tr = d_high/tr`
   then `pdm_tr = pos_tr/tr`), leaving 1/price units. The final ratio does not
   cancel it, because the smoothing happens between the two divisions.
4. **No mid-pipeline `drop_nulls()`.** The original called it inside
   `efficiency_ratio` and `rsi`, which dropped every warm-up row of whichever
   indicators had already been added -- so results depended on call order.
5. **The decay ran backwards.** polars applies `weights[0]` to the *oldest* bar
   in the window, and the original passed `decay ** arange(window)`, whose first
   entry is 1.0. So `decay=0.9` weighted the stalest bar most and the newest
   least. The weights are now reversed, giving the latest bar weight 1.

The volume-weighted quantile is also vectorised; the original looped in Python
and called `np.quantile(weights=...)`, which needs numpy >= 2.0 and so raised
`TypeError` on the old repo's numpy 1.26.

`metrics()` is imputed-aware -- see `on_imputed`.
"""

import numpy as np
import polars as pl

from numpy.lib.stride_tricks import sliding_window_view

DEFAULT_WINDOWS = [3, 7, 14, 28, 56]
QUANTILES = [0.1, 0.5, 0.9]


def _weights(window: int, decay: float) -> np.ndarray:
    """Exponential weights, most recent bar last (polars applies them in order)."""
    w = decay ** np.arange(window)[::-1]
    return w / w.sum()


def _true_range() -> pl.Expr:
    """max(high, prev close) - min(low, prev close), per ticker."""
    prev_close = pl.col("close").shift(1).over("ticker")
    return pl.max_horizontal(pl.col("high"), prev_close) - pl.min_horizontal(
        pl.col("low"), prev_close
    )


def _directional_movement() -> tuple[pl.Expr, pl.Expr]:
    """Wilder's +DM / -DM: only the larger of the two moves counts, and only
    when it is positive."""
    up = (pl.col("high") - pl.col("high").shift(1).over("ticker")).fill_null(0.0)
    down = (pl.col("low").shift(1).over("ticker") - pl.col("low")).fill_null(0.0)
    plus = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    minus = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    return plus, minus


def _rolling(col: str, window: int, decay: float, how: str = "sum") -> pl.Expr:
    w = _weights(window, decay)
    expr = pl.col(col)
    roll = expr.rolling_sum if how == "sum" else expr.rolling_mean
    return roll(window_size=window, weights=list(w)).over("ticker")


def trend_strength(
    df: pl.DataFrame, window: int = 28, decay: float = 0.9, volume: bool = False
) -> pl.DataFrame:
    """Directional-movement trend strength in [-1, 1].

    +1 means every recent bar extended the high, -1 every bar extended the low.
    With `volume=True` each day's directional movement is weighted by that day's
    volume, so quiet days count for less.
    """
    name = f"{'tsv' if volume else 'ts'}_{window}_{decay}"
    plus, minus = _directional_movement()
    if volume:
        plus, minus = plus * pl.col("volume"), minus * pl.col("volume")

    out = (
        df.with_columns(
            _true_range().alias("_tr"),
            plus.alias("_dm_plus"),
            minus.alias("_dm_minus"),
        )
        .with_columns(
            # normalise by true range so the measure is scale-free
            (pl.col("_dm_plus") / pl.col("_tr").clip(lower_bound=1e-8)).alias("_di_plus"),
            (pl.col("_dm_minus") / pl.col("_tr").clip(lower_bound=1e-8)).alias("_di_minus"),
        )
        .with_columns(
            _rolling("_di_plus", window, decay, "mean").alias("_di_plus_s"),
            _rolling("_di_minus", window, decay, "mean").alias("_di_minus_s"),
        )
        .with_columns(
            ((pl.col("_di_plus_s") - pl.col("_di_minus_s"))
             / (pl.col("_di_plus_s") + pl.col("_di_minus_s")).clip(lower_bound=1e-8)
             ).alias(name)
        )
    )
    return out.select(df.columns + [name])


def efficiency_ratio(
    df: pl.DataFrame, window: int = 28, decay: float = 0.9, volume: bool = False
) -> pl.DataFrame:
    """Signed Kaufman efficiency ratio in [-1, 1]: net move over total travel.

    Near +/-1 the move was a straight line; near 0 it was chop. Signed rather
    than absolute, matching the original.
    """
    name = f"{'erv' if volume else 'er'}_{window}_{decay}"
    change = pl.col("close") - pl.col("close").shift(1).over("ticker")
    if volume:
        change = change * pl.col("volume")

    out = (
        df.with_columns(change.fill_null(0.0).alias("_d"))
        .with_columns(
            _rolling("_d", window, decay).alias("_net"),
            pl.col("_d").abs().alias("_abs"),
        )
        .with_columns(_rolling("_abs", window, decay).alias("_travel"))
        .with_columns(
            (pl.col("_net") / pl.col("_travel").clip(lower_bound=1e-8)).alias(name)
        )
    )
    return out.select(df.columns + [name])


def rsi(
    df: pl.DataFrame, window: int = 28, decay: float = 0.9, volume: bool = False
) -> pl.DataFrame:
    """RSI on a 0-1 scale: weighted gains over weighted total movement."""
    name = f"{'rsiv' if volume else 'rsi'}_{window}_{decay}"
    change = pl.col("close") - pl.col("close").shift(1).over("ticker")
    if volume:
        change = change * pl.col("volume")

    out = (
        df.with_columns(change.fill_null(0.0).alias("_d"))
        .with_columns(
            pl.col("_d").clip(lower_bound=0).alias("_up"),
            pl.col("_d").clip(upper_bound=0).abs().alias("_down"),
        )
        .with_columns(
            _rolling("_up", window, decay).alias("_gains"),
            _rolling("_down", window, decay).alias("_losses"),
        )
        .with_columns(
            (pl.col("_gains")
             / (pl.col("_gains") + pl.col("_losses")).clip(lower_bound=1e-8)
             ).alias(name)
        )
    )
    return out.select(df.columns + [name])


def stochastic_oscillator(df: pl.DataFrame, window: int = 28) -> pl.DataFrame:
    """Where the close sits in its recent high-low range, in [0, 1]."""
    name = f"so_{window}"
    out = (
        df.with_columns(
            pl.col("low").rolling_min(window_size=window).over("ticker").alias("_lo"),
            pl.col("high").rolling_max(window_size=window).over("ticker").alias("_hi"),
        )
        .with_columns(
            ((pl.col("close") - pl.col("_lo"))
             / (pl.col("_hi") - pl.col("_lo")).clip(lower_bound=1e-8)
             ).alias(name)
        )
    )
    return out.select(df.columns + [name])


def _rolling_weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, window: int, quantiles: list[float]
) -> dict[float, np.ndarray]:
    """Rolling volume-weighted quantiles, vectorised.

    Matches numpy's `inverted_cdf`: sort by value, accumulate weight, and take
    the first value whose cumulative weight reaches q of the total. Windows with
    no volume at all fall back to the unweighted quantile, so a stretch of
    imputed (zero-volume) bars cannot collapse the result onto the minimum.
    """
    n = values.shape[0]
    out = {q: np.full(n, np.nan) for q in quantiles}
    if n < window:
        return out

    v = sliding_window_view(values, window)
    w = sliding_window_view(weights, window)

    order = np.argsort(v, axis=1, kind="stable")
    vs = np.take_along_axis(v, order, axis=1)
    ws = np.take_along_axis(w, order, axis=1)

    empty = ws.sum(axis=1) <= 0
    if empty.any():
        ws = np.where(empty[:, None], 1.0, ws)

    cum = np.cumsum(ws, axis=1)
    total = cum[:, -1:]

    for q in quantiles:
        idx = (cum < q * total).sum(axis=1).clip(0, window - 1)
        out[q][window - 1:] = np.take_along_axis(vs, idx[:, None], axis=1)[:, 0]
    return out


def volume_quantile_ratios(
    df: pl.DataFrame, window: int = 28, quantiles: list[float] | None = None
) -> pl.DataFrame:
    """Close relative to the volume-weighted quantiles of recent closes.

    A ratio above 1 against the 0.9 quantile means today's close is above where
    nearly all recent *volume* traded -- a breakout past the crowd's basis.
    """
    quantiles = QUANTILES if quantiles is None else quantiles
    names = [f"rvwq_{window}_{q}" for q in quantiles]

    frames = []
    for (ticker,), g in df.group_by(["ticker"], maintain_order=True):
        g = g.sort("dt")
        qs = _rolling_weighted_quantiles(
            g["close"].to_numpy().astype(float),
            g["volume"].to_numpy().astype(float),
            window,
            quantiles,
        )
        frames.append(
            g.with_columns(
                [
                    (pl.col("close") / pl.Series(qs[q])).alias(f"rvwq_{window}_{q}")
                    for q in quantiles
                ]
            )
        )
    return pl.concat(frames, how="vertical").select(df.columns + names)


def metric_columns(
    windows: list[int] | None = None, decay: float = 0.9,
    quantiles: list[float] | None = None,
) -> list[str]:
    """Names of every column `metrics()` adds, in the order it adds them."""
    windows = DEFAULT_WINDOWS if windows is None else windows
    quantiles = QUANTILES if quantiles is None else quantiles
    names = [
        f"{stem}_{w}_{decay}"
        for w in windows
        for stem in ("ts", "tsv", "er", "erv", "rsi", "rsiv")
    ]
    for w in windows:
        names.append(f"so_{w}")
        names += [f"rvwq_{w}_{q}" for q in quantiles]
    return names


def _apply_all(df: pl.DataFrame, windows: list[int], decay: float,
               quantiles: list[float], elliott: bool = False,
               elliott_kw: dict | None = None) -> pl.DataFrame:
    if elliott:
        from etfs.elliott import elliott_features

        df = elliott_features(df, **(elliott_kw or {}))
    for w in windows:
        for fn, kw in (
            (trend_strength, {}), (trend_strength, {"volume": True}),
            (efficiency_ratio, {}), (efficiency_ratio, {"volume": True}),
            (rsi, {}), (rsi, {"volume": True}),
        ):
            df = fn(df, window=w, decay=decay, **kw)
    for w in windows:
        df = stochastic_oscillator(df, window=w)
        df = volume_quantile_ratios(df, window=w, quantiles=quantiles)
    return df


def metrics(
    df: pl.DataFrame,
    windows: list[int] | None = None,
    decay: float = 0.9,
    quantiles: list[float] | None = None,
    on_imputed: str = "skip",
    drop_warmup: bool = False,
    elliott: bool = False,
    elliott_kw: dict | None = None,
) -> pl.DataFrame:
    """Every indicator, for every window, per ticker.

    Args:
        df: the panel from `Store.load()`. Must have a `ticker` column.
        windows: lookbacks in sessions.
        decay: exponential weight decay within each window.
        quantiles: for the volume-weighted quantile ratios.
        on_imputed: how to treat bars flagged `imputed` by
            `store.fill_missing_sessions` -- days the fund did not trade.

            * `"skip"` (default): drop them, compute on the real series, then
              carry the last real indicator value onto them. The panel stays
              rectangular and a non-trading day neither dampens a trend nor
              feeds a zero into a volume-weighted average.
            * `"include"`: treat them as ordinary bars. They are zero-return and
              zero-volume, so they will drag indicators toward neutral.
            * `"null"`: compute on the real series and leave the indicator null
              on imputed rows, so nothing is implied about them at all.

            Ignored when there is no `imputed` column.
        drop_warmup: drop rows where any indicator is still null (the leading
            `max(windows)` bars of each ticker).
        elliott: also add the `ew_*` structure features from `etfs.elliott`.
            They inherit `on_imputed`, so a no-trade day cannot create a phantom
            swing pivot.
        elliott_kw: passed to `elliott_features` (`atr_mult`, `atr_window`).

    Returns:
        `df` plus the columns listed by `metric_columns()`.
    """
    windows = DEFAULT_WINDOWS if windows is None else windows
    quantiles = QUANTILES if quantiles is None else quantiles
    if "ticker" not in df.columns:
        raise ValueError("metrics() needs a 'ticker' column; pass the full panel")
    if on_imputed not in ("skip", "include", "null"):
        raise ValueError(
            f"on_imputed must be 'skip', 'include' or 'null'; got {on_imputed!r}"
        )

    df = df.sort(["ticker", "dt"])
    names = metric_columns(windows, decay, quantiles)
    if elliott:
        from etfs.elliott import FEATURES as EW_FEATURES

        names = names + EW_FEATURES

    if "imputed" not in df.columns or on_imputed == "include":
        out = _apply_all(df, windows, decay, quantiles, elliott, elliott_kw)
    else:
        real = df.filter(~pl.col("imputed"))
        computed = _apply_all(real, windows, decay, quantiles, elliott, elliott_kw)
        out = df.join(
            computed.select(["ticker", "dt"] + names), on=["ticker", "dt"], how="left"
        ).sort(["ticker", "dt"])
        if on_imputed == "skip":
            out = out.with_columns(
                [pl.col(c).forward_fill().over("ticker").alias(c) for c in names]
            )

    if drop_warmup:
        out = out.drop_nulls(subset=names)
    return out.select(df.columns + names)
