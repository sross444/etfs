"""Turn the indicator panel into a point-in-time feature tensor for the game.

Two things here are easy to get wrong and both would inflate results silently.

**Alignment.** Indicator row `k` is computed from bars up to and including `k`,
so it is knowable at the close of day `k`. The pre-open decision for day `t`
therefore reads row `t-1`. The environment does that shift; this module only
guarantees the tensor is indexed the same way as the price matrices.

**Standardisation.** Features arrive on wildly different scales -- `rvwq` sits
near 1.0, `ts` spans [-1, 1], `ew_pivot_lag` counts sessions -- so a network
needs them normalised. Computing that mean and standard deviation over the whole
panel would leak test-set information into training. `standardise()` takes the
statistics from a slice you nominate, which must be the training window.
"""

import numpy as np
import polars as pl


def build_features(df: pl.DataFrame, elliott: bool = False, windows=None,
                   decay: float = 0.9) -> tuple[np.ndarray, list[str], list, list[str]]:
    """Compute indicators and lay them out as `[T, n_etfs, n_features]`.

    Args:
        df: rectangular panel from `Store.load(common_start=True, fill_gaps=True)`.
        elliott: include the 10 `ew_*` structure features as well as the 50
            technicals.
        windows, decay: passed to `etfs.technicals.metrics`.

    Returns:
        `(features, tickers, dates, names)`. Ticker order matches the sorted
        pivot used by `TradingGame.from_panel`, so the axes line up with the
        price matrices.
    """
    from etfs.technicals import metric_columns, metrics

    names = metric_columns(windows, decay)
    if elliott:
        from etfs.elliott import FEATURES as EW
        names = names + EW

    scored = metrics(df, windows=windows, decay=decay, elliott=elliott)

    frames = {c: scored.pivot(index="dt", on="ticker", values=c).sort("dt")
              for c in names}
    any_frame = next(iter(frames.values()))
    tickers = sorted(c for c in any_frame.columns if c != "dt")
    dates = any_frame["dt"].to_list()

    stacked = np.stack(
        [frames[c].select(tickers).to_numpy().astype(np.float32) for c in names],
        axis=-1,
    )
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0), tickers, dates, names


def standardise(features: np.ndarray, train_slice: slice, clip: float = 5.0):
    """Z-score each feature using statistics from the training window only.

    Args:
        features: `[T, n_etfs, n_features]`.
        train_slice: the rows the statistics may be computed from. Passing
            anything that reaches beyond the training window leaks.
        clip: bound on the standardised value, so a single outlier bar cannot
            dominate a gradient step.

    Returns:
        `(standardised, mean, std)` -- keep `mean`/`std` to apply to new data.
    """
    train = features[train_slice]
    if train.size == 0:
        raise ValueError("train_slice selects no rows")

    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    out = np.clip((features - mean) / std, -clip, clip)
    return out.astype(np.float32), mean, std


def apply_standardisation(features: np.ndarray, mean: np.ndarray,
                          std: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """Apply previously fitted statistics -- for validate and test."""
    return np.clip((features - mean) / std, -clip, clip).astype(np.float32)
