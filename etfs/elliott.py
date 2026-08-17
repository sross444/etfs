"""Mechanical Elliott-wave features.

Elliott wave analysis is discretionary: two analysts routinely label the same
chart differently, and counts get revised as new bars arrive. This module does
not pretend to resolve that. It extracts the parts that *are* mechanical --
swing pivots, Elliott's three hard rules, and the Fibonacci ratios between legs
-- and emits them as numeric features. Treat them as structure descriptors, not
as a wave count anyone would defend.

**Everything here is causal.** A swing pivot is not knowable when it happens;
it is only confirmed once price retraces far enough away from it. A naive
zigzag marks the pivot at its own bar, which back-dates information by however
many bars the confirmation took -- that is repainting, and it silently leaks
the future into any backtest built on it. Here every pivot carries the bar it
was *confirmed* on, features at bar t use only pivots confirmed at or before t,
and `test_elliott.py` asserts that computing on a truncated series reproduces
the full-series values exactly.

The price of that honesty is lag: `ew_pivot_lag` reports it per bar, and it is
typically 5-20 sessions. Nothing here identifies a turn as it happens.

The three hard rules (the only non-negotiable ones in the theory):

  1. Wave 2 never retraces more than 100% of wave 1.
  2. Wave 3 is never the shortest of waves 1, 3 and 5.
  3. Wave 4 never enters wave 1's price territory.

Guidelines -- wave 2 retracing 0.5/0.618, wave 3 extending 1.618, wave 4
retracing 0.382 -- are scored continuously in `ew_fib`, not enforced.
"""

import numpy as np
import polars as pl

# Canonical Fibonacci levels per leg relationship.
FIB_RETRACE = (0.382, 0.5, 0.618, 0.786)
FIB_EXTEND = (1.0, 1.618, 2.618)

FEATURES = [
    "ew_dir",
    "ew_leg",
    "ew_rules",
    "ew_w2_retr",
    "ew_w3_ext",
    "ew_w4_retr",
    "ew_fib",
    "ew_swing_pos",
    "ew_swing_age",
    "ew_pivot_lag",
]


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Wilder true range, simple-averaged. Causal: bar t uses bars <= t."""
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high, prev) - np.minimum(low, prev)
    out = np.full(tr.shape, np.nan)
    if len(tr) >= window:
        c = np.cumsum(np.insert(tr, 0, 0.0))
        out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def zigzag(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_mult: float = 3.0,
    atr_window: int = 14,
) -> list[tuple[int, float, int, int]]:
    """Confirmed swing pivots.

    A running extreme becomes a pivot only once price reverses from it by more
    than `atr_mult` ATRs. Scaling the threshold by ATR rather than a fixed
    percentage keeps swing counts comparable across a panel holding both TLT
    and EWZ.

    Returns:
        (pivot_index, pivot_price, kind, confirmed_index) per pivot, where kind
        is +1 for a swing high and -1 for a swing low. `confirmed_index` is the
        first bar on which the pivot was knowable, and is always > pivot_index.
    """
    n = len(close)
    atr = _atr(high, low, close, atr_window)
    pivots: list[tuple[int, float, int, int]] = []
    if n == 0:
        return pivots

    direction = 1  # +1: tracking a running high, -1: tracking a running low
    ext_price, ext_idx = high[0], 0

    for t in range(n):
        thr = atr[t]
        if not np.isfinite(thr) or thr <= 0:
            # still warming up: track extremes but confirm nothing
            if direction == 1 and high[t] >= ext_price:
                ext_price, ext_idx = high[t], t
            elif direction == -1 and low[t] <= ext_price:
                ext_price, ext_idx = low[t], t
            continue
        thr *= atr_mult

        if direction == 1:
            if high[t] >= ext_price:
                ext_price, ext_idx = high[t], t
            elif ext_price - low[t] > thr:
                pivots.append((ext_idx, float(ext_price), 1, t))
                direction = -1
                ext_price, ext_idx = low[t], t
        else:
            if low[t] <= ext_price:
                ext_price, ext_idx = low[t], t
            elif high[t] - ext_price > thr:
                pivots.append((ext_idx, float(ext_price), -1, t))
                direction = 1
                ext_price, ext_idx = high[t], t

    return pivots


def _fib_closeness(value: float, levels, tol: float = 0.2) -> float:
    """1.0 on a canonical level, decaying to 0 at `tol` away."""
    if not np.isfinite(value):
        return np.nan
    d = min(abs(value - lv) for lv in levels)
    return max(0.0, 1.0 - d / tol)


def _score_sequence(pivots: list[tuple[int, float, int, int]]) -> dict:
    """Rules and ratios for the last six pivots (a complete 5-leg attempt)."""
    blank = {
        "ew_dir": 0.0, "ew_rules": np.nan, "ew_w2_retr": np.nan,
        "ew_w3_ext": np.nan, "ew_w4_retr": np.nan, "ew_fib": np.nan,
    }
    if len(pivots) < 6:
        return blank

    p = [x[1] for x in pivots[-6:]]
    kinds = [x[2] for x in pivots[-6:]]
    # p0 must open the sequence: a low for a bullish impulse, high for bearish.
    direction = -kinds[0]  # starting at a low (-1) means an up-impulse (+1)
    if kinds != [(-direction if i % 2 == 0 else direction) for i in range(6)]:
        return blank

    legs = [abs(p[i + 1] - p[i]) for i in range(5)]
    w1, w2, w3, w4, w5 = legs
    if w1 <= 0 or w3 <= 0:
        return blank

    if direction > 0:
        r1 = p[2] > p[0]        # wave 2 holds above the start of wave 1
        r3 = p[4] > p[1]        # wave 4 stays clear of wave 1's territory
    else:
        r1 = p[2] < p[0]
        r3 = p[4] < p[1]
    r2 = not (w3 < w1 and w3 < w5)   # wave 3 is not the shortest

    w2_retr, w3_ext, w4_retr = w2 / w1, w3 / w1, w4 / w3
    fib = np.nanmean([
        _fib_closeness(w2_retr, FIB_RETRACE),
        _fib_closeness(w3_ext, FIB_EXTEND),
        _fib_closeness(w4_retr, FIB_RETRACE),
    ])

    return {
        "ew_dir": float(direction),
        "ew_rules": float(r1 + r2 + r3) / 3.0,
        "ew_w2_retr": float(w2_retr),
        "ew_w3_ext": float(w3_ext),
        "ew_w4_retr": float(w4_retr),
        "ew_fib": float(fib),
    }


def _features_for_series(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    atr_mult: float, atr_window: int,
) -> dict[str, np.ndarray]:
    """Per-bar features, using only pivots confirmed at or before each bar."""
    n = len(close)
    out = {name: np.full(n, np.nan) for name in FEATURES}
    pivots = zigzag(high, low, close, atr_mult, atr_window)

    # Walk bars forward, revealing each pivot on its confirmation bar.
    by_confirm: dict[int, int] = {}
    for i, (_, _, _, c) in enumerate(pivots):
        by_confirm.setdefault(c, i)

    revealed = 0
    cached: dict | None = None
    leg = 0.0
    for t in range(n):
        if t in by_confirm:
            revealed = by_confirm[t] + 1
            cached = _score_sequence(pivots[:revealed])
            # Count legs within the current impulse attempt; restart after five.
            leg = leg + 1 if leg < 5 else 1.0

        if revealed == 0:
            continue

        idx, price, kind, conf = pivots[revealed - 1]
        for k, v in (cached or {}).items():
            out[k][t] = v
        out["ew_leg"][t] = leg
        out["ew_swing_age"][t] = t - conf
        out["ew_pivot_lag"][t] = conf - idx

        # Where the current close sits between the last pivot and the running
        # extreme reached since it: 0 at the pivot, 1 at the extreme. Capped
        # above at 1, but deliberately allowed to go negative -- price back
        # through the pivot means the swing failed, which is worth knowing.
        window_hi = high[conf:t + 1].max() if t >= conf else high[t]
        window_lo = low[conf:t + 1].min() if t >= conf else low[t]
        if kind > 0:  # last pivot was a high, so we are swinging down
            span = price - window_lo
            out["ew_swing_pos"][t] = (
                (price - close[t]) / span if span > 0 else 0.0
            )
        else:
            span = window_hi - price
            out["ew_swing_pos"][t] = (
                (close[t] - price) / span if span > 0 else 0.0
            )

    return out


def elliott_features(
    df: pl.DataFrame, atr_mult: float = 3.0, atr_window: int = 14
) -> pl.DataFrame:
    """Add the `ew_*` columns, computed per ticker.

    Args:
        df: panel with ticker / dt / high / low / close.
        atr_mult: reversal size, in ATRs, needed to confirm a swing pivot.
            Larger means fewer, bigger waves and a longer confirmation lag.
        atr_window: lookback for the ATR that scales the threshold.

    Returns:
        `df` plus the columns in `FEATURES`:

        * `ew_dir` +1 if the last complete sequence was an up-impulse, -1 down,
          0 if the pivots do not form one
        * `ew_leg` which leg of the current attempt is forming, 1-5
        * `ew_rules` fraction of Elliott's three hard rules satisfied, 0-1
        * `ew_w2_retr`, `ew_w3_ext`, `ew_w4_retr` the leg ratios
        * `ew_fib` how close those ratios sit to canonical Fibonacci levels, 0-1
        * `ew_swing_pos` position of the close within the current swing: 0 at
          the last pivot, 1 at the extreme reached since. Bounded above by 1;
          goes negative when price retraces back through the pivot, i.e. the
          swing failed
        * `ew_swing_age` bars since the last pivot was confirmed
        * `ew_pivot_lag` bars that pivot took to confirm -- the information lag
    """
    if "ticker" not in df.columns:
        raise ValueError("elliott_features() needs a 'ticker' column")

    frames = []
    for (_,), g in df.group_by(["ticker"], maintain_order=True):
        g = g.sort("dt")
        feats = _features_for_series(
            g["high"].to_numpy().astype(float),
            g["low"].to_numpy().astype(float),
            g["close"].to_numpy().astype(float),
            atr_mult,
            atr_window,
        )
        frames.append(g.with_columns([pl.Series(k, v) for k, v in feats.items()]))

    return pl.concat(frames, how="vertical").select(df.columns + FEATURES)
