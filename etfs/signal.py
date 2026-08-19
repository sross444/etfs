"""Does the feature panel predict anything?

Before asking whether an agent can learn a policy, ask whether there is
anything to learn. This fits a linear model on the same features, with the same
train/validate discipline, and measures it the way the game would use it.

The target is chosen to match the game exactly. At the pre-open decision for
day `t` the agent knows features through `t-1`, buys at `open[t]` and marks at
`open[t+1]`. So features at row `t-1` are asked to predict the open-to-open
return from `t` to `t+1` -- one full day beyond the information cutoff.

The metric that matters is *cross-sectional*. Most of an ETF's daily move is
the market, which no amount of selection can capture, so the relevant question
is whether the features rank ETFs against each other on the same day. That is
the information coefficient.
"""

import numpy as np


def build_xy(features: np.ndarray, opens: np.ndarray, lo: int, hi: int):
    """Assemble `(X, y, day_index)` for bars in `[lo, hi]`.

    Returns:
        X: `[n_samples, n_features]` from row `t-1`.
        y: open-to-open return from `t` to `t+1`.
        day: the bar index each sample belongs to, for cross-sectional grouping.
    """
    n_steps = opens.shape[0]
    first = max(lo, 1)
    last = min(hi - 1, n_steps - 2)
    if last < first:
        raise ValueError("window too short to build a supervised sample")

    xs, ys, days = [], [], []
    for t in range(first, last + 1):
        xs.append(features[t - 1])                       # knowable before day t
        ys.append(opens[t + 1] / opens[t] - 1.0)         # what the game earns
        days.append(np.full(opens.shape[1], t))
    return (np.concatenate(xs).astype(np.float64),
            np.concatenate(ys).astype(np.float64),
            np.concatenate(days))


def demean_by_day(y: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Strip the market: what is left is the selectable part."""
    out = y.copy()
    for d in np.unique(day):
        m = day == d
        out[m] -= out[m].mean()
    return out


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 10.0):
    """Closed-form ridge with an intercept. Deliberately linear -- if a linear
    model finds nothing, the question is about the features, not the capacity."""
    Xc = np.hstack([X, np.ones((X.shape[0], 1))])
    n_features = Xc.shape[1]
    penalty = alpha * np.eye(n_features)
    penalty[-1, -1] = 0.0                                # never penalise the intercept
    beta = np.linalg.solve(Xc.T @ Xc + penalty, Xc.T @ y)
    return beta


def ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ beta


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R^2 against the mean of the evaluation set."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def information_coefficient(y: np.ndarray, pred: np.ndarray, day: np.ndarray,
                            rank: bool = True) -> dict:
    """Mean daily cross-sectional correlation between prediction and outcome.

    Args:
        rank: use Spearman (rank) rather than Pearson, which is the usual
            choice because it is not dominated by a handful of extreme days.

    Returns:
        `mean`, `std`, `t_stat` and `n_days`. A t-stat above ~2 is the
        conventional threshold for a signal worth acting on.
    """
    ics = []
    for d in np.unique(day):
        m = day == d
        a, b = pred[m], y[m]
        if a.size < 3 or a.std() == 0 or b.std() == 0:
            continue
        if rank:
            a = np.argsort(np.argsort(a)).astype(float)
            b = np.argsort(np.argsort(b)).astype(float)
        ics.append(float(np.corrcoef(a, b)[0, 1]))

    ics = np.asarray(ics)
    if ics.size < 2:
        return {"mean": 0.0, "std": 0.0, "t_stat": 0.0, "n_days": int(ics.size)}
    return {
        "mean": float(ics.mean()),
        "std": float(ics.std(ddof=1)),
        "t_stat": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(ics.size))),
        "n_days": int(ics.size),
    }


def top_n_portfolio(y: np.ndarray, pred: np.ndarray, day: np.ndarray,
                    n: int = 10, cost: float = 0.0005) -> np.ndarray:
    """Daily returns from equal-weighting the `n` highest-predicted ETFs.

    This is the decision-relevant test: it is exactly what the long-only game
    would do with a working forecast, and it is directly comparable to the
    random_portfolio control.
    """
    rets, prev = [], None
    for d in np.unique(day):
        m = day == d
        pick = np.argsort(pred[m])[::-1][:n]
        w = np.zeros(m.sum())
        w[pick] = 1.0 / n
        turnover = float(np.abs(w - prev).sum()) if prev is not None else 1.0
        rets.append(float(w @ y[m]) - turnover * cost)
        prev = w
    return np.asarray(rets)
