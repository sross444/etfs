"""Splits, baselines, and the bench that ranks agents.

Nothing produced here is meaningful without the split discipline: an agent is
trained only on `train`, tuned only against `validate`, and touches `test` once,
at the end. `TradingGame` enforces this structurally via its `window` -- an
episode cannot observe or trade outside its own range.

Everything is measured on realised **open-to-open** returns net of cost, so an
agent policy and a fixed-weight benchmark are compared on exactly the same
arithmetic.
"""

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from etfs.game import BUY, SELL, STOP, TradingGame
from etfs.reward import max_drawdown, sharpe, sortino

TRAIN_END = dt.date(2024, 1, 1)
VALIDATE_END = dt.date(2026, 1, 1)


@dataclass(frozen=True)
class Split:
    name: str
    lo: int
    hi: int
    start: dt.date
    end: dt.date

    @property
    def n_bars(self) -> int:
        return self.hi - self.lo + 1

    def __repr__(self) -> str:
        return (f"Split({self.name}: {self.start}..{self.end}, "
                f"{self.n_bars} bars, idx {self.lo}..{self.hi})")


def make_splits(dates, train_end: dt.date = TRAIN_END,
                validate_end: dt.date = VALIDATE_END) -> dict[str, Split]:
    """Chronological train / validate / test windows over the panel's dates.

    Boundaries are dates, not fractions, so the split does not move when the
    panel is re-synced with fresh bars -- only `test` grows.
    """
    d = [x.date() if hasattr(x, "date") else x for x in dates]
    n = len(d)
    bounds = {
        "train": (0, sum(x < train_end for x in d) - 1),
        "validate": (sum(x < train_end for x in d),
                     sum(x < validate_end for x in d) - 1),
        "test": (sum(x < validate_end for x in d), n - 1),
    }
    out = {}
    for name, (lo, hi) in bounds.items():
        if hi < lo:
            raise ValueError(f"split {name!r} is empty; check the date bounds")
        out[name] = Split(name, lo, hi, d[lo], d[hi])
    return out


@dataclass
class Report:
    """What a strategy did over one window."""

    name: str
    split: str
    n_days: int
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    avg_turnover: float
    avg_exposure: float
    final_nav: float
    returns: np.ndarray = field(repr=False, default=None)

    def row(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "returns"}


def summarise(name: str, split: str, returns, turnover, exposure,
              capital: float = 100_000.0, periods_per_year: int = 252) -> Report:
    r = np.asarray(returns, dtype=np.float64)
    equity = float(np.prod(1.0 + r)) if r.size else 1.0
    years = r.size / periods_per_year if r.size else 1.0
    return Report(
        name=name,
        split=split,
        n_days=int(r.size),
        total_return=equity - 1.0,
        ann_return=equity ** (1.0 / years) - 1.0 if years > 0 else 0.0,
        ann_vol=float(r.std(ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else 0.0,
        sharpe=sharpe(r, periods_per_year),
        sortino=sortino(r, periods_per_year),
        max_drawdown=max_drawdown(r),
        avg_turnover=float(np.mean(turnover)) if len(turnover) else 0.0,
        avg_exposure=float(np.mean(exposure)) if len(exposure) else 0.0,
        final_nav=capital * equity,
        returns=r,
    )


# ---------------------------------------------------------------------------
# fixed-weight benchmarks
# ---------------------------------------------------------------------------

def simulate_weights(opens: np.ndarray, target: np.ndarray, split: Split,
                     transaction_cost: float = 0.0005, name: str = "benchmark",
                     rebalance: bool = False, capital: float = 100_000.0) -> Report:
    """Run a fixed target allocation through the same accounting as the game.

    Args:
        target: weight vector summing to <= 1.
        rebalance: True to restore `target` every day (paying turnover), False
            to buy once and let the weights drift, which is what
            "buy and hold" actually means.
    """
    w = np.asarray(target, dtype=np.float64).copy()
    if w.sum() > 1.0 + 1e-9 or (w < 0).any():
        raise ValueError("target weights must be non-negative and sum to <= 1")

    held = np.zeros_like(w)
    rets, turns, exposures = [], [], []

    # settle at open[t+1], so stop at hi-1 and stay inside the window
    for t in range(split.lo, min(split.hi - 1, len(opens) - 2) + 1):
        want = w if rebalance else (w if t == split.lo else held)
        turnover = float(np.abs(want - held).sum())
        cost = turnover * transaction_cost

        period = opens[t + 1] / opens[t] - 1.0
        gross = float(want @ period)

        rets.append(gross - cost)
        turns.append(turnover)
        exposures.append(float(want.sum()))

        grown = want * (1.0 + period)
        held = grown / (1.0 + gross) if abs(1.0 + gross) > 1e-12 else grown

    return summarise(name, split.name, rets, turns, exposures, capital)


def equal_weight(opens: np.ndarray, split: Split, **kw) -> Report:
    """Equal weight across every ETF, bought once and held.

    Note this allocation is **not reachable through the action space**: with
    `trade_fraction = 0.10` an agent can hold at most 10 positions, so it can
    never express 1/76 in each of 76. It is a market benchmark, not a policy an
    agent is being asked to imitate.
    """
    n = opens.shape[1]
    kw.setdefault("name", "equal_weight")
    return simulate_weights(opens, np.full(n, 1.0 / n), split, **kw)


def all_cash(opens: np.ndarray, split: Split, **kw) -> Report:
    kw.setdefault("name", "all_cash")
    return simulate_weights(opens, np.zeros(opens.shape[1]), split, **kw)


# ---------------------------------------------------------------------------
# policy evaluation
# ---------------------------------------------------------------------------

class RandomPolicy:
    """Uniform over the action space. The floor any agent must clear."""

    def __init__(self, n_etfs: int, seed: int = 0, stop_prob: float = 0.2):
        self.n_etfs = n_etfs
        self.stop_prob = stop_prob
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs, deterministic: bool = False):
        if self.rng.random() < self.stop_prob:
            return STOP, -1, 0.0, 0.0
        side = BUY if self.rng.random() < 0.5 else SELL
        return side, int(self.rng.integers(self.n_etfs)), 0.0, 0.0


def make_env(opens, closes, split: Split, **kw) -> TradingGame:
    """A game confined to one split. This is where the separation is enforced."""
    return TradingGame(opens, closes, window=(split.lo, split.hi), **kw)


def evaluate_policy(env: TradingGame, policy, name: str,
                    deterministic: bool = True, max_steps: int = 2_000_000) -> Report:
    """Roll a policy once, start to finish, over the env's own window.

    The policy only needs `select_action(obs, deterministic) -> (side, etf, ...)`,
    which both `PPOAgent` and the baselines satisfy. Pass an env already built
    for the split you mean to measure -- see `make_env`.
    """
    split_name = getattr(env, "split_name", "window")
    obs, _ = env.reset()
    rets, turns, exposures = [], [], []

    for _ in range(max_steps):
        side, etf_id, *_ = policy.select_action(obs, deterministic=deterministic)
        obs, _, done, info = env.step(side, etf_id)
        if info["advances_market"]:
            rets.append(info["net_return"])
            turns.append(info["turnover"])
            exposures.append(info["invested"])
        if done:
            break

    return summarise(name, split_name, rets, turns, exposures, env.starting_capital)


def compare(reports) -> pl.DataFrame:
    """Reports as a sortable table, best risk-adjusted first."""
    return pl.DataFrame([r.row() for r in reports]).sort("sharpe", descending=True)


# ---------------------------------------------------------------------------
# multi-seed experiments
# ---------------------------------------------------------------------------

def run_seeds(opens, closes, splits: dict, agent_factory, train_fn,
              seeds=(0, 1, 2, 3, 4), env_kwargs: dict | None = None,
              train_kwargs: dict | None = None, eval_split: str = "validate",
              name: str = "ppo", verbose: bool = True):
    """Train one agent per seed on `train`, then evaluate each on `eval_split`.

    A single RL run is a sample from a wide distribution, so a lone seed says
    almost nothing. The spread across seeds is the result; the mean alone will
    flatter or damn a method at random.

    Returns:
        `(reports, table)` -- one report per seed, and a summary row per seed
        plus aggregate mean/std.
    """
    import torch

    env_kwargs = dict(env_kwargs or {})
    train_kwargs = dict(train_kwargs or {})
    reports = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_env = make_env(opens, closes, splits["train"], seed=seed, **env_kwargs)
        train_env.split_name = "train"
        agent = agent_factory(train_env, seed)
        train_fn(train_env, agent, **train_kwargs)

        eval_env = make_env(opens, closes, splits[eval_split], seed=seed, **env_kwargs)
        eval_env.split_name = eval_split
        report = evaluate_policy(eval_env, agent, name=f"{name}_seed{seed}")
        reports.append(report)
        if verbose:
            print(f"  seed {seed}: sharpe={report.sharpe:+.3f} "
                  f"ann_return={report.ann_return:+.2%} "
                  f"maxDD={report.max_drawdown:.2%} "
                  f"turnover={report.avg_turnover:.3f}")

    table = compare(reports)
    return reports, table


def aggregate(reports, name: str = "ppo") -> pl.DataFrame:
    """Mean and standard deviation across seeds for the metrics that matter."""
    cols = ["sharpe", "sortino", "ann_return", "ann_vol", "max_drawdown",
            "avg_turnover", "avg_exposure"]
    df = pl.DataFrame([r.row() for r in reports])
    return pl.DataFrame({
        "name": [name],
        "seeds": [len(reports)],
        **{f"{c}_mean": [float(df[c].mean())] for c in cols},
        **{f"{c}_std": [float(df[c].std(ddof=1)) if len(reports) > 1 else 0.0]
           for c in cols},
    })
