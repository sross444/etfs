"""Reward functions for the trading game.

Raw daily return is a poor training signal for a scoreboard that ranks agents
on risk-adjusted performance: it pays identically for a steady 10 bps and for a
coin flip between +5% and -5%. The differential Sharpe ratio prices the
volatility as it goes.
"""

import numpy as np


class DifferentialSharpe:
    """Online differential Sharpe ratio (Moody & Saffell, 1998).

    Keeps exponential estimates of the first two moments of the return stream,

        A_t = A_{t-1} + eta * (R_t - A_{t-1})
        B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})

    whose ratio `A / sqrt(B - A^2)` is a rolling Sharpe. The reward is that
    Sharpe's sensitivity to the newest return,

        D_t = (B_{t-1} dA - 0.5 A_{t-1} dB) / (B_{t-1} - A_{t-1}^2)^{3/2}

    which is positive when `R_t` improves the Sharpe and negative when it hurts
    it. A large gain that arrives with a large jump in the second moment can
    score *worse* than a smaller steady one -- that is the entire point.

    It is a *differential*, and that has a consequence which is easy to get
    wrong: in a stationary regime its expectation is approximately zero however
    good that regime is. It does not pay for sitting in a high-Sharpe state, it
    pays for moving to one. What makes it a valid objective is that it
    telescopes,

        sum_t D_t  ~=  (S_T - S_0) / eta

    so maximising *undiscounted* cumulative reward maximises terminal Sharpe.
    **This requires a discount close to 1.** At gamma = 0.99 the sum no longer
    telescopes and the agent is paid to improve Sharpe early and allowed to
    give it back later. Use gamma_daily >= 0.999 with this reward.

    The estimator is stateful, which makes the environment non-Markovian unless
    its moments are visible to the agent -- see `TradingGame._observation`.
    """

    def __init__(self, eta: float = 0.02, warmup: int = 20, clip: float = 5.0,
                 eps: float = 1e-12):
        """
        Args:
            eta: adaptation rate. Roughly a `1/eta`-period effective window, so
                0.02 is about 50 sessions.
            warmup: periods used to seed the moment estimates before any
                reward is emitted. Seeding from a real sample matters: starting
                from A = B = 0 leaves the variance near zero for the first
                periods, and `D` divides by variance^1.5, so the opening
                rewards would dwarf everything that follows.
            clip: absolute bound on the emitted reward; None disables.
            eps: variance floor guarding the denominator.
        """
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"eta must be in (0, 1]; got {eta}")
        self.eta, self.warmup, self.clip, self.eps = eta, warmup, clip, eps
        self.reset()

    def reset(self) -> None:
        self.a = 0.0
        self.b = 0.0
        self.n = 0
        self._seed = []

    def update(self, r: float) -> float:
        """Absorb one period's return and return its differential Sharpe."""
        r = float(r)
        self.n += 1

        if self.n <= self.warmup:
            # Accumulate a seed sample; emit nothing while ill-conditioned.
            self._seed.append(r)
            if self.n == self.warmup:
                sample = np.asarray(self._seed, dtype=np.float64)
                self.a = float(sample.mean())
                self.b = float((sample ** 2).mean())
                self._seed = []
            return 0.0

        da = r - self.a
        db = r * r - self.b
        variance = self.b - self.a * self.a

        if variance <= self.eps:
            d = 0.0
        else:
            d = (self.b * da - 0.5 * self.a * db) / variance ** 1.5
            if self.clip is not None:
                d = float(np.clip(d, -self.clip, self.clip))

        self.a += self.eta * da
        self.b += self.eta * db
        return d

    @property
    def variance(self) -> float:
        return max(0.0, self.b - self.a * self.a)

    @property
    def sharpe(self) -> float:
        """Rolling per-period Sharpe. Multiply by sqrt(252) to annualise."""
        v = self.variance
        return self.a / np.sqrt(v) if v > self.eps else 0.0


def sharpe(returns, periods_per_year: int = 252) -> float:
    """Annualised Sharpe of a realised return series. Evaluation, not training."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino(returns, periods_per_year: int = 252) -> float:
    """Annualised Sortino -- Sharpe counting only downside deviation."""
    r = np.asarray(returns, dtype=np.float64)
    downside = r[r < 0]
    if r.size < 2 or downside.size == 0:
        return 0.0
    dd = np.sqrt((downside ** 2).mean())
    return float(r.mean() / dd * np.sqrt(periods_per_year)) if dd > 0 else 0.0


def max_drawdown(returns) -> float:
    """Worst peak-to-trough decline of the compounded series, as a fraction."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())
