"""The trading game as a reinforcement-learning environment.

Two clocks, and keeping them apart is the whole job. See GAME.md.

  * The **decision clock** ticks on every BUY/SELL. No market time passes, no
    money moves, nothing is executed. Reward is exactly zero.
  * The **market clock** ticks once, on STOP. The accumulated allocation
    executes at the open, and the return it earns is the only reward in the
    episode.

`step()` therefore reports `advances_market` in its info dict. Anything doing
credit assignment over these transitions must discount by that flag rather than
by wall-clock step count -- discounting the construction of a portfolio as if
time were passing teaches an agent to be impatient about the order in which it
places its own trades, which is meaningless. See `etfs/ppo.py`.

Accounting runs open-to-open. The allocation committed for day `t` executes at
`open[t]` and is next touched at `open[t+1]`, so that interval is exactly the
holding period, and no unobservable intraday path is ever consulted.
"""

import numpy as np

from etfs.reward import DifferentialSharpe

BUY, SELL, STOP = 0, 1, 2
NUM_SIDES = 3


class TradingGame:
    """Long-only, no-leverage allocation game over a panel of ETFs."""

    def __init__(
        self,
        opens: np.ndarray,
        closes: np.ndarray,
        tickers: list[str] | None = None,
        trade_fraction: float = 0.10,
        transaction_cost: float = 0.0005,
        max_actions: int = 20,
        lookback: int = 20,
        starting_capital: float = 100_000.0,
        starting_weights: np.ndarray | None = None,
        reward_mode: str = "dsr",
        dsr_eta: float = 0.02,
        dsr_warmup: int = 50,
        seed: int | None = None,
    ):
        """
        Args:
            opens, closes: `[T, n_etfs]` price matrices, strictly positive.
                Executions use `opens`; features are built from `closes`.
            trade_fraction: capital moved per BUY/SELL, as a weight.
            transaction_cost: charged on executed turnover, one way.
            max_actions: cap on actions per session; STOP is forced at the cap.
            lookback: sessions of history in the observation.
            starting_capital: opening NAV.
            starting_weights: optional opening allocation, defaults to all cash.
            reward_mode: `"dsr"` for the differential Sharpe ratio, or
                `"return"` for the raw net return. DSR prices volatility as it
                goes, so the agent optimises what the scoreboard measures --
                but it is a differential and only telescopes to terminal Sharpe
                under a discount near 1. See `etfs.reward`.
            dsr_eta: DSR adaptation rate, roughly a `1/eta`-period window.
            dsr_warmup: periods used to seed the DSR moment estimates.
        """
        opens = np.asarray(opens, dtype=np.float64)
        closes = np.asarray(closes, dtype=np.float64)
        if opens.shape != closes.shape:
            raise ValueError("opens and closes must have the same shape")
        if opens.ndim != 2:
            raise ValueError("expected [T, n_etfs] price matrices")
        if not (np.all(opens > 0) and np.all(closes > 0)):
            raise ValueError("prices must be strictly positive")

        self.opens, self.closes = opens, closes
        self.n_steps, self.n_etfs = opens.shape
        self.tickers = list(tickers) if tickers else [str(i) for i in range(self.n_etfs)]

        self.trade_fraction = float(trade_fraction)
        self.transaction_cost = float(transaction_cost)
        self.max_actions = int(max_actions)
        self.lookback = int(lookback)
        self.starting_capital = float(starting_capital)
        self.starting_weights = (
            np.zeros(self.n_etfs) if starting_weights is None
            else np.asarray(starting_weights, dtype=np.float64).copy()
        )
        if self.starting_weights.sum() > 1.0 + 1e-9 or (self.starting_weights < 0).any():
            raise ValueError("starting_weights must be non-negative and sum to <= 1")

        if reward_mode not in ("dsr", "return"):
            raise ValueError(f"reward_mode must be 'dsr' or 'return'; got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.dsr = DifferentialSharpe(eta=dsr_eta, warmup=dsr_warmup)

        self.rng = np.random.default_rng(seed)

        # close-to-close returns, aligned so ret[k] belongs to bar k
        self.returns = np.zeros_like(closes)
        self.returns[1:] = closes[1:] / closes[:-1] - 1.0

        self.features_per_etf = 5
        self.obs_dim = self.n_etfs * self.features_per_etf + 4

        # Valid execution days: need `lookback` observable returns before t,
        # and open[t+1] after it to close out the holding period.
        self.first_step = self.lookback + 1
        self.last_step = self.n_steps - 2
        if self.last_step <= self.first_step:
            raise ValueError("price history too short for this lookback")

        self.t = None
        self.held_w = None
        self.work_w = None
        self.nav = None
        self.n_actions = None

    # -- episode lifecycle ------------------------------------------------

    def reset(self, start: int | None = None, random_start: bool = False):
        if random_start:
            start = int(self.rng.integers(self.first_step, self.last_step))
        self.t = self.first_step if start is None else int(
            np.clip(start, self.first_step, self.last_step)
        )
        self.held_w = self.starting_weights.copy()
        self.work_w = self.held_w.copy()
        self.nav = self.starting_capital
        self.n_actions = 0
        self.dsr.reset()
        return self._observation(), {"nav": self.nav, "t": self.t}

    def step(self, side: int, etf_id: int):
        """Apply one action.

        Returns:
            obs, reward, terminated, info. `info["advances_market"]` is True
            only on the transition that executed -- the only one carrying a
            reward, and the only one that should be discounted.
        """
        if side not in (BUY, SELL, STOP):
            raise ValueError(f"side must be BUY, SELL or STOP; got {side}")

        self.n_actions += 1
        forced = self.n_actions >= self.max_actions

        if side in (BUY, SELL):
            if not 0 <= etf_id < self.n_etfs:
                raise ValueError(f"etf_id out of range: {etf_id}")
            if side == BUY:
                # No leverage: buy only what uninvested cash supports.
                room = max(0.0, 1.0 - self.work_w.sum())
                self.work_w[etf_id] += min(self.trade_fraction, room)
            else:
                # Long only: sell no more than is held.
                self.work_w[etf_id] -= min(self.trade_fraction, self.work_w[etf_id])

        if side != STOP and not forced:
            # Still deciding. No time passes, no reward.
            return self._observation(), 0.0, False, {
                "advances_market": False,
                "nav": self.nav,
                "actions_used": self.n_actions,
            }

        return self._execute(forced=forced and side != STOP)

    def _execute(self, forced: bool = False):
        """Commit the allocation at the open and run the market forward one day."""
        turnover = float(np.abs(self.work_w - self.held_w).sum())
        cost = turnover * self.transaction_cost

        # Held from open[t] to open[t+1] -- exactly until the next rebalance.
        period = self.opens[self.t + 1] / self.opens[self.t] - 1.0
        gross = float(self.work_w @ period)
        net = gross - cost
        self.nav *= 1.0 + net

        # The DSR estimator must see every period, whichever reward is in use,
        # so its moments stay meaningful in the observation either way.
        differential = self.dsr.update(net)
        reward = differential if self.reward_mode == "dsr" else net

        # Weights drift with the assets they are invested in.
        grown = self.work_w * (1.0 + period)
        denominator = 1.0 + gross
        self.held_w = grown / denominator if abs(denominator) > 1e-12 else grown

        self.t += 1
        self.work_w = self.held_w.copy()
        self.n_actions = 0
        terminated = self.t >= self.last_step

        info = {
            "advances_market": True,
            "nav": self.nav,
            "gross_return": gross,
            "net_return": net,
            "turnover": turnover,
            "cost": cost,
            "invested": float(self.held_w.sum()),
            "forced_stop": forced,
            "dsr": differential,
            "rolling_sharpe": self.dsr.sharpe,
            "t": self.t,
        }
        return self._observation(), reward, terminated, info

    # -- observation ------------------------------------------------------

    def _observation(self) -> np.ndarray:
        """Point-in-time features: only bars strictly before the execution day.

        Includes the *working* allocation, which is what makes two different
        mid-session states distinguishable to a value function.
        """
        window = self.returns[self.t - self.lookback:self.t]

        last = window[-1]
        mean = window.mean(axis=0)
        vol = window.std(axis=0) + 1e-8
        momentum = np.prod(1.0 + window, axis=0) - 1.0

        per_etf = np.stack([last, mean, vol, momentum, self.work_w], axis=-1)
        # The DSR moments are part of the state: the reward is a function of
        # them, so without them the environment is not Markovian.
        globals_ = np.array([
            max(0.0, 1.0 - self.work_w.sum()),          # uninvested cash
            self.n_actions / max(1, self.max_actions),  # session progress
            np.clip(self.dsr.sharpe, -3.0, 3.0),        # rolling Sharpe
            np.sqrt(self.dsr.variance) * 100.0,         # rolling volatility
        ])
        return np.concatenate([per_etf.reshape(-1), globals_]).astype(np.float32)

    # -- construction from the panel --------------------------------------

    @classmethod
    def from_panel(cls, df, **kwargs):
        """Build from a polars panel with dt / ticker / open / close.

        Requires a rectangular panel -- use
        `Store.load(common_start=True, fill_gaps=True)`.
        """
        import polars as pl

        opens = df.pivot(index="dt", on="ticker", values="open").sort("dt")
        closes = df.pivot(index="dt", on="ticker", values="close").sort("dt")
        tickers = [c for c in opens.columns if c != "dt"]

        o = opens.select(tickers).to_numpy()
        c = closes.select(tickers).to_numpy()
        if np.isnan(o).any() or np.isnan(c).any():
            raise ValueError(
                "panel has gaps; load with common_start=True, fill_gaps=True"
            )
        return cls(o, c, tickers=tickers, **kwargs)
