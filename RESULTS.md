# Results

A running record of what has actually been measured. Every number here is
validate-split performance from agents trained only on `train` (< 2024). The
test split (2026) has not been touched.

**Nothing here beats equal weight.** That is the headline, and the rest of this
file is the evidence and the reasons.

---

## Baselines (validate, 2024-2025)

| strategy | Sharpe | ann. return | max drawdown | exposure |
|---|---|---|---|---|
| equal_weight | **1.04** | +12.2% | −12.7% | 1.00 |
| random policy (5 seeds) | 0.93 ± 0.23 | +11.5% | −11.7% | 0.98 |
| all cash | 0.00 | 0.0% | 0.0% | 0.00 |

Equal weight is a hard baseline. It is also **not reachable through the action
space** — at `trade_fraction = 0.10` an agent can hold at most 10 positions and
can never express 1/76 across 76 ETFs. It is a market reference, not a target.

The random policy is close behind it, because random buying fills to ~98%
exposure and becomes a diluted equal weight. Any agent that merely learns "be
invested" lands here.

---

## PPO: the definitive run

20 seeds, 150 updates, `entropy_coef = 0.0002`, real 50 features, 45-day
sampled episodes, `gamma = 1.0`. Validate split, all controls also 20 draws.

| strategy | Sharpe | ann. return | max drawdown | exposure |
|---|---|---|---|---|
| equal_weight | **1.04** | +12.2% | −12.7% | 1.00 |
| random_policy | 0.95 ± 0.32 | +11.8% | −13.6% | 0.98 |
| random_portfolio(10) | 0.93 ± 0.28 | +11.3% | −13.4% | 1.00 |
| **ppo** | **0.62 ± 0.73** | +10.6% | −17.4% | 0.86 |

Welch tests against the PPO seed population:

| comparison | difference | s.e. | t | p |
|---|---|---|---|---|
| ppo − random_portfolio(10) | **−0.31** | 0.176 | −1.78 | 0.075 |
| ppo − random_policy | **−0.33** | 0.179 | −1.84 | 0.066 |
| ppo − equal_weight | **−0.42** | 0.164 | −2.54 | — |

PPO Sharpe quartiles: **−0.60 / 0.16 / 0.60 / 0.98 / 2.45**.
Seeds beating equal weight: **4 of 20** — about what chance would give.

**PPO is strictly dominated.** Lower mean than both controls, 2.4x their seed
variance, and worse drawdown. It is not merely failing to add skill; it is
adding variance on top of a worse allocation. Every earlier result suggesting
otherwise -- the 1.23 at 60 updates, the 1.20 and 1.22 checkpoints, the 1.31
first seed of this very run -- was selection from a wide distribution.

The controls are the useful finding here. `random_portfolio(10)` holds ten
randomly chosen ETFs and scores 0.93, statistically indistinguishable from the
random policy at 0.95 and close to equal weight at 1.04. In this universe over
this window, **being invested is worth ~0.95 Sharpe and selection is worth
approximately nothing** -- which is the bar any agent has to clear.

### What this does not tell us

That the game is unwinnable. It tells us this configuration does not win. The
plausible explanations are not yet separated:

- **There may be no signal** in daily technical features at this horizon. This
  is the hypothesis that should be tested first and cheapest -- a supervised
  cross-sectional model predicting next-day returns will answer it in minutes.
  If a linear model cannot beat chance, no amount of RL engineering will.
- **Capacity vastly exceeds data.** 1.1M parameters, a 3,880-dimensional
  observation, and 1393 training bars whose 45-day episodes overlap heavily.
- **Regime mismatch.** Train is Sharpe 0.20 with a −31% drawdown; validate is
  1.04. A policy tuned to survive 2018-2023 should be expected to lag a bull
  run.
- **The action space is awkward for credit assignment.** Composing an
  allocation from repeated 10% increments plus a STOP is a harder problem than
  emitting target weights directly.

## Standing caveats

- The agent runs at ~1.0 exposure in every successful configuration, so it has
  learned *to be invested*, and the open question is whether its **selection**
  adds anything beyond that. Random-at-98%-exposure scores 0.93.
- The train split (Sharpe 0.20, −31% drawdown) is a far harder regime than
  validate (1.04) or test (1.06). An agent trained on 2018-2023 is learning a
  market that stopped existing.
- 45-day windows sampled from 1393 training bars overlap heavily, so episodes
  are not independent and the effective sample size is well below the episode
  count.
- The universe is survivorship-biased: delisted funds were dropped, so every
  ETF in the panel survived to 2026.
