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

## PPO

All runs: 45-day sampled episodes, `gamma = 1.0`, DSR reward
(`eta = 0.08`, `warmup = 10`), 5 seeds unless noted.

| configuration | updates | Sharpe | cash collapses |
|---|---|---|---|
| crude 4 features | 60 | 1.23 ± 0.73 | 0/5 |
| crude 4 features | 300 | 0.59 ± 0.55 | 2/5 |
| real 50 features | 300 | 0.81 ± 0.91 | 0/5 |

Three things follow.

**Training longer made it worse.** Same configuration, 60 → 300 updates, and
the mean roughly halved while two seeds collapsed to holding cash. An earlier
commit message claimed PPO beat both baselines on the strength of the 60-update
row; that claim does not survive replication and is withdrawn. It was one draw
from a wide distribution.

**Real features helped, and fixed the collapse.** At matched budget they beat
the crude built-ins (0.81 vs 0.59) and every seed stayed fully invested. But the
seed spread nearly doubled, with individual seeds at 2.40 and 0.16.

**Nothing reliably beats random.** Across every configuration tried, the
distribution of PPO outcomes overlaps the random policy's.

---

## Why: the entropy bonus was suppressing convergence

Entropy *rose* during training in both arms (crude 3.55 → 3.71, real
4.00 → 4.06) against a maximum of ln(3) + ln(76) = 5.43. The policy was becoming
more random as training proceeded — diffusing, not converging.

Held-out Sharpe measured every 60 updates, 3 seeds, real features:

| updates | `entropy_coef = 0.005` | `entropy_coef = 0.0002` |
|---|---|---|
| 60 | 0.26 ± 0.53 | 0.49 ± 0.71 |
| 120 | 0.85 ± 0.22 | **1.20 ± 0.18** |
| 180 | 0.53 ± 0.58 | 0.96 ± 1.33 |
| 240 | 0.56 ± 0.41 | **1.22 ± 0.32** |
| 300 | 0.50 ± 0.15 | 0.63 ± 0.48 |
| final entropy | 4.11 | 3.62 |

The lower coefficient is better at **every** checkpoint, and its entropy
actually falls. The default of 0.005 was too high for the size of the advantage
signal.

---

## The measurement is the bottleneck

Seed standard deviations run from 0.15 to 1.33. With 3-5 seeds the standard
error is roughly 0.3-0.8 Sharpe, so this bench **cannot resolve differences
below about 0.8 Sharpe**, and several comparisons above sit inside that band.

The validate curve is non-monotonic — 0.49, 1.20, 0.96, 1.22, 0.63 — with no
clean peak. That shape is what noise looks like, not what overfitting looks
like. Read it as "somewhere past 120 updates it stops improving", not as a
precise optimum.

Before any further tuning is worth doing, the error bars have to come down:
more seeds, not more hyperparameters.

---

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
