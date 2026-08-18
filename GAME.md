# The Game

A daily trading game played over the ETF panel in this repo. Agents are scored
on risk-adjusted returns.

The design stays inside what daily OHLCV can honestly support. Bars give four
prices and no sequence — you know the high and low were touched, but not in what
order, not when, and not what happened between. Every rule below follows from
refusing to pretend otherwise.

---

## The two clocks

This is the single most important structural fact about the game, and the thing
that makes it awkward for off-the-shelf RL.

**The market clock** ticks once per trading day. It is the only clock on which
money is made or lost.

**The decision clock** ticks many times within a single pre-open session. The
agent adjusts its intended allocation action by action, and *no market time
passes while it does so*. Nothing is executed, nothing is priced, nothing is
learned.

The two are joined by `STOP`. Until the agent stops, it is still deciding. When
it stops, the market opens, the whole accumulated allocation executes at once,
and the day plays out.

```
pre-open session (decision clock)          market clock
──────────────────────────────────         ────────────
  observe bars through day t-1
  BUY  XLK   ─┐
  BUY  XLE    │  no time passes,
  SELL XLK    │  no reward, nothing
  BUY  GLD    │  executed yet
  SELL VNQ   ─┘
  STOP       ─────────────────────────►  execute at day t OPEN
                                          hold the allocation
                                          ▼
  observe bars through day t              next rebalance at day t+1 OPEN
                                          (reward realised over that interval)
  ...
```

---

## The daily loop

Once per trading day, before the market opens:

1. **Observe.** All `open, high, low, close, volume` for every ETF, for every
   prior market day. Any features derived from them.
2. **Act, repeatedly.** Choose `BUY`, `SELL`, or `STOP`. `BUY` and `SELL` name an
   ETF and move a fixed fraction of capital into or out of it. The agent may act
   as many times as it likes.
3. **Stop.** `STOP` ends the session and commits the accumulated allocation.
4. **Execute.** Every net change transacts at that day's **open price**.
   Fractional shares are permitted, so the allocation is achieved exactly.
5. **Carry.** The resulting positions are held through the close and into the
   next pre-open decision.

### The information line

At the pre-open decision for day `t`, the agent knows everything through day
`t-1`'s close and **nothing about day `t`** — not its open, not its close.

This is the rule that silently destroys naive backtests, so it is structural
here rather than advisory: the decision is complete before the open exists, and
the open is the earliest thing it can touch.

---

## Actions

| action | effect |
|---|---|
| `BUY(etf)` | move `trade_fraction` of capital into that ETF |
| `SELL(etf)` | move `trade_fraction` of capital out of that ETF |
| `STOP` | commit; execute everything at the open |

`trade_fraction` is a fixed percentage of total capital — the same size every
time. The agent expresses conviction by acting on the same ETF repeatedly, not
by varying size. Three `BUY(XLK)` actions is a 3-unit position.

Actions **aggregate**. `BUY(XLK)` then `SELL(XLK)` within one session nets to
nothing, and — because nothing was sent to the market — costs nothing. Only the
net change between yesterday's allocation and the committed one ever reaches an
exchange.

### Constraints

**Long only.** No shorting. An ETF's allocation is floored at zero, and a `SELL`
on a flat position is a no-op rather than an error.

**No leverage.** Total allocation across ETFs cannot exceed capital. A `BUY`
takes whatever uninvested cash remains, up to `trade_fraction` — so it fills
partially when nearly fully invested, and is a no-op at zero cash. Whatever is
not allocated sits in cash.

Both constraints are enforced by clamping, not by rejecting the episode. An
agent that tries to over-buy simply finds the action did nothing, which is a
signal it can learn from.

### Session length

The number of actions per session is capped. On reaching the cap, `STOP` is
forced. Without this an agent can decline to stop forever, and the market clock
never advances.

---

### Accounting interval

An allocation committed for day `t` executes at `open[t]` and is next touched at
`open[t+1]`. That interval is exactly the holding period, so returns are measured
**open-to-open**. The close is never needed for PnL, which keeps the unobservable
intraday path out of the accounting entirely.

---

## Costs

A transaction cost is charged on the **turnover actually executed** — the net
change between the previous allocation and the committed one, measured at the
open.

```
turnover  = sum over ETFs of |new_weight - old_weight|
cost      = turnover x transaction_cost_rate
```

It is deliberately small. The intent is to discourage pointless churn, not to
dominate the strategy. Two consequences worth internalising:

- **Intra-session churn is free.** Buying and selling the same ETF before
  stopping costs nothing, because no order was ever sent.
- **Only day-over-day change is charged.** Holding is free. Rebalancing is not.

---

## Game setup

| input | default | meaning |
|---|---|---|
| `starting_capital` | `100_000` | cash at the start |
| `starting_positions` | all zero | optional opening allocation |
| `trade_fraction` | `0.10` | capital moved per BUY/SELL |
| `transaction_cost` | small | charged on executed turnover |
| `max_actions` | — | cap on actions per session |

`starting_capital` is not a cosmetic knob. At $1,000 a single share of a $190
ETF is 19% of the book and granularity dominates; at $100,000 selection
dominates. Fractional shares soften this, but an agent that only works at one
scale is worth knowing about.

---

## Why the rules are shaped this way

Each restriction removes something the data cannot support:

| the game forbids | because daily OHLCV cannot tell us |
|---|---|
| intraday action | when anything happened within a bar |
| limit or stop orders | whether a specific order would have been reached |
| fills at any price but the open | that a price was transactable at a knowable moment |
| acting on same-day data | anything about day `t` before day `t` opens |

What remains — decide pre-open on completed bars, execute the whole allocation at
the open — is fully determined by the data. No result depends on an unverifiable
assumption about market microstructure.

---

## Consequences for learning

The two-clock structure has direct implications for any RL agent, and they are
easy to get wrong:

**Most actions earn nothing.** Every `BUY` and `SELL` returns reward zero. Only
the transition through `STOP` carries a reward, because only then does the market
advance. Credit for a good allocation has to flow backwards through the whole
session to the actions that built it.

**Intra-session transitions must not be discounted.** No time passes between
`BUY(XLK)` and `BUY(XLE)`. Discounting them as if it did makes the agent
impatient about the *order* it constructs its portfolio in, which is meaningless.
Discounting belongs on the market clock only.

**Intermediate states must be observable.** The partially built allocation has to
appear in the observation. Otherwise two different mid-session states look
identical to the value function while having genuinely different futures.

**Sessions vary in length**, so a fixed-length rollout will slice through the
middle of them. That is fine, but the bootstrap has to be correct at the seam.

`etfs/game.py` reports which kind of transition just happened, and `etfs/ppo.py`
is a reference agent that handles all four correctly.

---

## Still to define

1. **Scoring.** The training reward is the differential Sharpe ratio (see
   `etfs/reward.py`), which prices volatility as it goes. The *evaluation*
   metric is still open — annualised Sharpe, Sortino and max drawdown are
   implemented, but which one ranks agents, over what window, is undecided.
   Reward and metric deliberately need not be the same thing.
2. **Dividends.** Carried positions in `TLT`, `HYG` and similar earn material
   distributions. `adj_close` is in the panel; `close` is what trades.
3. **Non-trading days.** 14 bars are flagged `imputed` — days a fund did not
   trade, padded with zero volume. Allocation changes on those should presumably
   be rejected rather than executed at a synthetic price.
4. **Liquidity.** Whether an allocation is capped relative to a fund's volume.
5. **Universe size.** The panel holds 76 ETFs; the reference PPO code is written
   for 25. Either is workable, but the action space scales with it.
