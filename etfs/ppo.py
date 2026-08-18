"""Hierarchical PPO for the two-clock trading game.

The game interleaves two kinds of transition (see GAME.md):

  * **decision transitions** -- BUY/SELL. Reward zero. No market time passes.
  * **market transitions** -- STOP. Carries the day's return. Time passes.

Standard PPO discounts every transition identically, which is wrong here: it
would make the agent impatient about the order in which it assembles its own
portfolio, a quantity with no economic meaning. Three changes fix it, and they
are the only substantive departures from textbook PPO.

1. **Per-transition discount.** GAE uses `gamma = 1.0` across decision
   transitions and `gamma_daily` only where the market advanced. A session of
   any length is then discounted exactly as much as a single day, because it
   *is* a single day.

2. **The working allocation is in the observation.** Otherwise two mid-session
   states with different partial portfolios are indistinguishable to the critic
   while having genuinely different futures, and the value function cannot fit.

3. **A hierarchical action head.** `side in {BUY, SELL, STOP}`, with a
   conditional ETF distribution that only exists for BUY and SELL. STOP has no
   ETF, so it contributes no ETF log-probability and no ETF entropy.

Consequence worth stating plainly: credit for a good allocation flows backwards
through the entire session to the actions that built it, undiscounted. The
critic does the work -- each intermediate state is valued by what the finished
portfolio is expected to earn.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from etfs.game import BUY, NUM_SIDES, SELL, STOP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HierarchicalActorCritic(nn.Module):
    """Shared encoder; a side head, a conditional ETF head, and a critic.

    The ETF head emits `[batch, 2, n_etfs]` -- row 0 conditional on BUY, row 1
    conditional on SELL. There is deliberately no ETF policy for STOP.
    """

    def __init__(self, obs_dim: int, n_etfs: int, hidden_dim: int = 256):
        super().__init__()
        self.n_etfs = n_etfs
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.side_head = nn.Linear(hidden_dim, NUM_SIDES)
        self.etf_head = nn.Linear(hidden_dim, 2 * n_etfs)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        z = self.encoder(obs)
        return (
            self.side_head(z),
            self.etf_head(z).view(-1, 2, self.n_etfs),
            self.value_head(z).squeeze(-1),
        )

    def _etf_dist(self, etf_logits, side, mask):
        """Conditional ETF distribution for the non-STOP rows only."""
        return Categorical(logits=etf_logits[mask, side[mask], :])

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        side_logits, etf_logits, values = self(obs)
        side_dist = Categorical(logits=side_logits)
        side = (torch.argmax(side_logits, -1) if deterministic
                else side_dist.sample())

        n = obs.shape[0]
        etf_id = torch.full((n,), -1, dtype=torch.long, device=obs.device)
        etf_logp = torch.zeros(n, device=obs.device)

        mask = side != STOP
        if mask.any():
            dist = self._etf_dist(etf_logits, side, mask)
            chosen = (torch.argmax(dist.logits, -1) if deterministic
                      else dist.sample())
            etf_id[mask] = chosen
            etf_logp[mask] = dist.log_prob(chosen)

        return side, etf_id, side_dist.log_prob(side) + etf_logp, values

    def evaluate_actions(self, obs, side, etf_id):
        side_logits, etf_logits, values = self(obs)
        side_dist = Categorical(logits=side_logits)
        side_logp = side_dist.log_prob(side)
        side_entropy = side_dist.entropy()

        etf_logp = torch.zeros_like(side_logp)
        etf_entropy = torch.zeros_like(side_entropy)

        mask = side != STOP
        if mask.any():
            dist = self._etf_dist(etf_logits, side, mask)
            etf_logp[mask] = dist.log_prob(etf_id[mask])
            etf_entropy[mask] = dist.entropy()

        return side_logp + etf_logp, side_entropy + etf_entropy, values


@dataclass
class RolloutBuffer:
    """Stores `advances` alongside each transition -- the flag that decides
    whether a step is discounted at all."""

    obs: list = field(default_factory=list)
    sides: list = field(default_factory=list)
    etf_ids: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    advances: list = field(default_factory=list)
    values: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)

    def add(self, obs, side, etf_id, reward, done, advances, value, log_prob):
        self.obs.append(np.asarray(obs).copy())
        self.sides.append(side)
        self.etf_ids.append(etf_id)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.advances.append(float(advances))
        self.values.append(value)
        self.log_probs.append(log_prob)

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value: float, gamma_daily: float, gae_lambda: float):
        """GAE with a per-transition discount.

            gamma_t = gamma_daily   if the transition advanced the market
                    = 1.0           if it was a decision within a session

        With `gamma_t = 1` and `r_t = 0` on decision steps, delta collapses to
        `V(s_{t+1}) - V(s_t)`: the advantage of a BUY is exactly how much it
        improved the critic's opinion of the portfolio being built. That is the
        credit-assignment signal, and it is why intermediate states must be
        observable.
        """
        rewards = np.asarray(self.rewards, dtype=np.float64)
        dones = np.asarray(self.dones, dtype=np.float64)
        advances = np.asarray(self.advances, dtype=np.float64)
        values = np.asarray(self.values + [last_value], dtype=np.float64)

        gammas = np.where(advances > 0, gamma_daily, 1.0)

        adv = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            g = gammas[t]
            delta = rewards[t] + g * values[t + 1] * nonterminal - values[t]
            running = delta + g * gae_lambda * nonterminal * running
            adv[t] = running

        as_t = lambda x, d=torch.float32: torch.as_tensor(np.asarray(x), dtype=d)
        return {
            "obs": as_t(self.obs),
            "sides": as_t(self.sides, torch.long),
            "etf_ids": as_t(self.etf_ids, torch.long),
            "old_log_probs": as_t(self.log_probs),
            "advantages": as_t(adv),
            "returns": as_t(adv + values[:-1]),
        }


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        n_etfs: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma_daily: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.20,
        value_coef: float = 0.50,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.50,
        target_kl: float = 0.015,
    ):
        self.model = HierarchicalActorCritic(obs_dim, n_etfs, hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.gamma_daily = gamma_daily
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

    @torch.no_grad()
    def select_action(self, obs, deterministic: bool = False):
        t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        side, etf_id, logp, value = self.model.act(t, deterministic)
        return int(side.item()), int(etf_id.item()), float(logp.item()), float(value.item())

    def update(self, rollout, last_value, update_epochs=10, minibatch_size=256):
        data = rollout.compute_gae(last_value, self.gamma_daily, self.gae_lambda)
        obs = data["obs"].to(DEVICE)
        sides = data["sides"].to(DEVICE)
        etf_ids = data["etf_ids"].to(DEVICE)
        old_log_probs = data["old_log_probs"].to(DEVICE)
        returns = data["returns"].to(DEVICE)

        advantages = data["advantages"].to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        stop = False

        for _ in range(update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, minibatch_size):
                batch = torch.as_tensor(idx[start:start + minibatch_size],
                                        dtype=torch.long, device=DEVICE)
                new_logp, entropy, values = self.model.evaluate_actions(
                    obs[batch], sides[batch], etf_ids[batch]
                )
                ratio = torch.exp(new_logp - old_log_probs[batch])
                a = advantages[batch]
                policy_loss = -torch.min(
                    ratio * a,
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * a,
                ).mean()
                value_loss = 0.5 * (returns[batch] - values).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy_bonus)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs[batch] - new_logp).mean().item()
                metrics = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy_bonus.item()),
                    "approx_kl": approx_kl,
                }
                if approx_kl > 1.5 * self.target_kl:
                    stop = True
                    break
            if stop:
                break
        return metrics


def train(env, agent, total_updates=100, rollout_steps=2048,
          update_epochs=10, minibatch_size=256, log_every=10, verbose=True):
    """Rollouts may slice through the middle of a session; that is fine, because
    the bootstrap value at the seam is a genuine state value either way."""
    obs, _ = env.reset()
    history = []

    for update in range(total_updates):
        rollout = RolloutBuffer()
        market_steps = 0
        rewards_seen, returns_seen = [], []

        for _ in range(rollout_steps):
            side, etf_id, logp, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(side, etf_id)

            rollout.add(obs, side, etf_id, reward, done,
                        info["advances_market"], value, logp)

            if info["advances_market"]:
                market_steps += 1
                rewards_seen.append(reward)
                # the reward may be a DSR; the realised return is separate
                returns_seen.append(info["net_return"])
            obs = next_obs
            if done:
                obs, _ = env.reset()

        _, _, _, last_value = agent.select_action(obs)
        metrics = agent.update(rollout, last_value, update_epochs, minibatch_size)

        from etfs.reward import sharpe as ann_sharpe

        mean_reward = float(np.mean(rewards_seen)) if rewards_seen else 0.0
        mean_return = float(np.mean(returns_seen)) if returns_seen else 0.0
        row = {"update": update + 1, "market_days": market_steps,
               "mean_reward": mean_reward, "mean_return": mean_return,
               "sharpe": ann_sharpe(returns_seen), "nav": info["nav"], **metrics}
        history.append(row)

        if verbose and (update + 1) % log_every == 0:
            print(f"update={row['update']:04d} days={market_steps:5d} "
                  f"reward={mean_reward:+.4f} ret={mean_return:+.5f} "
                  f"sharpe={row['sharpe']:+6.2f} nav={info['nav']:,.0f} "
                  f"pi={metrics['policy_loss']:+.4f} v={metrics['value_loss']:.4f} "
                  f"H={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f}")
    return history
