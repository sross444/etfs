import numpy as np
import pytest

torch = pytest.importorskip("torch")

from etfs.game import BUY, SELL, STOP, TradingGame  # noqa: E402
from etfs.ppo import HierarchicalActorCritic, PPOAgent, RolloutBuffer, train  # noqa: E402


def buf(steps, values=None):
    """steps: list of (advances, reward)."""
    b = RolloutBuffer()
    values = values or [0.0] * len(steps)
    for (adv, r), v in zip(steps, values):
        b.add(np.zeros(3), STOP if adv else BUY, -1 if adv else 0,
              r, False, adv, v, 0.0)
    return b


# -- the per-transition discount: the whole point of this module ------------

def test_decision_steps_are_not_discounted():
    """A session of any length is one day, so it takes one day of discount."""
    b = buf([(False, 0.0)] * 4 + [(True, 0.10)])
    adv = b.compute_gae(0.0, gamma_daily=0.99, gae_lambda=1.0)["advantages"].numpy()
    np.testing.assert_allclose(adv, 0.10, rtol=1e-6)


def test_session_length_does_not_change_credit():
    """Two sessions differing only in how many actions built them must assign
    the same advantage -- otherwise the agent learns to prefer short sessions
    for reasons with no economic content."""
    short = buf([(False, 0.0)] * 2 + [(True, 0.10)])
    long_ = buf([(False, 0.0)] * 15 + [(True, 0.10)])
    a = short.compute_gae(0.0, 0.99, 1.0)["advantages"].numpy()
    b = long_.compute_gae(0.0, 0.99, 1.0)["advantages"].numpy()
    assert a[0] == pytest.approx(b[0])


def test_market_steps_are_discounted_across_days():
    b = buf([(False, 0.0), (False, 0.0), (True, 0.0),
             (False, 0.0), (False, 0.0), (True, 1.0)])
    adv = b.compute_gae(0.0, gamma_daily=0.99, gae_lambda=1.0)["advantages"].numpy()
    np.testing.assert_allclose(adv[:3], 0.99, rtol=1e-6)   # one day away
    np.testing.assert_allclose(adv[3:], 1.00, rtol=1e-6)   # same day


def test_uniform_discounting_would_differ():
    """Guards the fix: a naive implementation gives a visibly different answer,
    so this test fails loudly if the per-step gamma is ever dropped."""
    b = buf([(False, 0.0)] * 4 + [(True, 0.10)])
    ours = b.compute_gae(0.0, 0.99, 1.0)["advantages"].numpy()
    naive = np.array([0.99 ** (4 - i) * 0.10 for i in range(4)] + [0.10])
    assert not np.allclose(ours, naive)
    assert ours[0] > naive[0]


def test_decision_advantage_is_the_critics_change_of_opinion():
    """With gamma=1 and r=0, delta collapses to V(s') - V(s)."""
    b = buf([(False, 0.0), (False, 0.0), (True, 0.0)], values=[1.0, 3.0, 7.0])
    adv = b.compute_gae(0.0, gamma_daily=1.0, gae_lambda=0.0)["advantages"].numpy()
    assert adv[0] == pytest.approx(3.0 - 1.0)
    assert adv[1] == pytest.approx(7.0 - 3.0)


def test_returns_are_advantages_plus_values():
    b = buf([(False, 0.0), (True, 0.05)], values=[0.2, 0.3])
    d = b.compute_gae(0.0, 0.99, 0.95)
    np.testing.assert_allclose(
        d["returns"].numpy(),
        d["advantages"].numpy() + np.array([0.2, 0.3]), rtol=1e-6,
    )


def test_terminal_step_does_not_bootstrap():
    b = RolloutBuffer()
    b.add(np.zeros(3), STOP, -1, 0.5, True, True, 0.0, 0.0)
    adv = b.compute_gae(99.0, 0.99, 0.95)["advantages"].numpy()
    assert adv[0] == pytest.approx(0.5)


# -- hierarchical action head ----------------------------------------------

def model(n_etfs=6, obs_dim=20):
    torch.manual_seed(0)
    return HierarchicalActorCritic(obs_dim, n_etfs, hidden_dim=32)


def test_stop_carries_no_etf_and_buy_sell_do():
    m = model()
    side, etf_id, logp, value = m.act(torch.randn(256, 20))
    assert set(side.unique().tolist()) <= {BUY, SELL, STOP}
    assert (etf_id[side == STOP] == -1).all()
    assert (etf_id[side != STOP] >= 0).all()
    assert torch.isfinite(logp).all() and torch.isfinite(value).all()


def test_stop_logprob_omits_the_etf_term():
    """STOP has no ETF policy, so its log-prob is the side term alone."""
    m = model()
    obs = torch.randn(64, 20)
    side = torch.full((64,), STOP, dtype=torch.long)
    etf = torch.full((64,), -1, dtype=torch.long)
    logp, entropy, _ = m.evaluate_actions(obs, side, etf)

    from torch.distributions import Categorical
    side_logits, _, _ = m(obs)
    expected = Categorical(logits=side_logits).log_prob(side)
    torch.testing.assert_close(logp, expected)


def test_buy_and_sell_use_separate_conditional_heads():
    m = model()
    obs = torch.randn(32, 20)
    etf = torch.zeros(32, dtype=torch.long)
    buy_lp, _, _ = m.evaluate_actions(obs, torch.full((32,), BUY), etf)
    sell_lp, _, _ = m.evaluate_actions(obs, torch.full((32,), SELL), etf)
    assert not torch.allclose(buy_lp, sell_lp)


def test_evaluate_actions_reproduces_act_logprobs():
    """The ratio in PPO is meaningless unless these two agree exactly."""
    m = model()
    obs = torch.randn(128, 20)
    side, etf_id, logp, _ = m.act(obs)
    again, _, _ = m.evaluate_actions(obs, side, etf_id)
    torch.testing.assert_close(logp, again, rtol=1e-5, atol=1e-6)


def test_deterministic_act_is_stable():
    m = model()
    obs = torch.randn(16, 20)
    a = m.act(obs, deterministic=True)
    b = m.act(obs, deterministic=True)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


# -- end to end -------------------------------------------------------------

def small_env():
    rng = np.random.default_rng(1)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 5)), axis=0))
    return TradingGame(p, p, lookback=10, max_actions=8, trade_fraction=0.25)


def test_training_runs_and_changes_the_policy():
    env = small_env()
    agent = PPOAgent(env.obs_dim, env.n_etfs, hidden_dim=32)
    before = [p.detach().clone() for p in agent.model.parameters()]
    history = train(env, agent, total_updates=2, rollout_steps=128,
                    update_epochs=2, minibatch_size=64, verbose=False)
    after = list(agent.model.parameters())
    assert len(history) == 2
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    assert all(np.isfinite(h["policy_loss"]) for h in history)


def test_rollout_records_both_transition_kinds():
    env = small_env()
    agent = PPOAgent(env.obs_dim, env.n_etfs, hidden_dim=32)
    obs, _ = env.reset()
    b = RolloutBuffer()
    for _ in range(200):
        side, etf, lp, v = agent.select_action(obs)
        obs, r, done, info = env.step(side, etf)
        b.add(obs, side, etf, r, done, info["advances_market"], v, lp)
        if done:
            obs, _ = env.reset()
    advances = np.array(b.advances)
    assert advances.sum() > 0, "no market step in 200 actions"
    assert (advances == 0).sum() > 0, "no decision step in 200 actions"
    # reward is zero exactly on the decision steps
    rewards = np.array(b.rewards)
    assert np.allclose(rewards[advances == 0], 0.0)
