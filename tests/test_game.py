import numpy as np
import pytest

from etfs.game import BUY, SELL, STOP, TradingGame


def prices(n=200, k=4, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    p = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, (n, k)), axis=0))
    return p


def game(**kw):
    p = prices()
    kw.setdefault("trade_fraction", 0.25)
    kw.setdefault("max_actions", 10)
    kw.setdefault("lookback", 20)
    return TradingGame(p, p, **kw)


# -- the two clocks ---------------------------------------------------------

def test_decision_steps_pay_nothing_and_do_not_advance_the_market():
    g = game()
    g.reset()
    t0 = g.t
    for a in [(BUY, 0), (BUY, 1), (SELL, 0)]:
        _, reward, _, info = g.step(*a)
        assert reward == 0.0
        assert info["advances_market"] is False
    assert g.t == t0, "market clock moved during a decision"


def test_stop_is_the_only_transition_that_pays():
    g = game()
    g.reset()
    t0 = g.t
    g.step(BUY, 0)
    _, reward, _, info = g.step(STOP, -1)
    assert info["advances_market"] is True
    assert reward != 0.0
    assert g.t == t0 + 1


def test_stop_is_forced_at_the_action_cap():
    """Without this the agent can decline to stop forever."""
    g = game(max_actions=4)
    g.reset()
    infos = [g.step(BUY, 0)[3] for _ in range(4)]
    assert [i["advances_market"] for i in infos] == [False, False, False, True]
    assert infos[-1]["forced_stop"] is True


def test_action_counter_resets_between_sessions():
    g = game(max_actions=3)
    g.reset()
    for _ in range(3):
        g.step(BUY, 0)
    assert g.n_actions == 0


# -- aggregation and costs --------------------------------------------------

def test_intra_session_churn_is_free():
    """Buying then selling before STOP never reached an exchange."""
    g = game()
    g.reset()
    for _ in range(6):
        g.step(BUY, 0)
        g.step(SELL, 0)
    _, _, _, info = g.step(STOP, -1)
    assert info["turnover"] == pytest.approx(0.0)
    assert info["cost"] == pytest.approx(0.0)


def test_repeated_buys_accumulate_into_one_position():
    g = game(trade_fraction=0.10)
    g.reset()
    for _ in range(3):
        g.step(BUY, 2)
    assert g.work_w[2] == pytest.approx(0.30)


def test_turnover_is_measured_against_yesterdays_allocation():
    g = game(trade_fraction=0.25, transaction_cost=0.001)
    g.reset()
    g.step(BUY, 0)
    _, _, _, first = g.step(STOP, -1)
    assert first["turnover"] == pytest.approx(0.25)

    # hold: change nothing, pay nothing
    _, _, _, second = g.step(STOP, -1)
    assert second["turnover"] == pytest.approx(0.0)
    assert second["cost"] == pytest.approx(0.0)


def test_cost_is_charged_on_executed_turnover():
    g = game(trade_fraction=0.5, transaction_cost=0.01)
    g.reset()
    g.step(BUY, 0)
    _, _, _, info = g.step(STOP, -1)
    assert info["cost"] == pytest.approx(info["turnover"] * 0.01)
    assert info["net_return"] == pytest.approx(info["gross_return"] - info["cost"])


# -- constraints ------------------------------------------------------------

def test_long_only_a_sell_cannot_go_short():
    g = game()
    g.reset()
    for _ in range(5):
        g.step(SELL, 1)
    assert g.work_w[1] == pytest.approx(0.0)
    assert (g.work_w >= 0).all()


def test_selling_a_flat_position_is_a_noop_not_an_error():
    g = game()
    g.reset()
    _, reward, _, info = g.step(SELL, 0)
    assert reward == 0.0 and info["advances_market"] is False
    assert g.work_w.sum() == pytest.approx(0.0)


def test_no_leverage_buying_stops_at_fully_invested():
    g = game(trade_fraction=0.25, max_actions=50)
    g.reset()
    for _ in range(20):
        g.step(BUY, 0)
    assert g.work_w.sum() <= 1.0 + 1e-9
    assert g.work_w[0] == pytest.approx(1.0)


def test_no_leverage_holds_across_several_etfs():
    g = game(trade_fraction=0.30, max_actions=50)
    g.reset()
    for etf in [0, 1, 2, 3, 0, 1]:
        g.step(BUY, etf)
    assert g.work_w.sum() <= 1.0 + 1e-9


def test_partial_buy_fills_the_remaining_cash():
    """0.3 + 0.3 + 0.3 leaves 0.1 of room; the next buy takes exactly that."""
    g = game(trade_fraction=0.30, max_actions=50)
    g.reset()
    for etf in [0, 1, 2]:
        g.step(BUY, etf)
    g.step(BUY, 3)
    assert g.work_w[3] == pytest.approx(0.10)
    assert g.work_w.sum() == pytest.approx(1.0)


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="strictly positive"):
        TradingGame(np.zeros((100, 3)), np.ones((100, 3)))
    with pytest.raises(ValueError, match="same shape"):
        TradingGame(prices(100, 3), prices(100, 4))
    g = game()
    g.reset()
    with pytest.raises(ValueError, match="etf_id"):
        g.step(BUY, 99)
    with pytest.raises(ValueError, match="side"):
        g.step(7, 0)


# -- accounting -------------------------------------------------------------

def test_all_cash_earns_nothing_and_costs_nothing():
    g = game()
    g.reset()
    _, reward, _, info = g.step(STOP, -1)
    assert reward == pytest.approx(0.0)
    assert info["nav"] == pytest.approx(g.starting_capital)


def test_fully_invested_in_one_etf_earns_its_open_to_open_return():
    p = prices()
    g = TradingGame(p, p, trade_fraction=1.0, max_actions=10, lookback=20)
    g.reset()
    t = g.t
    g.step(BUY, 0)
    _, reward, _, info = g.step(STOP, -1)
    expected = p[t + 1, 0] / p[t, 0] - 1.0
    assert info["gross_return"] == pytest.approx(expected)
    assert reward == pytest.approx(expected - info["cost"])


def test_weights_drift_with_returns_and_stay_normalised():
    g = game(trade_fraction=0.5, transaction_cost=0.0)
    g.reset()
    g.step(BUY, 0)
    g.step(BUY, 1)
    g.step(STOP, -1)
    assert g.held_w.sum() == pytest.approx(1.0)
    assert (g.held_w >= 0).all()


def test_nav_compounds_across_days():
    g = game(trade_fraction=0.5, transaction_cost=0.0)
    g.reset()
    g.step(BUY, 0)
    navs, rets = [], []
    for _ in range(5):
        _, r, _, info = g.step(STOP, -1)
        navs.append(info["nav"])
        rets.append(r)
    expected = g.starting_capital * np.prod([1 + r for r in rets])
    assert navs[-1] == pytest.approx(expected)


# -- the information line ---------------------------------------------------

def test_observation_uses_no_future_data():
    """Rewriting every bar from the execution day onward must not move the
    observation the agent decides on."""
    p = prices(seed=3)
    g1 = TradingGame(p, p, lookback=20, max_actions=10)
    obs1, _ = g1.reset()

    tampered = p.copy()
    tampered[g1.t:] *= 5.0          # scramble the present and the future
    g2 = TradingGame(tampered, tampered, lookback=20, max_actions=10)
    obs2, _ = g2.reset()

    np.testing.assert_allclose(obs1, obs2, rtol=1e-6)


def test_observation_reflects_the_working_allocation():
    """Mid-session states must be distinguishable, or the critic cannot fit."""
    g = game()
    before, _ = g.reset()
    after, _, _, _ = g.step(BUY, 0)
    assert not np.allclose(before, after)


def test_observation_is_finite_and_correctly_shaped():
    g = game()
    obs, _ = g.reset()
    assert obs.shape == (g.obs_dim,)
    assert np.isfinite(obs).all()
    for _ in range(5):
        obs, _, _, _ = g.step(BUY, 1)
        assert np.isfinite(obs).all()


def test_episode_terminates_at_the_end_of_the_panel():
    g = TradingGame(prices(60), prices(60), lookback=20, max_actions=5)
    g.reset(start=g.last_step - 2)
    done = False
    for _ in range(10):
        _, _, done, _ = g.step(STOP, -1)
        if done:
            break
    assert done
