import numpy as np
import pytest
from etfs.reward import DifferentialSharpe, max_drawdown, sharpe, sortino


def test_warmup_emits_nothing_then_seeds_from_the_sample():
    d = DifferentialSharpe(eta=0.02, warmup=10)
    r = np.random.default_rng(0).normal(0.001, 0.01, 10)
    assert [d.update(x) for x in r] == [0.0] * 10
    assert d.a == pytest.approx(r.mean())
    assert d.b == pytest.approx((r ** 2).mean())


def test_seeding_avoids_the_opening_spike():
    """From A=B=0 the variance starts near zero and D divides by var**1.5, so
    the first rewards would dwarf every later one."""
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.008, 400)
    d = DifferentialSharpe(eta=0.02, warmup=50)
    vals = np.array([d.update(x) for x in r])[50:]
    assert np.abs(vals).max() < 20 * np.abs(vals).mean()


def test_telescopes_to_the_change_in_sharpe():
    """The property that makes it a valid objective: sum D == (S_T - S_0)/eta."""
    rng = np.random.default_rng(1)
    r = np.concatenate([rng.normal(0.0, 0.008, 800), rng.normal(0.002, 0.005, 800)])
    eta = 0.005
    d = DifferentialSharpe(eta=eta, warmup=50, clip=None)
    total, s0 = 0.0, None
    for x in r:
        total += d.update(x)
        if s0 is None and d.n == d.warmup:
            s0 = d.sharpe
    assert total == pytest.approx((d.sharpe - s0) / eta, rel=0.15)


def test_pays_for_improvement_and_charges_for_decay():
    rng = np.random.default_rng(2)
    bad = rng.normal(0.0, 0.010, 700)
    good = rng.normal(0.0015, 0.004, 700)

    def cumulative(series):
        d = DifferentialSharpe(eta=0.02, warmup=50)
        return sum(d.update(x) for x in series)

    assert cumulative(np.concatenate([bad, good])) > 0
    assert cumulative(np.concatenate([good, bad])) < 0


def test_volatility_is_penalised_at_equal_mean():
    """Two streams, same mean, different variance: the calmer one ends higher."""
    rng = np.random.default_rng(5)
    base = rng.normal(0.001, 0.004, 900)
    noisy = base + rng.normal(0.0, 0.010, 900)
    noisy -= noisy.mean() - base.mean()          # equalise the means exactly

    def final_sharpe(series):
        d = DifferentialSharpe(eta=0.02, warmup=50)
        for x in series:
            d.update(x)
        return d.sharpe

    assert final_sharpe(base) > final_sharpe(noisy)


def test_clipping_bounds_the_reward():
    d = DifferentialSharpe(eta=0.5, warmup=3, clip=2.0)
    vals = [d.update(x) for x in [0.01, -0.01, 0.02, 0.5, -0.9, 1.5, -2.0]]
    assert all(abs(v) <= 2.0 for v in vals)


def test_rejects_bad_eta():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="eta"):
            DifferentialSharpe(eta=bad)


def test_evaluation_metrics():
    r = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert sortino(r) > sharpe(r)                 # upside vol not punished
    assert max_drawdown(np.array([0.1, -0.5, 0.1])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([0.01, 0.01])) == pytest.approx(0.0)
    assert sharpe(np.array([0.01])) == 0.0        # degenerate input
    assert sortino(np.array([0.01, 0.02])) == 0.0 # no downside at all
