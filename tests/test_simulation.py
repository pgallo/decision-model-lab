import numpy as np
import pytest

from decision_lab.core.distributions import Bernoulli, Normal, Uniform
from decision_lab.core.model import DecisionModel
from decision_lab.core.simulation import (
    evpi,
    first_order_share,
    sample_externals,
    simulate,
    tornado,
    value_of_information,
)


def newsvendor() -> DecisionModel:
    m = DecisionModel("newsvendor", parameters={"price": 10, "cost": 6})
    m.lever("order", [0, 50, 100, 150])
    m.external("demand", Normal(100, 30))
    m.external("noise", Uniform(-1, 1))
    m.intermediate("sold", "min(order, max(demand, 0))")
    m.outcome("profit", "sold * price - order * cost + noise")
    m.constraint("cap", "order <= 120")
    return m


def test_same_seed_same_result():
    a = simulate(newsvendor(), n=2000, seed=42)
    b = simulate(newsvendor(), n=2000, seed=42)
    c = simulate(newsvendor(), n=2000, seed=43)
    assert [p.expected_utility for p in a.policies] == [p.expected_utility for p in b.policies]
    assert a.policies[2].expected_utility != c.policies[2].expected_utility


def test_adding_an_external_does_not_change_existing_streams():
    m = newsvendor()
    before = sample_externals(m, 100, seed=1)
    m.external("extra", Normal(0, 1))
    after = sample_externals(m, 100, seed=1)
    np.testing.assert_array_equal(before["demand"], after["demand"])


def test_constraints_and_ranking():
    result = simulate(newsvendor(), n=20_000, seed=0)
    feasible = {p.levers["order"] for p in result.feasible()}
    assert feasible == {0, 50, 100}
    # Critical ratio (10 - 6) / 10 = 0.4 -> optimum near the 40th percentile (~92);
    # among the grid, 100 beats 50.
    assert result.best().levers == {"order": 100}
    assert result.best().expected_utility == pytest.approx(282, rel=0.03)


def test_tornado_ranks_the_real_driver_first():
    bars = tornado(newsvendor(), {"order": 100}, n=5000, seed=0)
    assert bars[0].external == "demand"
    assert bars[0].swing > 100 * bars[1].swing
    assert bars[0].variance_share > 0.9


def test_first_order_share_known_case():
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=50_000), rng.normal(size=50_000)
    y = 2 * x1 + x2  # Var = 5, shares 0.8 and 0.2
    assert first_order_share(x1, y, bins=50) == pytest.approx(0.8, abs=0.02)
    assert first_order_share(x2, y, bins=50) == pytest.approx(0.2, abs=0.02)


def gamble() -> DecisionModel:
    """Invest 1 to win 3 if a coin with p=0.5 comes up; otherwise lose the stake."""
    m = DecisionModel("gamble")
    m.lever("invest", [0, 1])
    m.external("win", Bernoulli(0.5))
    m.outcome("payoff", "invest * (win * 3 - 1)")
    return m


def test_value_of_information_hand_checked():
    # Prior: invest (E = 0.5 * 2 - 0.5 * 1 = 0.5). Perfect info: invest only on a win,
    # E = 0.5 * 2 = 1.0. EVPI = 0.5.
    result = simulate(gamble(), n=100_000, seed=3)
    voi = value_of_information(result, "win")
    assert voi.prior_choice == {"invest": 1}
    assert voi.value == pytest.approx(0.5, abs=0.02)
    assert evpi(result) == pytest.approx(0.5, abs=0.02)


def test_imperfect_study_is_worth_less_than_perfect_information():
    result = simulate(gamble(), n=100_000, seed=3)

    def study(rng, x):  # reports the truth 80 percent of the time
        return np.where(rng.random(len(x)) < 0.8, x, 1 - x)

    evsi = value_of_information(result, "win", signal=study)
    # Positive readout: P(win) = 0.8, invest, E = 0.8*2 - 0.2 = 1.4; negative readout:
    # P(win) = 0.2, E(invest) = 0.4 - 0.8 < 0, stay out. EVSI = 0.5*1.4 + 0.5*0 - 0.5 = 0.2.
    assert evsi.value == pytest.approx(0.2, abs=0.02)
    assert evsi.choices_by_signal == {0.0: {"invest": 0}, 1.0: {"invest": 1}}


def test_observed_data_externals():
    m = DecisionModel.from_yaml_text(
        """
name: cases
nodes:
  - {name: cut, kind: lever, options: [0.3, 0.7]}
  - {name: p, kind: external, distribution: {dist: data}}
  - {name: flagged, kind: outcome, expr: "if p >= cut then 1 else 0"}
"""
    )
    result = simulate(m, data={"p": np.array([0.1, 0.5, 0.9])})
    assert result.n == 3
    assert [p.expected_utility for p in result.policies] == pytest.approx([2 / 3, 1 / 3])
    with pytest.raises(ValueError, match="supplied by the caller"):
        simulate(m, n=10)
