import numpy as np
import pytest

from decision_lab.core.policy import (
    CostMatrix,
    Route,
    RoutingPolicy,
    binary_cost,
    optimise_routing,
    optimise_threshold,
    routing_cost,
)

# Hand-checked case: six scored cases, positives at 0.9, 0.6 and 0.3.
SCORES = np.array([0.9, 0.8, 0.6, 0.4, 0.3, 0.1])
LABELS = np.array([1, 0, 1, 0, 1, 0])


def test_bayes_threshold():
    assert CostMatrix(false_positive=1, false_negative=9).bayes_threshold() == pytest.approx(0.1)
    assert CostMatrix(false_positive=5, false_negative=5).bayes_threshold() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        CostMatrix(false_positive=1, false_negative=-5).bayes_threshold()


def test_threshold_optimiser_hand_checked():
    # FP costs 1, FN costs 3. Cost by cut (flag score >= t):
    #   t=0.1: 3 FP          -> 3      t=0.3: 2 FP          -> 2
    #   t=0.4: 2 FP + 1 FN   -> 5      t=0.6: 1 FP + 1 FN   -> 4
    #   t=0.8: 1 FP + 2 FN   -> 7      t=0.9: 2 FN          -> 6
    #   above max: 3 FN      -> 9
    costs = CostMatrix(false_positive=1, false_negative=3)
    best = optimise_threshold(SCORES, LABELS, costs)
    assert best.threshold == pytest.approx(0.3)
    assert best.total_cost == 2
    assert (best.true_positives, best.false_positives, best.false_negatives) == (3, 2, 0)
    assert binary_cost(SCORES, LABELS, 0.6, costs).total_cost == 4


def test_threshold_optimiser_with_capacity():
    costs = CostMatrix(false_positive=1, false_negative=3)
    # At most 3 flags: candidates t=0.6 (cost 4, 3 flags), 0.8 (7), 0.9 (6), none (9).
    best = optimise_threshold(SCORES, LABELS, costs, max_flagged=3)
    assert best.threshold == pytest.approx(0.6)
    assert best.flagged == 3 and best.total_cost == 4


def test_routing_policy_lanes():
    policy = RoutingPolicy(act_threshold=0.8, review_threshold=0.3)
    assert policy.route(0.95) is Route.AUTO_ACT
    assert policy.route(0.5) is Route.HUMAN_REVIEW
    assert policy.route(0.1) is Route.REJECT
    np.testing.assert_array_equal(
        policy.route_many(np.array([0.9, 0.3, 0.29])), ["auto-act", "human-review", "reject"]
    )
    with pytest.raises(ValueError):
        RoutingPolicy(act_threshold=0.2, review_threshold=0.5)


def test_routing_cost_hand_checked():
    costs = CostMatrix(false_positive=10, false_negative=20, review=2)
    r = routing_cost(SCORES, LABELS, RoutingPolicy(0.85, 0.35), costs)
    # auto: 0.9 (TP). review: 0.8, 0.6, 0.4 -> 3 * 2. reject: 0.3 (FN, 20), 0.1 (TN).
    assert r.total_cost == 26
    assert r.counts == {"auto-act": 1, "human-review": 3, "reject": 2}
    assert r.errors == {"auto_false_positive": 0, "missed_positive": 1}


def test_routing_optimiser_prefers_review_when_cheap_and_respects_capacity():
    costs = CostMatrix(false_positive=10, false_negative=20, review=1)
    best = optimise_routing(SCORES, LABELS, costs)
    # Optimum: auto-act 0.9 (a true positive, cost 0), review 0.8, 0.6, 0.4, 0.3 (4 x 1),
    # reject 0.1 (a true negative). Total 4, no errors.
    assert best.total_cost == 4
    assert best.counts == {"auto-act": 1, "human-review": 4, "reject": 1}
    capped = optimise_routing(SCORES, LABELS, costs, review_capacity=2)
    assert capped.counts["human-review"] <= 2
    assert capped.total_cost > best.total_cost
