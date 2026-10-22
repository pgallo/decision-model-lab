"""Human-in-the-loop routing and cost-sensitive threshold optimisation.

A calibrated probability p that a case is "positive" (fraudulent, risky, true) is
routed to one of three lanes:

    p >= act_threshold                  -> AUTO_ACT     (system acts on its own)
    review_threshold <= p < act_thr.    -> HUMAN_REVIEW (an analyst decides)
    p < review_threshold                -> REJECT       (no action: dismiss the alert)

Thresholds are chosen by minimising expected cost under a cost matrix rather than by
maximising accuracy, because the errors are rarely symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Route(StrEnum):
    AUTO_ACT = "auto-act"
    HUMAN_REVIEW = "human-review"
    REJECT = "reject"


@dataclass(frozen=True)
class CostMatrix:
    """Costs per case. Use negative values for benefits."""

    false_positive: float
    false_negative: float
    true_positive: float = 0.0
    true_negative: float = 0.0
    review: float = 0.0  # cost of one human review (human assumed to decide correctly)

    def bayes_threshold(self) -> float:
        """Probability above which acting has lower expected cost than not acting.

        Act when p*c_tp + (1-p)*c_fp < p*c_fn + (1-p)*c_tn, which rearranges to
        p > (c_fp - c_tn) / ((c_fp - c_tn) + (c_fn - c_tp)).
        """
        num = self.false_positive - self.true_negative
        den = num + self.false_negative - self.true_positive
        if den <= 0:
            raise ValueError("cost matrix does not favour acting on positives")
        return num / den


@dataclass(frozen=True)
class RoutingPolicy:
    act_threshold: float
    review_threshold: float

    def __post_init__(self):
        if not 0.0 <= self.review_threshold <= self.act_threshold <= 1.0 + 1e-12:
            raise ValueError("need 0 <= review_threshold <= act_threshold <= 1")

    def route(self, p: float) -> Route:
        if p >= self.act_threshold:
            return Route.AUTO_ACT
        if p >= self.review_threshold:
            return Route.HUMAN_REVIEW
        return Route.REJECT

    def route_many(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p)
        return np.where(
            p >= self.act_threshold,
            Route.AUTO_ACT.value,
            np.where(p >= self.review_threshold, Route.HUMAN_REVIEW.value, Route.REJECT.value),
        )


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    total_cost: float
    flagged: int
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        return self.true_positives / self.flagged if self.flagged else 0.0

    @property
    def recall(self) -> float:
        positives = self.true_positives + self.false_negatives
        return self.true_positives / positives if positives else 0.0


def _candidates(scores: np.ndarray) -> np.ndarray:
    # Every distinct score is a possible cut, plus one above the max (flag nothing).
    distinct = np.unique(scores)
    return np.append(distinct, np.nextafter(distinct[-1], np.inf))


def binary_cost(scores: np.ndarray, y: np.ndarray, threshold: float, costs: CostMatrix) -> ThresholdResult:
    scores, y = np.asarray(scores, dtype=float), np.asarray(y).astype(bool)
    flag = scores >= threshold
    tp = int(np.sum(flag & y))
    fp = int(np.sum(flag & ~y))
    fn = int(np.sum(~flag & y))
    tn = int(np.sum(~flag & ~y))
    total = (
        tp * costs.true_positive
        + fp * costs.false_positive
        + fn * costs.false_negative
        + tn * costs.true_negative
    )
    return ThresholdResult(float(threshold), float(total), int(flag.sum()), tp, fp, fn)


def optimise_threshold(
    scores: np.ndarray, y: np.ndarray, costs: CostMatrix, max_flagged: int | None = None
) -> ThresholdResult:
    """Empirical cost-minimising threshold, optionally under a capacity limit.

    ``max_flagged`` caps how many cases may be flagged (for example investigator
    capacity). Ties on cost are broken towards the higher threshold, which flags fewer
    cases for the same cost.
    """
    scores = np.asarray(scores, dtype=float)
    best: ThresholdResult | None = None
    for t in _candidates(scores):
        r = binary_cost(scores, y, t, costs)
        if max_flagged is not None and r.flagged > max_flagged:
            continue
        if (
            best is None
            or r.total_cost < best.total_cost
            or (r.total_cost == best.total_cost and r.threshold > best.threshold)
        ):
            best = r
    assert best is not None  # flagging nothing is always feasible
    return best


@dataclass(frozen=True)
class RoutingResult:
    policy: RoutingPolicy
    total_cost: float
    counts: dict[str, int]
    errors: dict[str, int]


def routing_cost(p: np.ndarray, y: np.ndarray, policy: RoutingPolicy, costs: CostMatrix) -> RoutingResult:
    p, y = np.asarray(p, dtype=float), np.asarray(y).astype(bool)
    act = p >= policy.act_threshold
    review = (p >= policy.review_threshold) & ~act
    reject = ~act & ~review
    # Reviewed cases are resolved correctly, at the review cost plus the outcome cost.
    total = (
        np.sum(act & y) * costs.true_positive
        + np.sum(act & ~y) * costs.false_positive
        + np.sum(review) * costs.review
        + np.sum(review & y) * costs.true_positive
        + np.sum(review & ~y) * costs.true_negative
        + np.sum(reject & y) * costs.false_negative
        + np.sum(reject & ~y) * costs.true_negative
    )
    counts = {
        Route.AUTO_ACT.value: int(act.sum()),
        Route.HUMAN_REVIEW.value: int(review.sum()),
        Route.REJECT.value: int(reject.sum()),
    }
    errors = {
        "auto_false_positive": int(np.sum(act & ~y)),
        "missed_positive": int(np.sum(reject & y)),
    }
    return RoutingResult(policy, float(total), counts, errors)


def optimise_routing(
    p: np.ndarray,
    y: np.ndarray,
    costs: CostMatrix,
    review_capacity: int | None = None,
    grid: int = 41,
) -> RoutingResult:
    """Grid search over (review, act) threshold pairs.

    Candidate cuts are the union of score quantiles (dense where most cases are) and an
    even grid on [0, 1] (dense where the rare positives are); with imbalanced classes,
    quantiles alone put every cut in the low tail. ``review_capacity`` limits the number
    of cases sent to humans.
    """
    p = np.asarray(p, dtype=float)
    cuts = np.unique(
        np.concatenate(
            [
                np.quantile(p, np.linspace(0, 1, grid)),
                np.linspace(0, 1, grid),
                [1.0 + 1e-9],
            ]
        )
    )
    best: RoutingResult | None = None
    for i, lo in enumerate(cuts):
        for hi in cuts[i:]:
            r = routing_cost(p, y, RoutingPolicy(float(min(hi, 1.0)), float(min(lo, 1.0))), costs)
            if review_capacity is not None and r.counts[Route.HUMAN_REVIEW.value] > review_capacity:
                continue
            if best is None or r.total_cost < best.total_cost - 1e-9:
                best = r
    assert best is not None
    return best
