"""Evidence fusion and probability calibration.

Binary decisions combine evidence in log-odds space: each item contributes
log(LR), where LR = P(e | H) / P(e | not H). Contributions add, which makes the
fused score directly explainable: the trace lists each item's share of the shift
from prior to posterior.

Calibration maps a raw model score to a probability that can be fed into a cost
model. Brier score and a reliability table measure how honest the probabilities are.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-12


def logit(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def discount_lr(lr: float, reliability: float) -> float:
    """Discount a likelihood ratio by source reliability r in [0, 1].

    The source reports faithfully with probability r and otherwise produces a report
    that carries no information about H. Then P(e|H) = r*a + (1-r)*b and
    P(e|not H) = b, so the effective ratio is r*LR + (1 - r). An r of 1 keeps the
    ratio, an r of 0 collapses it to 1 (no evidence).
    """
    if not 0.0 <= reliability <= 1.0:
        raise ValueError("reliability must be in [0, 1]")
    if lr <= 0:
        raise ValueError("likelihood ratio must be positive")
    return reliability * lr + (1.0 - reliability)


@dataclass(frozen=True)
class Evidence:
    name: str
    likelihood_ratio: float
    reliability: float = 1.0
    detail: str = ""

    @property
    def effective_lr(self) -> float:
        return discount_lr(self.likelihood_ratio, self.reliability)

    @property
    def log_lr(self) -> float:
        return math.log(self.effective_lr)


@dataclass(frozen=True)
class Contribution:
    name: str
    log_lr: float
    detail: str = ""

    @property
    def direction(self) -> str:
        return "supports" if self.log_lr > 0 else "weakens" if self.log_lr < 0 else "neutral"


@dataclass(frozen=True)
class FusionResult:
    prior: float
    posterior: float
    contributions: tuple[Contribution, ...]

    @property
    def log_odds_shift(self) -> float:
        return sum(c.log_lr for c in self.contributions)

    def top(self, k: int = 3) -> list[Contribution]:
        return sorted(self.contributions, key=lambda c: abs(c.log_lr), reverse=True)[:k]


def fuse(prior: float, evidence: Iterable[Evidence]) -> FusionResult:
    """Naive-Bayes fusion of conditionally independent evidence items."""
    if not 0.0 < prior < 1.0:
        raise ValueError("prior must be strictly between 0 and 1")
    contributions = tuple(Contribution(e.name, e.log_lr, e.detail) for e in evidence)
    posterior = float(sigmoid(logit(prior) + sum(c.log_lr for c in contributions)))
    return FusionResult(prior, posterior, contributions)


def update_hypotheses(
    priors: Sequence[float], likelihoods: Sequence[Sequence[float]], reliabilities: Sequence[float]
) -> np.ndarray:
    """Multi-hypothesis Bayesian update.

    ``likelihoods[k][j]`` is P(e_k | H_j). An unreliable source is modelled as a
    mixture with an uninformative report whose probability is the average likelihood
    across hypotheses, the multi-class analogue of :func:`discount_lr`.
    Returns the posterior over hypotheses.
    """
    log_post = np.log(np.asarray(priors, dtype=float))
    for row, r in zip(likelihoods, reliabilities, strict=True):
        lik = np.asarray(row, dtype=float)
        effective = r * lik + (1 - r) * lik.mean()
        log_post += np.log(np.clip(effective, _EPS, None))
    log_post -= log_post.max()
    post = np.exp(log_post)
    return post / post.sum()


# -- calibration -----------------------------------------------------------------


class Calibrator:
    """Platt scaling (logistic on the logit of the score) or isotonic regression."""

    def __init__(self, method: str = "platt"):
        if method not in ("platt", "isotonic"):
            raise ValueError("method must be 'platt' or 'isotonic'")
        self.method = method
        self._model: LogisticRegression | IsotonicRegression | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> Calibrator:
        scores = np.asarray(scores, dtype=float)
        if self.method == "platt":
            self._model = LogisticRegression(C=1e6).fit(self._features(scores), y)
        else:
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("calibrator is not fitted")
        scores = np.asarray(scores, dtype=float)
        if self.method == "platt":
            return self._model.predict_proba(self._features(scores))[:, 1]
        return self._model.predict(scores)

    @staticmethod
    def _features(scores: np.ndarray) -> np.ndarray:
        # Scores in [0, 1] are mapped to log-odds first; a sigmoid of a linear function
        # of the logit is the classic two-parameter Platt correction.
        if scores.min() >= 0.0 and scores.max() <= 1.0:
            return np.asarray(logit(scores)).reshape(-1, 1)
        return scores.reshape(-1, 1)


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    probs, y = np.asarray(probs, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((probs - y) ** 2))


@dataclass(frozen=True)
class ReliabilityRow:
    low: float
    high: float
    count: int
    mean_predicted: float
    observed_rate: float


def reliability_table(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> list[ReliabilityRow]:
    probs, y = np.asarray(probs, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append(
                ReliabilityRow(
                    float(edges[b]),
                    float(edges[b + 1]),
                    int(mask.sum()),
                    float(probs[mask].mean()),
                    float(y[mask].mean()),
                )
            )
    return rows


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    rows = reliability_table(probs, y, bins)
    n = sum(r.count for r in rows)
    return float(sum(r.count * abs(r.mean_predicted - r.observed_rate) for r in rows) / n)
