"""Monte Carlo evaluation of a decision model.

Three questions are answered over the same set of samples:

* expected utility per lever setting (policy), with constraint satisfaction;
* sensitivity: which external drives the outcome (tornado swings and a first-order
  variance share estimated by binning);
* value of information: how much better the decision would be if an uncertain input
  were known (EVPPI) or partially revealed by an imperfect study (EVSI).

All policies are evaluated with common random numbers: the externals are sampled once
and reused, so differences between policies are not polluted by sampling noise.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .model import DecisionModel


def sample_externals(
    model: DecisionModel, n: int, seed: int, data: Mapping[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """Sample every external with a child generator per node.

    Each external gets its own stream derived from (seed, node index), so adding a new
    external to a model does not change the samples drawn for the existing ones.
    Externals present in ``data`` are taken from it instead (observed cases).
    """
    data = data or {}
    children = np.random.SeedSequence(seed).spawn(len(model.externals))
    out: dict[str, np.ndarray] = {}
    for node, child in zip(model.externals, children, strict=True):
        if node.name in data:
            out[node.name] = np.asarray(data[node.name])
        else:
            out[node.name] = node.distribution.sample(np.random.default_rng(child), n)
    lengths = {len(v) for v in out.values()}
    if len(lengths) > 1:
        raise ValueError(f"externals have inconsistent lengths {sorted(lengths)}")
    return out


def lever_grid(model: DecisionModel) -> list[dict[str, Any]]:
    names = [lv.name for lv in model.levers]
    return [
        dict(zip(names, combo, strict=True))
        for combo in itertools.product(*[lv.options for lv in model.levers])
    ]


@dataclass
class PolicyResult:
    levers: dict[str, Any]
    utility: np.ndarray  # per-sample utility
    outcomes: dict[str, np.ndarray]
    constraint_rates: dict[str, float]
    feasible: bool

    @property
    def expected_utility(self) -> float:
        return float(np.mean(self.utility))

    @property
    def sd(self) -> float:
        return float(np.std(self.utility))

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.utility, q))

    def label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.levers.items())


@dataclass
class SimulationResult:
    model: DecisionModel
    externals: dict[str, np.ndarray]
    policies: list[PolicyResult]
    seed: int

    @property
    def n(self) -> int:
        return len(next(iter(self.externals.values())))

    def feasible(self) -> list[PolicyResult]:
        return [p for p in self.policies if p.feasible]

    def ranked(self) -> list[PolicyResult]:
        return sorted(self.feasible(), key=lambda p: p.expected_utility, reverse=True)

    def best(self) -> PolicyResult:
        ranked = self.ranked()
        if not ranked:
            raise ValueError("no lever setting satisfies the constraints")
        return ranked[0]

    def utility_matrix(self) -> np.ndarray:
        """Feasible policies x samples."""
        return np.vstack([p.utility for p in self.feasible()])


def evaluate_policy(
    model: DecisionModel, levers: Mapping[str, Any], externals: Mapping[str, np.ndarray]
) -> PolicyResult:
    values = model.evaluate(levers, externals)
    n = len(next(iter(externals.values())))
    utility = np.broadcast_to(np.asarray(model.utility(values), dtype=float), (n,)).copy()
    rates: dict[str, float] = {}
    feasible = True
    for c in model.constraints:
        ok = np.broadcast_to(np.asarray(values[c.name], dtype=bool), (n,))
        rates[c.name] = float(np.mean(ok))
        feasible &= rates[c.name] >= c.min_probability
    outcomes = {
        o.name: np.broadcast_to(np.asarray(values[o.name], dtype=float), (n,)) for o in model.outcomes
    }
    return PolicyResult(dict(levers), utility, outcomes, rates, bool(feasible))


def simulate(
    model: DecisionModel,
    n: int = 10_000,
    seed: int = 0,
    policies: Iterable[Mapping[str, Any]] | None = None,
    data: Mapping[str, np.ndarray] | None = None,
) -> SimulationResult:
    """Evaluate every lever setting (or the given ``policies``) on shared samples.

    With ``data``, externals found there are used as-is and ``n`` is taken from them.
    """
    model.validate()
    if data:
        n = len(next(iter(data.values())))
    externals = sample_externals(model, n, seed, data)
    grid = list(policies) if policies is not None else lever_grid(model)
    results = [evaluate_policy(model, levers, externals) for levers in grid]
    return SimulationResult(model, externals, results, seed)


# -- sensitivity ---------------------------------------------------------------


@dataclass
class TornadoBar:
    external: str
    low_value: float
    high_value: float
    utility_at_low: float
    utility_at_high: float
    variance_share: float

    @property
    def swing(self) -> float:
        return abs(self.utility_at_high - self.utility_at_low)


def first_order_share(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Estimate Var(E[Y|X]) / Var(Y) by averaging Y within quantile bins of X.

    This is the first-order (main effect) sensitivity index. Discrete inputs are
    grouped by their distinct values instead of quantile bins.
    """
    var_y = float(np.var(y))
    if var_y == 0.0:
        return 0.0
    groups = _group_ids(x, bins)
    means = np.bincount(groups, weights=y) / np.maximum(np.bincount(groups), 1)
    weights = np.bincount(groups) / len(y)
    share = float(np.sum(weights * (means - y.mean()) ** 2) / var_y)
    # Binning adds a small positive bias of roughly (bins - 1) / n; remove it.
    share -= (len(means) - 1) / len(y)
    return float(np.clip(share, 0.0, 1.0))


def _group_ids(x: np.ndarray, bins: int) -> np.ndarray:
    distinct = np.unique(x)
    if len(distinct) <= bins:
        return np.searchsorted(distinct, x)
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(edges, x, side="right")


def tornado(
    model: DecisionModel,
    levers: Mapping[str, Any],
    n: int = 10_000,
    seed: int = 0,
    low_q: float = 0.1,
    high_q: float = 0.9,
    externals: Iterable[str] | None = None,
) -> list[TornadoBar]:
    """Tornado analysis for one lever setting.

    For each external X, the expected utility is recomputed with X pinned at its
    ``low_q`` and ``high_q`` quantiles while every other external keeps its full
    distribution. Pinning one input and leaving the rest random (rather than pinning
    the rest at their medians) keeps interactions and non-linear payoffs in the bars.
    """
    samples = sample_externals(model, n, seed)
    base = evaluate_policy(model, levers, samples).utility
    names = list(externals) if externals is not None else [e.name for e in model.externals]
    bars = []
    for name in names:
        x = samples[name]
        lo, hi = float(np.quantile(x, low_q)), float(np.quantile(x, high_q))
        u_lo = evaluate_policy(model, levers, {**samples, name: np.full(n, lo)}).expected_utility
        u_hi = evaluate_policy(model, levers, {**samples, name: np.full(n, hi)}).expected_utility
        bars.append(TornadoBar(name, lo, hi, u_lo, u_hi, first_order_share(x, base)))
    return sorted(bars, key=lambda b: b.swing, reverse=True)


# -- value of information --------------------------------------------------------


@dataclass
class VoiResult:
    external: str
    prior_value: float  # E[U] of the best policy chosen without the information
    posterior_value: float  # E[U] when the policy is chosen after seeing the signal
    prior_choice: dict[str, Any]
    choices_by_signal: dict[Any, dict[str, Any]] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return max(self.posterior_value - self.prior_value, 0.0)


SignalFn = Callable[[np.random.Generator, np.ndarray], np.ndarray]


def value_of_information(
    result: SimulationResult,
    external: str,
    signal: SignalFn | None = None,
    bins: int = 20,
    seed: int = 1,
) -> VoiResult:
    """Expected value of (partial perfect or sample) information about one external.

    Without ``signal`` this is EVPPI: the decision maker learns which quantile bin the
    external falls into (a close approximation to learning its exact value). With
    ``signal`` it is EVSI: ``signal(rng, x)`` maps the true external samples to what an
    imperfect study would report, and the policy is re-chosen for each report.
    """
    feasible = result.feasible()
    if not feasible:
        raise ValueError("no feasible policies")
    u = result.utility_matrix()
    x = result.externals[external]
    if signal is None:
        groups = _group_ids(x, bins)
        labels: np.ndarray = groups
    else:
        labels = np.asarray(signal(np.random.default_rng(seed), x))
        _, groups = np.unique(labels, return_inverse=True)
    prior_idx = int(np.argmax(u.mean(axis=1)))
    posterior = 0.0
    choices: dict[Any, dict[str, Any]] = {}
    n = u.shape[1]
    for g in np.unique(groups):
        mask = groups == g
        means = u[:, mask].mean(axis=1)
        best = int(np.argmax(means))
        posterior += means[best] * mask.sum() / n
        key = labels[mask][0].item() if signal is not None else int(g)
        choices[key] = dict(feasible[best].levers)
    return VoiResult(
        external,
        float(u[prior_idx].mean()),
        float(posterior),
        dict(feasible[prior_idx].levers),
        choices,
    )


def evpi(result: SimulationResult) -> float:
    """Expected value of perfect information on all externals at once (an upper bound)."""
    u = result.utility_matrix()
    return float(u.max(axis=0).mean() - u.mean(axis=1).max())
