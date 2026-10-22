"""Probability distributions for external (uncertain) inputs.

Each distribution is a small frozen dataclass with ``sample(rng, n)``, so a model
stays declarative (YAML friendly) and sampling is reproducible from one seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Distribution:
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        raise NotImplementedError

    @property
    def is_discrete(self) -> bool:
        return False


@dataclass(frozen=True)
class Constant(Distribution):
    value: float

    def sample(self, rng, n):
        return np.full(n, float(self.value))

    @property
    def is_discrete(self) -> bool:
        return True


@dataclass(frozen=True)
class Normal(Distribution):
    mean: float
    sd: float

    def sample(self, rng, n):
        return rng.normal(self.mean, self.sd, n)


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Parameterised by the median and the geometric standard deviation factor."""

    median: float
    gsd: float

    def sample(self, rng, n):
        return rng.lognormal(np.log(self.median), np.log(self.gsd), n)


@dataclass(frozen=True)
class Uniform(Distribution):
    low: float
    high: float

    def sample(self, rng, n):
        return rng.uniform(self.low, self.high, n)


@dataclass(frozen=True)
class Triangular(Distribution):
    low: float
    mode: float
    high: float

    def sample(self, rng, n):
        return rng.triangular(self.low, self.mode, self.high, n)


@dataclass(frozen=True)
class Beta(Distribution):
    a: float
    b: float

    def sample(self, rng, n):
        return rng.beta(self.a, self.b, n)


@dataclass(frozen=True)
class Bernoulli(Distribution):
    p: float

    def sample(self, rng, n):
        return (rng.random(n) < self.p).astype(float)

    @property
    def is_discrete(self) -> bool:
        return True


@dataclass(frozen=True)
class Discrete(Distribution):
    values: tuple[float, ...]
    probs: tuple[float, ...]

    def __post_init__(self):
        if len(self.values) != len(self.probs) or not np.isclose(sum(self.probs), 1.0):
            raise ValueError("discrete distribution needs matching values and probs summing to 1")

    def sample(self, rng, n):
        return rng.choice(np.asarray(self.values, dtype=float), size=n, p=np.asarray(self.probs))

    @property
    def is_discrete(self) -> bool:
        return True


@dataclass(frozen=True)
class Data(Distribution):
    """Marks an external whose values come from a dataset at evaluation time.

    Used when a model is run over observed cases (a claims backlog, a day of
    transactions) instead of over sampled scenarios.
    """

    source: str = ""

    def sample(self, rng, n):
        raise ValueError("a 'data' external must be supplied by the caller, not sampled")


_REGISTRY: dict[str, type[Distribution]] = {
    "constant": Constant,
    "normal": Normal,
    "lognormal": LogNormal,
    "uniform": Uniform,
    "triangular": Triangular,
    "beta": Beta,
    "bernoulli": Bernoulli,
    "discrete": Discrete,
    "data": Data,
}


def from_spec(spec: dict | float | int) -> Distribution:
    """Build a distribution from a mapping like ``{"dist": "normal", "mean": 0, "sd": 1}``."""
    if isinstance(spec, int | float):
        return Constant(float(spec))
    params = dict(spec)
    kind = params.pop("dist", None)
    if kind not in _REGISTRY:
        raise ValueError(f"unknown distribution {kind!r}; choose from {sorted(_REGISTRY)}")
    if kind == "discrete":
        return Discrete(tuple(params["values"]), tuple(params["probs"]))
    return _REGISTRY[kind](**params)
