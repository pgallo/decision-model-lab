"""DMN-style decision tables with UNIQUE, FIRST and COLLECT hit policies.

Input entries are FEEL-lite unary tests, output entries are FEEL-lite expressions.
Evaluation returns both the result and the list of fired rules, so the explanation
layer can report exactly which rows drove a decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from . import feel


class HitPolicy(StrEnum):
    UNIQUE = "UNIQUE"
    FIRST = "FIRST"
    COLLECT = "COLLECT"


class Aggregation(StrEnum):
    LIST = "LIST"
    SUM = "SUM"
    MIN = "MIN"
    MAX = "MAX"
    COUNT = "COUNT"


class HitPolicyViolation(RuntimeError):
    """A UNIQUE table matched more than one rule."""


class NoRuleMatched(RuntimeError):
    """No rule matched and the table has no default output."""


@dataclass(frozen=True)
class Rule:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    annotation: str = ""


@dataclass(frozen=True)
class FiredRule:
    index: int  # 1-based, as in DMN tooling
    annotation: str
    output: dict[str, Any]


@dataclass
class TableResult:
    value: Any
    fired: list[FiredRule] = field(default_factory=list)


@dataclass
class DecisionTable:
    name: str
    inputs: list[str]  # FEEL expressions evaluated against the context
    outputs: list[str]  # output column names
    rules: list[Rule]
    hit_policy: HitPolicy = HitPolicy.UNIQUE
    aggregation: Aggregation = Aggregation.LIST
    default: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.hit_policy = HitPolicy(self.hit_policy)
        self.aggregation = Aggregation(self.aggregation)
        for i, rule in enumerate(self.rules, start=1):
            if len(rule.inputs) != len(self.inputs) or len(rule.outputs) != len(self.outputs):
                raise ValueError(f"{self.name}: rule {i} has the wrong number of entries")
            # Parse eagerly so syntax errors surface at load time, not at decision time.
            for entry in rule.inputs:
                feel.parse_unary_tests(entry)
            for entry in rule.outputs:
                feel.parse(entry)
        if self.aggregation is not Aggregation.LIST and self.hit_policy is not HitPolicy.COLLECT:
            raise ValueError(f"{self.name}: aggregation is only valid with COLLECT")
        if self.aggregation not in (Aggregation.LIST, Aggregation.COUNT) and len(self.outputs) != 1:
            raise ValueError(f"{self.name}: numeric aggregation needs exactly one output")

    @classmethod
    def from_dict(cls, data: dict) -> DecisionTable:
        policy = str(data.get("hit_policy", "UNIQUE")).upper()
        aggregation = "LIST"
        if " " in policy:  # e.g. "COLLECT SUM"
            policy, aggregation = policy.split(None, 1)
        return cls(
            name=data["name"],
            inputs=list(data["inputs"]),
            outputs=list(data["outputs"]),
            rules=[
                Rule(
                    tuple(str(x) for x in r["when"]),
                    tuple(str(x) for x in r["then"]),
                    r.get("annotation", ""),
                )
                for r in data["rules"]
            ],
            hit_policy=HitPolicy(policy),
            aggregation=Aggregation(data.get("aggregation", aggregation).upper()),
            default=data.get("default"),
        )

    def referenced_names(self) -> set[str]:
        names: set[str] = set()
        for expr in self.inputs:
            names |= feel.free_names(feel.parse(expr))
        for rule in self.rules:
            for entry in rule.inputs:
                names |= feel.free_names(feel.parse_unary_tests(entry))
            for entry in rule.outputs:
                names |= feel.free_names(feel.parse(entry))
        return names

    def _matches(self, rule: Rule, values: list[Any], context: dict) -> bool:
        for test, value in zip(rule.inputs, values, strict=True):
            if feel.match_unary(test, value, context) is not True:
                return False
        return True

    def _output(self, rule: Rule, context: dict) -> dict[str, Any]:
        return {
            name: feel.evaluate(expr, context) for name, expr in zip(self.outputs, rule.outputs, strict=True)
        }

    def evaluate(self, context: dict) -> TableResult:
        """Evaluate the table for a single case (scalar context values)."""
        values = [feel.evaluate(expr, context) for expr in self.inputs]
        fired = [
            FiredRule(i, rule.annotation, self._output(rule, context))
            for i, rule in enumerate(self.rules, start=1)
            if self._matches(rule, values, context)
        ]
        if not fired:
            if self.default is None:
                if self.hit_policy is HitPolicy.COLLECT:
                    return TableResult(0 if self.aggregation is Aggregation.COUNT else [], [])
                raise NoRuleMatched(
                    f"{self.name}: no rule matched {dict(zip(self.inputs, values, strict=True))}"
                )
            return TableResult(self._single(self.default), [])

        if self.hit_policy is HitPolicy.UNIQUE:
            if len(fired) > 1:
                ids = ", ".join(str(f.index) for f in fired)
                raise HitPolicyViolation(f"{self.name}: UNIQUE table matched rules {ids}")
            return TableResult(self._single(fired[0].output), fired)
        if self.hit_policy is HitPolicy.FIRST:
            return TableResult(self._single(fired[0].output), fired[:1])
        return TableResult(self._aggregate([f.output for f in fired]), fired)

    def evaluate_many(self, context: dict) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised evaluation over array-valued contexts (UNIQUE, FIRST, COLLECT SUM/MIN/MAX/COUNT).

        Returns ``(values, rule_index)`` where ``rule_index`` is the 1-based index of the
        selected rule (0 when the default applied). For COLLECT the index is the first
        matching rule. Every rule is tested once over the whole batch, so the cost is
        rules x inputs array operations instead of cases x rules x inputs scalar ones.
        """
        if self.hit_policy is HitPolicy.COLLECT and self.aggregation is Aggregation.LIST:
            raise ValueError(f"{self.name}: COLLECT LIST is not vectorised; use evaluate()")
        if len(self.outputs) != 1:
            raise ValueError(f"{self.name}: evaluate_many supports single-output tables")
        values = [feel.evaluate(expr, context) for expr in self.inputs]
        n = max(np.size(v) for v in values)
        masks = []
        for rule in self.rules:
            m = np.ones(n, dtype=bool)
            for test, value in zip(rule.inputs, values, strict=True):
                m &= np.broadcast_to(np.asarray(feel.match_unary(test, value, context), dtype=bool), (n,))
            masks.append(m)
        hits = np.vstack(masks) if masks else np.zeros((0, n), dtype=bool)
        matched = hits.any(axis=0)
        first = np.where(matched, hits.argmax(axis=0) + 1, 0)
        outs = [np.broadcast_to(np.asarray(feel.evaluate(r.outputs[0], context)), (n,)) for r in self.rules]

        if self.hit_policy is HitPolicy.COLLECT:
            if self.aggregation is Aggregation.COUNT:
                return hits.sum(axis=0), first
            stacked = np.vstack(outs).astype(float)
            if self.aggregation is Aggregation.SUM:
                return np.where(hits, stacked, 0.0).sum(axis=0), first
            fill = np.inf if self.aggregation is Aggregation.MIN else -np.inf
            reduce = np.min if self.aggregation is Aggregation.MIN else np.max
            agg = reduce(np.where(hits, stacked, fill), axis=0)
            return np.where(matched, agg, 0.0), first

        if self.hit_policy is HitPolicy.UNIQUE and (hits.sum(axis=0) > 1).any():
            case = int(np.argmax(hits.sum(axis=0) > 1))
            ids = ", ".join(str(i + 1) for i in np.flatnonzero(hits[:, case]))
            raise HitPolicyViolation(f"{self.name}: UNIQUE table matched rules {ids} for case {case}")
        if not matched.all() and self.default is None:
            case = int(np.argmin(matched))
            raise NoRuleMatched(f"{self.name}: no rule matched case {case}")
        result = np.empty(n, dtype=object)
        if self.default is not None:
            result[:] = self._single(self.default)
        for i, out in enumerate(outs, start=1):
            sel = first == i
            result[sel] = out[sel]
        # Return a typed array (str or float) rather than object when possible.
        return np.asarray(result.tolist()), first

    def _single(self, output: dict[str, Any]) -> Any:
        return next(iter(output.values())) if len(self.outputs) == 1 else dict(output)

    def _aggregate(self, outputs: list[dict[str, Any]]) -> Any:
        if self.aggregation is Aggregation.COUNT:
            return len(outputs)
        if self.aggregation is Aggregation.LIST:
            return [self._single(o) for o in outputs]
        values = [next(iter(o.values())) for o in outputs]
        if self.aggregation is Aggregation.SUM:
            return sum(values)
        if self.aggregation is Aggregation.MIN:
            return min(values)
        return max(values)
