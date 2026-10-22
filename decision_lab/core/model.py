"""Typed decision model: a DAG of levers, externals, intermediates, outcomes, constraints.

The node taxonomy follows causal decision diagrams and the DMN decision requirements
graph: levers are the choices we control, externals are uncertain inputs we do not,
intermediates are deterministic functions, outcomes carry utility, and constraints
restrict which lever settings are admissible.

Edges are not declared by hand: they are derived from the names referenced in each
node's FEEL-lite expression (or decision table), so the graph cannot drift away from
the logic it describes. Python callables declare ``depends_on`` explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import feel
from .decision_table import DecisionTable
from .distributions import Distribution, from_spec


class NodeKind(StrEnum):
    LEVER = "lever"
    EXTERNAL = "external"
    INTERMEDIATE = "intermediate"
    OUTCOME = "outcome"
    CONSTRAINT = "constraint"


class ModelError(ValueError):
    """Raised when a decision model is structurally invalid."""


# Which node kinds may feed which. Constraints are sinks; levers and externals are roots.
_ALLOWED_PARENTS = {
    NodeKind.LEVER: set(),
    NodeKind.EXTERNAL: set(),
    NodeKind.INTERMEDIATE: {NodeKind.LEVER, NodeKind.EXTERNAL, NodeKind.INTERMEDIATE},
    NodeKind.OUTCOME: {NodeKind.LEVER, NodeKind.EXTERNAL, NodeKind.INTERMEDIATE, NodeKind.OUTCOME},
    NodeKind.CONSTRAINT: {
        NodeKind.LEVER,
        NodeKind.EXTERNAL,
        NodeKind.INTERMEDIATE,
        NodeKind.OUTCOME,
    },
}


@dataclass
class Node:
    name: str
    kind: NodeKind
    description: str = ""
    # lever
    options: list[Any] = field(default_factory=list)
    # external
    distribution: Distribution | None = None
    # intermediate / outcome / constraint
    expr: str | None = None
    table: DecisionTable | None = None
    func: Callable[..., Any] | None = None
    depends_on: list[str] = field(default_factory=list)
    # outcome: utility weight applied to the outcome value (1.0 = maximise, -1.0 = minimise)
    utility_weight: float = 1.0
    # constraint: minimum share of Monte Carlo samples in which it must hold
    min_probability: float = 1.0

    def parents(self, known: Mapping[str, Any]) -> set[str]:
        if self.func is not None:
            return set(self.depends_on)
        names: set[str] = set()
        if self.expr is not None:
            names |= feel.free_names(feel.parse(self.expr))
        if self.table is not None:
            names |= self.table.referenced_names()
        # Parameters are constants, not graph nodes.
        return {n for n in names if n not in known}


@dataclass
class DecisionModel:
    name: str
    nodes: dict[str, Node] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    # -- construction -------------------------------------------------------
    def add(self, node: Node) -> DecisionModel:
        if node.name in self.nodes or node.name in self.parameters:
            raise ModelError(f"duplicate name {node.name!r}")
        self.nodes[node.name] = node
        return self

    def lever(self, name: str, options: list[Any], description: str = "") -> DecisionModel:
        return self.add(Node(name, NodeKind.LEVER, description, options=list(options)))

    def external(self, name: str, dist: Distribution | dict, description: str = "") -> DecisionModel:
        d = dist if isinstance(dist, Distribution) else from_spec(dist)
        return self.add(Node(name, NodeKind.EXTERNAL, description, distribution=d))

    def intermediate(self, name: str, expr: str | None = None, **kw) -> DecisionModel:
        return self.add(Node(name, NodeKind.INTERMEDIATE, expr=expr, **kw))

    def outcome(self, name: str, expr: str | None = None, utility_weight: float = 1.0, **kw):
        return self.add(Node(name, NodeKind.OUTCOME, expr=expr, utility_weight=utility_weight, **kw))

    def constraint(self, name: str, expr: str, min_probability: float = 1.0, **kw):
        return self.add(Node(name, NodeKind.CONSTRAINT, expr=expr, min_probability=min_probability, **kw))

    # -- structure ----------------------------------------------------------
    def of_kind(self, kind: NodeKind) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind is kind]

    @property
    def levers(self) -> list[Node]:
        return self.of_kind(NodeKind.LEVER)

    @property
    def externals(self) -> list[Node]:
        return self.of_kind(NodeKind.EXTERNAL)

    @property
    def outcomes(self) -> list[Node]:
        return self.of_kind(NodeKind.OUTCOME)

    @property
    def constraints(self) -> list[Node]:
        return self.of_kind(NodeKind.CONSTRAINT)

    def edges(self) -> list[tuple[str, str]]:
        return [(p, n.name) for n in self.nodes.values() for p in sorted(n.parents(self.parameters))]

    def validate(self) -> list[str]:
        """Check references, node typing and acyclicity. Returns the topological order."""
        errors: list[str] = []
        for node in self.nodes.values():
            if node.kind is NodeKind.LEVER and not node.options:
                errors.append(f"lever {node.name!r} has no options")
            if node.kind is NodeKind.EXTERNAL and node.distribution is None:
                errors.append(f"external {node.name!r} has no distribution")
            if node.kind in (NodeKind.INTERMEDIATE, NodeKind.OUTCOME, NodeKind.CONSTRAINT):
                defined = sum(x is not None for x in (node.expr, node.table, node.func))
                if defined != 1:
                    errors.append(f"{node.kind} {node.name!r} needs exactly one of expr/table/func")
                    continue
            try:
                parents = node.parents(self.parameters)
            except feel.FeelError as exc:
                errors.append(f"{node.name!r}: {exc}")
                continue
            for parent in sorted(parents):
                if parent not in self.nodes:
                    errors.append(f"{node.name!r} references unknown name {parent!r}")
                elif self.nodes[parent].kind not in _ALLOWED_PARENTS[node.kind]:
                    errors.append(
                        f"{node.kind} {node.name!r} may not depend on {self.nodes[parent].kind} {parent!r}"
                    )
        if not self.outcomes:
            errors.append("model has no outcome node")
        if errors:
            raise ModelError("; ".join(errors))
        return self._toposort()

    def _toposort(self) -> list[str]:
        # Kahn's algorithm; insertion order breaks ties so the order is deterministic.
        parents = {n: self.nodes[n].parents(self.parameters) for n in self.nodes}
        remaining = {n: set(ps) for n, ps in parents.items()}
        order: list[str] = []
        ready = [n for n, ps in remaining.items() if not ps]
        while ready:
            current = ready.pop(0)
            order.append(current)
            for n, ps in remaining.items():
                if current in ps:
                    ps.discard(current)
                    if not ps and n not in order and n not in ready:
                        ready.append(n)
        if len(order) != len(self.nodes):
            stuck = sorted(set(self.nodes) - set(order))
            raise ModelError(f"cycle detected among nodes {stuck}")
        return order

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, levers: Mapping[str, Any], externals: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate every node given lever settings and external values (scalars or arrays)."""
        ctx: dict[str, Any] = dict(self.parameters)
        for node in self.levers:
            if node.name not in levers:
                raise ModelError(f"missing setting for lever {node.name!r}")
            ctx[node.name] = levers[node.name]
        for node in self.externals:
            ctx[node.name] = externals[node.name]
        for name in self.validate():
            node = self.nodes[name]
            if node.kind in (NodeKind.LEVER, NodeKind.EXTERNAL):
                continue
            if node.func is not None:
                ctx[name] = node.func(*[ctx[p] for p in node.depends_on])
            elif node.table is not None:
                ctx[name] = _evaluate_table(node.table, ctx)
            else:
                ctx[name] = feel.evaluate(node.expr, ctx)
        return ctx

    def utility(self, values: Mapping[str, Any]) -> Any:
        return sum(node.utility_weight * values[node.name] for node in self.outcomes)

    # -- serialisation ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> DecisionModel:
        model = cls(
            data["name"],
            parameters=dict(data.get("parameters", {})),
            description=data.get("description", ""),
        )
        for spec in data.get("nodes", []):
            kind = NodeKind(spec["kind"])
            table = DecisionTable.from_dict(spec["table"]) if "table" in spec else None
            model.add(
                Node(
                    name=spec["name"],
                    kind=kind,
                    description=spec.get("description", ""),
                    options=list(spec.get("options", [])),
                    distribution=from_spec(spec["distribution"]) if "distribution" in spec else None,
                    expr=spec.get("expr"),
                    table=table,
                    utility_weight=float(spec.get("utility_weight", 1.0)),
                    min_probability=float(spec.get("min_probability", 1.0)),
                )
            )
        model.validate()
        return model

    @classmethod
    def from_yaml(cls, path: str | Path) -> DecisionModel:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def from_yaml_text(cls, text: str) -> DecisionModel:
        return cls.from_dict(yaml.safe_load(text))

    def to_mermaid(self, include: set[str] | None = None) -> str:
        """Mermaid flowchart of the graph, optionally restricted to a subset of nodes."""
        shapes = {
            NodeKind.LEVER: ("[/", "/]"),
            NodeKind.EXTERNAL: ("((", "))"),
            NodeKind.INTERMEDIATE: ("[", "]"),
            NodeKind.OUTCOME: ("{{", "}}"),
            NodeKind.CONSTRAINT: ("[[", "]]"),
        }
        lines = ["flowchart LR"]
        for node in self.nodes.values():
            if include is not None and node.name not in include:
                continue
            left, right = shapes[node.kind]
            lines.append(f'  {node.name}{left}"{node.name}"{right}')
        lines += [
            f"  {a} --> {b}" for a, b in self.edges() if include is None or (a in include and b in include)
        ]
        return "\n".join(lines)


def _evaluate_table(table: DecisionTable, ctx: dict) -> Any:
    if any(isinstance(v, np.ndarray) for v in ctx.values()):
        return table.evaluate_many(ctx)[0]
    return table.evaluate(ctx).value
