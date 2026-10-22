import numpy as np
import pytest

from decision_lab.core.distributions import Normal
from decision_lab.core.model import DecisionModel, ModelError, NodeKind

YAML = """
name: newsvendor
parameters: {price: 10, cost: 6}
nodes:
  - {name: order, kind: lever, options: [0, 50, 100]}
  - {name: demand, kind: external, distribution: {dist: normal, mean: 100, sd: 20}}
  - {name: sold, kind: intermediate, expr: "min(order, max(demand, 0))"}
  - {name: profit, kind: outcome, expr: "sold * price - order * cost"}
  - {name: cap, kind: constraint, expr: "order <= 80"}
"""


def newsvendor() -> DecisionModel:
    m = DecisionModel("newsvendor", parameters={"price": 10, "cost": 6})
    m.lever("order", [0, 50, 100])
    m.external("demand", Normal(100, 20))
    m.intermediate("sold", "min(order, max(demand, 0))")
    m.outcome("profit", "sold * price - order * cost")
    return m


def test_yaml_and_python_definitions_agree():
    a = DecisionModel.from_yaml_text(YAML)
    b = newsvendor()
    b.constraint("cap", "order <= 80")
    assert a.edges() == b.edges()
    assert [n.kind for n in a.nodes.values()] == [n.kind for n in b.nodes.values()]


def test_edges_are_derived_from_expressions():
    m = newsvendor()
    assert set(m.edges()) == {
        ("demand", "sold"),
        ("order", "sold"),
        ("order", "profit"),
        ("sold", "profit"),
    }
    order = m.validate()
    assert order.index("sold") < order.index("profit")


def test_cycle_detected():
    m = DecisionModel("cyclic")
    m.lever("x", [1])
    m.intermediate("a", "b + x")
    m.intermediate("b", "a + 1")
    m.outcome("o", "a")
    with pytest.raises(ModelError, match="cycle"):
        m.validate()


def test_unknown_reference():
    m = DecisionModel("bad")
    m.lever("x", [1])
    m.outcome("o", "x + y")
    with pytest.raises(ModelError, match="unknown name 'y'"):
        m.validate()


def test_type_rules():
    m = DecisionModel("types")
    m.lever("x", [1])
    m.outcome("o", "x")
    m.constraint("c", "o > 0")
    m.intermediate("i", "c")  # an intermediate may not depend on a constraint
    with pytest.raises(ModelError, match="may not depend on constraint"):
        m.validate()


def test_structural_requirements():
    m = DecisionModel("empty")
    m.lever("x", [])
    with pytest.raises(ModelError) as exc:
        m.validate()
    assert "no options" in str(exc.value) and "no outcome" in str(exc.value)
    with pytest.raises(ModelError, match="duplicate"):
        newsvendor().lever("order", [1])


def test_python_callable_nodes():
    m = DecisionModel("callable")
    m.lever("x", [1, 2])
    m.external("e", Normal(0, 1))
    m.outcome("o", func=lambda x, e: x * 10 + e, depends_on=["x", "e"])
    values = m.evaluate({"x": 2}, {"e": np.array([0.5, -0.5])})
    np.testing.assert_allclose(values["o"], [20.5, 19.5])


def test_decision_table_node_in_yaml():
    m = DecisionModel.from_yaml_text(
        """
name: tiered
nodes:
  - {name: band, kind: lever, options: [1, 2]}
  - {name: x, kind: external, distribution: {dist: uniform, low: 0, high: 10}}
  - name: fee
    kind: outcome
    table:
      name: fees
      hit_policy: FIRST
      inputs: [x]
      outputs: [fee]
      rules:
        - {when: ["< 5"], then: ["band * 1"]}
        - {when: ["-"], then: ["band * 3"]}
"""
    )
    assert m.nodes["fee"].kind is NodeKind.OUTCOME
    values = m.evaluate({"band": 2}, {"x": np.array([1.0, 7.0])})
    np.testing.assert_array_equal(values["fee"], [2, 6])
    assert ("band", "fee") in m.edges() and ("x", "fee") in m.edges()


def test_mermaid_export():
    text = newsvendor().to_mermaid()
    assert text.startswith("flowchart LR")
    assert 'order[/"order"/]' in text and "sold --> profit" in text
