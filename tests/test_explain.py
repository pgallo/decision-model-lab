from decision_lab.core.decision_table import FiredRule
from decision_lab.core.explain import Trace, md_table
from decision_lab.core.fusion import Contribution
from decision_lab.core.simulation import TornadoBar


def test_trace_dict_and_markdown():
    trace = Trace(
        subject="Case 1",
        recommendation="investigate",
        route="human-review",
        score=0.42,
        rules_fired=[FiredRule(2, "large amount", {"lane": "investigate"})],
        evidence=[Contribution("shared_contact", 3.0, "shared bank account")],
        drivers=[TornadoBar("demand", 1, 2, 10.0, 30.0, 0.6)],
        notes=["calibrated"],
    )
    d = trace.to_dict()
    assert d["rules_fired"][0] == {"rule": 2, "annotation": "large amount", "lane": "investigate"}
    assert d["drivers"][0]["swing"] == 20.0
    md = trace.to_markdown()
    assert "### Case 1" in md
    assert "rule 2: large amount" in md
    assert "shared_contact: +3.00 (supports)" in md
    assert "demand: swing 20.0, variance share 60%" in md


def test_md_table():
    text = md_table(["a", "b"], [(1, 0.5), ("x", 1234.5)])
    assert text.splitlines() == ["| a | b |", "|---|---|", "| 1 | 0.500 |", "| x | 1,234.5 |"]
