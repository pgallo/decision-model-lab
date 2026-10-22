import numpy as np
import pytest

from decision_lab.core.decision_table import (
    DecisionTable,
    HitPolicyViolation,
    NoRuleMatched,
)


def risk_table(policy: str, default=None) -> DecisionTable:
    return DecisionTable.from_dict(
        {
            "name": "risk",
            "hit_policy": policy,
            "inputs": ["age", "score"],
            "outputs": ["points"],
            "default": default,
            "rules": [
                {"when": ["< 25", "-"], "then": ["30"], "annotation": "young"},
                {"when": ["-", "< 600"], "then": ["50"], "annotation": "low score"},
                {"when": ["[25..65]", ">= 600"], "then": ["0"], "annotation": "standard"},
                {"when": ["> 65", "-"], "then": ["20"], "annotation": "senior"},
            ],
        }
    )


def test_unique_single_match():
    result = risk_table("UNIQUE").evaluate({"age": 40, "score": 700})
    assert result.value == 0
    assert [f.index for f in result.fired] == [3]


def test_unique_violation_when_rules_overlap():
    with pytest.raises(HitPolicyViolation, match="rules 1, 2"):
        risk_table("UNIQUE").evaluate({"age": 20, "score": 500})


def test_first_returns_first_match_in_order():
    result = risk_table("FIRST").evaluate({"age": 20, "score": 500})
    assert result.value == 30
    assert [f.annotation for f in result.fired] == ["young"]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("COLLECT", [30, 50]),
        ("COLLECT SUM", 80),
        ("COLLECT MIN", 30),
        ("COLLECT MAX", 50),
        ("COLLECT COUNT", 2),
    ],
)
def test_collect_aggregations(policy, expected):
    result = risk_table(policy).evaluate({"age": 20, "score": 500})
    assert result.value == expected
    assert [f.index for f in result.fired] == [1, 2]


def test_collect_with_no_match_is_empty():
    table = risk_table("COLLECT")
    assert table.evaluate({"age": 30, "score": None}).value == []


def test_default_and_no_match():
    ctx = {"age": 30, "score": None}  # null score matches neither "< 600" nor ">= 600"
    assert risk_table("FIRST", default={"points": -1}).evaluate(ctx).value == -1
    with pytest.raises(NoRuleMatched):
        risk_table("FIRST").evaluate(ctx)


def test_output_entries_are_expressions_and_tests_reference_context():
    table = DecisionTable.from_dict(
        {
            "name": "discount",
            "hit_policy": "FIRST",
            "inputs": ["amount"],
            "outputs": ["discount"],
            "rules": [
                {"when": ["> threshold"], "then": ["amount * rate"]},
                {"when": ["-"], "then": ["0"]},
            ],
        }
    )
    ctx = {"amount": 200, "threshold": 100, "rate": 0.1}
    assert table.evaluate(ctx).value == pytest.approx(20)
    assert table.referenced_names() == {"amount", "threshold", "rate"}


def test_vectorised_matches_scalar():
    rng = np.random.default_rng(0)
    ages = rng.integers(18, 90, 200)
    scores = rng.integers(400, 850, 200)
    for policy in ("FIRST", "COLLECT SUM", "COLLECT MAX", "COLLECT COUNT"):
        table = risk_table(policy, default={"points": -1} if policy == "FIRST" else None)
        values, _ = table.evaluate_many({"age": ages, "score": scores})
        scalar = [
            table.evaluate({"age": int(a), "score": int(s)}).value for a, s in zip(ages, scores, strict=True)
        ]
        np.testing.assert_array_equal(values, scalar)


def test_vectorised_unique_violation():
    with pytest.raises(HitPolicyViolation):
        risk_table("UNIQUE").evaluate_many({"age": np.array([40, 20]), "score": np.array([700, 500])})


def test_bad_tables_rejected_at_load_time():
    with pytest.raises(ValueError, match="wrong number"):
        DecisionTable.from_dict(
            {
                "name": "t",
                "inputs": ["a"],
                "outputs": ["o"],
                "rules": [{"when": ["-", "-"], "then": ["1"]}],
            }
        )
    with pytest.raises(ValueError):
        DecisionTable.from_dict(
            {
                "name": "t",
                "inputs": ["a"],
                "outputs": ["o"],
                "rules": [{"when": ["< "], "then": ["1"]}],
            }
        )
    with pytest.raises(ValueError, match="only valid with COLLECT"):
        DecisionTable.from_dict(
            {
                "name": "t",
                "inputs": ["a"],
                "outputs": ["o"],
                "hit_policy": "FIRST",
                "aggregation": "SUM",
                "rules": [{"when": ["-"], "then": ["1"]}],
            }
        )
