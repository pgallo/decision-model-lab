import numpy as np
import pytest

from decision_lab.core.feel import FeelError, evaluate, free_names, match_unary, parse


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1 + 2 * 3", 7),
        ("(1 + 2) * 3", 9),
        ("2 ** 3 ** 2", 512),  # right associative
        ("-2 ** 2", 4),  # unary minus binds tighter than ** in this subset
        ("10 / 4", 2.5),
        ("1 / 0", None),  # FEEL: division by zero is null
        ('"a" = "a"', True),
        ("3 != 3", False),
        ("true and not(false)", True),
        ("null = null", True),
        ("null > 1", None),
        ("false and null", False),
        ("true or null", True),
        ('if 2 > 1 then "yes" else "no"', "yes"),
        ("5 in [1..5]", True),
        ("5 in [1..5)", False),
        ("1 in (1..5]", False),
        ("1 in ]1..5]", False),
        ('"b" in ("a", "b")', True),
        ("3 between 1 and 3", True),
        ("min(4, 2, 9) + max([1, 7])", 9),
        ("sum([1, 2, 3])", 6),
        ("abs(-3) + floor(2.7) + ceil(0.2)", 6),
        ("1.5e2", 150.0),
    ],
)
def test_expressions(expr, expected):
    assert evaluate(expr) == expected


def test_context_names_and_paths():
    ctx = {"claim": {"amount": 1200}, "limit": 1000}
    assert evaluate("claim.amount > limit", ctx) is True
    with pytest.raises(FeelError, match="unknown name"):
        evaluate("missing + 1", ctx)
    with pytest.raises(FeelError, match="cannot resolve"):
        evaluate("claim.nope", ctx)


@pytest.mark.parametrize(
    "bad",
    [
        "__import__('os')",  # single quotes are not tokens
        '__import__("os")',  # unknown function
        "a.__class__",  # attribute access only through dicts
        "1 +",
        "(1 + 2",
        "if 1 then 2",
        "1 2",
        "[1..",
        "x" * 5000,
    ],
)
def test_rejects_unsafe_or_malformed(bad):
    with pytest.raises(FeelError):
        evaluate(bad, {"a": object(), "x": 1})


def test_vectorised_evaluation():
    ctx = {"x": np.array([1.0, 5.0, 10.0]), "lim": 4}
    np.testing.assert_array_equal(evaluate("x > lim", ctx), [False, True, True])
    np.testing.assert_array_equal(evaluate("if x > lim then x * 2 else 0", ctx), [0, 10, 20])
    np.testing.assert_array_equal(evaluate("min(x, 6)", ctx), [1, 5, 6])
    np.testing.assert_array_equal(evaluate("x in [2..9] and x > 1", ctx), [False, True, False])
    np.testing.assert_array_equal(evaluate('if x > lim then "hi" else "lo"', ctx), ["lo", "hi", "hi"])


@pytest.mark.parametrize(
    ("tests", "value", "expected"),
    [
        ("-", 42, True),
        ("", 42, True),
        ("< 10", 3, True),
        ("< 10", 10, False),
        (">= limit", 5, True),
        ("[1..5]", 5, True),
        ("(1..5)", 5, False),
        ('"a", "b"', "b", True),
        ('"a", "b"', "c", False),
        ('not("a")', "a", False),
        ('not("a", "b")', "c", True),
        ("true", True, True),
        ("-5", -5, True),  # a negative literal, not the "any" dash
        ("< 0, > 100", 150, True),
        ("< 10", None, False),  # null input never matches a comparison
    ],
)
def test_unary_tests(tests, value, expected):
    assert match_unary(tests, value, {"limit": 5}) is expected


def test_unary_tests_vectorised():
    v = np.array([1, 5, 9])
    np.testing.assert_array_equal(match_unary("[2..8]", v), [False, True, False])
    np.testing.assert_array_equal(match_unary("-", v), [True, True, True])


def test_free_names():
    assert free_names(parse("if a > b.c then min(d, 2) else e")) == {"a", "b", "d", "e"}
    assert free_names(parse('"text" = "x"')) == set()
