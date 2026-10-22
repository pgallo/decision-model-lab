"""FEEL-lite: a small, safe subset of the DMN FEEL expression language.

The parser is a hand-written tokenizer plus recursive-descent parser that produces
a tuple-based AST. Evaluation walks the AST against a context dict. There is no
``eval()``, no attribute access on Python objects and no import mechanism: names
resolve only through the supplied context, and calls only reach an allow-listed
function table.

Evaluation is array-aware. When a context value is an n-dimensional array, arithmetic,
comparisons, boolean operators and ``if/then/else`` are applied element-wise, so the
same expression can score one case or a hundred thousand Monte Carlo samples.

Supported expression syntax::

    1.5  "text"  true  false  null  a  a.b
    + - * / **   = != < <= > >=   and  or  not(x)
    if c then a else b      x in [1..5]     x in ("a", "b")     x between 1 and 3
    min(a, b)  max(...)  abs(x)  sqrt(x)  exp(x)  log(x)  floor(x)  ceil(x)  sum(list)

Unary tests (decision table input cells)::

    -    < 10    >= limit    [1..5]    "a", "b"    not("x")    true
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

KEYWORDS = {"and", "or", "not", "if", "then", "else", "in", "true", "false", "null", "between"}
COMPARATORS = {"=", "!=", "<", "<=", ">", ">="}
MAX_EXPRESSION_LENGTH = 4096

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>\*\*|\.\.|<=|>=|!=|[-+*/=<>(),.\[\]])
    """,
    re.VERBOSE,
)


class FeelError(ValueError):
    """Raised for syntax errors and evaluation errors in FEEL-lite expressions."""


@dataclass(frozen=True)
class Token:
    kind: str  # number, string, name, keyword, op, eof
    value: Any
    pos: int


def tokenize(text: str) -> list[Token]:
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise FeelError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise FeelError(f"unexpected character {text[pos]!r} at position {pos}")
        kind = m.lastgroup
        raw = m.group()
        if kind == "number":
            tokens.append(Token("number", float(raw) if any(c in raw for c in ".eE") else int(raw), pos))
        elif kind == "string":
            tokens.append(Token("string", bytes(raw[1:-1], "utf-8").decode("unicode_escape"), pos))
        elif kind == "name":
            tokens.append(Token("keyword" if raw in KEYWORDS else "name", raw, pos))
        elif kind == "op":
            tokens.append(Token("op", raw, pos))
        pos = m.end()
    tokens.append(Token("eof", None, pos))
    return tokens


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize(text)
        self.i = 0

    # -- token helpers -------------------------------------------------------
    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def _is(self, kind: str, value: Any = None) -> bool:
        t = self.tok
        return t.kind == kind and (value is None or t.value == value)

    def _accept(self, kind: str, value: Any = None) -> Token | None:
        if self._is(kind, value):
            t = self.tok
            self.i += 1
            return t
        return None

    def _expect(self, kind: str, value: Any = None) -> Token:
        t = self._accept(kind, value)
        if t is None:
            want = value if value is not None else kind
            raise FeelError(
                f"expected {want!r} at position {self.tok.pos}, found {self.tok.value!r} in {self.text!r}"
            )
        return t

    def _end(self) -> None:
        if not self._is("eof"):
            raise FeelError(f"unexpected {self.tok.value!r} at position {self.tok.pos} in {self.text!r}")

    # -- expressions ---------------------------------------------------------
    def expression(self):
        if self._accept("keyword", "if"):
            cond = self.expression()
            self._expect("keyword", "then")
            then = self.expression()
            self._expect("keyword", "else")
            other = self.expression()
            return ("if", cond, then, other)
        return self._or()

    def _or(self):
        node = self._and()
        while self._accept("keyword", "or"):
            node = ("or", node, self._and())
        return node

    def _and(self):
        node = self._comparison()
        while self._accept("keyword", "and"):
            node = ("and", node, self._comparison())
        return node

    def _comparison(self):
        node = self._additive()
        if self.tok.kind == "op" and self.tok.value in COMPARATORS:
            op = self._expect("op").value
            return ("cmp", op, node, self._additive())
        if self._accept("keyword", "in"):
            return ("in", node, self._in_target())
        if self._accept("keyword", "between"):
            low = self._additive()
            self._expect("keyword", "and")
            return ("between", node, low, self._additive())
        return node

    def _in_target(self):
        if self._is("op", "(") or self._is("op", "[") or self._is("op", "]"):
            node = self._primary()
            if node[0] in ("interval", "list"):
                return node
            return ("list", [node])
        return ("list", [self._additive()])

    def _additive(self):
        node = self._multiplicative()
        while self.tok.kind == "op" and self.tok.value in ("+", "-"):
            op = self._expect("op").value
            node = ("bin", op, node, self._multiplicative())
        return node

    def _multiplicative(self):
        node = self._power()
        while self.tok.kind == "op" and self.tok.value in ("*", "/"):
            op = self._expect("op").value
            node = ("bin", op, node, self._power())
        return node

    def _power(self):
        base = self._unary()
        if self._accept("op", "**"):
            return ("bin", "**", base, self._power())  # right associative
        return base

    def _unary(self):
        if self._accept("op", "-"):
            return ("neg", self._unary())
        if self._accept("keyword", "not"):
            self._expect("op", "(")
            inner = self.expression()
            self._expect("op", ")")
            return ("not", inner)
        return self._primary()

    def _primary(self):
        t = self.tok
        if t.kind in ("number", "string"):
            self.i += 1
            return ("lit", t.value)
        if t.kind == "keyword" and t.value in ("true", "false", "null"):
            self.i += 1
            return ("lit", {"true": True, "false": False, "null": None}[t.value])
        if t.kind == "name":
            self.i += 1
            if self._accept("op", "("):
                args = []
                if not self._is("op", ")"):
                    args.append(self.expression())
                    while self._accept("op", ","):
                        args.append(self.expression())
                self._expect("op", ")")
                return ("call", t.value, args)
            path = [t.value]
            while self._accept("op", "."):
                path.append(self._expect("name").value)
            return ("name", tuple(path))
        if t.kind == "op" and t.value in ("(", "[", "]"):
            self.i += 1
            if t.value == "[" and self._accept("op", "]"):
                return ("list", [])
            first = self.expression()
            if self._accept("op", ".."):
                high = self.expression()
                closer = self._expect("op").value
                if closer not in (")", "]", "["):
                    raise FeelError(f"bad interval end {closer!r} in {self.text!r}")
                return ("interval", t.value == "[", first, high, closer == "]")
            if t.value == "]":
                raise FeelError(f"']' can only open an interval in {self.text!r}")
            items = [first]
            while self._accept("op", ","):
                items.append(self.expression())
            self._expect("op", ")" if t.value == "(" else "]")
            if t.value == "(" and len(items) == 1:
                return first
            return ("list", items)
        raise FeelError(f"unexpected {t.value!r} at position {t.pos} in {self.text!r}")

    # -- unary tests ---------------------------------------------------------
    def unary_tests(self):
        if self._is("eof"):  # an empty cell behaves like "-"
            return ("any",)
        if self._accept("op", "-") and self._is("eof"):
            return ("any",)
        if self.i:  # a leading '-' that was not alone: re-parse as an expression
            self.i = 0
        if self._is("keyword", "not") and self.tokens[self.i + 1].value == "(":
            self.i += 2
            tests = self._test_list()
            self._expect("op", ")")
            return ("not", ("tests", tests))
        return ("tests", self._test_list())

    def _test_list(self):
        tests = [self._single_test()]
        while self._accept("op", ","):
            tests.append(self._single_test())
        return tests

    def _single_test(self):
        if self.tok.kind == "op" and self.tok.value in COMPARATORS:
            op = self._expect("op").value
            return ("cmp_input", op, self._additive())
        node = self.expression()
        if node[0] == "interval":
            return ("in_input", node)
        return ("cmp_input", "=", node)


@lru_cache(maxsize=4096)
def parse(text: str):
    """Parse a FEEL-lite expression into an AST (cached)."""
    p = _Parser(str(text))
    node = p.expression()
    p._end()
    return node


@lru_cache(maxsize=4096)
def parse_unary_tests(text: str):
    """Parse a decision-table input entry (unary tests) into an AST (cached)."""
    p = _Parser(str(text).strip())
    node = p.unary_tests()
    p._end()
    return node


def free_names(node) -> set[str]:
    """Root names referenced by an AST; used to derive DAG edges from expressions."""
    kind = node[0]
    if kind == "name":
        return {node[1][0]}
    if kind == "lit" or kind == "any":
        return set()
    out: set[str] = set()
    for child in node[1:]:
        if isinstance(child, tuple):
            out |= free_names(child)
        elif isinstance(child, list):
            for c in child:
                out |= free_names(c)
    return out


# -- evaluation --------------------------------------------------------------


def _is_array(*xs) -> bool:
    return any(isinstance(x, np.ndarray) for x in xs)


def _vector(fn_scalar, fn_array):
    def fn(*args):
        return fn_array(*args) if _is_array(*args) else fn_scalar(*args)

    return fn


def _fold(fn_array, fn_scalar):
    def fn(*args):
        if len(args) == 1 and isinstance(args[0], list):
            args = tuple(args[0])
        if not args:
            raise FeelError("function needs at least one argument")
        if _is_array(*args):
            out = args[0]
            for a in args[1:]:
                out = fn_array(out, a)
            return out
        return fn_scalar(args)

    return fn


def _sum(*args):
    if len(args) == 1 and isinstance(args[0], list):
        args = tuple(args[0])
    total: Any = 0
    for a in args:
        total = total + a
    return total


FUNCTIONS = {
    "min": _fold(np.minimum, min),
    "max": _fold(np.maximum, max),
    "sum": _sum,
    "count": lambda xs: len(xs),
    "abs": _vector(abs, np.abs),
    "sqrt": _vector(math.sqrt, np.sqrt),
    "exp": _vector(math.exp, np.exp),
    "log": _vector(math.log, np.log),
    "floor": _vector(math.floor, np.floor),
    "ceil": _vector(math.ceil, np.ceil),
}

_ARITH = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "**": lambda a, b: a**b,
}
_CMP = {
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _and(a, b):
    if _is_array(a, b):
        return np.logical_and(a, b)
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return bool(a) and bool(b)


def _or(a, b):
    if _is_array(a, b):
        return np.logical_or(a, b)
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return bool(a) or bool(b)


def _not(a):
    if isinstance(a, np.ndarray):
        return np.logical_not(a)
    return None if a is None else not a


def _compare(op: str, a, b):
    if a is None or b is None:
        if op in ("=", "!="):
            same = a is None and b is None
            return same if op == "=" else not same
        return None
    return _CMP[op](a, b)


def _in_interval(value, node, ctx):
    _, closed_low, low, high, closed_high = node
    lo, hi = _eval(low, ctx), _eval(high, ctx)
    left = _compare(">=" if closed_low else ">", value, lo)
    right = _compare("<=" if closed_high else "<", value, hi)
    return _and(left, right)


def _eval(node, ctx: dict):
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "name":
        path = node[1]
        if path[0] not in ctx:
            raise FeelError(f"unknown name {path[0]!r}")
        value = ctx[path[0]]
        for key in path[1:]:
            if not isinstance(value, dict) or key not in value:
                raise FeelError(f"cannot resolve {'.'.join(path)!r}")
            value = value[key]
        return value
    if kind == "bin":
        a, b = _eval(node[2], ctx), _eval(node[3], ctx)
        if a is None or b is None:
            return None
        if node[1] == "/" and not _is_array(a, b) and b == 0:
            return None  # FEEL semantics: division by zero yields null
        with np.errstate(divide="ignore", invalid="ignore"):
            return _ARITH[node[1]](a, b)
    if kind == "neg":
        v = _eval(node[1], ctx)
        return None if v is None else -v
    if kind == "cmp":
        return _compare(node[1], _eval(node[2], ctx), _eval(node[3], ctx))
    if kind == "and":
        return _and(_eval(node[1], ctx), _eval(node[2], ctx))
    if kind == "or":
        return _or(_eval(node[1], ctx), _eval(node[2], ctx))
    if kind == "not":
        return _not(_eval(node[1], ctx))
    if kind == "if":
        cond = _eval(node[1], ctx)
        if isinstance(cond, np.ndarray):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(cond, _eval(node[2], ctx), _eval(node[3], ctx))
        return _eval(node[2], ctx) if cond is True else _eval(node[3], ctx)
    if kind == "between":
        v = _eval(node[1], ctx)
        return _and(_compare(">=", v, _eval(node[2], ctx)), _compare("<=", v, _eval(node[3], ctx)))
    if kind == "in":
        v = _eval(node[1], ctx)
        target = node[2]
        if target[0] == "interval":
            return _in_interval(v, target, ctx)
        result: Any = False
        for item in target[1]:
            if item[0] == "interval":
                result = _or(result, _in_interval(v, item, ctx))
            else:
                result = _or(result, _compare("=", v, _eval(item, ctx)))
        return result
    if kind == "list":
        return [_eval(item, ctx) for item in node[1]]
    if kind == "interval":
        raise FeelError("an interval is only valid after 'in' or as a unary test")
    if kind == "call":
        fn = FUNCTIONS.get(node[1])
        if fn is None:
            raise FeelError(f"unknown function {node[1]!r}")
        return fn(*[_eval(a, ctx) for a in node[2]])
    raise FeelError(f"cannot evaluate node {kind!r}")


def evaluate(expr: str | tuple, context: dict | None = None):
    """Evaluate a FEEL-lite expression (text or pre-parsed AST) against a context."""
    node = parse(expr) if isinstance(expr, str) else expr
    return _eval(node, context or {})


def match_unary(tests: str | tuple, value, context: dict | None = None):
    """Evaluate unary tests against an input value. Returns bool or a boolean array."""
    node = parse_unary_tests(tests) if isinstance(tests, str) else tests
    ctx = context or {}
    return _run_tests(node, value, ctx)


def _run_tests(node, value, ctx):
    kind = node[0]
    if kind == "any":
        return np.ones(np.shape(value), dtype=bool) if isinstance(value, np.ndarray) else True
    if kind == "not":
        return _not(_run_tests(node[1], value, ctx))
    result: Any = False
    for test in node[1]:
        if test[0] == "in_input":
            hit = _in_interval(value, test[1], ctx)
        else:
            hit = _compare(test[1], value, _eval(test[2], ctx))
        result = _or(result, hit)
    if result is None:
        return False
    return result
