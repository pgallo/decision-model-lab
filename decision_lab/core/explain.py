"""Structured explanation traces and Markdown rendering.

Every recommendation produced by the experiments returns a :class:`Trace`. The trace is
data first (so tests and downstream systems can inspect it) and Markdown second.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .decision_table import FiredRule
from .fusion import Contribution
from .simulation import TornadoBar


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


@dataclass
class Trace:
    subject: str
    recommendation: str
    route: str | None = None
    score: float | None = None
    rules_fired: list[FiredRule] = field(default_factory=list)
    evidence: list[Contribution] = field(default_factory=list)
    drivers: list[TornadoBar] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "recommendation": self.recommendation,
            "route": self.route,
            "score": self.score,
            "rules_fired": [
                {"rule": r.index, "annotation": r.annotation, **r.output} for r in self.rules_fired
            ],
            "evidence": [{"name": c.name, "log_lr": c.log_lr, "detail": c.detail} for c in self.evidence],
            "drivers": [
                {"external": d.external, "swing": d.swing, "variance_share": d.variance_share}
                for d in self.drivers
            ],
            "notes": list(self.notes),
        }

    def to_markdown(self, level: int = 3) -> str:
        h = "#" * level
        out = [f"{h} {self.subject}", "", f"**Recommendation:** {self.recommendation}"]
        if self.route:
            out.append(f"**Route:** {self.route}")
        if self.score is not None:
            out.append(f"**Score:** {self.score:.3f}")
        if self.rules_fired:
            out += ["", "Rules fired:", ""]
            out += [f"- rule {r.index}: {r.annotation or r.output}" for r in self.rules_fired]
        if self.evidence:
            out += ["", "Top contributing evidence (log-likelihood ratio):", ""]
            out += [
                f"- {c.name}: {c.log_lr:+.2f} ({c.direction}){' ' + c.detail if c.detail else ''}"
                for c in self.evidence
            ]
        if self.drivers:
            out += ["", "Sensitivity drivers:", ""]
            out += [
                f"- {d.external}: swing {d.swing:,.1f}, variance share {d.variance_share:.0%}"
                for d in self.drivers
            ]
        if self.notes:
            out += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(out) + "\n"
