"""Analysis of competing hypotheses (ACH) with Admiralty-graded sources.

Scenario (fictional): a cargo-handling system at a port on an invented coastline
stopped for 14 hours. Four hypotheses compete. An evidence catalogue rates each item
against each hypothesis on the classic ACH scale (CC, C, N, I, II), which is mapped to
likelihoods P(e | H). A seeded generator picks the true hypothesis and simulates which
items are reported, by which sources, at which Admiralty grade; unreliable sources
report items regardless of the truth.

The analysis then:
1. scores hypotheses with the classic inconsistency count and with a Bayesian update
   that discounts each item by its source grade;
2. ranks evidence by diagnosticity;
3. asks what would change the conclusion: linchpin items (leave-one-out), source
   downgrades that flip the lead, and unreported items that collection could target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decision_lab.core import Trace, md_table
from decision_lab.core.fusion import Contribution, update_hypotheses
from experiments.common import Report

HYPOTHESES = {
    "H1": "equipment failure (worn controller, no hostile actor)",
    "H2": "insider sabotage by a disgruntled operator",
    "H3": "external network intrusion",
    "H4": "staged outage to cover cargo theft",
}
RATING_LIKELIHOOD = {"CC": 0.85, "C": 0.65, "N": 0.40, "I": 0.15, "II": 0.03}
INCONSISTENCY_WEIGHT = {"CC": 0, "C": 0, "N": 0, "I": 1, "II": 2}

# Admiralty grading: source reliability A to F, information credibility 1 to 6.
SOURCE_RELIABILITY = {"A": 0.95, "B": 0.80, "C": 0.60, "D": 0.35, "E": 0.15, "F": 0.50}
INFO_CREDIBILITY = {1: 1.00, 2: 0.85, 3: 0.65, 4: 0.40, 5: 0.20, 6: 0.50}

# Evidence catalogue: code, description, ratings for H1..H4.
CATALOGUE = [
    ("E01", "controller firmware log shows a watchdog reset", ("C", "C", "C", "C")),
    ("E02", "maintenance record: controller overdue for replacement", ("CC", "N", "N", "N")),
    ("E03", "outbound connections to an unfamiliar address before the stop", ("I", "N", "CC", "C")),
    ("E04", "operator badge used at an unusual hour", ("I", "CC", "N", "C")),
    ("E05", "two containers missing at the next inventory", ("N", "N", "N", "CC")),
    ("E06", "operator recently disciplined", ("N", "C", "N", "N")),
    ("E07", "same malware family seen at another regional port", ("II", "I", "CC", "N")),
    ("E08", "controller board shows heat damage", ("CC", "I", "I", "I")),
    ("E09", "CCTV gap during the outage window", ("I", "C", "N", "CC")),
    ("E10", "ransom note received", ("II", "I", "C", "II")),
    ("E11", "vendor bulletin on a defect in this controller model", ("C", "N", "N", "N")),
    ("E12", "remote-access account created two days before", ("II", "C", "CC", "C")),
    ("E13", "truck seen leaving the yard during the outage", ("I", "N", "N", "CC")),
    ("E14", "operator has debts and contact with a known fence", ("N", "C", "N", "CC")),
]
SOURCES = [
    ("S-alpha", "site engineer", "A"),
    ("S-bravo", "network sensor logs", "B"),
    ("S-charlie", "port security officer", "B"),
    ("S-delta", "anonymous tip line", "E"),
    ("S-echo", "liaison partner service", "C"),
    ("S-foxtrot", "new informant, no track record", "F"),
]


@dataclass(frozen=True)
class IntelReport:
    """One reported item of evidence, with its source and Admiralty grade."""

    item: str
    source: str
    grade: str  # e.g. "B2"

    @property
    def reliability(self) -> float:
        return SOURCE_RELIABILITY[self.grade[0]] * INFO_CREDIBILITY[int(self.grade[1:])]


def likelihood_matrix(items: list[str]) -> np.ndarray:
    lookup = {code: ratings for code, _, ratings in CATALOGUE}
    return np.array([[RATING_LIKELIHOOD[r] for r in lookup[i]] for i in items])


def diagnosticity(ratings: tuple[str, ...]) -> float:
    """Largest log-likelihood ratio between any two hypotheses for one item."""
    lik = np.array([RATING_LIKELIHOOD[r] for r in ratings])
    return float(np.log(lik.max() / lik.min()))


def generate_reports(seed: int, true_h: str = "H4") -> list[IntelReport]:
    rng = np.random.default_rng(seed)
    h = list(HYPOTHESES).index(true_h)
    reports = []
    for code, _, ratings in CATALOGUE:
        lik = np.array([RATING_LIKELIHOOD[r] for r in ratings])
        name, _, rel_letter = SOURCES[rng.integers(len(SOURCES))]
        r = SOURCE_RELIABILITY[rel_letter]
        faithful = rng.random() < r
        # A faithful source reports what is true; an unfaithful one reports noise.
        p_report = lik[h] if faithful else lik.mean()
        if rng.random() < p_report:
            cred = (
                int(rng.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1]))
                if faithful
                else int(rng.choice([2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.2]))
            )
            reports.append(IntelReport(code, name, f"{rel_letter}{cred}"))
    return reports


def posterior(reports: list[IntelReport], priors: np.ndarray, overrides: dict[str, float] | None = None):
    overrides = overrides or {}
    if not reports:
        return priors.copy()
    lik = likelihood_matrix([r.item for r in reports])
    rel = [overrides.get(r.item, r.reliability) for r in reports]
    return update_hypotheses(priors, lik, rel)


def run(seed: int = 5) -> Report:
    reports = generate_reports(seed)
    names = list(HYPOTHESES)
    priors = np.full(len(names), 1 / len(names))
    post = posterior(reports, priors)
    order = np.argsort(-post)
    lead, second = names[order[0]], names[order[1]]
    descr = {code: d for code, d, _ in CATALOGUE}
    ratings = {code: r for code, _, r in CATALOGUE}

    out = Report("intelligence", "Intelligence analysis: competing hypotheses with graded sources")
    out.add(
        "Hypotheses",
        "\n".join(f"- **{k}**: {v}" for k, v in HYPOTHESES.items())
        + "\n\nUniform prior. The scenario, sources and evidence are invented.",
    )

    # ACH matrix with classic inconsistency score.
    rows = []
    incons = np.zeros(len(names))
    for r in reports:
        rt = ratings[r.item]
        incons += np.array([INCONSISTENCY_WEIGHT[x] for x in rt])
        rows.append(
            (
                r.item,
                descr[r.item],
                r.source,
                r.grade,
                f"{r.reliability:.2f}",
                *rt,
                f"{diagnosticity(rt):.2f}",
            )
        )
    rows.sort(key=lambda row: -float(row[-1]))
    out.add(
        "ACH matrix (reported evidence, most diagnostic first)",
        md_table(["Item", "Evidence", "Source", "Grade", "Weight", *names, "Diagnosticity"], rows)
        + "\n\nDiagnosticity is the largest log-likelihood ratio between two hypotheses. An item "
        "rated the same against every hypothesis has diagnosticity 0 and cannot help, "
        "however reliable its source. Weight combines source reliability and information "
        "credibility from the Admiralty grade.",
    )

    out.add(
        "Scores",
        md_table(
            ["Hypothesis", "Inconsistency score (lower is better)", "Posterior"],
            [(h, int(incons[j]), f"{post[j]:.1%}") for j, h in enumerate(names)],
        )
        + "\n\nThe inconsistency count follows the classic ACH practice of looking for evidence "
        "against each hypothesis; the posterior uses all evidence and the source grades. When "
        "the two disagree, the reason is usually a low-grade source carrying an inconsistent "
        "rating.",
    )

    # What would change the conclusion?
    linchpins = []
    for i, r in enumerate(reports):
        p = posterior(reports[:i] + reports[i + 1 :], priors)
        linchpins.append((r.item, r.grade, names[int(np.argmax(p))], float(p[order[0]])))
    linchpins.sort(key=lambda t: t[3])
    downgrades = []
    for r in reports:
        p = posterior(reports, priors, overrides={r.item: 0.0})
        if names[int(np.argmax(p))] != lead:
            downgrades.append((r.item, r.source, r.grade, names[int(np.argmax(p))]))
    reported = {r.item for r in reports}
    collection = []
    for code, d, _ in CATALOGUE:
        if code in reported:
            continue
        p = posterior(reports + [IntelReport(code, "hypothetical", "B2")], priors)
        collection.append(
            (
                code,
                d,
                names[int(np.argmax(p))],
                float(p[order[0]]),
                "yes" if names[int(np.argmax(p))] != lead else "no",
            )
        )
    collection.sort(key=lambda t: t[3])
    flip_count = sum(1 for c in collection if c[4] == "yes")

    out.add(
        "What evidence would change the conclusion",
        "**Linchpin analysis.** Each reported item is removed in turn. Items whose removal "
        f"most lowers the posterior of {lead} are the assumptions to check first.\n\n"
        + md_table(
            ["Removed item", "Grade", "Leader without it", f"P({lead}) without it"],
            [(a, b, c, f"{d:.1%}") for a, b, c, d in linchpins[:5]],
        )
        + "\n\n**Deception check.** Items whose source, if shown to be fabricating "
        "(weight set to 0), would change the leading hypothesis:\n\n"
        + (
            md_table(["Item", "Source", "Grade", "New leader"], downgrades)
            if downgrades
            else f"None: no single source can overturn {lead}."
        )
        + "\n\n**Collection targets.** Unreported catalogue items, each added as if reported "
        "by a B2 source:\n\n"
        + md_table(
            ["Item", "Evidence", "Leader if reported", f"P({lead})", "Flips the lead"],
            [(a, b, c, f"{d:.1%}", e) for a, b, c, d, e in collection],
        ),
    )

    lead_j, second_j = order[0], order[1]
    contributions = []
    for r in reports:
        lik = likelihood_matrix([r.item])[0]
        eff = r.reliability * lik + (1 - r.reliability) * lik.mean()
        contributions.append(
            Contribution(f"{r.item} ({r.grade})", float(np.log(eff[lead_j] / eff[second_j])), descr[r.item])
        )
    contributions.sort(key=lambda c: abs(c.log_lr), reverse=True)
    trace = Trace(
        subject="Assessment",
        recommendation=f"{lead}: {HYPOTHESES[lead]}",
        score=float(post[lead_j]),
        evidence=contributions[:4],
        notes=[
            f"runner-up {second} at {post[second_j]:.1%}",
            f"evidence contributions are log-likelihood ratios {lead} vs {second}, after source discounting",
            f"{len(downgrades)} reported item(s) could overturn the lead if their source proved false",
            f"{flip_count} unreported item(s) would flip the lead if collected from a B2 source",
        ],
    )
    out.add("Explanation trace", trace.to_markdown())

    out.summary = [
        f"{len(reports)} reports from {len({r.source for r in reports})} sources",
        f"leading hypothesis {lead} ({post[lead_j]:.1%}), runner-up {second} ({post[second_j]:.1%})",
        "lowest inconsistency score: "
        + ", ".join(h for j, h in enumerate(names) if incons[j] == incons.min())
        + f" ({int(incons.min())})",
        f"top linchpin: {linchpins[0][0]} (without it P({lead}) = {linchpins[0][3]:.1%})",
        f"{flip_count} collection target(s) could flip the conclusion",
    ]
    out.metrics = {
        "posterior_lead": float(post[lead_j]),
        "lead_index": float(lead_j),
        "n_reports": float(len(reports)),
        "flip_targets": float(flip_count),
    }
    return out
