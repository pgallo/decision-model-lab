"""Portfolio prioritisation of fictional drug-discovery programmes.

Levers: fund or do not fund each programme, under a budget constraint.
Externals: probability of technical success (PTS) to launch, time to market,
peak sales, and for one preclinical asset a hidden toxicity flaw.
Outcome: portfolio NPV (risk is carried by Bernoulli success draws, so the
distribution of NPV, not only its mean, is available).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decision_lab.core import (
    DecisionModel,
    Trace,
    md_table,
    simulate,
    tornado,
    value_of_information,
)
from decision_lab.core.distributions import Bernoulli, Beta, LogNormal, Triangular, Uniform
from decision_lab.core.simulation import evpi
from experiments.common import Report

# Industry-style phase priors for PTS to launch (mean, concentration).
PHASE_PTS = {
    "preclinical": (0.07, 40),
    "phase1": (0.11, 40),
    "phase2": (0.22, 30),
    "phase3": (0.58, 25),
}
PHASE_YEARS = {
    "preclinical": (7, 9, 12),
    "phase1": (5, 7, 9),
    "phase2": (4, 5, 7),
    "phase3": (1.5, 2.5, 4),
}
TOX_PROGRAMME = "P1"  # the preclinical asset whose toxicity question a study could resolve
TOX_PRIOR = 0.7  # probability the asset has no disqualifying toxicity
STUDY_ACCURACY = 0.85
STUDY_COST = 6.0


@dataclass(frozen=True)
class Programme:
    code: str
    phase: str
    cost: float  # remaining development cost, in M (fictional currency units)
    peak_median: float  # median peak annual sales, in M
    peak_gsd: float


def generate_programmes(seed: int, n: int = 8) -> list[Programme]:
    rng = np.random.default_rng(seed)
    phases = [
        "preclinical",
        "preclinical",
        "phase1",
        "phase1",
        "phase2",
        "phase2",
        "phase3",
        "phase3",
    ]
    base_cost = {"preclinical": 45, "phase1": 80, "phase2": 150, "phase3": 260}
    programmes = []
    for i in range(n):
        phase = phases[i % len(phases)]
        programmes.append(
            Programme(
                code=f"P{i + 1}",
                phase=phase,
                cost=round(base_cost[phase] * rng.uniform(0.7, 1.3), 1),
                peak_median=round(float(rng.lognormal(np.log(900), 0.5)), 0),
                peak_gsd=round(float(rng.uniform(1.5, 2.2)), 2),
            )
        )
    return programmes


def build_model(programmes: list[Programme], budget: float) -> DecisionModel:
    m = DecisionModel("drug-portfolio", description="Fund a subset of programmes under a budget")
    # Simplified value of a launched asset: peak sales x margin x an annuity factor for
    # ten years of sales, discounted back over the time to market.
    m.parameters.update(budget=budget, margin=0.35, annuity=6.1, rate=0.10)
    for p in programmes:
        mean, k = PHASE_PTS[p.phase]
        low, mode, high = PHASE_YEARS[p.phase]
        c = p.code
        m.parameters[f"cost_{c}"] = p.cost
        m.lever(f"fund_{c}", [0, 1], f"fund programme {c} ({p.phase})")
        m.external(f"pts_{c}", Beta(mean * k, (1 - mean) * k), f"PTS to launch for {c}")
        m.external(f"ttm_{c}", Triangular(low, mode, high), f"years to market for {c}")
        m.external(f"peak_{c}", LogNormal(p.peak_median, p.peak_gsd), f"peak sales for {c}")
        m.external(f"draw_{c}", Uniform(0, 1), f"technical outcome draw for {c}")
        success = f"draw_{c} < pts_{c}"
        if c == TOX_PROGRAMME:
            # For the toxicity-flagged asset the PTS prior is conditional on no tox flaw.
            m.external(f"tox_ok_{c}", Bernoulli(TOX_PRIOR), f"no disqualifying toxicity for {c}")
            success = f"{success} and tox_ok_{c} = 1"
        m.intermediate(f"success_{c}", success)
        m.intermediate(
            f"npv_{c}",
            f"fund_{c} * ((if success_{c} then peak_{c} * margin * annuity / (1 + rate) ** ttm_{c} "
            f"else 0) - cost_{c})",
        )
    m.outcome("portfolio_npv", " + ".join(f"npv_{p.code}" for p in programmes))
    m.constraint(
        "within_budget",
        " + ".join(f"fund_{p.code} * cost_{p.code}" for p in programmes) + " <= budget",
    )
    return m


def run(seed: int = 7, n: int = 20_000) -> Report:
    programmes = generate_programmes(seed)
    budget = 420.0
    model = build_model(programmes, budget)
    result = simulate(model, n=n, seed=seed)
    ranked = result.ranked()
    best = ranked[0]
    funded = [p for p in programmes if best.levers[f"fund_{p.code}"] == 1]
    spend = sum(p.cost for p in funded)

    report = Report("life_sciences", "Life sciences: drug-discovery portfolio prioritisation")
    report.add(
        "Programmes",
        md_table(
            ["Programme", "Phase", "Remaining cost (M)", "Median peak sales (M)", "PTS prior mean"],
            [(p.code, p.phase, p.cost, p.peak_median, PHASE_PTS[p.phase][0]) for p in programmes],
        )
        + f"\n\nBudget: {budget:.0f} M. {len(result.policies)} lever settings evaluated, "
        f"{len(result.feasible())} within budget, {n:,} Monte Carlo samples each "
        "(common random numbers).",
    )

    def portfolio(p) -> str:
        levers = p if isinstance(p, dict) else p.levers
        return " ".join(k.removeprefix("fund_") for k, v in levers.items() if v == 1) or "(none)"

    report.add(
        "Top portfolios by expected NPV",
        md_table(
            ["Portfolio", "E[NPV] (M)", "SD", "P10", "P90", "P(NPV < 0)"],
            [
                (
                    portfolio(p),
                    p.expected_utility,
                    p.sd,
                    p.quantile(0.1),
                    p.quantile(0.9),
                    float(np.mean(p.utility < 0)),
                )
                for p in ranked[:6]
            ],
        ),
    )

    driver_names = [f"{kind}_{p.code}" for p in funded for kind in ("pts", "ttm", "peak")]
    if TOX_PROGRAMME in {p.code for p in funded}:
        driver_names.append(f"tox_ok_{TOX_PROGRAMME}")
    bars = tornado(model, best.levers, n=n // 2, seed=seed, externals=driver_names)
    report.add(
        "Tornado: what drives the NPV of the recommended portfolio",
        "Each external is pinned at its 10th and 90th percentile while the others stay random.\n\n"
        + md_table(
            [
                "External",
                "Low value",
                "High value",
                "E[NPV] at low",
                "E[NPV] at high",
                "Swing",
                "First-order variance share",
            ],
            [
                (
                    b.external,
                    b.low_value,
                    b.high_value,
                    b.utility_at_low,
                    b.utility_at_high,
                    b.swing,
                    f"{b.variance_share:.0%}",
                )
                for b in bars[:8]
            ],
        )
        + "\n\nThe variance shares are small because most NPV variance comes from the binary "
        "technical outcome of each programme (succeed or fail), which is not a lever-relevant "
        "input and is left out of the tornado. The swings show which estimates are worth "
        "refining.",
    )

    tox = f"tox_ok_{TOX_PROGRAMME}"

    def study(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
        correct = rng.random(len(x)) < STUDY_ACCURACY
        return np.where(correct, x, 1 - x)

    evsi = value_of_information(result, tox, signal=study, seed=seed)
    evppi = value_of_information(result, tox)
    top_driver = bars[0].external
    evppi_driver = value_of_information(result, top_driver) if top_driver != tox else evppi
    total_evpi = evpi(result)
    report.add(
        f"Value of information: extra preclinical toxicity study on {TOX_PROGRAMME}",
        md_table(
            ["Quantity", "Value (M)"],
            [
                ("EVPI, all uncertainty resolved (upper bound)", total_evpi),
                (f"EVPPI, toxicity of {TOX_PROGRAMME} known exactly", evppi.value),
                (f"EVSI, study with {STUDY_ACCURACY:.0%} accuracy", evsi.value),
                ("Study cost", STUDY_COST),
                ("Net value of running the study", evsi.value - STUDY_COST),
                (f"EVPPI of the top tornado driver ({top_driver})", evppi_driver.value),
            ],
        )
        + "\n\nPortfolio chosen after each study readout:\n\n"
        + "\n".join(
            f"- readout {'clean' if k == 1 else 'toxicity signal'}: fund {portfolio(v)}"
            for k, v in sorted(evsi.choices_by_signal.items(), reverse=True)
        ),
    )

    run_study = evsi.value > STUDY_COST
    gap = best.expected_utility - ranked[1].expected_utility
    p_loss = float(np.mean(best.utility < 0))
    trace = Trace(
        subject="Recommendation",
        recommendation=f"fund {', '.join(p.code for p in funded)} (spend {spend:.0f} of {budget:.0f} M)",
        score=best.expected_utility,
        drivers=bars[:3],
        notes=[
            f"expected NPV {best.expected_utility:,.0f} M, P(NPV < 0) {p_loss:.0%}",
            f"runner-up {portfolio(ranked[1])} is {gap:,.0f} M lower",
            f"{'run' if run_study else 'skip'} the toxicity study: "
            f"EVSI {evsi.value:.1f} M vs cost {STUDY_COST:.1f} M",
        ],
    )
    report.add("Explanation trace", trace.to_markdown())
    report.add("Decision graph", "```mermaid\n" + _graph_excerpt(model, programmes[:2]) + "\n```")

    report.summary = [
        f"recommended portfolio: {portfolio(best)} (spend {spend:.0f}/{budget:.0f} M)",
        f"expected NPV {best.expected_utility:,.0f} M, "
        f"P10 {best.quantile(0.1):,.0f}, P90 {best.quantile(0.9):,.0f}",
        f"top NPV driver: {bars[0].external} (swing {bars[0].swing:,.0f} M)",
        f"toxicity study on {TOX_PROGRAMME}: EVSI {evsi.value:.1f} M vs cost {STUDY_COST:.1f} M -> "
        f"{'run it' if run_study else 'skip it'}",
    ]
    report.metrics = {
        "expected_npv": best.expected_utility,
        "evsi": evsi.value,
        "evppi": evppi.value,
        "evpi": total_evpi,
        "spend": spend,
    }
    return report


def _graph_excerpt(model: DecisionModel, programmes: list[Programme]) -> str:
    """The full graph has ~40 nodes; show the structure for two programmes."""
    keep = {"portfolio_npv", "within_budget"}
    for p in programmes:
        keep |= {n for n in model.nodes if n.endswith(f"_{p.code}")}
    return model.to_mermaid(include=keep)
