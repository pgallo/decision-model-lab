"""Claims triage: DMN decision table + fraud-likelihood score + cost-sensitive thresholds.

The fraud likelihood comes from fusing red-flag indicators (naive Bayes in log-odds).
The generator correlates the red flags through a hidden "organised ring" factor, so the
naive-Bayes posterior is overconfident; isotonic calibration on a held-out split fixes
that before the probabilities reach the cost model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from decision_lab.core import (
    Calibrator,
    CostMatrix,
    DecisionModel,
    Evidence,
    Trace,
    brier_score,
    fuse,
    md_table,
    reliability_table,
    simulate,
)
from decision_lab.core.fusion import expected_calibration_error
from decision_lab.core.policy import optimise_threshold
from experiments.common import Report

MODEL_PATH = Path(__file__).with_name("triage_model.yaml")
BASE_FRAUD_RATE = 0.06

# Red flags: (name, P(flag | fraud), P(flag | legitimate), description)
RED_FLAGS = [
    ("early_claim", 0.45, 0.08, "claim within 60 days of policy start"),
    ("late_report", 0.40, 0.12, "reported more than 30 days after the loss"),
    ("prior_claims", 0.50, 0.15, "three or more claims in 24 months"),
    ("missing_police_report", 0.55, 0.25, "no third-party report for a theft or collision"),
    ("round_amount", 0.35, 0.10, "claimed amount is a round figure"),
    ("shared_contact", 0.20, 0.01, "phone or bank account shared with another claimant"),
]


def generate_claims(seed: int, n: int = 6000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    claim_type = rng.choice(["auto", "property", "liability"], size=n, p=[0.55, 0.30, 0.15])
    scale = np.select([claim_type == "auto", claim_type == "property"], [3500, 6000], 14000)
    is_fraud = (rng.random(n) < BASE_FRAUD_RATE).astype(float)
    # Fraudulent claims are inflated on average.
    amount = np.round(rng.lognormal(np.log(scale), 0.8) * np.where(is_fraud == 1, 1.6, 1.0), -1)
    injury = rng.random(n) < np.where(claim_type == "property", 0.01, 0.18)
    # A hidden ring factor makes red flags co-occur for a subset of frauds.
    ring = (is_fraud == 1) & (rng.random(n) < 0.4)
    flags = {}
    for name, p_fraud, p_legit, _ in RED_FLAGS:
        p = np.where(is_fraud == 1, p_fraud, p_legit)
        p = np.where(ring, np.minimum(p * 1.8, 0.95), np.where(is_fraud == 1, p * 0.6, p))
        flags[name] = rng.random(n) < p
    return {
        "claim_type": claim_type,
        "amount": amount,
        "injury": injury,
        "is_fraud": is_fraud,
        **flags,
    }


def fraud_evidence(claims: dict[str, np.ndarray], i: int) -> list[Evidence]:
    out = []
    for name, p_fraud, p_legit, desc in RED_FLAGS:
        present = bool(claims[name][i])
        lr = p_fraud / p_legit if present else (1 - p_fraud) / (1 - p_legit)
        out.append(Evidence(name if present else f"no {name}", lr, detail=desc if present else ""))
    return out


def raw_scores(claims: dict[str, np.ndarray]) -> np.ndarray:
    n = len(claims["amount"])
    return np.array([fuse(BASE_FRAUD_RATE, fraud_evidence(claims, i)).posterior for i in range(n)])


def run(seed: int = 11) -> Report:
    claims = generate_claims(seed)
    n = len(claims["amount"])
    y = claims["is_fraud"].astype(int)
    raw = raw_scores(claims)

    # Calibrate on the first 40 percent, evaluate the decision on the remaining 60.
    split = int(n * 0.4)
    cal = Calibrator("isotonic").fit(raw[:split], y[:split])
    test = slice(split, n)
    p_cal = cal.predict(raw[test])
    p_raw = raw[test]
    y_test = y[test]

    report = Report("insurance", "Insurance: claims triage with cost-sensitive thresholds")
    report.add(
        "Fraud likelihood calibration (held-out claims)",
        md_table(
            ["Score", "Brier", "ECE", "Mean predicted", "Observed fraud rate"],
            [
                (
                    "naive-Bayes fusion (raw)",
                    brier_score(p_raw, y_test),
                    expected_calibration_error(p_raw, y_test),
                    float(p_raw.mean()),
                    float(y_test.mean()),
                ),
                (
                    "after isotonic calibration",
                    brier_score(p_cal, y_test),
                    expected_calibration_error(p_cal, y_test),
                    float(p_cal.mean()),
                    float(y_test.mean()),
                ),
            ],
        )
        + "\n\nReliability of the raw fused score (red flags are correlated, so the "
        "independence assumption overstates high scores):\n\n"
        + md_table(
            ["Bin", "Claims", "Mean predicted", "Observed"],
            [
                (f"{r.low:.1f} to {r.high:.1f}", r.count, r.mean_predicted, r.observed_rate)
                for r in reliability_table(p_raw, y_test, bins=5)
            ],
        ),
    )

    model = DecisionModel.from_yaml(MODEL_PATH)
    data = {
        "amount": claims["amount"][test],
        "claim_type": claims["claim_type"][test],
        "injury": claims["injury"][test],
        "fraud_p": p_cal,
        "is_fraud": claims["is_fraud"][test],
    }
    result = simulate(model, data=data, seed=seed)

    def breakdown(policy) -> tuple[float, float, float, dict[str, int]]:
        values = model.evaluate(policy.levers, data)
        lanes, counts = np.unique(values["lane"], return_counts=True)
        return (
            float(values["handling_cost"].mean()),
            float(values["leakage"].mean()),
            float(values["friction"].mean()),
            dict(zip(lanes.tolist(), counts.tolist(), strict=True)),
        )

    ranked = result.ranked()
    best = ranked[0]
    rows = []
    frontier = sorted(
        [p for p in result.policies if p.levers["fast_track_max"] == best.levers["fast_track_max"]],
        key=lambda p: p.levers["investigate_at"],
    )
    for p in frontier:
        h, leak, fr, lanes = breakdown(p)
        rows.append(
            (
                p.levers["investigate_at"],
                h,
                leak,
                fr,
                -p.expected_utility,
                lanes.get("investigate", 0),
            )
        )
    report.add(
        "Leakage vs handling cost trade-off",
        f"Average cost per claim on {len(y_test):,} held-out claims, fast-track limit "
        f"{best.levers['fast_track_max']}. Lowering the investigation threshold buys lower "
        "leakage with more handling cost and friction.\n\n"
        + md_table(
            ["Investigate at p >=", "Handling", "Leakage", "Friction", "Total", "Investigations"],
            rows,
        ),
    )

    report.add(
        "Best threshold pairs",
        md_table(
            ["investigate_at", "fast_track_max", "Cost per claim", "Saving vs all-standard"],
            [
                (
                    p.levers["investigate_at"],
                    p.levers["fast_track_max"],
                    -p.expected_utility,
                    _baseline(result) + p.expected_utility,
                )
                for p in ranked[:5]
            ],
        ),
    )

    # Cost-sensitive threshold from first principles, for comparison with the grid search.
    avg_fraud_amount = float(claims["amount"][claims["is_fraud"] == 1].mean())
    params = model.parameters
    extra = params["cost_investigate"] - params["cost_standard"]
    costs = CostMatrix(
        false_positive=extra + params["friction_investigate"],
        false_negative=(params["catch_investigate"] - params["catch_standard"]) * avg_fraud_amount,
    )
    analytic = costs.bayes_threshold()
    empirical = optimise_threshold(p_cal, y_test, costs)
    report.add(
        "Cost-sensitive threshold check",
        f"Investigating instead of standard handling costs {costs.false_positive:.0f} on an honest "
        f"claim and saves {costs.false_negative:,.0f} on an average fraudulent one. The Bayes "
        f"threshold is FP / (FP + FN) = **{analytic:.3f}**; the empirical cost-minimising cut on "
        f"the calibrated scores is **{empirical.threshold:.3f}** (precision {empirical.precision:.0%}, "
        f"recall {empirical.recall:.0%}). The grid search over the full model picked "
        f"{best.levers['investigate_at']}, which also accounts for amount-dependent leakage "
        "and the table's hard routing rules.",
    )

    traces = _example_traces(model, claims, p_cal, split, best.levers)
    report.add("Explanation traces for three claims", "\n".join(t.to_markdown() for t in traces))
    report.add("Decision table (YAML)", "```yaml\n" + _table_excerpt() + "```")

    h, leak, fr, lanes = breakdown(best)
    report.summary = [
        f"fraud score Brier {brier_score(p_raw, y_test):.4f} -> {brier_score(p_cal, y_test):.4f}, "
        f"ECE {expected_calibration_error(p_raw, y_test):.3f} -> "
        f"{expected_calibration_error(p_cal, y_test):.3f} after isotonic calibration",
        f"best thresholds: investigate at p >= {best.levers['investigate_at']}, "
        f"fast-track up to {best.levers['fast_track_max']}",
        f"cost per claim {-best.expected_utility:,.0f} (handling {h:,.0f}, leakage {leak:,.0f}, "
        f"friction {fr:,.0f}) vs {_baseline(result):,.0f} if every claim went to standard",
        f"lanes: {', '.join(f'{k} {v}' for k, v in sorted(lanes.items()))}",
        f"Bayes threshold {analytic:.3f}, empirical {empirical.threshold:.3f}",
    ]
    report.metrics = {
        "cost_per_claim": -best.expected_utility,
        "baseline_cost": _baseline(result),
        "brier_raw": brier_score(p_raw, y_test),
        "brier_calibrated": brier_score(p_cal, y_test),
        "bayes_threshold": analytic,
    }
    return report


def _baseline(result) -> float:
    """Cost per claim when nothing is fast-tracked or investigated."""
    for p in result.policies:
        if p.levers == {"investigate_at": 1.01, "fast_track_max": 0}:
            return -p.expected_utility
    raise KeyError("baseline policy missing from lever grid")


def _example_traces(model, claims, p_cal, offset, levers) -> list[Trace]:
    table = model.nodes["lane"].table
    picks = [int(np.argmax(p_cal)), int(np.argmin(np.abs(p_cal - 0.2))), int(np.argmin(p_cal))]
    traces = []
    for j in picks:
        i = offset + j
        ctx = {
            **model.parameters,
            **levers,
            "amount": float(claims["amount"][i]),
            "claim_type": str(claims["claim_type"][i]),
            "injury": bool(claims["injury"][i]),
            "fraud_p": float(p_cal[j]),
        }
        res = table.evaluate(ctx)
        fused = fuse(BASE_FRAUD_RATE, fraud_evidence(claims, i))
        traces.append(
            Trace(
                subject=f"Claim C{i:05d} ({ctx['claim_type']}, amount {ctx['amount']:,.0f})",
                recommendation=res.value,
                score=float(p_cal[j]),
                rules_fired=res.fired,
                evidence=fused.top(3),
                notes=[f"raw fused probability {fused.posterior:.3f}, calibrated {p_cal[j]:.3f}"],
            )
        )
    return traces


def _table_excerpt() -> str:
    text = MODEL_PATH.read_text()
    start = text.index("  - name: lane")
    end = text.index("  - name: handling_cost")
    return text[start:end]
