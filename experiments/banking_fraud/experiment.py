"""Transaction alert disposition under investigator capacity.

Synthetic transactions -> gradient boosting score -> Platt calibration -> threshold
optimisation with a cap on alerts per day. Three strategies are compared on the same
held-out days: rules only, model only, and model plus rules (a COLLECT decision table
of hard rules that can force an alert regardless of the score).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from decision_lab.core import (
    Calibrator,
    CostMatrix,
    DecisionTable,
    Route,
    Trace,
    brier_score,
    md_table,
    optimise_routing,
    reliability_table,
)
from decision_lab.core.fusion import Contribution, expected_calibration_error
from decision_lab.core.policy import binary_cost, optimise_threshold, routing_cost
from experiments.common import Report

FEATURES = [
    "amount_z",  # amount vs the customer's own history (z-score)
    "new_payee",
    "foreign",
    "night",
    "device_change",
    "velocity_1h",
    "account_age_days",
    "merchant_risk",
]
INVESTIGATION_COST = 25.0
FALSE_BLOCK_COST = 400.0
CAPACITY_PER_DAY = 40  # alerts the investigation team can work per day
TX_PER_DAY = 5000

RULES = DecisionTable.from_dict(
    {
        "name": "hard-rules",
        "hit_policy": "COLLECT MAX",
        "inputs": [
            "amount",
            "new_payee",
            "device_change",
            "velocity_1h",
            "foreign",
            "merchant_risk",
        ],
        "outputs": ["priority"],
        "rules": [
            {
                "when": ["> 5000", "true", "-", "-", "-", "-"],
                "then": ["3"],
                "annotation": "large payment to a new payee",
            },
            {
                "when": ["-", "-", "true", ">= 4", "-", "-"],
                "then": ["3"],
                "annotation": "burst of transactions after a device change",
            },
            {
                "when": ["> 1500", "-", "-", "-", "true", "> 0.6"],
                "then": ["2"],
                "annotation": "foreign payment to a high-risk merchant category",
            },
            {
                "when": ["-", "true", "true", "-", "-", "-"],
                "then": ["1"],
                "annotation": "new payee from a new device",
            },
        ],
    }
)
RULE_ALERT_PRIORITY = 2  # rules-only strategy alerts on priority >= 2


def generate_transactions(seed: int, days: int = 20) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = days * TX_PER_DAY
    day = np.repeat(np.arange(days), TX_PER_DAY)
    account_age = rng.gamma(2.0, 400.0, n)
    # Fraud propensity rises for young accounts; base rate around 0.4 percent.
    logit = -5.0 + 1.2 * (account_age < 90)
    is_fraud = rng.random(n) < 1 / (1 + np.exp(-logit))
    f = is_fraud
    amount = np.round(rng.lognormal(np.log(np.where(f, 900, 120)), np.where(f, 1.1, 1.0)), 2)
    out = {
        "day": day,
        "amount": amount,
        "amount_z": (np.log(amount) - np.log(120)) / 1.0 + rng.normal(0, 0.3, n),
        "new_payee": rng.random(n) < np.where(f, 0.65, 0.12),
        "foreign": rng.random(n) < np.where(f, 0.35, 0.06),
        "night": rng.random(n) < np.where(f, 0.40, 0.15),
        "device_change": rng.random(n) < np.where(f, 0.45, 0.03),
        "velocity_1h": rng.poisson(np.where(f, 3.0, 0.8)),
        "account_age_days": account_age,
        "merchant_risk": np.clip(rng.beta(np.where(f, 4, 2), np.where(f, 3, 6)), 0, 1),
        "is_fraud": is_fraud.astype(int),
    }
    return out


def feature_matrix(tx: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.asarray(tx[k], dtype=float) for k in FEATURES])


def subset(tx: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in tx.items()}


def rule_priority(tx: dict[str, np.ndarray]) -> np.ndarray:
    ctx = {k: tx[k] for k in RULES.inputs}
    return RULES.evaluate_many(ctx)[0]


def daily_cap(candidate: np.ndarray, priority: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Keep at most CAPACITY_PER_DAY candidates per day, highest priority first."""
    keep = np.zeros_like(candidate, dtype=bool)
    for d in np.unique(day):
        idx = np.flatnonzero(candidate & (day == d))
        keep[idx[np.argsort(-priority[idx])[:CAPACITY_PER_DAY]]] = True
    return keep


def strategy_outcome(alert: np.ndarray, tx: dict[str, np.ndarray], days: int) -> dict[str, float]:
    y = tx["is_fraud"].astype(bool)
    prevented = float(tx["amount"][alert & y].sum())
    missed = float(tx["amount"][~alert & y].sum())
    cost = alert.sum() * INVESTIGATION_COST
    return {
        "alerts_per_day": alert.sum() / days,
        "precision": float((alert & y).sum() / max(alert.sum(), 1)),
        "recall_value": prevented / max(prevented + missed, 1e-9),
        "prevented_per_day": prevented / days,
        "net_per_day": (prevented - cost) / days,
        "peak_day_alerts": float(np.bincount(tx["day"][alert] - tx["day"].min()).max())
        if alert.any()
        else 0.0,
    }


def run(seed: int = 23) -> Report:
    tx = generate_transactions(seed)
    days_total = int(tx["day"].max()) + 1
    train_days, cal_days = 10, 4
    train = tx["day"] < train_days
    calib = (tx["day"] >= train_days) & (tx["day"] < train_days + cal_days)
    test = tx["day"] >= train_days + cal_days
    test_days = days_total - train_days - cal_days
    tr, ca, te = subset(tx, train), subset(tx, calib), subset(tx, test)

    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=1.0,
        random_state=seed,
    )
    clf.fit(feature_matrix(tr), tr["is_fraud"])
    raw_ca = clf.predict_proba(feature_matrix(ca))[:, 1]
    raw_te = clf.predict_proba(feature_matrix(te))[:, 1]
    cal = Calibrator("platt").fit(raw_ca, ca["is_fraud"])
    p_te = cal.predict(raw_te)
    y_te = te["is_fraud"]

    report = Report("banking_fraud", "Banking fraud: alert disposition under investigator capacity")
    report.add(
        "Data and model",
        f"{len(tx['amount']):,} synthetic transactions over {days_total} days "
        f"(fraud rate {tx['is_fraud'].mean():.2%}). Days 0 to {train_days - 1} train a gradient "
        f"boosting classifier, days {train_days} to {train_days + cal_days - 1} fit Platt "
        f"scaling, the last {test_days} days are held out for every comparison below.\n\n"
        + md_table(
            ["Score", "Brier", "ECE"],
            [
                (
                    "gradient boosting (raw)",
                    brier_score(raw_te, y_te),
                    expected_calibration_error(raw_te, y_te),
                ),
                (
                    "after Platt scaling",
                    brier_score(p_te, y_te),
                    expected_calibration_error(p_te, y_te),
                ),
            ],
        )
        + "\n\nReliability (calibrated, held-out):\n\n"
        + md_table(
            ["Bin", "Transactions", "Mean predicted", "Observed"],
            [
                (f"{r.low:.1f} to {r.high:.1f}", r.count, r.mean_predicted, r.observed_rate)
                for r in reliability_table(p_te, y_te, bins=5)
            ],
        ),
    )

    # Threshold under capacity: cost of a false alert is the investigation; cost of a
    # missed fraud is its amount. Using the mean fraud amount keeps the matrix scalar;
    # the strategy table below then scores real amounts.
    mean_fraud_amount = float(tr["amount"][tr["is_fraud"] == 1].mean())
    costs = CostMatrix(
        false_positive=INVESTIGATION_COST,
        false_negative=mean_fraud_amount,
        true_positive=INVESTIGATION_COST,
    )
    # Thresholds are chosen on the calibration days and only then applied to the test
    # days, so the comparison below is out of sample.
    p_ca = cal.predict(raw_ca)
    unconstrained = optimise_threshold(p_ca, ca["is_fraud"], costs)
    constrained = optimise_threshold(p_ca, ca["is_fraud"], costs, max_flagged=CAPACITY_PER_DAY * cal_days)
    test_unc = binary_cost(p_te, y_te, unconstrained.threshold, costs)
    test_con = binary_cost(p_te, y_te, constrained.threshold, costs)
    expected_value = p_te * te["amount"]  # expected loss if the alert is not worked
    report.add(
        "Threshold optimisation",
        md_table(
            ["Setting", "Threshold", "Alerts/day (test)", "Precision (test)", "Recall (test)"],
            [
                ("Bayes threshold from the cost matrix", costs.bayes_threshold(), "", "", ""),
                (
                    "cost-optimal, no capacity limit",
                    unconstrained.threshold,
                    test_unc.flagged / test_days,
                    test_unc.precision,
                    test_unc.recall,
                ),
                (
                    f"cost-optimal, at most {CAPACITY_PER_DAY} alerts/day",
                    constrained.threshold,
                    test_con.flagged / test_days,
                    test_con.precision,
                    test_con.recall,
                ),
            ],
        )
        + f"\n\nThresholds are fitted on the calibration days. With a {CAPACITY_PER_DAY}/day team "
        "the binding constraint is capacity, not cost: the unconstrained optimum would generate "
        f"{test_unc.flagged / test_days:.0f} alerts/day, so the optimiser raises the threshold "
        "until the queue fits.",
    )

    priority = rule_priority(te)
    rules_alert = priority >= RULE_ALERT_PRIORITY
    model_alert = daily_cap(p_te >= constrained.threshold, p_te, te["day"])
    # Model + rules: hard rules always alert; the remaining daily capacity goes to the
    # highest expected value (probability x amount) instead of the raw probability.
    forced = priority >= 3
    combined = daily_cap(
        forced | (p_te >= constrained.threshold / 3), expected_value + 1e9 * forced, te["day"]
    )

    strategies = {
        "rules only": rules_alert,
        "model only (probability threshold)": model_alert,
        "model + rules (hard rules, then expected value)": combined,
    }
    outcomes = {name: strategy_outcome(a, te, test_days) for name, a in strategies.items()}
    report.add(
        "Strategy comparison on held-out days",
        md_table(
            [
                "Strategy",
                "Alerts/day",
                "Peak day",
                "Precision",
                "Value recall",
                "Loss prevented/day",
                "Net of investigation cost/day",
            ],
            [
                (
                    name,
                    o["alerts_per_day"],
                    o["peak_day_alerts"],
                    o["precision"],
                    o["recall_value"],
                    o["prevented_per_day"],
                    o["net_per_day"],
                )
                for name, o in outcomes.items()
            ],
        )
        + f"\n\nCapacity is {CAPACITY_PER_DAY} alerts/day. Model strategies keep the top of "
        "each day's queue up to capacity (alerts beyond it are never worked). The rules-only "
        "strategy has no capacity control: its queue is whatever the rules produce. The "
        "combined strategy lowers the score cut to a third and ranks by probability x amount, "
        "so capacity goes to the alerts with the largest expected loss.",
    )

    routing_costs = CostMatrix(
        false_positive=FALSE_BLOCK_COST, false_negative=mean_fraud_amount, review=INVESTIGATION_COST
    )
    fitted = optimise_routing(
        p_ca, ca["is_fraud"], routing_costs, review_capacity=CAPACITY_PER_DAY * cal_days
    )
    routing = routing_cost(p_te, y_te, fitted.policy, routing_costs)
    report.add(
        "Human-in-the-loop routing",
        "Three lanes: auto-act (block the payment without review; wrongly blocking a "
        f"legitimate payment costs {FALSE_BLOCK_COST:.0f} in friction and attrition), human "
        f"review (costs {INVESTIGATION_COST:.0f}, assumed to resolve the case correctly), reject "
        f"(let it through). Review volume is capped at {CAPACITY_PER_DAY}/day; thresholds are "
        "fitted on the calibration days and applied to the test days.\n\n"
        + md_table(
            ["Lane", "Probability band", "Transactions"],
            [
                (
                    Route.AUTO_ACT.value,
                    f">= {routing.policy.act_threshold:.3f}",
                    routing.counts[Route.AUTO_ACT.value],
                ),
                (
                    Route.HUMAN_REVIEW.value,
                    f"{routing.policy.review_threshold:.3f} to {routing.policy.act_threshold:.3f}",
                    routing.counts[Route.HUMAN_REVIEW.value],
                ),
                (
                    Route.REJECT.value,
                    f"< {routing.policy.review_threshold:.3f}",
                    routing.counts[Route.REJECT.value],
                ),
            ],
        )
        + f"\n\nOn the test days: {routing.errors['auto_false_positive']} legitimate payments "
        f"auto-blocked, {routing.errors['missed_positive']} frauds let through.",
    )

    report.add(
        "Explanation traces",
        "\n".join(t.to_markdown() for t in _traces(te, p_te, routing.policy, clf)),
    )

    best_name = max(outcomes, key=lambda k: outcomes[k]["net_per_day"])
    report.summary = [
        f"calibrated score: Brier {brier_score(p_te, y_te):.4f}, "
        f"ECE {expected_calibration_error(p_te, y_te):.4f}",
        f"capacity-constrained threshold {constrained.threshold:.3f} "
        f"({test_con.flagged / test_days:.0f} alerts/day, precision {test_con.precision:.0%} on test days)",
        *[
            f"{name}: net {o['net_per_day']:,.0f}/day, {o['alerts_per_day']:.0f} alerts/day"
            for name, o in outcomes.items()
        ],
        f"best strategy: {best_name}",
    ]
    report.metrics = {
        "threshold": constrained.threshold,
        "alerts_per_day_model": outcomes["model only (probability threshold)"]["alerts_per_day"],
        **{f"net_{i}": o["net_per_day"] for i, o in enumerate(outcomes.values())},
        "brier": brier_score(p_te, y_te),
    }
    return report


def _traces(te, p_te, policy, clf) -> list[Trace]:
    """Explain three transactions; feature contributions are measured by ablation.

    Each feature is replaced by its population median and the change in the model's
    log-odds is reported. This is a cheap, model-agnostic local attribution that is good
    enough to tell an investigator why an alert fired.
    """
    X = feature_matrix(te)
    medians = np.median(X, axis=0)
    lanes = policy.route_many(p_te)
    picks = []
    for lane in (Route.AUTO_ACT, Route.HUMAN_REVIEW, Route.REJECT):
        idx = np.flatnonzero(lanes == lane.value)
        if len(idx):
            picks.append(int(idx[np.argsort(-p_te[idx])[len(idx) // 3]]))
    traces = []
    for i in picks:
        x = X[i : i + 1]  # ablation explains the raw model score, before calibration
        base = _log_odds(clf.predict_proba(x)[0])
        contribs = []
        for j, name in enumerate(FEATURES):
            x2 = x.copy()
            x2[0, j] = medians[j]
            contribs.append((name, base - _log_odds(clf.predict_proba(x2)[0])))
        contribs.sort(key=lambda c: abs(c[1]), reverse=True)
        ctx = {k: (te[k][i].item() if hasattr(te[k][i], "item") else te[k][i]) for k in RULES.inputs}
        fired = RULES.evaluate(ctx)
        traces.append(
            Trace(
                subject=f"Transaction T{i:06d} (amount {te['amount'][i]:,.2f})",
                recommendation=policy.route(float(p_te[i])).value,
                route=policy.route(float(p_te[i])).value,
                score=float(p_te[i]),
                rules_fired=fired.fired,
                evidence=[Contribution(n, v, f"(value {_fmt(te[n][i])})") for n, v in contribs[:3]],
            )
        )
    return traces


def _log_odds(proba: np.ndarray) -> float:
    p = float(np.clip(proba[1], 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _fmt(v) -> str:
    if isinstance(v, np.bool_ | bool):
        return "yes" if v else "no"
    if isinstance(v, np.floating | float):
        return f"{v:.2f}"
    return str(v)
