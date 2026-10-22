"""One smoke test per domain plus an end-to-end CLI run."""

import pytest

from decision_lab.cli import DOMAINS, main
from experiments.banking_fraud import experiment as banking
from experiments.insurance import experiment as insurance
from experiments.intelligence import experiment as intelligence
from experiments.life_sciences import experiment as life


def test_life_sciences():
    report = life.run(seed=7, n=4000)
    m = report.metrics
    assert m["spend"] <= 420
    assert m["expected_npv"] > 0
    assert 0 <= m["evsi"] <= m["evppi"] + 1.0 <= m["evpi"] + 1.0
    assert "Tornado" in report.to_markdown(7)


def test_insurance():
    report = insurance.run(seed=11)
    m = report.metrics
    assert m["cost_per_claim"] < m["baseline_cost"]
    assert m["brier_calibrated"] <= m["brier_raw"] + 1e-3
    assert 0 < m["bayes_threshold"] < 0.5
    assert "rule 1" in report.to_markdown(11)


def test_banking_fraud():
    report = banking.run(seed=23)
    m = report.metrics
    assert m["alerts_per_day_model"] <= banking.CAPACITY_PER_DAY
    rules_only, model_only, combined = m["net_0"], m["net_1"], m["net_2"]
    assert model_only > rules_only
    assert combined >= model_only * 0.98


def test_intelligence():
    report = intelligence.run(seed=5)
    m = report.metrics
    assert 0.25 < m["posterior_lead"] <= 1.0
    assert m["n_reports"] >= 3
    assert "What evidence would change the conclusion" in report.to_markdown(5)


def test_intelligence_recovers_truth_on_average():
    """With many seeds, the true hypothesis (H4) should lead more often than any other."""
    leads = [int(intelligence.run(seed=s).metrics["lead_index"]) for s in range(30)]
    assert max(range(4), key=leads.count) == 3


@pytest.mark.parametrize("domain", list(DOMAINS))
def test_cli_writes_report(domain, tmp_path, capsys):
    assert main(["run", domain, "--out", str(tmp_path)]) == 0
    text = (tmp_path / f"{domain}.md").read_text()
    assert text.startswith("# ") and "## Summary" in text
    assert "->" in capsys.readouterr().out
