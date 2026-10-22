import math

import numpy as np
import pytest

from decision_lab.core.fusion import (
    Calibrator,
    Evidence,
    brier_score,
    discount_lr,
    expected_calibration_error,
    fuse,
    reliability_table,
    update_hypotheses,
)


def test_log_odds_fusion_hand_checked():
    # Prior 0.2 -> odds 0.25. LRs 4 and 2 -> posterior odds 2 -> probability 2/3.
    result = fuse(0.2, [Evidence("a", 4.0), Evidence("b", 2.0)])
    assert result.posterior == pytest.approx(2 / 3)
    assert result.log_odds_shift == pytest.approx(math.log(8))
    assert [c.name for c in result.top(1)] == ["a"]


def test_order_does_not_matter_and_lr_one_is_neutral():
    ev = [Evidence("a", 3.0), Evidence("b", 0.25), Evidence("c", 1.0)]
    assert fuse(0.1, ev).posterior == pytest.approx(fuse(0.1, ev[::-1]).posterior)
    assert fuse(0.1, [Evidence("c", 1.0)]).posterior == pytest.approx(0.1)
    assert fuse(0.1, ev).contributions[2].direction == "neutral"


def test_reliability_discount():
    assert discount_lr(9.0, 1.0) == 9.0
    assert discount_lr(9.0, 0.0) == 1.0
    assert discount_lr(9.0, 0.5) == pytest.approx(5.0)
    assert Evidence("x", 0.1, reliability=0.5).effective_lr == pytest.approx(0.55)
    with pytest.raises(ValueError):
        discount_lr(2.0, 1.5)


def test_multi_hypothesis_update():
    # One reliable item: P(e|H) = [0.8, 0.2, 0.2]; uniform prior -> [2/3, 1/6, 1/6].
    post = update_hypotheses([1 / 3] * 3, [[0.8, 0.2, 0.2]], [1.0])
    np.testing.assert_allclose(post, [2 / 3, 1 / 6, 1 / 6])
    # A completely unreliable source leaves the prior unchanged.
    np.testing.assert_allclose(update_hypotheses([0.5, 0.5], [[0.9, 0.1]], [0.0]), [0.5, 0.5])
    # Two hypotheses with full reliability agree with the binary log-odds fusion.
    binary = fuse(0.3, [Evidence("e", 0.6 / 0.2)]).posterior
    assert update_hypotheses([0.3, 0.7], [[0.6, 0.2]], [1.0])[0] == pytest.approx(binary)


def test_brier_score_and_reliability_table():
    p = np.array([0.0, 1.0, 0.5, 0.5])
    y = np.array([0, 1, 1, 0])
    assert brier_score(p, y) == pytest.approx(0.125)
    rows = reliability_table(p, y, bins=2)
    assert [(r.count, r.observed_rate) for r in rows] == [(1, 0.0), (3, 2 / 3)]


def _miscalibrated(seed=0, n=20_000):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < true_p).astype(int)
    logit = np.log(true_p / (1 - true_p))
    overconfident = 1 / (1 + np.exp(-3 * logit))  # scores pushed towards 0 and 1
    return overconfident, y


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_improves_overconfident_scores(method):
    scores, y = _miscalibrated()
    cal = Calibrator(method).fit(scores[:10_000], y[:10_000])
    fixed = cal.predict(scores[10_000:])
    raw_ece = expected_calibration_error(scores[10_000:], y[10_000:])
    new_ece = expected_calibration_error(fixed, y[10_000:])
    assert new_ece < raw_ece / 3
    assert brier_score(fixed, y[10_000:]) < brier_score(scores[10_000:], y[10_000:])


def test_calibrator_guards():
    with pytest.raises(ValueError):
        Calibrator("magic")
    with pytest.raises(RuntimeError):
        Calibrator().predict(np.array([0.5]))
