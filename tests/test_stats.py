"""Unit tests for the paired significance stats (no GPU/network)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from src.stats import mcnemar, paired_delta_ci, wilson_ci  # noqa: E402


def _correctness(n_both_right, n_b, n_c, n_both_wrong):
    """Build aligned (base, ft) correctness lists from a 2x2 contingency.
    n_b = ft right / base wrong ; n_c = ft wrong / base right."""
    base, ft = [], []
    base += [True] * n_both_right;  ft += [True] * n_both_right
    base += [False] * n_b;          ft += [True] * n_b
    base += [True] * n_c;           ft += [False] * n_c
    base += [False] * n_both_wrong; ft += [False] * n_both_wrong
    return base, ft


def test_mcnemar_counts_discordant_pairs():
    base, ft = _correctness(n_both_right=100, n_b=40, n_c=10, n_both_wrong=50)
    r = mcnemar(base, ft)
    assert r.n == 200
    assert r.ft_right_base_wrong == 40
    assert r.ft_wrong_base_right == 10
    assert r.base_correct == 110  # both_right + c
    assert r.ft_correct == 140    # both_right + b
    assert r.accuracy_delta == pytest.approx((140 - 110) / 200, abs=1e-6)


def test_mcnemar_large_lopsided_is_significant():
    # 38 net improvements on 1273 (the full-test-set scenario) -> significant
    base, ft = _correctness(n_both_right=560, n_b=120, n_c=82, n_both_wrong=511)
    r = mcnemar(base, ft)
    assert r.accuracy_delta > 0
    assert r.p_value < 0.05
    assert r.significant_05 is True


def test_mcnemar_small_sample_not_significant():
    # 8 net improvements on 300 (what we originally did) -> NOT significant
    base, ft = _correctness(n_both_right=130, n_b=24, n_c=16, n_both_wrong=130)
    r = mcnemar(base, ft)
    assert r.significant_05 is False


def test_mcnemar_no_discordant_pairs_is_pvalue_one():
    base, ft = _correctness(n_both_right=50, n_b=0, n_c=0, n_both_wrong=50)
    r = mcnemar(base, ft)
    assert r.p_value == 1.0
    assert r.significant_05 is False


def test_mcnemar_symmetric_split_not_significant():
    base, ft = _correctness(n_both_right=100, n_b=30, n_c=30, n_both_wrong=100)
    r = mcnemar(base, ft)
    assert r.accuracy_delta == pytest.approx(0.0, abs=1e-9)
    assert r.p_value > 0.5


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError):
        mcnemar([True, False], [True])


def test_paired_delta_ci_brackets_point_estimate():
    lo, hi = paired_delta_ci(b=120, c=82, n=1273)
    delta = (120 - 82) / 1273
    assert lo < delta < hi
    assert lo > 0  # significant improvement excludes zero


def test_wilson_ci_within_unit_interval_and_brackets():
    lo, hi = wilson_ci(143, 300)
    assert 0.0 <= lo < 143 / 300 < hi <= 1.0


def test_exact_vs_known_binomial_pvalue():
    # b=10, c=0 discordant: exact two-sided p = 2 * (0.5^10) = 0.001953125
    base, ft = _correctness(n_both_right=5, n_b=10, n_c=0, n_both_wrong=5)
    r = mcnemar(base, ft)
    assert r.p_value == pytest.approx(2 * (0.5 ** 10), abs=1e-6)
