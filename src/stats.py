"""Paired significance testing for the base-vs-fine-tuned comparison.

Both models answer the *same* questions, so the correct analysis is a paired
one. McNemar's test looks only at the questions where the two models disagree
(the discordant pairs), which is far more powerful than comparing two
independent accuracies. All functions here are pure and unit-tested.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Sequence


@dataclass
class PairedResult:
    n: int                    # total examples
    base_correct: int
    ft_correct: int
    base_accuracy: float
    ft_accuracy: float
    accuracy_delta: float     # ft - base
    # discordant pairs
    ft_right_base_wrong: int  # "b": fine-tune fixed these
    ft_wrong_base_right: int  # "c": fine-tune broke these
    mcnemar_statistic: float
    p_value: float
    delta_ci_low: float       # 95% CI on accuracy_delta
    delta_ci_high: float
    significant_05: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _binom_two_sided_p(b: int, c: int) -> float:
    """Exact McNemar p-value: under H0 each discordant pair is a fair coin,
    so min(b,c) ~ Binomial(b+c, 0.5). Two-sided exact p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) for X ~ Binom(n, 0.5), doubled and capped at 1 (symmetric).
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _chi2_mcnemar_p(b: int, c: int) -> float:
    """McNemar chi-square with continuity correction (used for large samples
    where the exact binomial is unwieldy). Survival function of chi2_1."""
    n = b + c
    if n == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / n
    # SF of chi-square with df=1 is erfc(sqrt(stat/2)).
    return math.erfc(math.sqrt(stat / 2.0))


def mcnemar(base_correct: Sequence[bool], ft_correct: Sequence[bool]) -> "PairedResult":
    """Run the full paired analysis on aligned per-question correctness.

    Uses the exact binomial test when the number of discordant pairs is small
    (<= 1000) and the continuity-corrected chi-square otherwise.
    """
    if len(base_correct) != len(ft_correct):
        raise ValueError("base_correct and ft_correct must be the same length")
    n = len(base_correct)
    if n == 0:
        raise ValueError("need at least one example")

    b = sum(1 for bc, fc in zip(base_correct, ft_correct) if fc and not bc)
    c = sum(1 for bc, fc in zip(base_correct, ft_correct) if bc and not fc)

    base_n = sum(1 for x in base_correct if x)
    ft_n = sum(1 for x in ft_correct if x)
    base_acc = base_n / n
    ft_acc = ft_n / n
    delta = ft_acc - base_acc

    n_disc = b + c
    if n_disc == 0:
        statistic, p = 0.0, 1.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / n_disc  # continuity-corrected
        p = _binom_two_sided_p(b, c) if n_disc <= 1000 else _chi2_mcnemar_p(b, c)

    lo, hi = paired_delta_ci(b, c, n)

    return PairedResult(
        n=n, base_correct=base_n, ft_correct=ft_n,
        base_accuracy=round(base_acc, 4), ft_accuracy=round(ft_acc, 4),
        accuracy_delta=round(delta, 4),
        ft_right_base_wrong=b, ft_wrong_base_right=c,
        mcnemar_statistic=round(statistic, 4), p_value=round(p, 6),
        delta_ci_low=round(lo, 4), delta_ci_high=round(hi, 4),
        significant_05=p < 0.05,
    )


def paired_delta_ci(b: int, c: int, n: int, z: float = 1.96):
    """95% CI for the paired accuracy difference (b - c) / n.

    Variance of the difference of paired proportions:
        Var = [ (b + c) - (b - c)^2 / n ] / n^2
    (standard result; the second term corrects for the pairing).
    """
    if n == 0:
        return (0.0, 0.0)
    delta = (b - c) / n
    var = ((b + c) - (b - c) ** 2 / n) / (n ** 2)
    se = math.sqrt(max(var, 0.0))
    return (delta - z * se, delta + z * se)


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a single proportion k/n (better than Wald
    for small n or extreme proportions)."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))
