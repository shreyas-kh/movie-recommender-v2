"""
Paired bootstrap significance testing for per-user ranking-metric differences.

Point estimates (as printed by run_evaluation.py's tables) can't tell you
whether an apparent gap between two models is real or sampling noise. A paired
bootstrap answers that: resample the same set of users with replacement many
times, recompute the mean paired difference each time, and read off where the
resulting distribution's 95% interval falls. If it excludes zero, the
difference is unlikely to be noise at the 95% level.

"Paired" matters here: each resample draws the SAME users for both models
(rather than resampling each model's scores independently), so the comparison
correctly accounts for per-user correlation — e.g. a user who's easy for
everyone to satisfy shouldn't inflate the apparent gap between two models that
both do well on them.
"""
from typing import List, Sequence, Tuple

import numpy as np


def paired_bootstrap_ci(
    diffs: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean of paired per-user differences (a - b).

    `diffs` is one value per user: metric_a(user) - metric_b(user). Returns
    (observed_mean, ci_low, ci_high) using a percentile bootstrap: n_resamples
    samples of len(diffs) users drawn with replacement, the mean taken each
    time, and the (1-ci)/2 / 1-(1-ci)/2 percentiles of that distribution as
    the interval. A CI that excludes 0 means the difference is significant at
    the `ci` level under this test.

    Returns (0.0, 0.0, 0.0) for an empty input (nothing to resample).
    """
    arr = np.asarray(diffs, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = arr[idx].mean(axis=1)

    alpha = 1.0 - ci
    lo, hi = np.percentile(resample_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def paired_bootstrap_report(
    rec_ids_a: List[Sequence[int]],
    rec_ids_b: List[Sequence[int]],
    relevant_ids: List[set],
    relevant_ratings: List[dict],
    k: int,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict:
    """Convenience wrapper: bootstrap CIs for precision/recall/NDCG@k
    differences (a - b) across a list of users, given each user's top-k lists
    for model a and b plus their ground truth. All five list/dict arguments
    must be aligned by index (one entry per user, same order).

    Returns {"precision": (mean, lo, hi), "recall": (...), "ndcg": (...)}.
    """
    from models.evaluate import ndcg_at_k, precision_at_k, recall_at_k

    p_diffs, r_diffs, g_diffs = [], [], []
    for rec_a, rec_b, rel_ids, rel_ratings in zip(
        rec_ids_a, rec_ids_b, relevant_ids, relevant_ratings
    ):
        p_diffs.append(precision_at_k(rec_a, rel_ids, k) - precision_at_k(rec_b, rel_ids, k))
        r_diffs.append(recall_at_k(rec_a, rel_ids, k) - recall_at_k(rec_b, rel_ids, k))
        g_diffs.append(
            ndcg_at_k(rec_a, rel_ratings, k) - ndcg_at_k(rec_b, rel_ratings, k)
        )

    return {
        "precision": paired_bootstrap_ci(p_diffs, n_resamples=n_resamples, seed=seed),
        "recall": paired_bootstrap_ci(r_diffs, n_resamples=n_resamples, seed=seed),
        "ndcg": paired_bootstrap_ci(g_diffs, n_resamples=n_resamples, seed=seed),
    }
