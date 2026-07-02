"""
Offline ranking evaluation: SVD-only vs Content-only vs Hybrid.

Protocol
--------
* Same 80/20 split as train.py — per-user temporal by default (each user's
  earliest 80% trains, most recent 20% is held out; the model never sees a
  user's future), or the original shuffled split via --split random for
  comparison. All three recommenders see ONLY the train split: SVD is fit on
  it, the hybrid's ratings_df is it, and the content model's liked-movie
  inputs are derived from it. Relevance is judged purely on the held-out test
  split (movies the user rated >= 4.0).
* Because SVDRecommender's rated_mask comes from its fit data, test-set movies
  remain recommendable — no leakage in either direction.
* Sampled users need >= MIN_TEST_RELEVANT relevant test movies (so recall and
  NDCG aren't decided by a single item) and at least one liked train movie
  (so the content model has an input).

Run from the project root:
    python models/run_evaluation.py                  # temporal split (default)
    python models/run_evaluation.py --split random   # original shuffled split
"""
import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.content_recommender import ContentRecommender
from models.evaluate import ndcg_at_k, precision_at_k, recall_at_k
from models.hybrid_recommender import HybridRecommender
from models.recommender import SVDRecommender
from models.split import split_ratings

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
N_COMPONENTS = 50
K = 10                  # evaluate top-10 lists
N_USERS = 100           # users sampled for evaluation
MIN_TEST_RELEVANT = 5   # user must have >= this many 4.0+ test ratings
ALPHA = 0.5             # hybrid blend weight for the three-model comparison
SEED = 42
RELEVANT_THRESHOLD = 4.0
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]  # sweep: pure content -> pure SVD
SPARSE_THRESHOLD = 20   # "sparse-history" = fewer than this many TRAIN ratings
# The dedicated sparse-user protocol relaxes the relevance requirement: sparse
# users rarely have 5+ relevant test ratings (that filter is itself biased
# against low-activity users), so >= 2 keeps the sample meaningful in size.
SPARSE_MIN_TEST_RELEVANT = 2


def _liked_from_train(train_df: pd.DataFrame, user_id: int) -> Tuple[List[int], List[float]]:
    """A user's liked movies from the TRAIN split only (>=4.0, fallback >=3.5),
    plus their ratings as content-model weights. Mirrors the hybrid's rule."""
    user = train_df[train_df["userId"] == user_id]
    for thresh in (4.0, 3.5):
        liked = user[user["rating"] >= thresh]
        if not liked.empty:
            return (
                [int(m) for m in liked["movieId"].tolist()],
                [float(r) for r in liked["rating"].tolist()],
            )
    return [], []


def _sweep_per_user(
    hybrid: HybridRecommender,
    users: List[int],
    relevant_by_user: Dict[int, Dict[int, float]],
) -> Dict[float, Dict[int, Tuple[float, float, float]]]:
    """Per-user (precision, recall, ndcg) at each alpha in ALPHAS, so results
    can be aggregated over any user subset without recomputing."""
    per_alpha: Dict[float, Dict[int, Tuple[float, float, float]]] = {}
    for alpha in ALPHAS:
        per_user: Dict[int, Tuple[float, float, float]] = {}
        for uid in users:
            rel_ratings = relevant_by_user[uid]
            rel_ids = set(rel_ratings)
            rec_ids = [m for m, _ in hybrid.recommend(user_id=uid, n=K, alpha=alpha)]
            per_user[uid] = (
                precision_at_k(rec_ids, rel_ids, K),
                recall_at_k(rec_ids, rel_ids, K),
                ndcg_at_k(rec_ids, rel_ratings, K),
            )
        per_alpha[alpha] = per_user
    return per_alpha


def _print_sweep(
    per_alpha: Dict[float, Dict[int, Tuple[float, float, float]]],
    user_subset: List[int],
    title: str,
) -> None:
    print(f"\n{title}")
    header = f"{'alpha':>6} {'Precision@' + str(K):>13} {'Recall@' + str(K):>11} {'NDCG@' + str(K):>9}"
    print(header)
    print("-" * len(header))
    n_sub = float(len(user_subset))
    for alpha in ALPHAS:
        metrics = [per_alpha[alpha][uid] for uid in user_subset]
        p = sum(m[0] for m in metrics) / n_sub
        r = sum(m[1] for m in metrics) / n_sub
        g = sum(m[2] for m in metrics) / n_sub
        print(f"{alpha:>6.2f} {p:>13.4f} {r:>11.4f} {g:>9.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ranking evaluation: SVD vs Content vs Hybrid.")
    parser.add_argument(
        "--split", choices=["temporal", "random"], default="temporal",
        help="temporal = per-user chronological 80/20 (realistic, default); "
             "random = shuffled split (leaks future ratings; for comparison)",
    )
    args = parser.parse_args()

    print("Loading data...")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")

    train_df, test_df = split_ratings(ratings, method=args.split, test_size=0.2, seed=SEED)
    print(f"  {args.split} split: train={len(train_df):,} ratings | test={len(test_df):,} ratings")

    print("Fitting models on the train split...")
    svd = SVDRecommender(n_components=N_COMPONENTS).fit(train_df)
    # Item features only (no ratings): genre multi-hot + overview embeddings
    # (beta=0.5 combined scoring; silently genre-only if the npz is absent).
    content = ContentRecommender().fit(
        movies, embeddings_path=DATA_DIR.parent / "overview_embeddings.npz"
    )
    hybrid = HybridRecommender(svd, content, train_df)

    # Ground truth: per user, the test-set movies they rated >= 4.0 (+ ratings
    # for NDCG's gain weighting).
    relevant_test = test_df[test_df["rating"] >= RELEVANT_THRESHOLD]
    relevant_by_user: Dict[int, Dict[int, float]] = {
        int(uid): dict(zip(grp["movieId"].astype(int), grp["rating"].astype(float)))
        for uid, grp in relevant_test.groupby("userId")
    }

    # Eligible users: known to SVD, enough relevant test items, and at least
    # one liked train movie to drive the content model.
    eligible = []
    for uid, rel in relevant_by_user.items():
        if uid not in svd.user_index or len(rel) < MIN_TEST_RELEVANT:
            continue
        liked_ids, _ = _liked_from_train(train_df, uid)
        if liked_ids:
            eligible.append(uid)

    rng = random.Random(SEED)
    sampled = sorted(rng.sample(eligible, min(N_USERS, len(eligible))))
    print(f"  {len(eligible)} eligible users; evaluating {len(sampled)} "
          f"(>= {MIN_TEST_RELEVANT} relevant test movies each)\n")

    models = ["SVD-only", "Content-only", f"Hybrid (α={ALPHA})"]
    sums: Dict[str, Dict[str, float]] = {
        m: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for m in models
    }

    for uid in sampled:
        rel_ratings = relevant_by_user[uid]
        rel_ids = set(rel_ratings)
        liked_ids, weights = _liked_from_train(train_df, uid)
        seen_train = [int(m) for m in
                      train_df.loc[train_df["userId"] == uid, "movieId"]]

        recs = {
            "SVD-only": [m for m, _ in svd.recommend(uid, n=K)],
            "Content-only": [m for m, _ in content.recommend(
                liked_ids, n=K, weights=weights, exclude=seen_train)],
            models[2]: [m for m, _ in hybrid.recommend(
                user_id=uid, n=K, alpha=ALPHA)],
        }

        for name, rec_ids in recs.items():
            sums[name]["precision"] += precision_at_k(rec_ids, rel_ids, K)
            sums[name]["recall"] += recall_at_k(rec_ids, rel_ids, K)
            sums[name]["ndcg"] += ndcg_at_k(rec_ids, rel_ratings, K)

    # --- Summary table -------------------------------------------------------
    n = float(len(sampled))
    header = f"{'Model':<16} {'Precision@' + str(K):>13} {'Recall@' + str(K):>11} {'NDCG@' + str(K):>9}"
    print(header)
    print("-" * len(header))
    for name in models:
        s = sums[name]
        print(f"{name:<16} {s['precision'] / n:>13.4f} "
              f"{s['recall'] / n:>11.4f} {s['ndcg'] / n:>9.4f}")

    print(f"\nAveraged over {len(sampled)} users · top-{K} lists · "
          f"relevance = test-set rating >= {RELEVANT_THRESHOLD}")

    # --- Alpha sweep: is there a sweet spot above pure content? --------------
    per_alpha = _sweep_per_user(hybrid, sampled, relevant_by_user)
    _print_sweep(per_alpha, sampled,
                 f"Alpha sweep — all {len(sampled)} sampled users "
                 f"(1.0 = pure SVD, 0.0 = pure content):")

    # Sparse-history slice WITHIN the main sample (kept for continuity; the
    # dedicated protocol below is the statistically meaningful version).
    train_counts = train_df.groupby("userId").size()
    sparse_in_sample = [u for u in sampled
                        if int(train_counts.get(u, 0)) < SPARSE_THRESHOLD]
    if sparse_in_sample:
        counts = [int(train_counts.get(u, 0)) for u in sparse_in_sample]
        _print_sweep(per_alpha, sparse_in_sample,
                     f"Alpha sweep — {len(sparse_in_sample)} sparse-history users "
                     f"within the main sample (< {SPARSE_THRESHOLD} train ratings; "
                     f"range {min(counts)}-{max(counts)}):")

    # =========================================================================
    # Dedicated sparse-user protocol — separate sampling pass.
    #
    # The main protocol's >= MIN_TEST_RELEVANT relevant-test-ratings filter is
    # structurally biased against low-activity users (few ratings overall means
    # few test ratings), so it can't test the "content helps sparse users"
    # hypothesis. Here we sample ONLY users with < SPARSE_THRESHOLD train
    # ratings and relax relevance to >= SPARSE_MIN_TEST_RELEVANT. The main
    # 100-user evaluation above is untouched.
    # =========================================================================
    sparse_pop: List[int] = []
    for uid, rel in relevant_by_user.items():
        if uid not in svd.user_index:
            continue
        if int(train_counts.get(uid, 0)) >= SPARSE_THRESHOLD:
            continue
        if len(rel) < SPARSE_MIN_TEST_RELEVANT:
            continue
        liked_ids, _ = _liked_from_train(train_df, uid)
        if liked_ids:
            sparse_pop.append(uid)
    sparse_pop.sort()

    print("\n" + "=" * 52)
    print(f"DEDICATED SPARSE-USER EVALUATION  (n = {len(sparse_pop)})")
    print("=" * 52)
    if not sparse_pop:
        print(f"No users with < {SPARSE_THRESHOLD} train ratings and "
              f">= {SPARSE_MIN_TEST_RELEVANT} relevant test ratings exist.")
        return

    s_counts = [int(train_counts.get(u, 0)) for u in sparse_pop]
    s_rel = [len(relevant_by_user[u]) for u in sparse_pop]
    print(f"Population: every user with < {SPARSE_THRESHOLD} train ratings and "
          f">= {SPARSE_MIN_TEST_RELEVANT} relevant (4.0+) test ratings.")
    print(f"Train ratings per user: {min(s_counts)}-{max(s_counts)} "
          f"(mean {sum(s_counts) / len(s_counts):.1f}) · "
          f"relevant test movies: {min(s_rel)}-{max(s_rel)} "
          f"(mean {sum(s_rel) / len(s_rel):.1f})")
    if len(sparse_pop) < 30:
        print(f"⚠ Small sample (n={len(sparse_pop)} < 30) — treat as directional only.")

    sparse_per_alpha = _sweep_per_user(hybrid, sparse_pop, relevant_by_user)
    _print_sweep(sparse_per_alpha, sparse_pop,
                 f"Alpha sweep — sparse users only (n={len(sparse_pop)}):")


if __name__ == "__main__":
    main()
