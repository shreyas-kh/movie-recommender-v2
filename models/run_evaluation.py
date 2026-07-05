"""
Offline ranking evaluation: Popularity baseline vs SVD-only vs Content-only
vs Hybrid.

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
* The popularity baseline is the scientific control: recommend the K most-rated
  movies (by TRAIN rating count) the user hasn't rated in train. Entirely
  non-personalised — any model worth deploying should beat it, and every table
  includes it so the personalised numbers have an anchor.
* The long-tail stratified table re-scores each model's SAME top-10 lists
  against only the relevant test items OUTSIDE the HEAD_SIZE most-rated movies.
  Aggregate hit metrics are popularity-biased (held-out loved movies are mostly
  famous ones), so a most-popular list scores well without personalising; in
  the long-tail stratum that shortcut is unavailable by construction, making
  any hits there unambiguous personalisation signal.

Run from the project root:
    python models/run_evaluation.py                  # temporal split (default)
    python models/run_evaluation.py --split random   # original shuffled split
"""
import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# "Head" = this many most-rated train movies. Relevant test items outside the
# head form the long-tail stratum for the stratified table.
HEAD_SIZE = 100


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


def _popularity_ranking(train_df: pd.DataFrame) -> List[int]:
    """Every movieId in the train split, ranked by rating COUNT (descending;
    ties broken by ascending movieId for determinism).

    Count, not average rating, on purpose: ranking by mean rating needs a
    minimum-support threshold (one lone 5.0 rating would top the list) and,
    once damped, measures "highest quality" rather than what this baseline is
    for — the control question "how well does a completely non-personalised
    most-popular list do?". Interaction count is the standard control and uses
    only train data, exactly like the models.
    """
    counts = train_df.groupby("movieId").size()
    return sorted((int(m) for m in counts.index),
                  key=lambda m: (-int(counts[m]), m))


def _popularity_recommend(ranking: List[int], seen: set, n: int) -> List[int]:
    """Top-n most-rated movies the user hasn't rated in train."""
    recs: List[int] = []
    for mid in ranking:
        if mid not in seen:
            recs.append(mid)
            if len(recs) == n:
                break
    return recs


def _popularity_per_user(
    ranking: List[int],
    seen_by_user: Dict[int, set],
    users: List[int],
    relevant_by_user: Dict[int, Dict[int, float]],
) -> Dict[int, Tuple[float, float, float]]:
    """Per-user (precision, recall, ndcg) for the popularity baseline, keyed
    like _sweep_per_user's inner dicts so it can anchor any user subset."""
    per_user: Dict[int, Tuple[float, float, float]] = {}
    for uid in users:
        rel_ratings = relevant_by_user[uid]
        rel_ids = set(rel_ratings)
        rec_ids = _popularity_recommend(ranking, seen_by_user.get(uid, set()), K)
        per_user[uid] = (
            precision_at_k(rec_ids, rel_ids, K),
            recall_at_k(rec_ids, rel_ids, K),
            ndcg_at_k(rec_ids, rel_ratings, K),
        )
    return per_user


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
    baseline: Optional[Dict[int, Tuple[float, float, float]]] = None,
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
    if baseline is not None:
        # Constant reference row: the non-personalised control every alpha
        # setting should be compared against.
        metrics = [baseline[uid] for uid in user_subset]
        p = sum(m[0] for m in metrics) / n_sub
        r = sum(m[1] for m in metrics) / n_sub
        g = sum(m[2] for m in metrics) / n_sub
        print(f"{'pop':>6} {p:>13.4f} {r:>11.4f} {g:>9.4f}   <- popularity baseline")


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

    # Popularity control: one global most-rated ranking (train counts only),
    # personalised only by excluding each user's train-rated movies.
    pop_ranking = _popularity_ranking(train_df)
    seen_by_user: Dict[int, set] = {
        int(uid): set(int(m) for m in grp)
        for uid, grp in train_df.groupby("userId")["movieId"]
    }

    models = ["Popularity", "SVD-only", "Content-only", f"Hybrid (α={ALPHA})"]
    sums: Dict[str, Dict[str, float]] = {
        m: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for m in models
    }
    # Each user's top-10 per model, kept for the long-tail stratified table
    # (same lists, re-scored against a restricted relevant set).
    recs_by_user: Dict[int, Dict[str, List[int]]] = {}

    for uid in sampled:
        rel_ratings = relevant_by_user[uid]
        rel_ids = set(rel_ratings)
        liked_ids, weights = _liked_from_train(train_df, uid)
        seen_train = [int(m) for m in
                      train_df.loc[train_df["userId"] == uid, "movieId"]]

        recs = {
            "Popularity": _popularity_recommend(
                pop_ranking, seen_by_user.get(uid, set()), K),
            "SVD-only": [m for m, _ in svd.recommend(uid, n=K)],
            "Content-only": [m for m, _ in content.recommend(
                liked_ids, n=K, weights=weights, exclude=seen_train)],
            models[3]: [m for m, _ in hybrid.recommend(
                user_id=uid, n=K, alpha=ALPHA)],
        }
        recs_by_user[uid] = recs

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

    # --- Long-tail stratified table ------------------------------------------
    # Same top-10 lists, but only relevant test items OUTSIDE the HEAD_SIZE
    # most-rated train movies count as hits (recall/NDCG denominators shrink
    # accordingly). The popularity baseline recommends almost exclusively head
    # movies, so it scores ~0 here by construction — any real score from the
    # personalised models is signal the aggregate table cannot see.
    head = set(pop_ranking[:HEAD_SIZE])
    lt_relevant: Dict[int, Dict[int, float]] = {
        uid: {m: r for m, r in relevant_by_user[uid].items() if m not in head}
        for uid in sampled
    }
    lt_users = [uid for uid in sampled if lt_relevant[uid]]
    # The sweep's best alpha is the interesting long-tail question; its lists
    # aren't in recs_by_user, so compute them here.
    lt_models = models + ["Hybrid (α=0.75)"]
    for uid in lt_users:
        recs_by_user[uid]["Hybrid (α=0.75)"] = [
            m for m, _ in hybrid.recommend(user_id=uid, n=K, alpha=0.75)
        ]

    n_lt_items = [len(lt_relevant[uid]) for uid in lt_users]
    print(f"\nLong-tail stratum — relevant test items outside the "
          f"{HEAD_SIZE} most-rated train movies")
    print(f"({len(lt_users)}/{len(sampled)} sampled users have >= 1 such item; "
          f"{min(n_lt_items)}-{max(n_lt_items)} each, "
          f"mean {sum(n_lt_items) / len(lt_users):.1f})")
    header = (f"{'Model':<16} {'Precision@' + str(K):>13} "
              f"{'Recall@' + str(K):>11} {'NDCG@' + str(K):>9}")
    print(header)
    print("-" * len(header))
    n_lt = float(len(lt_users))
    for name in lt_models:
        p = r = g = 0.0
        for uid in lt_users:
            rel_ratings = lt_relevant[uid]
            rel_ids = set(rel_ratings)
            rec_ids = recs_by_user[uid][name]
            p += precision_at_k(rec_ids, rel_ids, K)
            r += recall_at_k(rec_ids, rel_ids, K)
            g += ndcg_at_k(rec_ids, rel_ratings, K)
        print(f"{name:<16} {p / n_lt:>13.4f} {r / n_lt:>11.4f} {g / n_lt:>9.4f}")

    # --- Alpha sweep: is there a sweet spot above pure content? --------------
    per_alpha = _sweep_per_user(hybrid, sampled, relevant_by_user)
    pop_metrics = _popularity_per_user(
        pop_ranking, seen_by_user, sampled, relevant_by_user)
    _print_sweep(per_alpha, sampled,
                 f"Alpha sweep — all {len(sampled)} sampled users "
                 f"(1.0 = pure SVD, 0.0 = pure content):",
                 baseline=pop_metrics)

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
                     f"range {min(counts)}-{max(counts)}):",
                     baseline=pop_metrics)

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
    sparse_pop_metrics = _popularity_per_user(
        pop_ranking, seen_by_user, sparse_pop, relevant_by_user)
    _print_sweep(sparse_per_alpha, sparse_pop,
                 f"Alpha sweep — sparse users only (n={len(sparse_pop)}):",
                 baseline=sparse_pop_metrics)


if __name__ == "__main__":
    main()
