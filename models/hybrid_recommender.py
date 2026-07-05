"""
Hybrid recommender: blends collaborative filtering (SVDRecommender) with
content-based filtering (ContentRecommender).

Why blend:
  * SVD captures latent taste patterns from *who rated what* — strong for users
    with history, but useless for a user (or visitor) the model never saw.
  * Content filtering scores movies by item-feature similarity to what someone
    liked — a weighted blend of genre overlap and plot-embedding similarity
    (see ContentRecommender) — weaker at capturing subtle taste, but works
    from a single liked movie.

The blend gives us the best of both while degrading gracefully: warm users get a
weighted mix; cold-start users and brand-new visitors fall back to content-only.

This class only *reads* SVDRecommender's public attributes (predicted, rated_mask,
user_index, movie_ids); it never modifies SVDRecommender or ContentRecommender.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.content_recommender import ContentRecommender
from models.recommender import SVDRecommender

# Threshold for what counts as a "liked" movie when deriving a warm user's taste
# vector from their history. Mirrors the app's existing 4.0-with-3.5-fallback rule.
_LIKE_THRESHOLD = 4.0
_LIKE_FALLBACK = 3.5


def _minmax(scores: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Scale scores into [0, 1] using the range over the *candidate* movies.

    SVD predictions live on the ~0.5-5 rating scale and cosine scores on 0-1, so
    we must put them on a common footing before blending. The min/max are taken
    over candidate (unseen) movies only, so already-rated movies don't skew the
    range. The whole vector stays finite (masking to -inf happens after the
    blend, so we never do 0 * -inf). Degenerate all-equal case -> neutral 0.5.
    """
    vals = scores[candidate]
    if vals.size == 0:
        return np.zeros_like(scores)
    lo, hi = float(vals.min()), float(vals.max())
    if hi == lo:
        return np.full_like(scores, 0.5)
    return (scores - lo) / (hi - lo)


class HybridRecommender:
    def __init__(
        self,
        svd_model: SVDRecommender,
        content_model: ContentRecommender,
        ratings_df: pd.DataFrame,
    ):
        self.svd = svd_model
        self.content = content_model
        self.ratings_df = ratings_df
        # movieId -> column index in SVD's predicted matrix (built once).
        self._svd_col: Dict[int, int] = {
            int(mid): j for j, mid in enumerate(svd_model.movie_ids)
        }

    # -- helpers -------------------------------------------------------------

    def _user_liked_movies(self, user_id: int) -> Tuple[List[int], List[float]]:
        """A warm user's liked movie IDs and the ratings to weight them by
        (>=4.0, falling back to >=3.5). Empty lists if the user has no history."""
        user = self.ratings_df[self.ratings_df["userId"] == user_id]
        if user.empty:
            return [], []
        for thresh in (_LIKE_THRESHOLD, _LIKE_FALLBACK):
            liked = user[user["rating"] >= thresh]
            if not liked.empty:
                return (
                    [int(m) for m in liked["movieId"].tolist()],
                    [float(r) for r in liked["rating"].tolist()],
                )
        return [], []

    def _content_scores_aligned(
        self,
        liked_ids: Sequence[int],
        weights: Optional[Sequence[float]],
    ) -> np.ndarray:
        """Content scores for every movie in SVD's universe, in SVD's movie order.

        SVD's movie set is a subset of the content model's, so every SVD movie has
        a content row; the .get default of 0.0 is a defensive no-op here.
        """
        full = self.content.score_movies(liked_ids, weights)
        c_index = self.content.movie_index
        aligned = np.zeros(len(self.svd.movie_ids), dtype=np.float64)
        for j, mid in enumerate(self.svd.movie_ids):
            idx = c_index.get(int(mid))
            if idx is not None:
                aligned[j] = full[idx]
        return aligned

    def has_liked_history(self, user_id: int) -> bool:
        """True if the user has liked movies (the >=4.0-falling-back-to->=3.5
        rule) to drive the content half of the blend. False means the content
        scores are all-zero, so warm-blend output is pure SVD at ANY alpha —
        callers showing blend percentages should say so instead."""
        return bool(self._user_liked_movies(user_id)[0])

    def predicted_rating(self, user_id: int, movie_id: int) -> Optional[float]:
        """SVD's raw predicted rating (~0.5-5 scale) for a warm user, or None if
        the user or movie is unknown to the collaborative model.

        For display only: the hybrid ranks by normalised blended score, but the
        ★ rating shown on a card is the interpretable SVD prediction.
        """
        if user_id not in self.svd.user_index:
            return None
        col = self._svd_col.get(int(movie_id))
        if col is None:
            return None
        idx = self.svd.user_index[user_id]
        return float(self.svd.predicted[idx][col])

    def _rank(
        self, scores: np.ndarray, movie_ids: Sequence[int], n: int
    ) -> List[Tuple[int, float]]:
        top = np.argsort(scores)[::-1][:n]
        return [
            (int(movie_ids[i]), float(scores[i]))
            for i in top
            if np.isfinite(scores[i])
        ]

    # -- public API ----------------------------------------------------------

    def recommend(
        self,
        user_id: Optional[int] = None,
        liked_movie_ids: Optional[Sequence[int]] = None,
        n: int = 10,
        alpha: float = 0.5,
    ) -> List[Tuple[int, float]]:
        """Blended top-n recommendations.

        Three modes, chosen automatically:
          1. Warm user  (user_id known to SVD): blend
             alpha * svd + (1 - alpha) * content, each min-max normalised.
          2. Cold-start user (user_id given but no SVD history): content-only,
             driven by liked_movie_ids (or the user's sub-threshold ratings).
          3. New visitor (no user_id, liked_movie_ids given): content-only.

        alpha in [0, 1]: 1.0 = pure collaborative, 0.0 = pure content.
        """
        # -- Mode 3: new visitor, no user at all -----------------------------
        if user_id is None:
            if not liked_movie_ids:
                raise ValueError(
                    "Provide either a user_id or a non-empty liked_movie_ids."
                )
            return self.content.recommend(liked_movie_ids, n=n)

        # -- Mode 1 vs 2: is this user known to the collaborative model? ------
        warm = user_id in self.svd.user_index

        # Assemble the liked set + weights that drive the content half.
        if liked_movie_ids:
            liked_ids: List[int] = [int(m) for m in liked_movie_ids]
            weights: Optional[List[float]] = None
        else:
            liked_ids, weights = self._user_liked_movies(user_id)

        if not warm:
            # -- Mode 2: cold-start user -> content-only ---------------------
            if not liked_ids:
                raise ValueError(
                    f"User {user_id} has no history and no liked_movie_ids were "
                    "given, so there's nothing to base recommendations on."
                )
            # Exclude anything the user has already rated, if we have history.
            seen = self.ratings_df.loc[
                self.ratings_df["userId"] == user_id, "movieId"
            ].tolist()
            return self.content.recommend(
                liked_ids, n=n, weights=weights, exclude=seen
            )

        # -- Mode 1: warm user -> blend --------------------------------------
        idx = self.svd.user_index[user_id]
        svd_scores = self.svd.predicted[idx].astype(np.float64).copy()
        content_scores = self._content_scores_aligned(liked_ids, weights)

        # Normalise each over the unseen candidates, blend (all finite), then
        # drop already-rated movies. Masking last avoids any 0 * -inf = NaN.
        rated = self.svd.rated_mask[idx]
        candidate = ~rated
        blended = (
            alpha * _minmax(svd_scores, candidate)
            + (1.0 - alpha) * _minmax(content_scores, candidate)
        )
        blended[rated] = -np.inf  # never recommend already-rated movies
        return self._rank(blended, self.svd.movie_ids, n)
