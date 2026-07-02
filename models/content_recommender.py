"""
Content-based recommender built on movie genres, optionally combined with
semantic overview embeddings.

Complements the collaborative-filtering SVDRecommender: because it scores movies
purely from item features, it needs no rating history for the *user* — a single
liked movie is enough to make recommendations. That makes it the cold-start half
of the HybridRecommender.

Two item-feature signals, combined as a weighted score:

    combined = beta * genre_similarity + (1 - beta) * embedding_similarity

* Genre similarity — cosine over L2-normalised multi-hot genre vectors.
* Embedding similarity — cosine over sentence-transformer embeddings of TMDB
  plot overviews (data/overview_embeddings.npz from scripts/embed_overviews.py).

Why a combination, not a replacement: the two signals fail in complementary
ways. Genres can't rank within a tag combo (Toy Story 2 ties ~200 other
Animation|Children movies at 1.0), which embeddings fix (franchise/plot
kinship). But embeddings encode PLOT, not TONE — Toy Story's raw embedding
neighbours include Child's Play ("toys come to life", as a slasher), which the
genre term vetoes (Horror|Thriller shares no tags). Movies without an overview
embedding fall back to pure genre scoring rather than being zeroed out.

We never materialise a full movie x movie similarity matrix (~9.7k^2 would be
~750 MB dense); all vectors are L2-normalised at fit time so cosine against
every movie is a single matvec computed on demand.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_NO_GENRES = "(no genres listed)"


class ContentRecommender:
    def __init__(self, beta: float = 0.5):
        # Weight of the genre term in the combined score; (1 - beta) goes to
        # the overview-embedding term. Only meaningful when embeddings are
        # loaded — without them scoring is pure genre regardless of beta.
        self.beta = beta
        self.movie_ids: List[int] = []
        self.movie_index: Dict[int, int] = {}
        self.genres: List[str] = []
        # L2-normalised multi-hot genre matrix: (n_movies, n_genres). Rows for
        # movies with no listed genres are all-zero (norm 0), which correctly
        # yields a cosine similarity of 0 against everything.
        self.features: Optional[np.ndarray] = None
        # Overview embeddings aligned to movie_ids: (n_movies, dim), zero rows
        # where no embedding exists; has_embedding marks the real rows.
        self.embeddings: Optional[np.ndarray] = None
        self.has_embedding: Optional[np.ndarray] = None

    def fit(
        self,
        movies_df: pd.DataFrame,
        embeddings_path: Optional[Union[str, Path]] = None,
    ) -> "ContentRecommender":
        """Build the normalised genre feature matrix from movies.csv, and
        optionally load overview embeddings to enable combined scoring.

        Needs only movieId + genres; no ratings, so this is independent of both
        SVDRecommender and any particular user. If embeddings_path is None or
        missing, the recommender behaves exactly as before (genre-only).
        """
        movies_df = movies_df.drop_duplicates(subset="movieId")
        self.movie_ids = [int(m) for m in movies_df["movieId"].tolist()]
        self.movie_index = {mid: i for i, mid in enumerate(self.movie_ids)}

        # Discover the genre vocabulary (excluding the "no genres" sentinel).
        genre_lists: List[List[str]] = []
        vocab: Dict[str, int] = {}
        for raw in movies_df["genres"].fillna(""):
            gs = [g for g in raw.split("|") if g and g != _NO_GENRES]
            genre_lists.append(gs)
            for g in gs:
                if g not in vocab:
                    vocab[g] = len(vocab)
        self.genres = list(vocab.keys())

        # Multi-hot encode.
        mat = np.zeros((len(self.movie_ids), len(self.genres)), dtype=np.float64)
        for row, gs in enumerate(genre_lists):
            for g in gs:
                mat[row, vocab[g]] = 1.0

        # L2-normalise rows so a dot product == cosine similarity. Guard the
        # zero-norm rows (movies with no genres) to avoid divide-by-zero.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.features = mat / norms

        if embeddings_path is not None:
            self._load_embeddings(embeddings_path)
        return self

    def _load_embeddings(self, path: Union[str, Path]) -> None:
        """Align the saved (movie_ids, embeddings) arrays to this model's movie
        order. Movies absent from the npz keep a zero row + has_embedding=False
        and will be scored by genre alone."""
        path = Path(path)
        if not path.exists():
            return  # stay genre-only; caller may not have run the pipeline yet
        data = np.load(path)
        ids, emb = data["movie_ids"], data["embeddings"]

        self.embeddings = np.zeros((len(self.movie_ids), emb.shape[1]), dtype=np.float32)
        self.has_embedding = np.zeros(len(self.movie_ids), dtype=bool)
        for i, mid in enumerate(ids):
            idx = self.movie_index.get(int(mid))
            if idx is not None:
                self.embeddings[idx] = emb[i]
                self.has_embedding[idx] = True

    # -- internals -----------------------------------------------------------

    def _taste_vector(
        self,
        matrix: np.ndarray,
        liked_movie_ids: Sequence[int],
        weights: Optional[Sequence[float]] = None,
        valid: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Weighted mean of the liked movies' rows of `matrix` (genre features
        or embeddings), re-normalised to unit length. Rows where `valid` is
        False (e.g. movies with no embedding) are skipped. Returns None if no
        liked movie contributes any signal."""
        rows: List[int] = []
        ws: List[float] = []
        for i, mid in enumerate(liked_movie_ids):
            idx = self.movie_index.get(int(mid))
            if idx is None:
                continue
            if valid is not None and not valid[idx]:
                continue
            rows.append(idx)
            ws.append(float(weights[i]) if weights is not None else 1.0)

        if not rows:
            return None

        w = np.asarray(ws, dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones_like(w)
        profile = (matrix[rows].astype(np.float64) * w[:, np.newaxis]).sum(axis=0)

        norm = np.linalg.norm(profile)
        if norm == 0.0:  # liked movies carried no signal in this matrix
            return None
        return profile / norm

    # -- public API ----------------------------------------------------------

    def score_movies(
        self,
        liked_movie_ids: Sequence[int],
        weights: Optional[Sequence[float]] = None,
        beta: Optional[float] = None,
    ) -> np.ndarray:
        """Similarity of every movie against the liked-movie taste profile.

        With embeddings loaded:
            combined = beta * genre_cos + (1 - beta) * embedding_cos
        applied only to movies that HAVE an embedding; movies without one are
        scored by genre alone (full weight) so missing data isn't a penalty.
        Without embeddings (or when the liked movies have none), this is the
        original pure-genre cosine.

        Returns an array aligned to self.movie_ids, in [0, 1]. All zeros if
        there's no usable signal from the liked movies. `beta` overrides
        self.beta for this call (used by sweeps/tuning).
        """
        if self.features is None:
            raise RuntimeError("ContentRecommender must be fit before use.")
        b = self.beta if beta is None else float(beta)

        genre_profile = self._taste_vector(self.features, liked_movie_ids, weights)
        genre_scores = (
            np.clip(self.features.dot(genre_profile), 0.0, 1.0)
            if genre_profile is not None
            else np.zeros(len(self.movie_ids), dtype=np.float64)
        )

        if self.embeddings is None:
            return genre_scores
        emb_profile = self._taste_vector(
            self.embeddings, liked_movie_ids, weights, valid=self.has_embedding
        )
        if emb_profile is None:  # none of the liked movies has an embedding
            return genre_scores

        # MiniLM cosines for unrelated text hover near 0 and can dip slightly
        # negative; clip into [0, 1] to match the genre term's range.
        emb_scores = np.clip(self.embeddings.astype(np.float64).dot(emb_profile), 0.0, 1.0)

        # Movies without an embedding get the query's MEAN embedding score
        # imputed. Zeroing their term would punish missing data; keeping full
        # genre weight would reward it (embedding cosines run lower than genre
        # cosines, so un-blended movies float to the top — measured 17x
        # over-representation). A neutral "average plot match" does neither.
        imputed = float(emb_scores[self.has_embedding].mean()) if self.has_embedding.any() else 0.0
        emb_term = np.where(self.has_embedding, emb_scores, imputed)

        combined = b * genre_scores + (1.0 - b) * emb_term
        return np.clip(combined, 0.0, 1.0)

    def recommend(
        self,
        liked_movie_ids: Sequence[int],
        n: int = 10,
        weights: Optional[Sequence[float]] = None,
        exclude: Optional[Sequence[int]] = None,
        beta: Optional[float] = None,
    ) -> List[Tuple[int, float]]:
        """Top-n most similar movies to the liked set.

        The liked movies themselves are always excluded; pass `exclude` to also
        drop already-seen movies (e.g. a warm user's full rating history).
        """
        scores = self.score_movies(liked_movie_ids, weights, beta=beta)

        blocked = {int(m) for m in liked_movie_ids}
        if exclude is not None:
            blocked.update(int(m) for m in exclude)
        for mid in blocked:
            idx = self.movie_index.get(mid)
            if idx is not None:
                scores[idx] = -np.inf

        top = np.argsort(scores)[::-1][:n]
        return [
            (self.movie_ids[i], float(scores[i]))
            for i in top
            if np.isfinite(scores[i])
        ]

    def similar_movies(
        self, movie_id: int, n: int = 10, beta: Optional[float] = None
    ) -> List[Tuple[int, float]]:
        """'More like this' — the n movies most similar to a single movie."""
        return self.recommend([movie_id], n=n, beta=beta)

    # -- persistence (optional; content model is cheap to rebuild) ------------

    def save(self, path: Union[str, Path]) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ContentRecommender":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
