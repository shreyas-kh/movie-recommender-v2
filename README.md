# 🎬 Movie Recommender — Hybrid SVD + Content Filtering, Rigorously Evaluated

A hybrid movie recommender (collaborative filtering + content-based similarity) built on
MovieLens 100k, served as a Streamlit app — **with an offline evaluation that caught its own
lies.** The headline of this project isn't the model; it's the evaluation story: an apparent
"+33% NDCG for sparse users" result that turned out to be train/test leakage, a popularity
baseline that matched the personalized models on aggregate metrics, and a stratified analysis
that finally isolated statistically significant personalization value in the long tail.
Every claim in the app's UI is backed by (and limited to) what the evaluation actually supports.

**Live demo:** https://movie-recommender-v2-shreyaskh.streamlit.app/

## Architecture

```
data/raw/ (MovieLens 100k: 610 users, ~9.7k movies, ~100k ratings)
    │
    ├── TMDB API (offline, one-time) ──> poster URLs, plot overviews
    │       └── sentence-transformers (all-MiniLM-L6-v2, offline)
    │               └── data/overview_embeddings.npz  (9,617 × 384)
    │
    ├── SVDRecommender          TruncatedSVD(50, arpack) on mean-centered sparse ratings
    ├── ContentRecommender      β·genre-cosine + (1−β)·plot-embedding-cosine, β=0.5
    └── HybridRecommender       α·SVD + (1−α)·content, min-max normalized per user
            │
            └── Streamlit app: Existing User mode (blend slider, personas, taste profile)
                               New Visitor mode (cold-start: pick 3–5 movies, no history)
```

The three models are composed, not merged: the SVD layer never learned about the content
layer, and the hybrid only reads their public outputs. The deployed app has no ML training
dependencies beyond scikit-learn — embeddings are precomputed offline and shipped as a 14 MB
artifact (`sentence-transformers` is deliberately **not** in `requirements.txt`).

## Key technical decisions

- **Mean-centered sparse SVD.** Ratings are centered per user and factorized as a sparse
  matrix (implicit zeros), so reconstruction + user mean gives interpretable ~0.5–5
  predicted ratings.
- **ARPACK instead of randomized SVD.** The randomized solver's final `U = Q @ Uhat` DGEMM
  triggered spurious BLAS `RuntimeWarning`s on macOS even with numerically clean inputs;
  switching `TruncatedSVD(algorithm="arpack")` eliminated the noise at equal quality. Found
  by running the test suite with warnings promoted to errors.
- **Genre + plot-embedding blend, not replacement.** The two content signals fail in
  complementary ways: genres can't rank within a category (hundreds of Animation|Children
  movies tie at cosine 1.0), embeddings can't see tone ("toys come to life" matches both
  *Toy Story 2* and the slasher *Child's Play* — the genre term vetoes it). Movies without
  an overview get the query's *mean* embedding score imputed: zeroing them punishes missing
  data, and full genre weight rewards it (measured 17× over-representation before the fix).
- **Per-user temporal splitting.** Each user's earliest 80% of ratings trains; their most
  recent 20% is held out. A random split lets a user's *future* ratings inform "predictions"
  of their past — see below for what that did to the numbers.
- **Normalize, blend, then mask.** The hybrid min-max normalizes each score vector over
  unseen candidates, blends, and only then masks rated movies to −∞. Masking first produced
  `0 × −inf = NaN` at slider extremes.

## The evaluation story (read this part)

This is the project's differentiator: each round of evaluation contradicted the previous
round's conclusion, and the corrections are all preserved in the code and the app.

1. **Random split said the hybrid works.** Under a shuffled 80/20 split, an alpha sweep
   showed the α=0.75 blend beating pure SVD, with **+33% NDCG@10 for sparse-history users**.
   The app briefly shipped α=0.75 as an "empirically tuned" default.
2. **Suspicion: that split leaks time.** A user's later ratings sat in training while
   earlier ones were "predicted." Re-evaluating under a per-user **temporal** split
   (`models/split.py`) reversed the finding: the sparse-user effect vanished and pure SVD
   won. The tuned default was a leakage artifact. Default reverted to α=1.0, with an honest
   caption in the UI.
3. **Semantic embeddings improved the hybrid — but not significantly.** After adding plot
   embeddings to the content model, α=0.75 beat α=1.0 on all three metrics
   (P@10 .0750 vs .0700), but a paired bootstrap (100 users, 10k resamples) put zero inside
   every 95% CI. The default stayed at 1.0 — same trap, not repeated.
4. **The popularity control humbled everything.** A non-personalized "most-rated movies you
   haven't seen" baseline scored P@10 .0680 / NDCG@10 **.0800** — statistically
   indistinguishable from SVD (.0700/.0758; every CI crosses zero) and *ahead* on NDCG.
   Aggregate hit metrics reward recommending famous movies, because famous movies dominate
   what users happen to rate next.
5. **The long-tail stratum found the real signal.** Re-scoring the same top-10 lists against
   only relevant test items *outside the 100 most-rated movies*: popularity scores exactly
   0.0000 (by construction), while SVD scores P@10 .0230 with **95% CIs excluding zero on
   all three metrics** — the project's first fully significant personalization result.
   **20% of users received at least one long-tail recommendation they went on to rate ≥4.0 —
   a recommendation a popularity system cannot produce.** SVD's lists overlap the popularity
   top-10 by only 2.2/10 movies; they were personalized all along, just invisible to
   aggregate hit-rate.

The honest conclusion, as shipped in the app's "How it works" panel: on aggregate offline
metrics, personalization is indistinguishable from popularity; its measurable value is
discovery in the long tail, and fully resolving "how much value" would take online evidence
(an A/B test), not another offline metric.

Reproduce all tables with `python models/run_evaluation.py` (temporal split; add
`--split random` to see the flattering leaky numbers for comparison).

## Features

- **Two modes.** *Existing User*: pick any of the 610 MovieLens users (or a persona), see
  their taste profile, and dial the collaborative↔content blend. *New Visitor*: pick 3–5
  movies and get content-only recommendations with zero rating history — the cold-start path.
- **Personas** — one-click demo users (Action Fan, Romance Lover, Indie Buff, Family Night).
- **Family Night content filter.** Pure SVD happily recommends *American Beauty* to a family
  persona — it ranks by taste similarity and has no concept of appropriateness. The Family
  Night persona adds a policy layer: rank the full candidate set, keep only Children-tagged
  movies ("Animation" is not trustworthy — it admits *Akira* and *Heavy Metal*), and say so
  visibly in the UI. The ranker ranks; policy filters.
- **Predicted star ratings** on each card (SVD's interpretable estimate, warm users only)
  and **"Matches your interest in …"** genre labels, honestly captioned as a post-hoc
  transparency aid, not the model's reasoning.
- **Edge-case honesty**: a user with no ratings ≥3.5 (user 442 exists) gets a caption
  explaining the blend isn't contributing for them, instead of fake percentages.

## Setup

```bash
git clone https://github.com/shreyas-kh/movie-recommender-v2.git
cd movie-recommender-v2
pip install -r requirements.txt
python models/train.py            # prints temporal RMSE, saves models/svd_model.pkl
streamlit run app/app.py
```

Python 3.9+. The MovieLens data, poster URLs, overviews, and embeddings are committed, so
the app runs with no API keys. Optional offline refresh scripts (need a free TMDB key in
`TMDB_API_KEY` or `.streamlit/secrets.toml`): `scripts/fetch_posters.py`,
`scripts/fetch_overviews.py`, then `scripts/embed_overviews.py` (needs
`pip install sentence-transformers`, dev-only).

## Project structure

```
app/
  app.py                  # Streamlit UI: modes, personas, blend slider, family filter
  utils.py                # taste-profile & genre-overlap helpers (explanation layer)
models/
  recommender.py          # SVDRecommender (TruncatedSVD + mean-centering)
  content_recommender.py  # genre ⊕ plot-embedding weighted similarity
  hybrid_recommender.py   # α-blend with cold-start & new-visitor fallbacks
  split.py                # per-user temporal split (default) / random split
  evaluate.py             # Precision@k, Recall@k, NDCG@k (pure, model-agnostic)
  run_evaluation.py       # baseline + 3 models + alpha sweeps + long-tail stratum
  train.py                # RMSE eval + final fit -> svd_model.pkl
scripts/                  # one-time TMDB fetches + offline embedding
data/                     # MovieLens raw + poster/overview/embedding artifacts
notebooks/01_eda.ipynb    # exploratory data analysis
```

## Future work

- **Global-time split**: the per-user temporal split still trains on *other* users' future
  ratings; a strict before-date-T split would close that last leak.
- **Regularized matrix factorization** (biased MF via SGD/ALS) — likely improves RMSE, but
  deliberately deprioritized: the current evaluation shows ranking gains of the plausible
  size would fall inside the noise band, so evaluation power comes first.
- **Beta sweep** for the genre/embedding weight (β=0.5 was validated qualitatively, not swept
  offline like α was).
- **Committed test suite** for the regressions found during development (NaN blend guard,
  temporal-split invariants, embedding imputation).
- **Online evaluation** — the only way to truly answer the popularity-vs-personalization
  question the offline analysis surfaced.
