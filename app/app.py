import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.content_recommender import ContentRecommender
from models.hybrid_recommender import HybridRecommender
from models.recommender import SVDRecommender
from utils import genre_overlap, get_user_profile_stats, get_user_taste_profile

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
MODEL_PATH = Path(__file__).parent.parent / "models" / "svd_model.pkl"
EMBEDDINGS_PATH = Path(__file__).parent.parent / "data" / "overview_embeddings.npz"

# Semi-transparent indigo badge: readable in both dark and light Streamlit themes
# because it uses rgba() rather than a hard-coded hex colour.
_CSS = """
<style>
.match-label {
    display: inline-block;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.78em;
    font-weight: 500;
    line-height: 1.8;
    margin-top: 2px;
}
</style>
"""


@st.cache_resource(show_spinner="Training recommender model...")
def load_model() -> SVDRecommender:
    if MODEL_PATH.exists():
        return SVDRecommender.load(MODEL_PATH)
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    model = SVDRecommender(n_components=50)
    model.fit(ratings)
    return model


@st.cache_resource(show_spinner="Building hybrid recommender...")
def load_hybrid(_svd: SVDRecommender, _ratings: pd.DataFrame, _movies: pd.DataFrame) -> HybridRecommender:
    # Content model is cheap to fit (genre encoding + loading pre-computed
    # overview embeddings), so we build it at startup rather than pickling a
    # second artifact. beta=0.5 (constructor default): equal weight to genre
    # cosine and plot-embedding cosine; degrades to genre-only if the npz is
    # absent. Leading underscores tell Streamlit not to hash unhashable args.
    content = ContentRecommender().fit(_movies, embeddings_path=EMBEDDINGS_PATH)
    return HybridRecommender(_svd, content, _ratings)


@st.cache_data
def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    links = pd.read_csv(DATA_DIR / "links.csv")
    return movies, links


@st.cache_data
def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ratings.csv")


@st.cache_data
def load_posters() -> Dict[int, str]:
    path = DATA_DIR.parent / "poster_urls.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[df["poster_url"].notna() & (df["poster_url"] != "")]
    return dict(zip(df["movieId"].astype(int), df["poster_url"]))


@st.cache_data
def movie_options(_ratings: pd.DataFrame, _movies: pd.DataFrame) -> List[int]:
    """Movie IDs ordered by popularity (rating count, descending) so the
    new-visitor picker surfaces well-known titles first."""
    counts = _ratings.groupby("movieId").size()
    ids = [int(m) for m in _movies["movieId"].tolist()]
    ids.sort(key=lambda m: int(counts.get(m, 0)), reverse=True)
    return ids


def _render_rec_card(
    row: pd.Series,
    liked_genres: Set[str],
    show_posters: bool,
    poster_map: Dict[int, str],
    predicted: Optional[float] = None,
) -> None:
    # Poster is rendered ABOVE the bordered box (not inside it) so it fills the
    # full column width — st.image clamps to a thumbnail inside a container.
    # Movies with no poster simply get no image; we never show a placeholder.
    poster = poster_map.get(int(row["movieId"])) if row.get("movieId") else None
    if show_posters and poster:
        st.image(poster, width="stretch")

    with st.container(border=True):
        st.markdown(f"**{row.get('title', 'Unknown')}**")

        # Predicted rating is SVD's interpretable estimate (warm users only).
        # New-visitor / pure-content picks have no rating, only a genre match.
        if predicted is not None:
            val = max(0.5, min(5.0, float(predicted)))  # clamp to MovieLens scale
            st.markdown(f"Predicted rating: **{val:.1f}** ★")

        overlap = genre_overlap(row.get("genres", ""), liked_genres)
        if overlap:
            st.markdown(
                f'<div class="match-label">Matches your interest in: {", ".join(overlap)}</div>',
                unsafe_allow_html=True,
            )
        else:
            genres = row.get("genres", "")
            if genres:
                st.caption(genres.replace("|", " · "))


def _render_picks_grid(
    recs: List[Tuple[int, float]],
    movie_meta: pd.DataFrame,
    liked_genres: Set[str],
    show_posters: bool,
    poster_map: Dict[int, str],
    hybrid: Optional[HybridRecommender] = None,
    uid: Optional[int] = None,
) -> None:
    """Render the recommendation grid, preserving the hybrid's ranking order.

    When `hybrid` and `uid` are given (warm user), each card shows SVD's
    predicted ★ rating; otherwise (new visitor) only the genre match is shown.
    """
    rec_ids = [mid for mid, _ in recs]
    order = {mid: i for i, (mid, _) in enumerate(recs)}
    rec_df = movie_meta[movie_meta["movieId"].isin(rec_ids)].copy()
    rec_df["_order"] = rec_df["movieId"].map(order)
    rec_df = rec_df.sort_values("_order")

    cols = st.columns(5, gap="small")
    for i, (_, row) in enumerate(rec_df.iterrows()):
        predicted = (
            hybrid.predicted_rating(uid, int(row["movieId"]))
            if hybrid is not None and uid is not None
            else None
        )
        with cols[i % 5]:
            _render_rec_card(row, liked_genres, show_posters, poster_map, predicted)


# Preset personas → representative MovieLens user IDs. Data-driven picks: each
# user's highly-rated history leans strongly toward the named taste. Swap freely.
# Third field: apply the family content filter when this persona is active.
PERSONAS = [
    ("🎬 Action Fan", 493, False),
    ("💕 Romance Lover", 358, False),
    ("🧠 Indie Buff", 74, False),
    ("👨‍👩‍👧 Family Night", 401, True),
]

# Family Night soft content filter. SVD has no concept of appropriateness — it
# ranks purely by rating-pattern similarity, so user 401's neighbours' love of
# American Beauty/Hannibal leaks into a "family" persona. When the Family Night
# persona is active we keep SVD's ranking but restrict candidates to movies
# tagged "Children". Children (not Animation) is the trustworthy tag: Animation
# alone admits adult titles like Heavy Metal, Akira and Beavis and Butt-Head.
FAMILY_TAG = "Children"
FAMILY_POOL_DEPTH = 300  # how deep to rank before filtering down to n


def _render_taste_profile(uid: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
    stats = get_user_profile_stats(uid, ratings_df, movies_df)
    st.markdown("#### 🎭 User Taste Profile")
    genre_counts = stats["genre_counts"]
    if not genre_counts.empty:
        st.caption("Genres in highly-rated movies")
        st.bar_chart(genre_counts.head(8), horizontal=True, height=220)
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Movies rated", stats["total_rated"])
    mcol2.metric("Avg rating", f"{stats['avg_rating']:.1f}")
    st.caption(f"Top genre: **{stats['top_genre']}**")


def _genres_of(movie_ids: List[int], movies_df: pd.DataFrame) -> Set[str]:
    """Union of genres across a set of movies (for new-visitor match labels)."""
    sub = movies_df[movies_df["movieId"].isin(movie_ids)]
    liked: Set[str] = set()
    for g in sub["genres"].dropna():
        liked.update(x for x in g.split("|") if x != "(no genres listed)")
    return liked


def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    st.title("🎬 Movie Recommender")
    st.caption("Hybrid recommender · SVD collaborative filtering + content-based genre similarity · MovieLens (~100k ratings)")

    st.info(
        "👋 **New here?** This app blends two recommenders. As an **existing user** "
        "(610 real MovieLens users), pick a User ID or persona and dial the blend "
        "between collaborative filtering and genre similarity. As a **new visitor**, "
        "just pick a few movies you like and get instant content-based picks — no "
        "history needed. Choose a mode in the sidebar to start."
    )

    with st.expander("How it works", expanded=False):
        st.markdown(
            """
This app combines **two** recommendation approaches:

**1. Collaborative filtering (SVD).** *Singular Value Decomposition* factorizes the MovieLens table of
~100,000 real ratings into a small set of hidden "taste factors" — patterns discovered purely from who
rated what and how highly, with no knowledge of genres or plot. It predicts how a user would rate films
they haven't seen, then surfaces the highest predictions.

**2. Content-based filtering.** Each movie is encoded as a multi-hot vector of its genres. Recommendations
come from **cosine similarity** between what you liked and everything else. Because it needs no rating
history for the *user*, it works from a single liked movie — this is the **cold-start** fix that pure
collaborative filtering can't handle (a brand-new visitor has no factors to look up).

**The hybrid** normalizes both score ranges and blends them: `alpha × collaborative + (1 − alpha) × content`.
Slide toward collaborative for taste patterns, toward content for genre-driven similarity.

The **"Matches your interest in: …"** labels are a post-hoc transparency aid comparing genres — for the
collaborative half, they are **not** how SVD chose the movie.
            """
        )

    svd = load_model()
    movies_df, links_df = load_metadata()
    ratings_df = load_ratings()
    poster_map = load_posters()
    hybrid = load_hybrid(svd, ratings_df, movies_df)

    min_uid = int(min(svd.user_ids))
    max_uid = int(max(svd.user_ids))

    movie_meta = movies_df.merge(links_df[["movieId", "tmdbId"]], on="movieId", how="left")
    title_map = dict(zip(movies_df["movieId"].astype(int), movies_df["title"]))

    # --- Session state: persona buttons + recommendation trigger -------------
    if "user_id" not in st.session_state:
        st.session_state.user_id = min_uid
    if "show_recs" not in st.session_state:
        st.session_state.show_recs = False
    if "nv_show_recs" not in st.session_state:
        st.session_state.nv_show_recs = False
    if "active_persona" not in st.session_state:
        st.session_state.active_persona = None

    def _select_persona(pid: int) -> None:
        st.session_state.user_id = pid
        st.session_state.active_persona = pid
        st.session_state.show_recs = True  # personas auto-trigger recommendations

    def _on_user_change() -> None:
        st.session_state.show_recs = False  # manual edit: show profile, wait to recommend
        st.session_state.active_persona = None  # leaving the persona drops its filter

    def _on_picks_change() -> None:
        st.session_state.nv_show_recs = False  # picks edited: wait for re-submit

    # Locals populated per-mode inside the sidebar, read again in the body below.
    alpha = 0.5
    liked_ids: List[int] = []

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.caption(f"{len(svd.user_ids)} users · {len(svd.movie_ids):,} movies")

        mode = st.radio(
            "Who's watching?",
            ["Existing User", "New Visitor"],
            help=(
                "Existing users get a collaborative + content blend. "
                "New visitors get content-based picks from movies they choose "
                "— no rating history required (the cold-start case)."
            ),
        )
        st.divider()

        n_recs = st.slider("Recommendations", min_value=5, max_value=20, value=10)
        show_posters = st.toggle("Show posters", value=True)
        if show_posters and not poster_map:
            st.caption("Run `scripts/fetch_posters.py` to enable posters.")
        st.divider()

        if mode == "Existing User":
            st.markdown("**Try a persona**")
            pcols = st.columns(2)
            for i, (label, pid, _fam) in enumerate(PERSONAS):
                pcols[i % 2].button(
                    label,
                    key=f"persona_{pid}",
                    width="stretch",
                    on_click=_select_persona,
                    args=(pid,),
                )

            user_id = st.number_input(
                f"User ID ({min_uid}–{max_uid})",
                min_value=min_uid,
                max_value=max_uid,
                step=1,
                key="user_id",
                on_change=_on_user_change,
            )

            # Default 1.0 (pure SVD): under the realistic per-user temporal
            # split (models/run_evaluation.py), blending in content showed no
            # measurable benefit for warm users — an earlier 0.75 default was
            # based on a random split, whose apparent gains turned out to be
            # leakage artifacts. Content's real value is cold-start (New
            # Visitor mode); the slider stays for exploration.
            alpha = st.slider(
                "Blend",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="1.0 = pure collaborative (SVD) · 0.0 = pure content (genres)",
            )
            st.caption(f"{int(round(alpha * 100))}% collaborative · {int(round((1 - alpha) * 100))}% content")
            st.caption(
                "Content-based blending showed no measurable benefit for warm "
                "users under rigorous (temporal) evaluation — its real value is "
                "for brand-new users with no rating history, demonstrated in "
                "New Visitor mode. Adjust this slider to explore the blend yourself."
            )

            if st.button("Get Recommendations", type="primary", width="stretch"):
                st.session_state.show_recs = True

            # Taste profile renders immediately for the current user (context
            # first), whether or not recommendations have been requested yet.
            uid = int(user_id)
            if uid in svd.user_index:
                st.divider()
                _render_taste_profile(uid, ratings_df, movies_df)
        else:
            st.markdown("**Pick 3–5 movies you like**")
            liked_ids = st.multiselect(
                "Movies you like",
                options=movie_options(ratings_df, movies_df),
                format_func=lambda mid: title_map.get(mid, str(mid)),
                max_selections=5,
                key="nv_liked",
                on_change=_on_picks_change,
                help="Search by title. Popular movies are listed first.",
            )
            st.caption("No account or rating history needed.")
            if st.button("Get Recommendations", type="primary", width="stretch", key="nv_go"):
                st.session_state.nv_show_recs = True

        st.divider()
        st.caption("Built with [Streamlit](https://streamlit.io) · Hybrid SVD + content filtering")

    # --- New visitor: content-only, driven purely by the picked movies -------
    if mode == "New Visitor":
        if not liked_ids:
            st.info("👈 Pick 3–5 movies you like, then click **Get Recommendations**.")
            return
        if not st.session_state.nv_show_recs:
            st.info("👈 Click **Get Recommendations** when you're happy with your picks.")
            return
        recs = hybrid.recommend(liked_movie_ids=liked_ids, n=n_recs)
        if not recs:
            st.warning("Couldn't find similar movies for that selection — try adding another title.")
            return
        liked_genres = _genres_of(liked_ids, movies_df)
        st.subheader("🍿 Because you like those, try these")
        st.caption(
            "Recommendations based on your picks using content-based filtering — "
            "no rating history needed. This demonstrates how the system handles new users."
        )
        _render_picks_grid(recs, movie_meta, liked_genres, show_posters, poster_map)
        return

    # --- Existing user: blended recommendations ------------------------------
    if not st.session_state.show_recs:
        st.info("👈 Pick a persona or enter a User ID, then click **Get Recommendations**.")
        return

    uid = int(st.session_state.user_id)
    if uid not in svd.user_index:
        st.error(f"User ID {uid} isn't in the training data. Try {min_uid}–{max_uid}.")
        return

    # Family Night persona: rank as usual, but keep only family-tagged movies.
    # A post-filter (rather than a lower alpha) preserves SVD's ranking quality
    # within the family-safe set and guarantees every card fits the framing.
    active = st.session_state.active_persona
    family_mode = any(pid == active == uid and fam for _, pid, fam in PERSONAS)
    if family_mode:
        family_ids = set(
            movies_df.loc[
                movies_df["genres"].str.contains(FAMILY_TAG, na=False), "movieId"
            ].astype(int)
        )
        deep = hybrid.recommend(user_id=uid, n=FAMILY_POOL_DEPTH, alpha=alpha)
        recs = [(m, s) for m, s in deep if m in family_ids][:n_recs]
    else:
        recs = hybrid.recommend(user_id=uid, n=n_recs, alpha=alpha)
    if not recs:
        st.warning("No new recommendations found — this user may have already rated most movies.")
        return

    # --- "Because you rated these highly" — the user's own evidence ----------
    top_rated, liked_genres = get_user_taste_profile(uid, ratings_df, movies_df)
    if not top_rated.empty:
        threshold = top_rated["threshold_used"].iloc[0]
        heading = (
            "Because you rated these highly"
            if threshold >= 4.0
            else "Your top-rated movies"
        )
        st.subheader(heading)
        taste_cols = st.columns(len(top_rated), gap="small")
        for col, (_, row) in zip(taste_cols, top_rated.iterrows()):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{row['title']}**")
                    filled = "★" * int(row["rating"])
                    empty = "☆" * (5 - int(row["rating"]))
                    st.caption(f"{filled}{empty}  {row['rating']:.1f}")
        st.divider()

    # Connecting narrative: ties the taste profile above to the picks below.
    st.subheader(f"🍿 Top {len(recs)} picks for User {uid}")
    st.caption(
        f"Blending rating history ({int(round(alpha * 100))}% collaborative) "
        f"with genre similarity ({int(round((1 - alpha) * 100))}% content)."
    )
    if family_mode:
        st.caption(
            f"👨‍👩‍👧 **Family Night filter on:** picks are limited to movies tagged "
            f"*{FAMILY_TAG}*. Collaborative filtering ranks by taste similarity alone "
            f"and has no concept of content appropriateness, so this persona adds a "
            f"soft genre-based filter on top. Edit the User ID to turn it off."
        )
    _render_picks_grid(recs, movie_meta, liked_genres, show_posters, poster_map, hybrid=hybrid, uid=uid)


if __name__ == "__main__":
    main()
