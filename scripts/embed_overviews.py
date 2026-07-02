"""
Embed movie overview text into semantic vectors — one-time offline step.

Reads data/movie_overviews.csv (from scripts/fetch_overviews.py), embeds each
non-empty overview with sentence-transformers' all-MiniLM-L6-v2 (384-dim,
~90MB, CPU-friendly), and saves data/overview_embeddings.npz containing:

    movie_ids   (n,)     int64   — movieIds WITH an overview, ascending
    embeddings  (n, 384) float32 — L2-normalised, row i <-> movie_ids[i]

Embeddings are unit-length, so dot product == cosine similarity — the same
contract ContentRecommender already uses for its genre vectors.

Movies with no overview text are left out; the model layer decides the
fallback (e.g. genre-only) for those.

Requires sentence-transformers (offline tooling only — NOT in requirements.txt,
the deployed app never embeds anything):
    python3 -m pip install --user sentence-transformers

Run from the project root:
    python3 scripts/embed_overviews.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
IN_CSV = ROOT / "data" / "movie_overviews.csv"
OUT_NPZ = ROOT / "data" / "overview_embeddings.npz"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"{IN_CSV} not found — run scripts/fetch_overviews.py first.")

    df = pd.read_csv(IN_CSV)
    df["overview"] = df["overview"].fillna("").astype(str).str.strip()
    total = len(df)
    df = df[df["overview"] != ""].sort_values("movieId")
    print(f"{len(df)}/{total} movies have overview text; embedding those.")

    from sentence_transformers import SentenceTransformer  # heavy import, defer

    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        df["overview"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit vectors: dot product == cosine
    ).astype(np.float32)

    movie_ids = df["movieId"].to_numpy(dtype=np.int64)
    np.savez_compressed(OUT_NPZ, movie_ids=movie_ids, embeddings=embeddings)

    print(f"\nSaved {embeddings.shape[0]} x {embeddings.shape[1]} embeddings "
          f"to {OUT_NPZ} ({OUT_NPZ.stat().st_size / 1e6:.1f} MB)")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Norm check: min={norms.min():.4f} max={norms.max():.4f} (expect ~1.0)")


if __name__ == "__main__":
    main()
