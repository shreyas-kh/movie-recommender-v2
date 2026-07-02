"""
Pre-fetch movie overview (plot summary) text from the TMDB API.

Saves data/movie_overviews.csv (movieId, overview) for the full catalog. The
overviews feed the offline embedding step (scripts/embed_overviews.py); like
posters, this is a one-time local fetch — nothing at app runtime calls TMDB.

TMDB's /movie/{id} details endpoint — the SAME endpoint fetch_posters.py
already calls — includes both `overview` and `poster_path`, so this script
also opportunistically backfills data/poster_urls.csv for any movie not
already in it, at zero extra API cost.

Usage:
    TMDB_API_KEY=your_key python scripts/fetch_overviews.py
    # or with the key in .streamlit/secrets.toml:
    python scripts/fetch_overviews.py

Resume-safe: movies already in movie_overviews.csv are skipped; results are
checkpointed periodically; requests that fail after retries are left out so a
re-run picks them up. Safe to interrupt.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))  # fetch_posters helpers

from fetch_posters import (
    CHECKPOINT_EVERY,
    MAX_RETRIES,
    MAX_WORKERS,
    OUT_CSV as POSTER_CSV,
    POSTER_BASE,
    TMDB_BASE,
    _api_key,
    _save as _save_posters,
)

LINKS_CSV = ROOT / "data" / "raw" / "links.csv"
OUT_CSV = ROOT / "data" / "movie_overviews.csv"


def _fetch_details(tmdb_id: int, api_key: str) -> Optional[Tuple[str, str]]:
    """Return (overview, poster_url) — either may be "" (definitively absent) —
    or None if the request failed after retries (caller leaves it out of the
    CSV so a re-run retries it). Mirrors fetch_posters._fetch_poster."""
    backoff = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                f"{TMDB_BASE}/movie/{tmdb_id}",
                params={"api_key": api_key},
                timeout=10,
            )
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", backoff))
                time.sleep(wait)
                backoff *= 2
                continue
            if resp.status_code == 404:
                return "", ""  # stale TMDB ID — definitively nothing there
            resp.raise_for_status()
            data = resp.json()
            overview = (data.get("overview") or "").strip()
            path = data.get("poster_path")
            poster_url = f"{POSTER_BASE}{path}" if path else ""
            return overview, poster_url
        except Exception:
            time.sleep(backoff)
            backoff *= 2
    return None


def _save_overviews(rows: list) -> None:
    df = pd.DataFrame(rows, columns=["movieId", "overview"])
    df = df.sort_values("movieId").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)


def main() -> None:
    api_key = _api_key()
    if not api_key:
        print("Error: TMDB_API_KEY not found (env var or .streamlit/secrets.toml).")
        sys.exit(1)

    links = pd.read_csv(LINKS_CSV).dropna(subset=["tmdbId"])
    links["movieId"] = links["movieId"].astype(int)
    links["tmdbId"] = links["tmdbId"].astype(int)

    # Resume: skip movies whose overview we already have.
    overview_rows: list = []
    have_overview: set = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        existing["overview"] = existing["overview"].fillna("")
        overview_rows = existing.to_dict("records")
        have_overview = set(existing["movieId"].astype(int))
        print(f"Resuming: {len(have_overview)} overviews already in {OUT_CSV.name}")

    # Poster backfill state (same endpoint, free data — never re-fetch).
    poster_rows: list = []
    have_poster: set = set()
    if POSTER_CSV.exists():
        pexist = pd.read_csv(POSTER_CSV)
        poster_rows = pexist.to_dict("records")
        have_poster = set(pexist["movieId"].astype(int))

    to_fetch = links[~links["movieId"].isin(have_overview)]
    total = len(to_fetch)
    if total == 0:
        print("Nothing to fetch — all movies already have overview entries.")
        return

    print(f"Fetching overviews for {total} movies across {MAX_WORKERS} workers...")

    tmdb_by_movie = dict(zip(to_fetch["movieId"], to_fetch["tmdbId"]))
    new_overviews: list = []
    new_posters = 0
    failed = 0
    done = 0

    def _checkpoint() -> None:
        _save_overviews(overview_rows + new_overviews)
        _save_posters(poster_rows)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_details, int(row["tmdbId"]), api_key): int(row["movieId"])
            for _, row in to_fetch.iterrows()
        }
        for future in as_completed(futures):
            movie_id = futures[future]
            result = future.result()
            done += 1

            if result is None:
                failed += 1  # leave out of CSV; a re-run will retry it
            else:
                overview, poster_url = result
                new_overviews.append({"movieId": movie_id, "overview": overview})
                # Backfill the poster table only where it has no entry yet.
                if movie_id not in have_poster and poster_url:
                    poster_rows.append({
                        "movieId": movie_id,
                        "tmdbId": int(tmdb_by_movie[movie_id]),
                        "poster_url": poster_url,
                    })
                    have_poster.add(movie_id)
                    new_posters += 1

            if done % CHECKPOINT_EVERY == 0:
                _checkpoint()
                got = sum(1 for r in new_overviews if r["overview"])
                print(f"  {done}/{total} — checkpoint  ({got} with overview, "
                      f"{new_posters} posters backfilled, {failed} to retry)")

    _checkpoint()

    all_rows = overview_rows + new_overviews
    got = sum(1 for r in all_rows if r.get("overview"))
    print(f"\nDone. {got}/{len(all_rows)} movies have overview text.")
    print(f"Backfilled {new_posters} poster URLs into {POSTER_CSV.name}.")
    if failed:
        print(f"{failed} request(s) failed after retries — re-run to fill them in.")
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
