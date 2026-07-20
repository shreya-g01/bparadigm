"""
reddit_loader.py

Utilities for loading Reddit submission archives (.zst),
filtering relevant subreddits, and exporting a structured CSV.

Workflow
--------
RS_2019-04.zst
    ↓
load_reddit_dataset()
    ↓
pandas.DataFrame
    ↓
reddit_raw.csv
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Generator

import pandas as pd
import zstandard as zstd
from tqdm import tqdm


DEFAULT_SAMPLE_SIZE = 20_000

TARGET_SUBREDDITS = {
    "apple",
    "android",
    "gadgets",
    "googlepixel",
    "oneplus",
    "samsung",
    "technology",
}

OUTPUT_COLUMNS = [
    "title",
    "selftext",
    "text",
    "subreddit",
    "score",
    "created_utc",
]

def reddit_stream(input_file: Path) -> Generator[dict, None, None]:
    """
    Stream a Reddit .zst archive one submission at a time.

    Parameters
    ----------
    input_file : Path
        Path to the Reddit submissions archive.

    Yields
    ------
    dict
        One Reddit submission represented as a Python dictionary.
    """

    with open(input_file, "rb") as file:

        decompressor = zstd.ZstdDecompressor()

        with decompressor.stream_reader(file) as reader:

            text_stream = io.TextIOWrapper(
                reader,
                encoding="utf-8"
            )

            for line in text_stream:

                try:

                    yield json.loads(line)

                except json.JSONDecodeError:

                    continue

from pathlib import Path

from bparadigm.reddit_loader import reddit_stream

INPUT_FILE = Path("../raw_data/reddit/RS_2019-04.zst")

stream = reddit_stream(INPUT_FILE)

first_post = next(stream)

print(type(first_post))
print(first_post["subreddit"])
print(first_post["title"])

def filter_post(
    post: dict,
    target_subreddits: set[str],
) -> dict | None:
    """
    Validate and filter a Reddit submission.

    Parameters
    ----------
    post : dict
        Reddit submission.

    target_subreddits : set[str]
        Allowed subreddit names.

    Returns
    -------
    dict | None

        Returns a cleaned dictionary if the post is valid.
        Returns None if the post should be discarded.
    """

    subreddit = post.get("subreddit", "").lower()

    if subreddit not in target_subreddits:
        return None

    title = str(post.get("title", "")).strip()
    selftext = str(post.get("selftext", "")).strip()

    return {
        "title": title,
        "selftext": selftext,
        "subreddit": subreddit,
        "score": post.get("score", 0),
        "created_utc": post.get("created_utc"),
    }
def load_reddit_dataset(
    input_file: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    target_subreddits: set[str] = TARGET_SUBREDDITS,
) -> pd.DataFrame:
    """
    Load a Reddit .zst archive into a pandas DataFrame.

    Parameters
    ----------
    input_file : Path
        Path to Reddit archive.

    sample_size : int
        Maximum number of posts to return.

    target_subreddits : set[str]
        Subreddits to keep.

    Returns
    -------
    pandas.DataFrame
    """

    rows = []

    scanned_posts = 0
    matched_posts = 0

    for post in tqdm(
        reddit_stream(input_file),
        desc="Scanning Reddit archive",
    ):

        scanned_posts += 1

        cleaned_post = filter_post(
            post,
            target_subreddits,
        )

        if cleaned_post is None:
            continue

        matched_posts += 1

        rows.append(cleaned_post)

        if len(rows) >= sample_size:
            break

    dataframe = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Posts scanned     : {scanned_posts:,}")
    print(f"Posts matched     : {matched_posts:,}")
    print(f"Rows returned     : {len(dataframe):,}")
    print("-" * 40)

    return dataframe

def save_reddit_dataset(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Save Reddit dataframe to CSV.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print(f"\nDataset saved to:\n{output_file}")
