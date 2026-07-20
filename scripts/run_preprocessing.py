"""
run_preprocessing.py

Preprocess the Reddit dataset.

Workflow
--------
reddit_raw.csv
        ↓
Cleaning
        ↓
reddit_clean.csv
"""

from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "reddit"
    / "reddit_raw.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "processed"
    / "reddit"
    / "reddit_clean.csv"
)


# ------------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------------

print("Loading Reddit dataset...")

df = pd.read_csv(INPUT_FILE)

# Store the original row count before any preprocessing
original_rows = len(df)

print(f"Loaded {original_rows:,} rows")


# ------------------------------------------------------------------
# Remove duplicate posts
# ------------------------------------------------------------------

df = df.drop_duplicates()

print(f"Rows after duplicate removal: {len(df):,}")


# ------------------------------------------------------------------
# Fill missing values
# ------------------------------------------------------------------

df["title"] = df["title"].fillna("")
df["selftext"] = df["selftext"].fillna("")


# ------------------------------------------------------------------
# Remove deleted / removed posts
# ------------------------------------------------------------------

df = df[
    ~df["selftext"].isin(
        [
            "[removed]",
            "[deleted]",
        ]
    )
]


# ------------------------------------------------------------------
# Clean whitespace
# ------------------------------------------------------------------

df["title"] = (
    df["title"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["selftext"] = (
    df["selftext"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ------------------------------------------------------------------
# Create text column
# ------------------------------------------------------------------

df["text"] = (
    df["title"]
    + " "
    + df["selftext"]
)

df["text"] = (
    df["text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ------------------------------------------------------------------
# Remove empty text
# ------------------------------------------------------------------

df = df[df["text"] != ""]


# ------------------------------------------------------------------
# Remove very short posts
# ------------------------------------------------------------------

MIN_TEXT_LENGTH = 10

df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]


# ------------------------------------------------------------------
# Keep required columns
# ------------------------------------------------------------------

df = df[
    [
        "title",
        "selftext",
        "text",
        "subreddit",
        "score",
        "created_utc",
    ]
]


# ------------------------------------------------------------------
# Reset index
# ------------------------------------------------------------------

df = df.reset_index(drop=True)


# ------------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------------
# Save cleaned dataset
# ------------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()

print("=" * 60)
print("Preprocessing Complete")
print("=" * 60)

print(f"Original rows        : {original_rows:,}")
print(f"Rows after cleaning  : {len(df):,}")
print(f"Rows removed         : {original_rows - len(df):,}")
print(f"Saved file           : {OUTPUT_FILE}")

print("=" * 60)
