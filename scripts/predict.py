from pathlib import Path
import pandas as pd
from tqdm import tqdm
from bparadigm.sentiment_transformer import (
    predict_sentiment,
    predict_sentiments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "raw_data" / "processed" / "reddit" / "reddit_clean.csv"
OUTPUT_FILE = PROJECT_ROOT / "raw_data" / "processed" / "reddit" / "reddit_sentiment.csv"

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} posts.")

# Predict sentiment for every post

print(f"Loaded {len(df)} posts.")

texts = df["text"].fillna("").tolist()

predictions = []

BATCH_SIZE = 64

for i in tqdm(range(0, len(texts), BATCH_SIZE)):

    batch = texts[i:i + BATCH_SIZE]

    predictions.extend(
        predict_sentiments(batch, batch_size=BATCH_SIZE)
    )

df["sentiment"] = [
    p["label"]
    for p in predictions
]

df["confidence"] = [
    p["confidence"]
    for p in predictions
]

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(df[["sentiment", "confidence"]].head())

print(f"\nSaved predictions to:\n{OUTPUT_FILE}")
