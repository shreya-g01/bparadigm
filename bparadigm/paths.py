# bparadigm/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA     = PROJECT_ROOT / "raw_data"
MODELS       = PROJECT_ROOT / "models"
REDDIT_RAW   = RAW_DATA / "reddit" / "reddit_raw.csv"
AMAZON_RAW = RAW_DATA / "Amazon_Reviews.csv"

PROCESSED     = RAW_DATA / "processed"
TWEETEVAL_TEST = RAW_DATA / "tweeteval" / "sentiment_test.csv"

AMAZON_CLEAN  = PROCESSED / "amazon_clean.csv"
TRAIN_CSV     = PROCESSED / "train.csv"
TEST_CSV      = PROCESSED / "test.csv"

BASELINE_DIR      = MODELS / "baseline"
BASELINE_MODEL    = BASELINE_DIR / "model.joblib"
BASELINE_METRICS  = BASELINE_DIR / "metrics.json"
TWEETEVAL_METRICS = BASELINE_DIR / "tweeteval_metrics.json"
