from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- directories ---
RAW_DATA  = PROJECT_ROOT / "raw_data"
MODELS    = PROJECT_ROOT / "models"
PROCESSED = RAW_DATA / "processed"

# --- amazon ---
AMAZON_RAW   = RAW_DATA / "Amazon_Reviews.csv"
AMAZON_CLEAN = PROCESSED / "amazon_clean.csv"
TRAIN_CSV    = PROCESSED / "train.csv"
TEST_CSV     = PROCESSED / "test.csv"

# --- tweeteval ---
TWEETEVAL_TEST = RAW_DATA / "tweeteval" / "sentiment_test.csv"

# --- reddit ---
REDDIT_DIR       = RAW_DATA / "reddit"
REDDIT_RAW       = REDDIT_DIR / "reddit_raw.csv"
REDDIT_PROCESSED = REDDIT_DIR / "reddit_processed"
REDDIT_BRAND     = REDDIT_PROCESSED / "reddit_brand.csv"
REDDIT_TOPICS    = REDDIT_PROCESSED / "reddit_topics.csv"

# --- baseline model ---
BASELINE_DIR      = MODELS / "baseline"
BASELINE_MODEL    = BASELINE_DIR / "model.joblib"
BASELINE_METRICS  = BASELINE_DIR / "metrics.json"
TWEETEVAL_METRICS = BASELINE_DIR / "tweeteval_metrics.json"

# --- topic model ---
TOPIC_DIR     = MODELS / "topic_model"
TOPIC_RESULTS = PROCESSED / "topic_results.csv"
