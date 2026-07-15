"""
api/fast.py — FastAPI wrapper around the baseline predict() function.

Contains NO machine-learning logic. It imports predict()/predict_batch() from
bparadigm.sentiment and exposes them over HTTP. Because the prediction logic
(including the text cleaning) lives in ONE place, the API can't drift out of
sync with training. When the model changes, this file does not.

Run it (from the repo root):
    uvicorn api.fast:app --reload
Then open  http://localhost:8000/docs  for the interactive API docs.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from bparadigm.sentiment import predict, predict_batch, load_model

app = FastAPI(title="BParadigm Sentiment API")

# Warm the model once, at boot — not on the first user request. If the model
# file is missing, the server fails to start (fail fast) instead of erroring
# on a live user.
load_model()


@app.get("/")
def root():
    """Health check: confirms the service is up without running a prediction."""
    return {"status": "ok", "service": "sentiment-baseline"}


@app.get("/predict")
def predict_endpoint(text: str):
    """Score a single text. Example: /predict?text=great service"""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Provide non-empty 'text'.")
    return predict(text)


class BatchIn(BaseModel):
    texts: list[str]


@app.post("/predict_batch")
def predict_batch_endpoint(batch: BatchIn):
    """Score many texts at once. POST body: {"texts": ["...", "..."]}
    POST (not GET) because a list of long texts doesn't belong in a URL."""
    if not batch.texts:
        raise HTTPException(status_code=400, detail="'texts' must be a non-empty list.")
    return predict_batch(batch.texts)
