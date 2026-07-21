from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

SENTIMENT_MODEL_PATH = Path("models/sentiment/ft_model")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    SENTIMENT_MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL_PATH
)

model.to(DEVICE)
model.eval()

LABEL_MAP = {
    0: "negative",
    1: "positive",
}

import torch.nn.functional as F

def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for a single piece of text.

    Returns:
        {
            "label": "positive" | "negative",
            "confidence": float
        }
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)

    return {
        "label": LABEL_MAP[prediction.item()],
        "confidence": round(confidence.item(), 4),
    }

def predict_sentiments(texts, batch_size=32):
    """
    Predict sentiment for multiple texts efficiently.

    Returns:
        List[dict]
    """

    results = []

    for i in range(0, len(texts), batch_size):

        batch = [
            "" if text is None else str(text)
            for text in texts[i : i + batch_size]
        ]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = F.softmax(outputs.logits, dim=1)

        confidences, predictions = torch.max(
            probabilities,
            dim=1,
        )

        for pred, conf in zip(predictions, confidences):
            results.append(
                {
                    "label": LABEL_MAP[pred.item()],
                    "confidence": round(conf.item(), 4),
                }
            )

    return results
