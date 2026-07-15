"""
bparadigm/sentiment.py — baseline sentiment inference (the deployment seam).

One job: text in -> {label, confidence}. It loads the trained baseline pipeline
(TF-IDF + LogReg) and applies the SAME cleaning used at training time.

Everything else in the serving stack sits on top of this: the FastAPI endpoint,
a Streamlit demo, and the CLI below all call predict(). When the fine-tuned
transformer is ready, ONLY the internals of load_model()/predict() change —
callers never need to.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import joblib

# Path to the trained model. Computed from this file's location, not hardcoded,
# so it works on every teammate's machine. Point it at wherever model.joblib
# actually lives in the new repo (the models/*.pkl there are empty placeholders).
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline" / "model.joblib"

_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")
_LABELS = {0: "Negative", 1: "Positive"}


def clean_text(t: str) -> str:
    """IDENTICAL to notebook 1's cleaning. This is the load-bearing part:
    the model learned its vocabulary from text cleaned this exact way, so
    serving must clean the same way or predictions silently degrade."""
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = _URL.sub(" ", t)
    return _WS.sub(" ", t).strip()


_model = None  # module-level cache: load the pipeline once, not per request


def load_model(path: Path = MODEL_PATH):
    global _model
    if _model is None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No model at {path}. The trained baseline isn't in this repo yet — "
                f"copy model.joblib over, or re-run the training notebook, or fix MODEL_PATH."
            )
        _model = joblib.load(path)
    return _model


def predict(text: str) -> dict:
    """text -> {'label': 'Positive'|'Negative', 'label_id': 0|1, 'confidence': float}."""
    pipe = load_model()
    proba = pipe.predict_proba([clean_text(text)])[0]   # [P(neg), P(pos)]
    label_id = int(proba.argmax())
    return {
        "label": _LABELS[label_id],
        "label_id": label_id,
        "confidence": round(float(proba[label_id]), 4),
    }


def predict_batch(texts: list[str]) -> list[dict]:
    """Same as predict() but for many texts at once (one vectorization pass)."""
    pipe = load_model()
    probas = pipe.predict_proba([clean_text(t) for t in texts])
    results = []
    for p in probas:
        i = int(p.argmax())
        results.append({"label": _LABELS[i], "label_id": i, "confidence": round(float(p[i]), 4)})
    return results


if __name__ == "__main__":
    import sys
    txt = " ".join(sys.argv[1:]) or "this is the worst service I have ever used"
    print(predict(txt))
