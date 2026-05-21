"""Sentiment analysis wrapper - scikit-learn LogReg + TF-IDF.

Loads a pre-trained pipeline from `SENTIMENT_MODEL_PATH`, retrains
this on UCI Drug Reviews via `scripts/train_sentiment.py` (stratified 80/20
on the cleaned drug-review files). The wrapper itself is model-agnostic —
it just expects an sklearn classifier with `predict_proba` + `classes_`.

"""
import joblib

from src.utils.config import SENTIMENT_MODEL_PATH

_pipeline = None


def _load():
    global _pipeline
    if _pipeline is None:
        if not SENTIMENT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{SENTIMENT_MODEL_PATH} not found. Run "
                f"`python -m src.pipeline.sentiment` for a dummy, or "
                f"`scripts/train_sentiment.py` (Phase 4) for the real model."
            )
        _pipeline = joblib.load(SENTIMENT_MODEL_PATH)
    return _pipeline


def predict(text: str) -> dict:
    """Return {"label", "score"} for one input. Score is the probability
    of the predicted class."""
    pipe = _load()
    proba = pipe.predict_proba([text])[0]
    idx = int(proba.argmax())
    return {"label": str(pipe.classes_[idx]), "score": float(proba[idx])}


def predict_batch(texts: list[str]) -> list[dict]:
    pipe = _load()
    probas = pipe.predict_proba(texts)
    idxs = probas.argmax(axis=1)
    return [
        {"label": str(pipe.classes_[i]), "score": float(probas[r, i])}
        for r, i in enumerate(idxs)
    ]


def _train_dummy():
    """Tiny placeholder pipeline so Phase 3 wrappers are exercisable.
    Phase 4 overwrites this file with the real Drug-Reviews-trained model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    texts = [
        "this drug worked great, no side effects, would recommend",
        "amazing relief from my symptoms, life changing",
        "completely cured my condition, excellent medication",
        "horrible side effects, made me very sick",
        "useless, did nothing for me, waste of money",
        "worst medication I have ever taken, terrible",
        "okay, not great but tolerable, mixed results",
        "kind of helped but had some nausea, average",
        "neither good nor bad, neutral experience overall",
    ]
    labels = ["positive"] * 3 + ["negative"] * 3 + ["neutral"] * 3

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf",   LogisticRegression(max_iter=200)),
    ])
    pipe.fit(texts, labels)
    SENTIMENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, SENTIMENT_MODEL_PATH)
    return pipe


if __name__ == "__main__":
    if not SENTIMENT_MODEL_PATH.exists():
        print(f"Training dummy -> {SENTIMENT_MODEL_PATH}")
        _train_dummy()
    for s in [
        "this medication completely changed my life for the better",
        "awful drug, gave me migraines and stomach pain",
        "did not really notice any difference one way or another",
    ]:
        print(f"  {s!r}\n    -> {predict(s)}")