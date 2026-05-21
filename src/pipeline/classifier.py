"""Medical specialty classifier — S-PubMedBert embeddings + MLP head.

Loads a pre-trained sklearn classifier from `CLASSIFIER_MODEL_PATH`.
retrains it on Medical Transcriptions via
`scripts/train_classifier.py` (encodes `text + description + keywords`
with PubMedBert, fits an MLPClassifier on the dense vectors). The wrapper
encodes incoming text via `src.pipeline.embeddings` before predicting,
so the saved model expects 768-dim L2-normalised input vectors — NOT raw
text.

The wrapper is otherwise model-agnostic — it just expects any sklearn
classifier with `predict_proba` + `classes_`. Swap MLP for SVC/XGBoost
in the training script and the wrapper keeps working.

"""
import joblib

from src.pipeline import embeddings
from src.utils.config import CLASSIFIER_MODEL_PATH

_pipeline = None


def _load():
    global _pipeline
    if _pipeline is None:
        if not CLASSIFIER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{CLASSIFIER_MODEL_PATH} not found. Run "
                f"`python -m src.pipeline.classifier` for a dummy, or "
                f"`scripts/train_classifier.py` (Phase 4) for the real model."
            )
        _pipeline = joblib.load(CLASSIFIER_MODEL_PATH)
    return _pipeline


def predict(text: str) -> dict:
    """Return {"label", "score"} for one input. Label is the predicted
    medical specialty; score is the probability of that class.
    Encodes the input via PubMedBert before classification."""
    clf = _load()
    vec = embeddings.encode(text)
    proba = clf.predict_proba(vec.reshape(1, -1))[0]
    idx = int(proba.argmax())
    return {"label": str(clf.classes_[idx]), "score": float(proba[idx])}


def predict_batch(texts: list[str]) -> list[dict]:
    clf = _load()
    vecs = embeddings.encode(texts)
    probas = clf.predict_proba(vecs)
    idxs = probas.argmax(axis=1)
    return [
        {"label": str(clf.classes_[i]), "score": float(probas[r, i])}
        for r, i in enumerate(idxs)
    ]


def _train_dummy():
    """Tiny placeholder model so Phase 3 wrappers are exercisable.
    Phase 4 overwrites this file with the real mtsamples-trained model.
    Encodes via PubMedBert just like the real model so the wrapper's
    encode-then-predict path is exercised."""
    from sklearn.neural_network import MLPClassifier

    texts = [
        "patient presents with chest pain and shortness of breath, ECG shows ST elevation",
        "history of myocardial infarction, on beta blockers, blood pressure controlled",
        "echocardiogram reveals reduced ejection fraction, congestive heart failure",
        "knee pain after fall, MRI shows ACL tear, scheduled for arthroscopic surgery",
        "lower back pain radiating down the leg, lumbar disc herniation suspected",
        "fracture of the distal radius, cast applied, follow up in 6 weeks",
        "patient reports headaches with visual aura, neurological exam normal",
        "seizure activity observed, EEG ordered, history of epilepsy in family",
        "tingling in extremities, suspect peripheral neuropathy, EMG scheduled",
    ]
    labels = (
        ["Cardiovascular / Pulmonary"] * 3
        + ["Orthopedic"] * 3
        + ["Neurology"] * 3
    )

    X = embeddings.encode(texts)
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    clf.fit(X, labels)
    CLASSIFIER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLASSIFIER_MODEL_PATH)
    return clf


if __name__ == "__main__":
    if not CLASSIFIER_MODEL_PATH.exists():
        print(f"Training dummy -> {CLASSIFIER_MODEL_PATH}")
        _train_dummy()
    for s in [
        "patient with crushing substernal chest pain radiating to left arm, diaphoretic",
        "twisted ankle playing soccer, swollen and painful, x-ray shows hairline fracture",
        "recurrent migraines with photophobia, has not responded to over-the-counter meds",
    ]:
        print(f"  {s!r}\n    -> {predict(s)}")
