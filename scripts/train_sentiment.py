"""train sentiment model on UCI Drug Reviews.

Reads the pre-split cleaned files from `SENTIMENT_TRAINING_DATA`
(train + test; validation reserved for future hyperparameter tuning),
bins the 1-10 rating into 3 classes, preprocesses text via
`src.pipeline.preprocessing.preprocess`, fits a TfidfVectorizer +
LogisticRegression pipeline, prints per-class metrics on the held-out
test split, and joblib-dumps the fitted pipeline to
`SENTIMENT_MODEL_PATH` (consumed by `src/pipeline/sentiment.py`).

Run from the project root:
    python scripts/train_sentiment.py

"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.pipeline.preprocessing import preprocess
from src.utils.config import (
    RESULTS_DIR,
    SENTIMENT_MODEL_PATH,
    SENTIMENT_TRAINING_DATA,
)


# Rating-to-class binning. Edit here if you want different thresholds.
# 1-4 -> negative, 5-6 -> neutral, 7-10 -> positive.
def bin_rating(r: int) -> str:
    if r <= 4:
        return "negative"
    if r <= 6:
        return "neutral"
    return "positive"


def load_split(path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    df = df.dropna(subset=["review", "rating"]).copy()
    df["label"] = df["rating"].astype(int).map(bin_rating)
    texts = [preprocess(t) for t in df["review"].astype(str).tolist()]
    return texts, df["label"].tolist()


def _write_report(
    n_train: int,
    n_test: int,
    train_dist: dict,
    test_dist: dict,
    train_time: float,
    y_true,
    y_pred,
    accuracy: float,
) -> Path:
    """Write a markdown training report to results/training_sentiment.md."""
    out = RESULTS_DIR / "training_sentiment.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    labels = sorted(
        k for k in report if k not in ("accuracy", "macro avg", "weighted avg")
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    md: list[str] = []
    md.append("# Sentiment Training Report")
    md.append("")
    md.append(f"- **Run date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **Train file**: `{SENTIMENT_TRAINING_DATA['train']}`")
    md.append(f"- **Test file**: `{SENTIMENT_TRAINING_DATA['test']}`")
    md.append(f"- **Train rows**: {n_train:,}")
    md.append(f"- **Test rows**: {n_test:,}")
    md.append("- **Binning**: 1-4 -> negative, 5-6 -> neutral, 7-10 -> positive")
    md.append(
        "- **Model**: TF-IDF (1-2 grams, min_df=5, max_df=0.95, sublinear_tf) "
        "+ LogisticRegression(class_weight=balanced, max_iter=1000)"
    )
    md.append(f"- **Train time**: {train_time:.1f}s")
    md.append(f"- **Saved model**: `{SENTIMENT_MODEL_PATH}`")
    md.append(f"- **Test accuracy**: **{accuracy:.4f}**")
    md.append("")

    md.append("## Class distribution")
    md.append("")
    md.append("| Label | Train | Test |")
    md.append("|---|---:|---:|")
    for label in labels:
        md.append(
            f"| {label} | {train_dist.get(label, 0):,} | {test_dist.get(label, 0):,} |"
        )
    md.append("")

    md.append("## Per-class metrics (test split)")
    md.append("")
    md.append("| Class | Precision | Recall | F1 | Support |")
    md.append("|---|---:|---:|---:|---:|")
    for label in labels:
        r = report[label]
        md.append(
            f"| {label} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1-score']:.4f} | {int(r['support']):,} |"
        )
    for avg in ("macro avg", "weighted avg"):
        r = report[avg]
        md.append(
            f"| **{avg}** | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1-score']:.4f} | {int(r['support']):,} |"
        )
    md.append("")

    md.append("## Confusion matrix")
    md.append("")
    md.append("Rows = actual, columns = predicted.")
    md.append("")
    md.append("| actual \\ predicted | " + " | ".join(labels) + " |")
    md.append("|---|" + "|".join(["---:"] * len(labels)) + "|")
    for i, label in enumerate(labels):
        row = " | ".join(str(int(x)) for x in cm[i])
        md.append(f"| {label} | {row} |")
    md.append("")

    out.write_text("\n".join(md), encoding="utf-8")
    return out


def main() -> None:
    print(f"Loading train split: {SENTIMENT_TRAINING_DATA['train']}")
    X_train, y_train = load_split(SENTIMENT_TRAINING_DATA["train"])
    print(f"Loading test split:  {SENTIMENT_TRAINING_DATA['test']}")
    X_test, y_test = load_split(SENTIMENT_TRAINING_DATA["test"])

    print(
        f"\nTrain: {len(X_train):,} reviews | "
        f"label dist: {pd.Series(y_train).value_counts().to_dict()}"
    )
    print(
        f"Test:  {len(X_test):,} reviews | "
        f"label dist: {pd.Series(y_test).value_counts().to_dict()}\n"
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    print("Fitting LogReg + TF-IDF ...")
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  done in {train_time:.1f}s")

    print("\nEvaluating on test split ...")
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    SENTIMENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, SENTIMENT_MODEL_PATH)
    print(f"\nSaved model  -> {SENTIMENT_MODEL_PATH}")

    train_dist = pd.Series(y_train).value_counts().to_dict()
    test_dist = pd.Series(y_test).value_counts().to_dict()
    report_path = _write_report(
        n_train=len(X_train),
        n_test=len(X_test),
        train_dist=train_dist,
        test_dist=test_dist,
        train_time=train_time,
        y_true=y_test,
        y_pred=y_pred,
        accuracy=accuracy,
    )
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
