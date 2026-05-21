"""train medical specialty classifier on Medical Transcriptions.

Loads the cleaned mtsamples CSV (one row per transcription, ~4,966 rows,
40 specialties), concatenates `text + description + keywords` for richer
per-row signal, preprocesses via `src.pipeline.preprocessing`, drops
specialties with fewer than `MIN_SAMPLES_PER_CLASS` rows, encodes each
input with `src.pipeline.embeddings` (S-PubMedBert, 768-dim L2-normalised
vectors), runs a stratified 80/20 split by specialty, fits a plain
LogisticRegression on the dense vectors, prints per-class metrics on the
held-out test split, and joblib-dumps the fitted model to
`CLASSIFIER_MODEL_PATH` (consumed by `src/pipeline/classifier.py`, which
encodes incoming text via the same embeddings module before predicting).

Run from the project root:
    python scripts/train_classifier.py
"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.pipeline import embeddings
from src.pipeline.preprocessing import preprocess
from src.utils.config import CLASSIFIER_MODEL_PATH, RAG_SOURCES, RESULTS_DIR

# Drop specialties with fewer than this many rows. Stratified split needs
# at least 2; <10 gives metrics that are noise. Tune if you want to keep
# more long-tail classes at the cost of test reliability.
MIN_SAMPLES_PER_CLASS = 10
RANDOM_STATE = 42

# Document-type and overly-generic labels in mtsamples that aren't real
# medical specialties — they're note formats ("Consult - History and Phy."),
# admin doc types ("Letters", "IME-QME"), venues ("Emergency Room Reports"),
# or vague catch-alls ("Surgery", "General Medicine"). They absorb content
# from real specialties (a cardiac consult is labeled "Consult", not
# "Cardiology"), making the classifier collapse predictions onto them.
# Filtered out before training so the model focuses on actual specialty
# signal. ~47% of mtsamples rows.
DROP_LABELS = {
    "Surgery",
    "Consult - History and Phy.",
    "General Medicine",
    "SOAP / Chart / Progress Notes",
    "Discharge Summary",
    "Emergency Room Reports",
    "Office Notes",
    "Letters",
    "IME-QME-Work Comp etc.",
}


def _write_report(
    n_rows_after: int,
    n_classes: int,
    drop_classes: int,
    drop_rows: int,
    n_train: int,
    n_test: int,
    train_time: float,
    y_true,
    y_pred,
    accuracy: float,
) -> Path:
    """Write a markdown training report to results/training_classifier.md."""
    out = RESULTS_DIR / "training_classifier.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true, y_pred, digits=4, output_dict=True, zero_division=0
    )
    labels = [
        k for k in report if k not in ("accuracy", "macro avg", "weighted avg")
    ]
    # Sort by support descending so head specialties appear first.
    labels.sort(key=lambda c: -report[c]["support"])

    md: list[str] = []
    md.append("# Classifier Training Report")
    md.append("")
    md.append(f"- **Run date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **Source file**: `{RAG_SOURCES['medical_transcriptions']}`")
    md.append(
        f"- **Non-specialty labels dropped**: {len(DROP_LABELS)} "
        "(document-types and generics — see DROP_LABELS in train_classifier.py): "
        + ", ".join(f"`{x}`" for x in sorted(DROP_LABELS))
    )
    md.append(f"- **Min samples per class**: {MIN_SAMPLES_PER_CLASS}")
    md.append(
        f"- **Specialties dropped**: {drop_classes} ({drop_rows:,} rows below threshold)"
    )
    md.append(f"- **Rows after filter**: {n_rows_after:,}")
    md.append(f"- **Classes after filter**: {n_classes}")
    md.append(f"- **Train rows**: {n_train:,}")
    md.append(f"- **Test rows**: {n_test:,}")
    md.append(
        f"- **Split**: stratified 80/20 by specialty (random_state={RANDOM_STATE})"
    )
    md.append(
        "- **Input fields**: `text + description + keywords` (concatenated)"
    )
    md.append(
        "- **Encoder**: S-PubMedBert (`pritamdeka/S-PubMedBert-MS-MARCO`, 768d, L2-normalised)"
    )
    md.append(
        "- **Head**: MLPClassifier(hidden=(256,), activation=relu, "
        "solver=adam, early_stopping=True, max_iter=200, random_state=42)"
    )
    md.append(f"- **Train time**: {train_time:.1f}s")
    md.append(f"- **Saved model**: `{CLASSIFIER_MODEL_PATH}`")
    md.append(f"- **Test accuracy**: **{accuracy:.4f}**")
    md.append("")

    md.append("## Per-class metrics (test split)")
    md.append("")
    md.append("Sorted by support (descending). 40-class problem with a long tail "
              "— expect low F1 on small specialties.")
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

    out.write_text("\n".join(md), encoding="utf-8")
    return out


def main() -> None:
    src = RAG_SOURCES["medical_transcriptions"]
    print(f"Loading {src}")
    df = pd.read_csv(src)
    print(f"Rows: {len(df):,} | Specialties: {df['specialty'].nunique()}")

    before = len(df)
    df = df[~df["specialty"].isin(DROP_LABELS)].reset_index(drop=True)
    dropped_rows = before - len(df)
    print(
        f"Dropped {dropped_rows:,} rows from {len(DROP_LABELS)} non-specialty "
        f"labels (Surgery, Consult, SOAP, ...) -> {len(df):,} rows | "
        f"{df['specialty'].nunique()} specialties remain"
    )

    counts = df["specialty"].value_counts()
    keep = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    drop_classes = (counts < MIN_SAMPLES_PER_CLASS).sum()
    drop_rows = (~df["specialty"].isin(keep)).sum()
    if drop_rows:
        print(
            f"Dropping {drop_rows} rows from {drop_classes} specialties "
            f"with <{MIN_SAMPLES_PER_CLASS} samples"
        )
        df = df[df["specialty"].isin(keep)].reset_index(drop=True)

    df = df.dropna(subset=["text"]).copy()
    print(f"After cleanup: {len(df):,} rows | {df['specialty'].nunique()} classes")

    print("Building input (text + description + keywords) and preprocessing ...")
    t0 = time.time()
    raw = (
        df["text"].fillna("").astype(str) + " "
        + df["description"].fillna("").astype(str) + " "
        + df["keywords"].fillna("").astype(str)
    )
    df["clean"] = raw.map(preprocess)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Encoding inputs with S-PubMedBert (this is the slow step) ...")
    t0 = time.time()
    X = embeddings.encode(df["clean"].tolist(), show_progress_bar=True)
    encode_time = time.time() - t0
    print(f"  encoded {X.shape[0]:,} inputs -> shape {X.shape} in {encode_time:.1f}s")

    # MLPClassifier(early_stopping=True) hits a sklearn bug with string
    # labels: its internal validation-set scorer calls np.isnan(y_pred),
    # which fails on object arrays. Workaround: encode labels to ints,
    # restore string classes_ after fit so the wrapper sees specialty names.
    le = LabelEncoder()
    y_int = le.fit_transform(df["specialty"].tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_int,
        test_size=0.2,
        stratify=y_int,
        random_state=RANDOM_STATE,
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}\n")

    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        solver="adam",
        early_stopping=True,
        max_iter=200,
        random_state=RANDOM_STATE,
    )

    print("Fitting MLPClassifier on PubMedBert embeddings ...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  done in {train_time:.1f}s")

    # Swap integer classes_ for the original string specialty names. The
    # wrapper does `classes_[predict_proba.argmax()]` so this gives back
    # specialty strings. (Note: clf.predict() still returns ints because
    # MLPClassifier uses an internal _label_binarizer fit on integer
    # labels — we use predict_proba + argmax + classes_ instead.)
    clf.classes_ = le.classes_

    print("\nEvaluating on test split ...")
    y_test_str = le.inverse_transform(y_test)
    proba = clf.predict_proba(X_test)
    y_pred = clf.classes_[proba.argmax(axis=1)]
    accuracy = accuracy_score(y_test_str, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test_str, y_pred, digits=4, zero_division=0))

    CLASSIFIER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLASSIFIER_MODEL_PATH)
    print(f"\nSaved model  -> {CLASSIFIER_MODEL_PATH}")

    report_path = _write_report(
        n_rows_after=len(df),
        n_classes=df["specialty"].nunique(),
        drop_classes=int(drop_classes),
        drop_rows=int(drop_rows),
        n_train=len(X_train),
        n_test=len(X_test),
        train_time=train_time,
        y_true=y_test_str,
        y_pred=y_pred,
        accuracy=accuracy,
    )
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
