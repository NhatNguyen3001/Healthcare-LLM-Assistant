# Models

Trained classical ML models live here as `.pkl` files. All are gitignored — regenerate locally with the training scripts.

| File | Trained by | Source data |
|---|---|---|
| `sentiment_model.pkl` | `scripts/train_sentiment.py` | UCI Drug Reviews |
| `classifier_model.pkl` | `scripts/train_classifier.py` | Medical Transcriptions |

Models are loaded once at app startup. Never retrain at runtime.
