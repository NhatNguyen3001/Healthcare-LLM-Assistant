"""Preprocessing - runs after Railguard 1, feeds NER / sentiment / classifier.

- Strip HTML (carried over from data-cleaning notes — left to this phase)
- Expand common, low-ambiguity medical abbreviations
- Collapse whitespace

Case and punctuation are preserved so scispaCy NER sees natural text.
Sentiment / classifier vectorizers handle their own lowercasing.
"""
import re

import spacy

# Conservative medical abbreviation map — case-sensitive matching to avoid
# accidental hits on lowercase tokens. Add new entries only if they're
# unambiguous in clinical context.
_ABBREVIATIONS = {
    "HTN":  "hypertension",
    "DM":   "diabetes mellitus",
    "T1DM": "type 1 diabetes mellitus",
    "T2DM": "type 2 diabetes mellitus",
    "MI":   "myocardial infarction",
    "CHF":  "congestive heart failure",
    "COPD": "chronic obstructive pulmonary disease",
    "CAD":  "coronary artery disease",
    "CKD":  "chronic kidney disease",
    "GERD": "gastroesophageal reflux disease",
    "URI":  "upper respiratory infection",
    "UTI":  "urinary tract infection",
    "DVT":  "deep vein thrombosis",
    "SOB":  "shortness of breath",
    "CXR":  "chest x-ray",
    "ECG":  "electrocardiogram",
    "EKG":  "electrocardiogram",
    "BP":   "blood pressure",
    "HR":   "heart rate",
    "RR":   "respiratory rate",
    "Pt":   "patient",
    "Hx":   "history",
    "Dx":   "diagnosis",
    "Tx":   "treatment",
    "Rx":   "prescription",
    "Fx":   "fracture",
    "Sx":   "symptoms",
}

_ABBR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ABBREVIATIONS) + r")\b"
)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_WS   = re.compile(r"\s+")

# Reuse a single shared pipeline for sentence-split + token/lemma helpers.
# NER is disabled here — scispaCy `en_ner_bc5cdr_md` runs in src/pipeline/ner.py.
_nlp = spacy.load("en_core_web_sm", disable=["ner"])


def expand_abbreviations(text: str) -> str:
    return _ABBR_PATTERN.sub(lambda m: _ABBREVIATIONS[m.group(1)], text)


def preprocess(text: str) -> str:
    """Clean text for downstream NLP. Preserves case + punctuation."""
    text = _RE_HTML.sub(" ", text)
    text = expand_abbreviations(text)
    text = _RE_WS.sub(" ", text).strip()
    return text


def sentences(text: str) -> list[str]:
    """Sentence-segment via spaCy. Run on already-preprocessed text."""
    doc = _nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def tokens(text: str) -> list[str]:
    """Lowercased lemma tokens with punct / whitespace / stopwords removed.
    Used for TF-IDF prep in sentiment / classifier training."""
    doc = _nlp(text)
    return [
        t.lemma_.lower()
        for t in doc
        if not (t.is_punct or t.is_space or t.is_stop)
    ]


if __name__ == "__main__":
    sample = (
        "<p>Patient has HTN and DM with SOB on exertion.</p> "
        "CXR was normal. Pt reports Hx of MI in 2018. "
        "BP 145/92, HR 78. Plan: continue Rx, follow up in 2 weeks."
    )
    cleaned = preprocess(sample)
    print("INPUT:    ", sample)
    print("CLEANED:  ", cleaned)
    print("SENTENCES:", sentences(cleaned))
    print("TOKENS:   ", tokens(cleaned))
