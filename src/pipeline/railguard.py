"""Railguard — strip/generalize PII before pipeline processing

This is the only module that sees raw PII. Everything downstream operates
on the railguarded output.

- Remove: names, phone, email, IDs, addresses
- Generalize: exact age -> age band, exact date -> relative time
- Keep: gender, symptoms, medications
"""
import re
from datetime import datetime

import spacy
from dateutil import parser as date_parser

from src.utils.config import AGE_BANDS

_nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "parser"])

# Order matters: more specific patterns first to avoid collisions
_RE_EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_RE_ID    = re.compile(r"\b(?:medicare|patient|mrn|id)\s*[#:]?\s*\d{4,}\b", re.IGNORECASE)
_RE_PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?0?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
_RE_AGE_YO = re.compile(r"\b(\d{1,3})[\s-]?year[\s-]?old\b", re.IGNORECASE)
_RE_AGE_ED = re.compile(r"\baged?\s+(\d{1,3})\b", re.IGNORECASE)
_RE_DATE = re.compile(
    r"\b(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)


def _age_band(age: int) -> str:
    for (lo, hi), label in AGE_BANDS.items():
        if lo <= age <= hi:
            return label
    return "patient"


def _relative_date(date_str: str) -> str:
    try:
        d = date_parser.parse(date_str, fuzzy=False)
    except (ValueError, OverflowError):
        return "[DATE]"
    days = (datetime.now() - d).days
    if days < 0:
        return "[DATE]"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    if days < 730:
        return f"{days // 30} months ago"
    return f"{days // 365} years ago"


def railguard(text: str) -> str:
    """Strip/generalize PII. Used identically for railguard 1 (pre-preprocessing)
    and railguard 2 (post-LLM-generation)."""
    text = _RE_EMAIL.sub("[EMAIL]", text)
    text = _RE_ID.sub("[ID]", text)
    text = _RE_PHONE.sub("[PHONE]", text)
    text = _RE_DATE.sub(lambda m: _relative_date(m.group()), text)
    text = _RE_AGE_YO.sub(lambda m: _age_band(int(m.group(1))), text)
    text = _RE_AGE_ED.sub(lambda m: _age_band(int(m.group(1))), text)

    doc = _nlp(text)
    spans = sorted(
        (
            (ent.start_char, ent.end_char, ent.label_)
            for ent in doc.ents
            if ent.label_ in {"PERSON", "GPE", "LOC"}
        ),
        reverse=True,
    )
    for start, end, label in spans:
        repl = "[PATIENT NAME]" if label == "PERSON" else "[LOCATION]"
        text = text[:start] + repl + text[end:]

    return text


# Aliases per pipeline naming
railguard_1 = railguard


if __name__ == "__main__":
    sample = (
        "John Smith, a 23-year-old male, called 0412 345 678 on April 20, 2013. "
        "He emailed john@example.com from Sydney. Medicare: 1234567890. "
        "He reports chest pain and is taking aspirin."
    )
    print("INPUT: ", sample)
    print("OUTPUT:", railguard(sample))
